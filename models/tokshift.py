import torch.nn as nn
from torch.nn.init import normal_, constant_
from models.basic_ops import ConsensusModule
import numpy as np
from torch.nn import Linear

from models.decoder import Decoder

class PositionWiseFeedForward(nn.Module):

    """
    w2(relu(w1(layer_norm(x))+b1))+b2
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output

class VideoNet(nn.Module):
	def __init__(self, num_class, with_decoder,
			  	sub_action_token_num,
				decoder_layer_num,
				with_spatial_conv,
				d_model=768,
			    modality='RGB',
				backbone='ViT-B_16', net='TokShift', consensus_type='avg',
				dropout=0.1, partial_bn=False, pretrain='imagenet',
				is_shift=True, shift_div=4,
			        drop_block=0, vit_img_size=224,
				vit_pretrain="", LayerNormFreeze=2, cfg=None):
		super(VideoNet, self).__init__()
		self.with_decoder = with_decoder
		self.sub_action_token_num = sub_action_token_num
		self.decoder_layer_num = decoder_layer_num
		self.with_spatial_conv = with_spatial_conv
		self.modality = modality
		self.backbone = backbone
		self.net = net
		self.dropout = dropout
		self.pretrain = pretrain
		self.consensus_type = consensus_type
		self.drop_block = drop_block
		self.init_crop_size = 256
		self.vit_img_size=vit_img_size
		self.vit_pretrain=vit_pretrain

		self.is_shift = is_shift
		self.shift_div = shift_div
		self.backbone = backbone
		
		self.num_class = num_class
		self.cfg = cfg
		self._prepare_base_model(backbone)
		if "resnet" in self.backbone:
			self._prepare_fc(num_class)
		self.consensus = ConsensusModule(consensus_type)
		self._enable_pbn = partial_bn
		self.LayerNormFreeze = LayerNormFreeze
		if partial_bn:
			self.partialBN(True)

		self.decoder = Decoder(num_queries=self.sub_action_token_num, with_spatial_conv=self.with_spatial_conv, num_encoder_layers=self.decoder_layer_num)
		self.temporal_conv = nn.Conv1d(d_model, d_model, kernel_size=self.sub_action_token_num, stride=1, padding=0)
		self.dropout_layer = nn.Dropout(p=dropout)

	def _prepare_base_model(self, backbone):
		if 'ViT' in backbone:
			if self.net == 'ViT':
				print('=> base model: ViT, with backbone: {}'.format(backbone))
				from models.vit_models.modeling import VisionTransformer, CONFIGS
				vit_cfg = CONFIGS[backbone]
				self.base_model = VisionTransformer(vit_cfg, self.vit_img_size,
									zero_head=True, num_classes=self.num_class)
			elif self.net == 'TokShift':
				print('=> base model: TokShift, with backbone: {}'.format(backbone))
				# from vit_models.modeling_tokshift import VisionTransformer, CONFIGS
				from models.vit_models.modeling_tokshift import VisionTransformer, CONFIGS
				vit_cfg = CONFIGS[backbone]
				vit_cfg.fold_div = self.shift_div
				self.base_model = VisionTransformer(vit_cfg, self.vit_img_size, vis=True,
									zero_head=True, num_classes=self.num_class)
			if self.vit_pretrain != "":
				print("ViT pretrain weights: {}".format(self.vit_pretrain))
				self.base_model.load_from(np.load(self.vit_pretrain))
			self.feature_dim=self.num_class
			self.head = Linear(vit_cfg.hidden_size, self.num_class)

		else:
			raise ValueError('Unknown backbone: {}'.format(backbone))


	def _prepare_fc(self, num_class):
		if self.dropout == 0:
			setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(self.feature_dim, num_class))
			self.new_fc = None
		else:
			setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
			self.new_fc = nn.Linear(self.feature_dim, num_class)

		std = 0.001
		if self.new_fc is None:
			normal_(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
			constant_(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
		else:
			if hasattr(self.new_fc, 'weight'):
				normal_(self.new_fc.weight, 0, std)
				constant_(self.new_fc.bias, 0)

	#
	def train(self, mode=True):
		# Override the default train() to freeze the BN parameters
		super(VideoNet, self).train(mode)
		count = 0
		if self._enable_pbn and mode:
			print("Freezing LayerNorm.")
			for m in self.base_model.modules():
				if isinstance(m, nn.LayerNorm):
					count += 1
					if count >= (self.LayerNormFreeze if self._enable_pbn else 1):
						m.eval()
						print("Freeze {}".format(m))
						# shutdown update in frozen mode
						m.weight.requires_grad = False
						m.bias.requires_grad = False


	#
	def partialBN(self, enable):
		self._enable_pbn = enable


	def forward(self, input):
		bs = input.size(0)
		input = input.reshape((-1, 3) + input.size()[-2:])
		if 'ViT' in self.backbone:
			base_out, atten = self.base_model(input, bs)
			if self.with_decoder:
				base_out = base_out[:, 0]
				base_out = base_out.view((bs,-1)+base_out.size()[1:])
				base_out = self.decoder(base_out)
				base_out = self.temporal_conv(base_out.transpose(1, 2)).squeeze(2)
			else:
				base_out = base_out[:, 0]
				base_out = base_out.view((bs,-1)+base_out.size()[1:])
				base_out = self.consensus(base_out)

		base_out = self.dropout_layer(base_out)
		output = self.head(base_out)
		return output