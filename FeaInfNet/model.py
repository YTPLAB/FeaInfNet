import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet_features import resnet18_features, resnet34_features, resnet50_features, resnet101_features, resnet152_features
from models.densenet_features import densenet121_features, densenet161_features, densenet169_features, densenet201_features
from models.vgg_features import vgg11_features, vgg11_bn_features, vgg13_features, vgg13_bn_features, vgg16_features, vgg16_bn_features,\
                         vgg19_features, vgg19_bn_features
from settings import feature_masks_num, setting_map_HW, positive_prototype_shape, negative_prototype_shape, positive_prototype_num, negative_prototype_num, share_positive_prototype_num
from receptive_field import compute_proto_layer_rf_info_v2
import numpy as np

base_architecture_to_features = {'resnet18': resnet18_features,
                                 'resnet34': resnet34_features,
                                 'resnet50': resnet50_features,
                                 'resnet101': resnet101_features,
                                 'resnet152': resnet152_features,
                                 'densenet121': densenet121_features,
                                 'densenet161': densenet161_features,
                                 'densenet169': densenet169_features,
                                 'densenet201': densenet201_features,
                                 'vgg11': vgg11_features,
                                 'vgg11_bn': vgg11_bn_features,
                                 'vgg13': vgg13_features,
                                 'vgg13_bn': vgg13_bn_features,
                                 'vgg16': vgg16_features,
                                 'vgg16_bn': vgg16_bn_features,
                                 'vgg19': vgg19_features,
                                 'vgg19_bn': vgg19_bn_features}

class PPNet(nn.Module):

    def __init__(self, features, img_size, prototype_shape,
                 proto_layer_rf_info, num_classes, init_weights=True,
                 prototype_activation_function='log',
                 add_on_layers_type='bottleneck'):

        super(PPNet, self).__init__()
        self.img_size = img_size
        self.prototype_shape = prototype_shape
        self.num_prototypes = prototype_shape[0]
        self.num_classes = num_classes
        self.epsilon = 1e-4
        self.map_HW = setting_map_HW
        self.positive_prototype_shape = positive_prototype_shape
        self.negative_prototype_shape = negative_prototype_shape
        self.positive_num_prototypes = positive_prototype_num
        self.negative_num_prototypes = negative_prototype_num
        self.prototype_activation_function = prototype_activation_function

        assert (self.num_prototypes % self.num_classes == 0)
        self.prototype_class_identity = torch.zeros(self.num_prototypes,
                                                    self.num_classes)

        num_prototypes_per_class = self.num_prototypes // self.num_classes

        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // num_prototypes_per_class] = 1

        self.proto_layer_rf_info = proto_layer_rf_info
        self.features = features
        features_name = str(self.features).upper()

        if features_name.startswith('VGG') or features_name.startswith('RES'):
            first_add_on_layer_in_channels = \
                [i for i in features.modules() if isinstance(i, nn.Conv2d)][-1].out_channels

        elif features_name.startswith('DENSE'):
            first_add_on_layer_in_channels = \
                [i for i in features.modules() if isinstance(i, nn.BatchNorm2d)][-1].num_features
        else:
            raise Exception('other base base_architecture NOT implemented')

        if add_on_layers_type == 'bottleneck':
            add_on_layers = []
            current_in_channels = first_add_on_layer_in_channels

            while (current_in_channels > self.prototype_shape[1]) or (
                    len(add_on_layers) == 0):
                current_out_channels = max(self.prototype_shape[1], (
                        current_in_channels // 2))
                add_on_layers.append(nn.Conv2d(in_channels=current_in_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))

                add_on_layers.append(nn.ReLU())
                add_on_layers.append(nn.Conv2d(in_channels=current_out_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))

                if current_out_channels > self.prototype_shape[1]:
                    add_on_layers.append(nn.ReLU())
                else:
                    assert (current_out_channels == self.prototype_shape[1])
                    add_on_layers.append(nn.Sigmoid())

                current_in_channels = current_in_channels // 2

            self.add_on_layers = nn.Sequential(*add_on_layers)

        else:
            self.add_on_layers = nn.Sequential(
                nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=self.prototype_shape[1], kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=self.prototype_shape[1], out_channels=self.prototype_shape[1], kernel_size=1),
                nn.Sigmoid()
            )

        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape), requires_grad=True)
        self.ones = nn.Parameter(torch.ones(self.prototype_shape), requires_grad=False)
        self.activation_num = feature_masks_num
        self.feature_masks_num = feature_masks_num
        self.share_positive_prototype_num = share_positive_prototype_num
        feature_masks = torch.abs(torch.randn(self.activation_num, self.prototype_shape[1], self.map_HW, self.map_HW))

        for height in range(self.map_HW):
            for width in range(self.map_HW):
                feature_masks[height * self.map_HW + width, :, height, width] += 3

        upsample_function = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=None)
        up_feature_masks = upsample_function(feature_masks)
        original_img_masks = torch.mean(up_feature_masks, dim=1).unsqueeze(1).repeat(1, 3, 1, 1)
        self.original_img_masks = nn.Parameter(original_img_masks, requires_grad=False)
        self.feature_masks = nn.Parameter(feature_masks, requires_grad=False)
        self.positive_prototype_vectors = nn.Parameter(torch.rand(self.share_positive_prototype_num, self.positive_prototype_shape[0], self.positive_prototype_shape[1]), requires_grad=True)
        self.negative_prototype_vectors = nn.Parameter(torch.rand(self.feature_masks_num, self.negative_prototype_shape[0], self.negative_prototype_shape[1]), requires_grad=True)
        self.positive_layer = nn.Parameter(torch.randn(self.positive_prototype_shape[0]), requires_grad=True)
        self.negative_layer = nn.Parameter(torch.randn(self.feature_masks_num, self.negative_prototype_shape[0]), requires_grad=True)
        self.last_layer = nn.Linear(in_features=self.num_prototypes, out_features=self.num_classes, bias=False)
        self.positive_image_index = np.zeros([feature_masks_num, positive_prototype_num])
        self.negative_image_index = np.zeros([feature_masks_num, negative_prototype_num])

        if init_weights:
            self._initialize_weights()

    def conv_features(self, x):
        '''
        the feature input to prototype layer
        '''
        x = self.features(
            x)
        x = self.add_on_layers(
            x)
        return x

    @staticmethod
    def _weighted_l2_convolution(input, filter, weights):
        input2 = input ** 2
        input_patch_weighted_norm2 = F.conv2d(input=input2, weight=weights)
        filter2 = filter ** 2
        weighted_filter2 = filter2 * weights
        filter_weighted_norm2 = torch.sum(weighted_filter2, dim=(1, 2, 3))
        filter_weighted_norm2_reshape = filter_weighted_norm2.view(-1, 1, 1)
        weighted_filter = filter * weights
        weighted_inner_product = F.conv2d(input=input, weight=weighted_filter)
        intermediate_result = - 2 * weighted_inner_product + filter_weighted_norm2_reshape
        distances = F.relu(input_patch_weighted_norm2 + intermediate_result)

        return distances

    def _l2_convolution(self, x):
        '''
        apply self.prototype_vectors as l2-convolution filters on input x
        '''

        feature_masks_tmp = self.feature_masks.unsqueeze(0)
        x = x.unsqueeze(1)
        feature_map_mask = torch.mean(x * feature_masks_tmp, dim=(-1, -2))
        positive_square = torch.sum((feature_map_mask.unsqueeze(2) - self.positive_prototype_vectors.unsqueeze(0)) ** 2, dim=-1)
        negative_square = torch.sum((feature_map_mask.unsqueeze(2) - self.negative_prototype_vectors.unsqueeze(0)) ** 2, dim=-1)
        positive_activation = self.distance_2_similarity(positive_square)
        negative_activation = self.distance_2_similarity(negative_square)
        feature_logits = torch.sum(torch.abs(self.positive_layer) * positive_activation, dim=-1) - torch.sum(torch.abs(self.negative_layer) * negative_activation, dim=-1)

        return feature_logits, positive_square, negative_square

    def prototype_distances(self, x):
        '''
        x is the raw input
        '''
        conv_features = self.conv_features(x)
        feature_logits, positive_square, negative_square = self._l2_convolution(conv_features)
        return feature_logits, positive_square, negative_square

    def distance_2_similarity(self, distances):
        if self.prototype_activation_function == 'log':
            return torch.log(
                (distances + 1) / (distances + self.epsilon))

        elif self.prototype_activation_function == 'linear':
            return -distances

        else:
            return self.prototype_activation_function(distances)

    def mdm_forward(self, x):
        x1 = self.conv_features(x)
        feature_masks_tmp1 = self.feature_masks.unsqueeze(0)
        x1 = x1.unsqueeze(1)
        analyze_img_feature_map = torch.mean(x1 * feature_masks_tmp1, dim=(-1, -2))
        feature_logits, positive_square, negative_square = self.prototype_distances(x)
        positive_prototype_vectors = self.positive_prototype_vectors
        logits, logits_index = torch.max(feature_logits, dim=-1)

        return logits, logits_index, feature_logits, positive_square, negative_square, positive_prototype_vectors, analyze_img_feature_map

    def forward(self, x):
        feature_masks_tmp1 = self.feature_masks.unsqueeze(0)
        original_x_activation = torch.mean(self.conv_features(x).unsqueeze(1) * feature_masks_tmp1, dim=(-1, -2))
        original_x_activation = original_x_activation[0:1]
        tmp_img_masks = self.original_img_masks.unsqueeze(0)
        mask_x = x.unsqueeze(1) * tmp_img_masks
        mask_x = mask_x[0:1]
        mask_x_shape_batch = mask_x.shape[0]
        mask_x_shape_feature = mask_x.shape[1]

        re_mask_x = mask_x.view(mask_x_shape_batch * mask_x_shape_feature, 3, 224, 224)
        re_mask_x_conv = self.conv_features(re_mask_x)
        mask_x_conv = re_mask_x_conv.view(mask_x_shape_batch, mask_x_shape_feature, 128, 7, 7)
        mask_x_activation = torch.mean(mask_x_conv * feature_masks_tmp1, dim=(-1, -2))
        feature_masks_consistency_activation = torch.mean((original_x_activation - mask_x_activation) ** 2, dim=(-1, -2))
        feature_logits, positive_square, negative_square = self.prototype_distances(x)
        positive_distance = positive_square
        negative_distance = negative_square

        logits, _ = torch.max(feature_logits, dim=-1)
        logits = logits.unsqueeze(-1).repeat(1, 2)
        logits[:, 0] = 1 - logits[:, 1]
        logits = torch.softmax(logits, dim=-1)

        return logits, positive_distance, negative_distance, feature_masks_consistency_activation

    def push_forward(self, x):
        ''' this method is needed for the pushing operation '''
        x = self.conv_features(x)
        feature_masks_tmp = self.feature_masks.unsqueeze(0)
        x = x.unsqueeze(1)
        feature_map_mask = torch.mean(x * feature_masks_tmp, dim=(-1, -2))
        positive_square = torch.sum((feature_map_mask.unsqueeze(2) - self.positive_prototype_vectors.unsqueeze(0)) ** 2, dim=-1)
        negative_square = torch.sum((feature_map_mask.unsqueeze(2) - self.negative_prototype_vectors.unsqueeze(0)) ** 2, dim=-1)
        return feature_map_mask, positive_square, negative_square

    def prune_prototypes(self, prototypes_to_prune):

        prototypes_to_keep = list(
            set(range(self.num_prototypes)) - set(prototypes_to_prune))

        self.prototype_vectors = nn.Parameter(self.prototype_vectors.data[prototypes_to_keep, ...], requires_grad=True)
        self.prototype_shape = list(self.prototype_vectors.size())
        self.num_prototypes = self.prototype_shape[0]
        self.last_layer.in_features = self.num_prototypes
        self.last_layer.out_features = self.num_classes
        self.last_layer.weight.data = self.last_layer.weight.data[:, prototypes_to_keep]
        self.ones = nn.Parameter(self.ones.data[prototypes_to_keep, ...], requires_grad=False)
        self.prototype_class_identity = self.prototype_class_identity[prototypes_to_keep, :]

    def __repr__(self):

        rep = (
            'PPNet(\n'
            '\tfeatures: {},\n'
            '\timg_size: {},\n'
            '\tprototype_shape: {},\n'
            '\tproto_layer_rf_info: {},\n'
            '\tnum_classes: {},\n'
            '\tepsilon: {}\n'
            ')'
        )

        return rep.format(self.features,
                          self.img_size,
                          self.prototype_shape,
                          self.proto_layer_rf_info,
                          self.num_classes,
                          self.epsilon)

    def set_last_layer_incorrect_connection(self, incorrect_strength):
        '''
        the incorrect strength will be actual strength if -0.5 then input -0.5
        '''

        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations
        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength

        self.last_layer.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)

    def _initialize_weights(self):
        for m in self.add_on_layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(tensor=m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(tensor=m.bias, val=0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(tensor=m.weight, val=1)
                nn.init.constant_(tensor=m.bias, val=0)

        self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)

def construct_PPNet(base_architecture, pretrained=True, img_size=224,
                    prototype_shape=(2000, 512, 1, 1), num_classes=200,
                    prototype_activation_function='log',
                    add_on_layers_type='bottleneck'):
    features = base_architecture_to_features[base_architecture](pretrained=pretrained)
    layer_filter_sizes, layer_strides, layer_paddings = features.conv_info()
    proto_layer_rf_info = compute_proto_layer_rf_info_v2(img_size=img_size,
                                                         layer_filter_sizes=layer_filter_sizes,
                                                         layer_strides=layer_strides,
                                                         layer_paddings=layer_paddings,
                                                         prototype_kernel_size=prototype_shape[2])

    return PPNet(features=features,
                 img_size=img_size,
                 prototype_shape=prototype_shape,
                 proto_layer_rf_info=proto_layer_rf_info,
                 num_classes=num_classes,
                 init_weights=True,
                 prototype_activation_function=prototype_activation_function,
                 add_on_layers_type=add_on_layers_type)
