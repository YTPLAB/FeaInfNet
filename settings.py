datasets_name_txt = ''
training_set_size = 5000
push_set_size = 5000
test_set_size = 1000
train_batch_size = 20
test_batch_size = 20
train_push_batch_size = 20
train_push_original_batch_size = 1
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
base_architecture = 'resnet50'
img_size = 224
num_classes = 2
train_iteration = 10
share_positive_prototype_num = 1
feature_masks_num = 49
attention_activation_num = 49
setting_map_HW = 7
prototype_channel = 128
prototype_shape = (2000, prototype_channel, 1, 1)
positive_prototype_num = 10
negative_prototype_num = 4
positive_prototype_shape = (positive_prototype_num, prototype_channel)
negative_prototype_shape = (negative_prototype_num, prototype_channel)
all_prototype_shape = (feature_masks_num, positive_prototype_num, prototype_channel)
all_positive_prototype_shape = (share_positive_prototype_num, positive_prototype_num, prototype_channel)
all_negative_prototype_shape = (feature_masks_num, negative_prototype_num, prototype_channel)

prototype_activation_function = 'log'
add_on_layers_type = 'regular'
experiment_run = '004'
data_path = './datasets/cub200_cropped/'
train_dir = data_path + 'train_cropped/'
test_dir = data_path + 'test_cropped/'
train_push_dir = data_path + 'train_cropped/'
train_push_original_dir = data_path + 'train_cropped/'

joint_optimizer_lrs = {'features': 1e-4,
                       'add_on_layers': 3e-3,
                       'prototype_vectors': 3e-3,
                       'positive_prototype_vectors': 1e-4,
                       'negative_prototype_vectors': 1e-4}

joint_lr_step_size = 5

warm_optimizer_lrs = {'add_on_layers': 3e-3,
                      'prototype_vectors': 3e-3,
                      'positive_prototype_vectors': 1e-4,
                      'negative_prototype_vectors': 1e-4}

last_layer_optimizer_lr = 1e-4

coefs = {
    'crs_ent': 1,
    'positive_clst': 1e-3,
    'positive_sep': 1e-3,
    'negative_clst': 1e-3,
    'negative_sep': 1e-3,
    'feature_mask_consistency': 1e-2,
}

num_train_epochs = 200
num_warm_epochs = 5
push_start = 10
push_epochs = [i for i in range(num_train_epochs) if i % 10 == 0]


