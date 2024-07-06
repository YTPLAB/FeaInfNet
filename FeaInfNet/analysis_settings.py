datasets_name_txt = ''
training_set_size = 5000
push_set_size = 5000
test_set_size = 1000
train_batch_size = 20
test_batch_size = 20
train_push_batch_size = 20
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

base_architecture = 'resnet50'
img_size = 224
prototype_shape = (20, 128, 1, 1)
num_classes = 2
prototype_activation_function = 'log'
add_on_layers_type = 'regular'
experiment_run = '003'
data_path = './datasets/cub200_cropped/'
train_dir = data_path + 'train_cropped/'
test_dir = data_path + 'test_cropped/'
train_push_dir = data_path + 'train_cropped/'
joint_optimizer_lrs = {'features': 1e-4,
                       'add_on_layers': 3e-3,
                       'prototype_vectors': 3e-3}

joint_lr_step_size = 5

warm_optimizer_lrs = {'add_on_layers': 3e-3,
                      'prototype_vectors': 3e-3}

last_layer_optimizer_lr = 1e-4
coefs = {
    'crs_ent': 1,
    'clst': 0.8,
    'sep': -0.08,
    'l1': 1e-4,
}

num_train_epochs = 100
num_warm_epochs = 5
push_start = 10
push_epochs = [i for i in range(num_train_epochs) if i % 10 == 0]
original_img_path_ = './bird_datasets'
img_size_ = 224
dynamic_img_batch_ = 2
dynamic_mask_lrs = {'mse': 1e-1, 'l1': 1}
base_mask_size_ = 10
best_loss_ = 1e8
best_epoch_ = -1
best_mask_position_ = -1
iteration_epoch_ = 20000
iteration_epoch_min_ = 400
patient_ = 200
mask_optimizer_lr_ = 3e-2
check_epoch_ = 400
dynamic_img_batch_ = 2
dynamic_mask_batch_ = 80
base_mask_size_list_ = [7, 8]


