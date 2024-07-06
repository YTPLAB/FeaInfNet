mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

base_architecture = 'resnet50'
img_size = 224
prototype_shape = (20, 128, 1, 1)
num_classes = 2
prototype_activation_function = 'log'
add_on_layers_type = 'regular'
experiment_run = '003'

img_size_ = 224
dynamic_img_batch_ = 2

dynamic_mask_lrs = {'mse': 1e-1,
                    'l1': 1}

base_mask_size_ = 10
best_loss_ = 1e8
best_epoch_ = -1
best_mask_position_ = -1
iteration_epoch_ = 200
iteration_epoch_min_ = 200
patient_ = 150
mask_optimizer_lr_ = 1e-3
patient_epoch_num_max_ = 7
check_epoch_ = 20
dynamic_mask_batch_ = 80
base_mask_size_list_ = [7, 8]
lamuda = [1e-1, 1]


