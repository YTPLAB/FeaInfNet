import os
import random
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import re
import numpy as np
from utils.helpers import makedir
import model as model
import train_and_test as tnt
import utils.save as save
from utils.log import create_logger
from utils.preprocess import preprocess_input_function
import datetime
import csv
from settings import train_batch_size, test_batch_size, train_push_batch_size
from settings import mean, std
from settings import train_dir, test_dir, train_push_dir, train_iteration
import FeaInfNet_push as push
from settings import base_architecture, img_size, prototype_shape, num_classes, prototype_activation_function, add_on_layers_type, experiment_run

def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if seed == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print(os.environ['CUDA_VISIBLE_DEVICES'])
    base_architecture_type = re.match('^[a-z]*', base_architecture).group(0)
    model_dir = './saved_models/' + base_architecture + '/' + experiment_run + '/'
    makedir(model_dir)

    result_path = './Result/'
    time_path = str(datetime.datetime.now()).replace(':', '.')
    makedir(os.path.join(result_path, time_path))
    train_test_path = 'Train_and_Test'
    push_test_path = 'Push_Test'
    iteration_train_path = 'Iteration_train'

    log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
    img_dir = os.path.join(model_dir, 'img')
    makedir(img_dir)
    weight_matrix_filename = 'outputL_weights'
    prototype_img_filename_prefix = 'prototype-img'
    prototype_self_act_filename_prefix = 'prototype-self-act'
    proto_bound_boxes_filename_prefix = 'bb'
    normalize = transforms.Normalize(mean=mean, std=std)

    train_dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True,
        num_workers=4, pin_memory=False)

    train_push_dataset = datasets.ImageFolder(
        train_push_dir,
        transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ]))
    train_push_loader = torch.utils.data.DataLoader(
        train_push_dataset, batch_size=train_push_batch_size, shuffle=False,
        num_workers=4, pin_memory=False)

    test_dataset = datasets.ImageFolder(
        test_dir,
        transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ]))
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False,
        num_workers=4, pin_memory=False)

    datasets_name_txt = 'PALM'
    training_set_size_txt = len(train_loader.dataset)
    push_set_size_txt = len(train_push_loader.dataset)
    test_set_size_txt = len(test_loader.dataset)
    batch_size_txt = train_batch_size

    log('training set size: {0}'.format(training_set_size_txt))
    log('push set size: {0}'.format(push_set_size_txt))
    log('test set size: {0}'.format(test_set_size_txt))
    log('batch size: {0}'.format(train_batch_size))

    with open(file=os.path.join(result_path, time_path, datasets_name_txt) + str('.txt'), mode='w') as f:
        f.write("datasets_name:" + datasets_name_txt)
        f.write("\n")
        f.write("training_set_size_txt = " + str(training_set_size_txt))
        f.write("\n")
        f.write("push_set_size_txt = " + str(push_set_size_txt))
        f.write("\n")
        f.write("test_set_size_txt = " + str(test_set_size_txt))
        f.write("\n")
        f.write("batch_size_txt = " + str(batch_size_txt))
        f.write("\n")

    with open(file=os.path.join(result_path, time_path, train_test_path) + '.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(
            ['epoch', 'train_time', 'train_cross_ent', 'train_negative_cluster', 'train_positive_cluster', 'train_negative_separation',
             'train_accu', \
             'train_positive_separation', 'train_feature_consistency', 'test_time', 'test_cross_ent', 'test_negative_cluster', 'test_positive_cluster',
             'test_negative_separation', \
             'test_accu', 'test_positive_separation', 'test_feature_consistency'])

    with open(file=os.path.join(result_path, time_path, push_test_path) + '.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(
            ['epoch', 'test_time', 'test_cross_ent', 'test_negative_cluster', 'test_positive_cluster',
             'test_negative_separation', 'test_accu', 'test_positive_separation', 'test_feature_consistency'])

    with open(file=os.path.join(result_path, time_path, iteration_train_path) + '.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(
            ['epoch', 'iteration', 'train_time', 'train_cross_ent', 'train_negative_cluster', 'train_positive_cluster',
             'train_negative_separation',
             'train_accu', \
             'train_positive_separation', 'train_feature_consistency', 'test_time', 'test_cross_ent', 'test_negative_cluster', 'test_positive_cluster',
             'test_negative_separation', \
             'test_accu', 'test_positive_separation', 'test_feature_consistency'])

    ppnet = model.construct_PPNet(base_architecture=base_architecture,
                                  pretrained=True, img_size=img_size,
                                  prototype_shape=prototype_shape,
                                  num_classes=num_classes,
                                  prototype_activation_function=prototype_activation_function,
                                  add_on_layers_type=add_on_layers_type)

    ppnet = ppnet.cuda()
    ppnet_multi = torch.nn.DataParallel(ppnet)
    class_specific = True

    from settings import joint_optimizer_lrs, joint_lr_step_size
    joint_optimizer_specs = \
        [{'params': ppnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3},
         {'params': ppnet.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
         {'params': ppnet.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
         {'params': ppnet.positive_prototype_vectors, 'lr': joint_optimizer_lrs['positive_prototype_vectors']},
         {'params': ppnet.negative_prototype_vectors, 'lr': joint_optimizer_lrs['negative_prototype_vectors']},
         ]

    joint_optimizer = torch.optim.Adam(params=joint_optimizer_specs)
    joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=joint_optimizer, step_size=joint_lr_step_size,
                                                         gamma=0.1)

    from settings import warm_optimizer_lrs
    warm_optimizer_specs = \
        [{'params': ppnet.add_on_layers.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
         {'params': ppnet.prototype_vectors, 'lr': warm_optimizer_lrs['prototype_vectors']},
         {'params': ppnet.positive_prototype_vectors, 'lr': warm_optimizer_lrs['positive_prototype_vectors']},
         {'params': ppnet.negative_prototype_vectors, 'lr': warm_optimizer_lrs['negative_prototype_vectors']},
         ]

    warm_optimizer = torch.optim.Adam(params=warm_optimizer_specs)

    from settings import last_layer_optimizer_lr
    last_layer_optimizer_specs = [{'params': ppnet.positive_layer, 'lr': last_layer_optimizer_lr},
                                  {'params': ppnet.negative_layer, 'lr': last_layer_optimizer_lr}]
    last_layer_optimizer = torch.optim.Adam(params=last_layer_optimizer_specs)

    log('start training')
    from settings import coefs
    from settings import num_train_epochs, num_warm_epochs, push_start, push_epochs

    for epoch in range(num_train_epochs):

        log('epoch: \t{0}'.format(epoch))

        if epoch < num_warm_epochs:
            tnt.warm_only(model=ppnet_multi, log=log)
            train_accu, train_result = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=warm_optimizer,
                                                 class_specific=class_specific, coefs=coefs, log=log)

        else:
            tnt.joint(model=ppnet_multi, log=log)
            joint_lr_scheduler.step()
            train_accu, train_result = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=joint_optimizer,
                                                 class_specific=class_specific, coefs=coefs, log=log)

        test_accu, test_result = tnt.tst(model=ppnet_multi, dataloader=test_loader,
                                         class_specific=class_specific, log=log)

        save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'nopush', accu=test_accu,
                                    target_accu=0.60, log=log)

        all_result = [str(epoch)]
        all_result = all_result + train_result
        all_result = all_result + test_result

        with open(file=os.path.join(result_path, time_path, train_test_path) + '.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(all_result)

        if epoch >= push_start and epoch in push_epochs:

            push.push_prototypes(
                 dataloader=train_push_loader,
                 prototype_network_parallel=ppnet_multi,
                 preprocess_input_function=preprocess_input_function,
                 root_dir_for_saving_prototypes=img_dir,
                 epoch_number=epoch,
                 proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
                 save_prototype_class_identity=True,
                 log=log)

            test_accu, test_result = tnt.tst(model=ppnet_multi, dataloader=test_loader,
                                             class_specific=class_specific, log=log)

            push_result = [str(epoch)]
            push_result = push_result + test_result

            with open(file=os.path.join(result_path, time_path, push_test_path) + '.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(push_result)

            save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'push', accu=test_accu,
                                        target_accu=0.60, log=log)

            if prototype_activation_function != 'linear':
                tnt.last_only(model=ppnet_multi, log=log)

                iteration = train_iteration
                for i in range(iteration):
                    log('iteration: \t{0}'.format(i))
                    train_accu, train_result = tnt.train(model=ppnet_multi, dataloader=train_loader,
                                                         optimizer=last_layer_optimizer,
                                                         class_specific=class_specific, coefs=coefs, log=log)
                    test_accu, test_result = tnt.tst(model=ppnet_multi, dataloader=test_loader,
                                                     class_specific=class_specific, log=log)

                    all_result = [str(epoch), str(i)]
                    all_result = all_result + train_result
                    all_result = all_result + test_result

                    with open(file=os.path.join(result_path, time_path, iteration_train_path) + '.csv', mode='a',
                              newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(all_result)

                    save.save_model_w_condition(model=ppnet, model_dir=model_dir,
                                                model_name=str(epoch) + '_' + str(i) + 'push', accu=test_accu,
                                                target_accu=0.60, log=log)

    logclose()
