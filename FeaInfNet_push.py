import os
import time
from utils.helpers import makedir
import numpy as np
import torch
from settings import all_positive_prototype_shape, all_negative_prototype_shape

def push_prototypes(dataloader,
                    prototype_network_parallel,
                    preprocess_input_function=None,
                    root_dir_for_saving_prototypes=None,
                    epoch_number=None,
                    proto_bound_boxes_filename_prefix=None,
                    save_prototype_class_identity=True,
                    log=print):

    prototype_network_parallel.eval()
    log('\tpush')
    start = time.time()
    positive_prototype_shape = prototype_network_parallel.module.positive_prototype_shape
    negative_prototype_shape = prototype_network_parallel.module.negative_prototype_shape
    n_prototypes = prototype_network_parallel.module.num_prototypes
    positive_n_prototypes = prototype_network_parallel.module.positive_num_prototypes
    negative_n_prototypes = prototype_network_parallel.module.negative_num_prototypes
    feature_masks_num = prototype_network_parallel.module.feature_masks_num
    share_positive_prototype_num = prototype_network_parallel.module.share_positive_prototype_num
    global_positive_min_proto_dist = np.full([share_positive_prototype_num, positive_n_prototypes], np.inf)
    global_negative_min_proto_dist = np.full([feature_masks_num, negative_n_prototypes], np.inf)
    global_positive_min_fmap_patches = np.zeros([share_positive_prototype_num, positive_n_prototypes, positive_prototype_shape[1]])
    global_negative_min_fmap_patches = np.zeros([feature_masks_num, negative_n_prototypes, negative_prototype_shape[1]])
    positive_image_index = np.zeros([share_positive_prototype_num, positive_n_prototypes])
    negative_image_index = np.zeros([feature_masks_num, negative_n_prototypes])
    positive_prototype_path = './prototype_info/positive_prototype_file.txt'
    negative_prototype_path = './prototype_info/negative_prototype_file.txt'

    if save_prototype_class_identity:
        proto_rf_boxes = np.full(shape=[n_prototypes, 6],
                                 fill_value=-1)

        proto_bound_boxes = np.full(shape=[n_prototypes, 6],
                                    fill_value=-1)

    else:
        proto_rf_boxes = np.full(shape=[n_prototypes, 5],
                                 fill_value=-1)

        proto_bound_boxes = np.full(shape=[n_prototypes, 5],
                                    fill_value=-1)

    if root_dir_for_saving_prototypes != None:
        if epoch_number != None:
            proto_epoch_dir = os.path.join(root_dir_for_saving_prototypes,
                                           'epoch-' + str(epoch_number))
            makedir(proto_epoch_dir)
        else:
            proto_epoch_dir = root_dir_for_saving_prototypes

    else:
        proto_epoch_dir = None

    search_batch_size = dataloader.batch_size

    for push_iter, (search_batch_input, search_y) in enumerate(dataloader):

        start_index_of_search_batch = push_iter * search_batch_size
        update_prototypes_on_batch(search_batch_input=search_batch_input,
                                   start_index_of_search_batch=start_index_of_search_batch,
                                   prototype_network_parallel=prototype_network_parallel,
                                   positive_image_index=positive_image_index,
                                   negative_image_index=negative_image_index,
                                   global_positive_min_proto_dist=global_positive_min_proto_dist,
                                   global_negative_min_proto_dist=global_negative_min_proto_dist,
                                   global_positive_min_fmap_patches=global_positive_min_fmap_patches,
                                   global_negative_min_fmap_patches=global_negative_min_fmap_patches,
                                   search_y=search_y,
                                   preprocess_input_function=preprocess_input_function)

    if proto_epoch_dir != None and proto_bound_boxes_filename_prefix != None:
        np.save(file=os.path.join(proto_epoch_dir,
                                  proto_bound_boxes_filename_prefix + '-receptive_field' + str(epoch_number) + '.npy'),
                arr=proto_rf_boxes)

        np.save(file=os.path.join(proto_epoch_dir, proto_bound_boxes_filename_prefix + str(epoch_number) + '.npy'),
                arr=proto_bound_boxes)

    log('\tExecuting push ...')

    positive_prototype_update = np.reshape(global_positive_min_fmap_patches,
                                           tuple(all_positive_prototype_shape))

    negative_prototype_update = np.reshape(global_negative_min_fmap_patches,
                                           tuple(all_negative_prototype_shape))

    prototype_network_parallel.module.positive_prototype_vectors.data.copy_(
        torch.tensor(positive_prototype_update, dtype=torch.float32).cuda())

    prototype_network_parallel.module.negative_prototype_vectors.data.copy_(
        torch.tensor(negative_prototype_update, dtype=torch.float32).cuda())

    makedir('./prototype_info')
    with open(positive_prototype_path, 'w') as f:
        for i in range(share_positive_prototype_num):
            if (i < share_positive_prototype_num - 1):
                f.write(str(positive_image_index[i]))
                f.write("\n")
            else:
                f.write(str(positive_image_index[i]))

    with open(negative_prototype_path, 'w') as f:
        for i in range(feature_masks_num):
            if (i < feature_masks_num - 1):
                f.write(str(negative_image_index[i]))
                f.write("\n")
            else:
                f.write(str(negative_image_index[i]))

    prototype_network_parallel.module.positive_image_index = positive_image_index
    prototype_network_parallel.module.negative_image_index = negative_image_index
    end = time.time()
    log('\tpush time: \t{0}'.format(end - start))

def update_prototypes_on_batch(search_batch_input,
                               start_index_of_search_batch,
                               prototype_network_parallel,
                               positive_image_index,
                               negative_image_index,
                               global_positive_min_proto_dist,
                               global_negative_min_proto_dist,
                               global_positive_min_fmap_patches,
                               global_negative_min_fmap_patches,
                               search_y=None,
                               preprocess_input_function=None):

    prototype_network_parallel.eval()
    if preprocess_input_function is not None:
        search_batch = search_batch_input

    else:
        search_batch = search_batch_input

    with torch.no_grad():

        search_batch = search_batch.cuda()
        feature_map_mask_torch, positive_distance_torch, negative_distance_torch = prototype_network_parallel.module.push_forward(search_batch)
        feature_map_mask = np.copy(feature_map_mask_torch.detach().cpu().numpy())
        positive_distance = np.copy(positive_distance_torch.detach().cpu().numpy())
        negative_distance = np.copy(negative_distance_torch.detach().cpu().numpy())
        positive_index = []
        negative_index = []

        for i in range(search_y.shape[0]):
            if(search_y[i].item() == 0):
                negative_index.append(i)
            else:
                positive_index.append(i)

        for positive_i in positive_index:
            for pos_feature_num in range(positive_distance.shape[1]):
                for pos_prototype_num in range(positive_distance.shape[2]):
                    if(positive_distance[positive_i, pos_feature_num, pos_prototype_num] < global_positive_min_proto_dist[0, pos_prototype_num]):
                        global_positive_min_proto_dist[0, pos_prototype_num] = positive_distance[positive_i, pos_feature_num, pos_prototype_num]
                        global_positive_min_fmap_patches[0, pos_prototype_num] = feature_map_mask[positive_i, pos_feature_num]
                        positive_image_index[0, pos_prototype_num] = start_index_of_search_batch + positive_i

        for negative_i in negative_index:
            for neg_feature_num in range(negative_distance.shape[1]):
                for neg_prototype_num in range(negative_distance.shape[2]):
                    if (negative_distance[negative_i, neg_feature_num, neg_prototype_num] < global_negative_min_proto_dist[neg_feature_num, neg_prototype_num]):
                        global_negative_min_proto_dist[neg_feature_num, neg_prototype_num] = negative_distance[negative_i, neg_feature_num, neg_prototype_num]
                        global_negative_min_fmap_patches[neg_feature_num, neg_prototype_num] = feature_map_mask[negative_i, neg_feature_num]
                        negative_image_index[neg_feature_num, neg_prototype_num] = start_index_of_search_batch + negative_i

