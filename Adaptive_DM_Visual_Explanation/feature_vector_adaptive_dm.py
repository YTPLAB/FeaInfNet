import shutil
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from utils.helpers import makedir
import os
import cv2
from utils.preprocess import preprocess_input_function
import dynamic_mask_learning
from analysis_settings import img_size_, dynamic_mask_lrs, best_loss_, \
    iteration_epoch_, iteration_epoch_min_, mask_optimizer_lr_, check_epoch_, \
    base_mask_size_list_, lamuda
from Monotonicity import Monotonicity_Modify_Rate
from find_high_activation import find_high_activation_mask_y

def FeatureVector_CL(model_path, Feature_vector_path):
    load_model_path = model_path
    resnet = torch.load(load_model_path)
    model1 = resnet.eval().cuda()
    model_multi = torch.nn.DataParallel(model1)
    model = model_multi
    each_saleincy_map_path = './feature_vector_feature_saliency_map'
    final_saleincy_map_path = './feature_vector_final_feature_saliency_map'

    feature_vector_path = Feature_vector_path
    original_img1 = cv2.imread(feature_vector_path)[..., ::-1]
    original_img1 = original_img1 / 255
    original_img1 = original_img1.astype('float32')
    x_torch = torch.tensor(original_img1)
    x_torch = x_torch.permute(2, 0, 1)
    x_torch = x_torch.unsqueeze(0)
    x = preprocess_input_function(x_torch)
    x = x.cuda()
    i = 0

    img_size = img_size_
    current_mask_img_root_path = './feature_vector_current_mask_img_root_path'
    base_mask_size_list = base_mask_size_list_
    logits, logits_index_torch, feature_logits, positive_square, negative_square, positive_prototype_vectors, analyze_img_feature_map = model.module.mdm_forward(x[i].unsqueeze(0))
    logits_index = logits_index_torch.item()
    predict_logits, _, _, _ = model(x[i].unsqueeze(0))
    _, positive_position_torch = torch.min(positive_square[0][logits_index], dim=-1)
    analyze_x_vector = analyze_img_feature_map[0, logits_index]
    current_all_mask_save_path = './'
    original_img = np.zeros([224, 224, 3])

    for base_mask_num in range(len(base_mask_size_list)):
        for lambda_e in lamuda:
            current_mask_img_path = os.path.join(current_mask_img_root_path)
            current_mask_img_path = os.path.join(current_mask_img_path)
            makedir(os.path.join(current_mask_img_path))

            d_mask = dynamic_mask_learning.Dynamic_Mask(img_size=img_size,
                                                        base_mask_size=base_mask_size_list[base_mask_num])
            d_mask = d_mask.cuda()

            best_loss = best_loss_
            iteration_epoch = iteration_epoch_
            iteration_epoch_min = iteration_epoch_min_
            mask_optimizer_lr = mask_optimizer_lr_
            check_epoch = check_epoch_
            base_mask_size = d_mask.base_mask_size
            d_mask = torch.nn.DataParallel(d_mask)
            mask_optimizer_specs = [{'params': d_mask.module.mask, 'lr': mask_optimizer_lr}]
            optimizer = torch.optim.Adam(mask_optimizer_specs)
            patient_epoch_num = 0

            for epoch in range(iteration_epoch):
                if (epoch > iteration_epoch_min):
                    break
                else:
                    x_mask = d_mask(x[i])
                    _, _, _, _, _, _, mask_analyze_img_feature_map = model.module.mdm_forward(x_mask)
                    mask_analyze_x_vector = mask_analyze_img_feature_map[0, logits_index]
                    original_act_patch = analyze_x_vector
                    mask_act_patch = mask_analyze_x_vector
                    act_mse_loss = (mask_act_patch - original_act_patch) ** 2
                    mse_loss = torch.sum(act_mse_loss, dim=0)
                    mse_loss = mse_loss.cuda()
                    mean_l1_loss = d_mask.module.mask.norm(p=1, dim=(1, 2)) / ((base_mask_size) ** 2)

                    if (epoch == 0):
                        dynamic_mask_lrs['mse'] = mean_l1_loss.data / (mse_loss.data + 1e-12) * lambda_e

                    loss = dynamic_mask_lrs['mse'] * mse_loss + dynamic_mask_lrs['l1'] * mean_l1_loss
                    loss = loss.sum()
                    loss = loss.cuda()

                    if (epoch % check_epoch == 0):
                        dynamic_mask_lrs['mse'] = mean_l1_loss.data / (mse_loss.data + 1e-12) * lambda_e
                        current_mse_loss = mse_loss.item()
                        current_l1_loss = mean_l1_loss.item()
                        threshold_mse = 0.4
                        threshold_l1 = 0.45
                        from analysis_settings import patient_epoch_num_max_
                        patient_epoch_num_max = patient_epoch_num_max_
                        patient_epoch_max_num = patient_epoch_num_max * check_epoch

                        if (epoch > 0):
                            if(current_mse_loss < threshold_mse and current_l1_loss < threshold_l1):
                                upsample_t = nn.Upsample(size=(img_size, img_size), mode='bilinear', align_corners=None)
                                current_mask_img = upsample_t(d_mask.module.mask.unsqueeze(0)).squeeze(0).permute(1, 2, 0).detach().cpu().numpy()[:, :, 0]
                                current_mask_img = current_mask_img - np.min(current_mask_img)
                                current_mask_img = current_mask_img / np.max(current_mask_img)
                                current_all_mask_save_path = current_mask_img_path
                                makedir(os.path.join(current_mask_img_path, str(base_mask_size_list[base_mask_num])))
                                plt.imsave(os.path.join(current_mask_img_path, str(base_mask_size_list[base_mask_num]), str(epoch) + '-' + str(lambda_e) + '-' + str(base_mask_size_list[base_mask_num]) + 'x' + '.jpg'), current_mask_img)
                                patient_epoch_num = 0
                            else:
                                patient_epoch_num += check_epoch
                                if(patient_epoch_num > patient_epoch_max_num):
                                    break

                    if (np.min(loss.detach().cpu().numpy()) < best_loss):
                        best_loss = np.min(loss.detach().cpu().numpy())

                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()

            img = x[i].permute(1, 2, 0).detach().cpu().numpy()
            original_img = img

        if(os.path.exists(os.path.join(current_all_mask_save_path, str(base_mask_size_list[base_mask_num])))):
            pass
        else:
            break

        all_mask_name = os.listdir(os.path.join(current_all_mask_save_path, str(base_mask_size_list[base_mask_num])))
        cur_dot = []
        cur_mask_rate = []
        cur_l1 = []
        cur_img_name_list = []
        cur_saliency_map_list = []

        for mask_x_name in all_mask_name:
            if(mask_x_name == '.ipynb_checkpoints'):
                continue

            mask_x = cv2.imread(os.path.join(current_all_mask_save_path, str(base_mask_size_list[base_mask_num]), mask_x_name), cv2.IMREAD_GRAYSCALE)
            mask_x = mask_x - np.min(mask_x)
            mask_x = mask_x / np.max(mask_x)
            local_mask_percentile = 0
            mask_x_binary, threshold = find_high_activation_mask_y(mask_x, local_mask_percentile)
            mask_x = mask_x_binary * mask_x
            mask_x = mask_x - threshold
            mask_x[mask_x < 0] = 0
            mask_x = mask_x / np.max(mask_x)
            cur_saliency_map_list.append(mask_x)
            cur_mask_l1 = torch.norm(torch.tensor(mask_x), dim=[0, 1]).item()
            Rt_mask_x = Monotonicity_Modify_Rate(mask_x)
            cur_dot.append(cur_mask_l1 * Rt_mask_x)
            cur_l1.append(cur_mask_l1)
            cur_mask_rate.append(Rt_mask_x)
            cur_img_name_list.append(mask_x_name)

        select_dot_min_index = np.argmin(cur_dot)
        select_original_img = original_img
        select_saliency_map = cur_saliency_map_list[select_dot_min_index]
        select_heatmap = cv2.applyColorMap(np.uint8(255 * select_saliency_map), cv2.COLORMAP_JET)
        select_heatmap = np.float32(select_heatmap) / 255
        select_heatmap = select_heatmap[..., ::-1]
        select_cam = 0.5 * select_original_img + 0.3 * select_heatmap
        select_cam = (select_cam - np.min(select_cam)) / (np.max(select_cam) - np.min(select_cam))
        store_path = os.path.join(each_saleincy_map_path, str(base_mask_size_list[base_mask_num]))
        makedir(store_path)
        mask_rgb = np.zeros([224, 224, 3])
        mask_rgb[:, :, 0] = select_saliency_map
        mask_rgb[:, :, 1] = select_saliency_map
        mask_rgb[:, :, 2] = select_saliency_map
        mask_img = mask_rgb * select_original_img

        from bbox_function import save_prototype_original_img_with_bbox1
        from utils.helpers import find_high_activation_crop

        original_img = original_img - np.min(original_img)
        original_img = original_img / np.max(original_img)
        original_img1 = np.zeros([224, 224, 3])
        original_img_bgr = original_img * 255
        original_img_bgr = original_img_bgr.astype('int32')
        original_img1[:, :, 0] = original_img_bgr[:, :, 0]
        original_img1[:, :, 1] = original_img_bgr[:, :, 0]
        original_img1[:, :, 2] = original_img_bgr[:, :, 0]
        mask_img = mask_img - np.min(mask_img)
        mask_img = mask_img / np.max(mask_img)
        plt.imsave(os.path.join(store_path, 'saliency_map.jpg'), select_saliency_map)
        plt.imsave(os.path.join(store_path, 'mask_img.jpg'), mask_img)
        plt.imsave(os.path.join(store_path, 'cam.jpg'), select_cam)

        select_original_img = select_original_img - np.min(select_original_img)
        select_original_img = select_original_img / np.max(select_original_img)
        plt.imsave(os.path.join(store_path, 'original_img.jpg'), select_original_img)
        o_img = cv2.imread(os.path.join(store_path, 'original_img.jpg'))[..., ::-1]
        o_heatmap = cv2.imread(os.path.join(store_path, 'cam.jpg'))[..., ::-1]

        crop_threshold = 99
        s1, x1, z1, y1 = find_high_activation_crop(select_saliency_map, crop_threshold)
        crop_patch = o_img[s1:x1, z1:y1]
        crop_o_img = save_prototype_original_img_with_bbox1(img_rgb=o_img, bbox_height_start=s1,
                                                            bbox_height_end=x1,
                                                            bbox_width_start=z1, bbox_width_end=y1,
                                                            color=(0, 255, 255))

        crop_o_heatmap = save_prototype_original_img_with_bbox1(img_rgb=o_heatmap, bbox_height_start=s1,
                                                                bbox_height_end=x1,
                                                                bbox_width_start=z1, bbox_width_end=y1,
                                                                color=(0, 255, 255))

        plt.imsave(os.path.join(store_path, 'crop_patch.jpg'), crop_patch)
        plt.imsave(os.path.join(store_path, 'crop_o_img.jpg'), crop_o_img)
        plt.imsave(os.path.join(store_path, 'crop_cam.jpg'), crop_o_heatmap)

    size_list = os.listdir(each_saleincy_map_path)
    sum_each_saliency = np.zeros([img_size, img_size])
    final_original_img = np.zeros([img_size, img_size, 3])

    for size in size_list:
        each_saliency_map = cv2.imread(os.path.join(each_saleincy_map_path, size, 'saliency_map.jpg'), cv2.IMREAD_GRAYSCALE)
        each_saliency_map = each_saliency_map - np.min(each_saliency_map)
        each_saliency_map = each_saliency_map / np.max(each_saliency_map)
        sum_each_saliency += each_saliency_map
        final_original_img = cv2.imread(os.path.join(each_saleincy_map_path, size, 'original_img.jpg'))[..., ::-1]

    makedir(os.path.join(final_saleincy_map_path))
    sum_each_saliency = sum_each_saliency - np.min(sum_each_saliency)
    sum_each_saliency = sum_each_saliency / np.max(sum_each_saliency)

    final_local_mask_percentile = 95

    final_mask_x_binary, final_threshold = find_high_activation_mask_y(sum_each_saliency, final_local_mask_percentile)
    sum_each_saliency = final_mask_x_binary * sum_each_saliency
    sum_each_saliency = sum_each_saliency - final_threshold
    sum_each_saliency[sum_each_saliency < 0] = 0
    sum_each_saliency = sum_each_saliency / np.max(sum_each_saliency)

    plt.imsave(os.path.join(final_saleincy_map_path, 'fianl_saliency.jpg'), sum_each_saliency)
    plt.imsave(os.path.join(final_saleincy_map_path, 'original_img.jpg'), final_original_img)
    final_heatmap = cv2.applyColorMap(np.uint8(255 * sum_each_saliency), cv2.COLORMAP_JET)
    final_heatmap = np.float32(final_heatmap) / 255
    final_heatmap = final_heatmap[..., ::-1]
    final_cam = 0.5 * final_original_img / 255 + 0.3 * final_heatmap
    final_cam = (final_cam - np.min(final_cam)) / (np.max(final_cam) - np.min(final_cam))
    plt.imsave(os.path.join(final_saleincy_map_path, 'cam.jpg'), final_cam)

    from bbox_function import save_prototype_original_img_with_bbox1
    from utils.helpers import find_high_activation_crop

    final_crop_threshold = 99
    sf, xf, zf, yf = find_high_activation_crop(sum_each_saliency, final_crop_threshold)
    final_crop_patch = final_original_img[sf:xf, zf:yf]
    plt.imsave(os.path.join(final_saleincy_map_path, 'final_crop_patch.jpg'), final_crop_patch)
    read_final_original_img = cv2.imread(os.path.join(final_saleincy_map_path, 'original_img.jpg'))[..., ::-1]
    read_final_heatmap = cv2.imread(os.path.join(final_saleincy_map_path, 'cam.jpg'))[..., ::-1]

    crop_final_img = save_prototype_original_img_with_bbox1(img_rgb=read_final_original_img, bbox_height_start=sf,
                                                        bbox_height_end=xf,
                                                        bbox_width_start=zf, bbox_width_end=yf,
                                                        color=(0, 255, 255))

    crop_final_heatmap = save_prototype_original_img_with_bbox1(img_rgb=read_final_heatmap, bbox_height_start=sf,
                                                            bbox_height_end=xf,
                                                            bbox_width_start=zf, bbox_width_end=yf,
                                                            color=(0, 255, 255))

    plt.imsave(os.path.join(final_saleincy_map_path, 'crop_original_img.jpg'), crop_final_img)
    plt.imsave(os.path.join(final_saleincy_map_path, 'crop_cam.jpg'), crop_final_heatmap)

    if os.path.exists(current_mask_img_root_path):
        shutil.rmtree(current_mask_img_root_path)
        print("fire_remove")
    else:
        print("fire_not_exist")

    if os.path.exists(each_saleincy_map_path):
        shutil.rmtree(each_saleincy_map_path)
        print("fire_remove")
    else:
        print("fire_not_exist")

    print("The saliency map is generated.")
    print("Max region t:", logits_index)
    print("feature_vector_saliency_map_path:")
    print(os.path.join(final_saleincy_map_path))





