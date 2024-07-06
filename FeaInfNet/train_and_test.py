import torch
from utils.helpers import list_of_distances

def _train_or_test(model, dataloader, optimizer=None, class_specific=True,
                   coefs=None, log=print):
    import time

    is_train = optimizer is not None
    start = time.time()
    epsilon = 1e-12
    n_examples = 0
    n_correct = 0
    n_batches = 0
    total_cross_entropy = 0
    total_cluster_cost = 0
    total_Negative_Clst = 0
    total_Positive_Clst = 0
    total_Negative_Sep = 0
    total_Positive_Sep = 0
    total_Feature_Consistency = 0
    total_separation_cost = 0
    total_avg_separation_cost = 0

    for i, (image, label) in enumerate(dataloader):
        image = image.cuda()
        target = label.cuda()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()

        with grad_req:
            output, positive_distance, negative_distance, feature_masks_consistency_activation = model(image)
            pos_loss, neg_loss = 0, 0
            pos_loss1 = torch.log(output[:, 1] + epsilon) * (1 - output[:, 1]) ** 2 * target.float()
            pos_loss += (pos_loss1.sum() / (target.sum() + epsilon)) if target.sum() != 0 else 0
            neg_loss1 = torch.log(1 - output[:, 1] + epsilon) * (output[:, 1]) ** 2 * (1 - target.float())
            neg_loss += (neg_loss1.sum() / ((1 - target).sum()) + epsilon) if (1 - target).sum() != 0 else 0
            focal_cross_entropy = - pos_loss - neg_loss
            cross_entropy = focal_cross_entropy
            batch_num = positive_distance.shape[0]
            Feature_Consistency_num = feature_masks_consistency_activation.shape[0]
            target_positve_num = torch.sum(target)
            target_negative_num = batch_num - target_positve_num
            target_positve = target
            target_negative = 1 - target

            if class_specific:
                min_negative_distance, _ = torch.min(negative_distance, dim=-1)
                max_min_negative_distance, _ = torch.max(min_negative_distance, dim=-1)
                Negative_Clst = torch.sum(max_min_negative_distance * target_negative) / (target_negative_num + epsilon)

                min_negative_distance, _ = torch.min(negative_distance, dim=-1)
                min_min_negative_distance, _ = torch.min(min_negative_distance, dim=-1)
                Negative_Sep = - torch.sum(min_min_negative_distance * target_negative) / (target_negative_num + epsilon)

                min_positive_distance, _ = torch.min(positive_distance, dim=-1)
                min_min_positive_distance, _ = torch.min(min_positive_distance, dim=-1)
                Positive_Clst = torch.sum(min_min_positive_distance * target_positve) / (target_positve_num + epsilon)

                max_positive_distance, _ = torch.max(positive_distance, dim=-1)
                max_max_positive_distance, _ = torch.max(max_positive_distance, dim=-1)
                Positive_Sep = - torch.sum(max_max_positive_distance * target_positve) / (target_positve_num + epsilon)

                Feature_Consistency = torch.sum(feature_masks_consistency_activation) / (Feature_Consistency_num + epsilon)

            _, predicted = torch.max(output.data, 1)
            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()
            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            total_Negative_Clst += Negative_Clst.item()
            total_Positive_Clst += Positive_Clst.item()
            total_Negative_Sep += Negative_Sep.item()
            total_Positive_Sep += Positive_Sep.item()
            total_Feature_Consistency += Feature_Consistency.item()
            total_cluster_cost += 0
            total_separation_cost += 0
            total_avg_separation_cost += 0

        cluster_cost = 0
        separation_cost = 0
        l1 = 0
        if is_train:
            if class_specific:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy
                            + coefs['positive_clst'] * Positive_Clst
                            + coefs['positive_sep'] * Positive_Sep
                            + coefs['negative_clst'] * Negative_Clst
                            + coefs['negative_sep'] * Negative_Sep
                            + coefs['feature_mask_consistency'] * Feature_Consistency)
                else:
                    loss = cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost + 1e-4 * l1
            else:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy
                            + coefs['clst'] * cluster_cost
                            + coefs['l1'] * l1)
                else:
                    loss = cross_entropy + 0.8 * cluster_cost + 1e-4 * l1

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        del image
        del target
        del output
        del predicted

    end = time.time()
    time = end - start
    cross_ent = total_cross_entropy / n_batches
    negative_cluster = total_Negative_Clst / n_batches
    positive_cluster = total_Positive_Clst / n_batches
    negative_separation = total_Negative_Sep / n_batches
    positive_separation = total_Positive_Sep / n_batches
    feature_consistency = total_Feature_Consistency / n_batches
    accu = n_correct / n_examples * 100

    log('\ttime: \t{0}'.format(time))
    log('\tcross ent: \t{0}'.format(cross_ent))
    log('\tnegative_cluster: \t{0}'.format(negative_cluster))
    log('\tpositive_cluster:\t{0}'.format(positive_cluster))
    log('\tnegative_separation:\t{0}'.format(negative_separation))
    log('\taccu: \t\t{0}%'.format(accu))
    log('\tpositive_separation: \t\t{0}'.format(positive_separation))
    log('\tfeature_consistency: \t\t{0}'.format(feature_consistency))

    p = model.module.prototype_vectors.view(model.module.num_prototypes, -1).cpu()

    with torch.no_grad():
        p_avg_pair_dist = torch.mean(list_of_distances(p, p))
    p_dist_pair = p_avg_pair_dist.item()
    log('\tp dist pair: \t{0}'.format(p_dist_pair))

    accuracy = n_correct / n_examples
    result = []
    result.append(time)
    result.append(cross_ent)
    result.append(negative_cluster)
    result.append(positive_cluster)
    result.append(negative_separation)
    result.append(accu)
    result.append(positive_separation)
    result.append(feature_consistency)

    return accuracy, result

def train(model, dataloader, optimizer, class_specific=False, coefs=None, log=print):
    assert (optimizer is not None)
    log('\ttrain')
    model.train()
    accuracy, train_result = _train_or_test(model=model, dataloader=dataloader, optimizer=optimizer,
                                            class_specific=class_specific, coefs=coefs, log=log)
    return accuracy, train_result

def tst(model, dataloader, class_specific=False, log=print):
    log('\ttest')
    model.eval()
    accuracy, test_result = _train_or_test(model=model, dataloader=dataloader, optimizer=None,
                                           class_specific=class_specific, log=log)
    return accuracy, test_result

def last_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False

    for p in model.module.add_on_layers.parameters():
        p.requires_grad = False

    model.module.prototype_vectors.requires_grad = False

    for p in model.module.last_layer.parameters():
        p.requires_grad = True

    model.module.positive_prototype_vectors.requires_grad = False
    model.module.negative_prototype_vectors.requires_grad = False
    model.module.positive_layer.requires_grad = True
    model.module.negative_layer.requires_grad = True

    log('\tlast layer')

def warm_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False

    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True

    model.module.prototype_vectors.requires_grad = True

    for p in model.module.last_layer.parameters():
        p.requires_grad = True

    model.module.positive_prototype_vectors.requires_grad = True
    model.module.negative_prototype_vectors.requires_grad = True
    model.module.positive_layer.requires_grad = True
    model.module.negative_layer.requires_grad = True

    log('\twarm')

def joint(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = True

    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True

    model.module.positive_prototype_vectors.requires_grad = True
    model.module.negative_prototype_vectors.requires_grad = True
    model.module.positive_layer.requires_grad = True
    model.module.negative_layer.requires_grad = True

    log('\tjoint')