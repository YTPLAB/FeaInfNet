import os
import re
import torchvision.transforms as transforms
from settings import mean, std
from settings import train_dir
from utils.helpers import makedir
import torchvision.datasets as datasets
import torch
import numpy as np
import matplotlib.pyplot as plt

negative_prototype_file = './prototype_info1/negative_prototype_file.txt'
positive_prototype_file = './prototype_info1/positive_prototype_file.txt'

def read_and_parse_file(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            numbers = re.findall(r'[\d.]+', line)
            data.append([int(float(num)) for num in numbers])
    return data

negative_prototype_data = np.array(read_and_parse_file(negative_prototype_file))
positive_prototype_data = np.array(read_and_parse_file(positive_prototype_file))
print(negative_prototype_data)
print(positive_prototype_data)
img_size = 224
normalize = transforms.Normalize(mean=mean, std=std)
train_dataset = datasets.ImageFolder(
    train_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ]))

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=1, shuffle=False,
    num_workers=4, pin_memory=False)

positive_prototype_path = './positive_prototype_img_path'
positive_feature_num = 0

for feature_t in positive_prototype_data:
    print(feature_t)

    positive_prototype_num = 0
    for feature_x in feature_t:
        print(feature_x)
        makedir(os.path.join(positive_prototype_path, str(positive_feature_num), str(positive_prototype_num)))

        for i, (image, label) in enumerate(train_loader):
            image = image.cuda()
            target = label.cuda()
            print(i)
            if(i == positive_prototype_data[positive_feature_num, positive_prototype_num]):
                original_img = image[0].permute(1, 2, 0).detach().cpu().numpy()
                original_img = original_img - np.min(original_img)
                original_img = original_img / np.max(original_img)
                plt.imsave(os.path.join(positive_prototype_path, str(positive_feature_num), str(positive_prototype_num),
                                        str(positive_feature_num) + '-' + str(positive_prototype_num) + '.jpg'), original_img)

        positive_prototype_num += 1
    positive_feature_num += 1


