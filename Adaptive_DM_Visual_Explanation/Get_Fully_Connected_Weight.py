import torch
import numpy as np

def Get_Model_Weight(Model_Path, Region_t, Prototype_Num, Prototype_Class):
    feainfnet = torch.load(Model_Path)

    print("positive_weights:")
    print(np.abs(feainfnet.positive_layer.cpu().detach().numpy()))
    print('\n')
    print("negative_weights:")
    print(np.abs(feainfnet.negative_layer.cpu().detach().numpy()))
    print('\n')

    if(Prototype_Class == 'Positive'):
        positive_layer = feainfnet.positive_layer.cpu().detach().numpy()
        print("Prototype_Class: ", Prototype_Class)
        print("Prototype_Num: ", Prototype_Num)
        print(np.abs(positive_layer)[Prototype_Num])
    else:
        negative_layer = feainfnet.negative_layer.cpu().detach().numpy()
        print("Prototype_Class: ", Prototype_Class)
        print("Region_t: ", Region_t)
        print("Prototype_Num: ", Prototype_Num)
        print(np.abs(negative_layer)[Region_t, Prototype_Num])

if __name__ == '__main__':
    Model_Path = './trained_model_path/'
    Region_t = 0 # Region t
    Prototype_Num = 1 # Prototype_Num
    Prototype_Class = 'Positive' # Positive/Negative

    Get_Model_Weight(Model_Path=Model_Path, Region_t=Region_t, Prototype_Num=Prototype_Num, Prototype_Class=Prototype_Class)