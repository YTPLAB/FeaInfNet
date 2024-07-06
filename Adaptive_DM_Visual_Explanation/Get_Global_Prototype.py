import os
from Get_Local_Prototype import get_local_prototype

def get_global_prototype(Global_Porotype_path, Model_Path, Prototype_Class):
    feature_t_list = os.listdir(Global_Porotype_path)

    for feature_t in feature_t_list:
        prototpye_num_list = os.listdir(os.path.join(Global_Porotype_path, feature_t))

        for prototpye_num in prototpye_num_list:
            get_local_prototype(Local_Porotype_path=Global_Porotype_path, Feature_Region_t=feature_t,
                                Prototype_Num=prototpye_num, Model_Path=Model_Path, Prototype_Class=Prototype_Class)

if __name__ == '__main__':
    Prototype_Class = 'Positive'  # Positive/Negative
    Porotype_path = './positive_prototype_img_path/' # positive_prototype_img_path/negative_prototype_img_path
    Model_Path = 'trained_models_path/'  # trained_model_file

    get_global_prototype(Global_Porotype_path=Porotype_path, Model_Path=Model_Path, Prototype_Class=Prototype_Class)