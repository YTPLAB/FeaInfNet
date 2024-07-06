import os
from prototype_adaptive_dm import Prototype_CL

def get_local_prototype(Local_Porotype_path, Feature_Region_t, Prototype_Num, Model_Path, Prototype_Class):
    Analysis_Path = os.path.join(Local_Porotype_path, Feature_Region_t, Prototype_Num)
    Analysis_Path_Name = os.listdir(Analysis_Path)[0]
    Complete_Analysis_Positive_Path = os.path.join(Local_Porotype_path, Feature_Region_t, Prototype_Num, Analysis_Path_Name)
    load_model_path = Model_Path

    print("The saliency map is being generated, please wait.")
    Prototype_CL(model_path=load_model_path, Prototype_path=Complete_Analysis_Positive_Path,
                 Feature_region_t=Feature_Region_t, prototype_num=Prototype_Num, prototype_class=Prototype_Class)

if __name__ == '__main__':
    Feature_Region_t = '0' # Prototype_Feature_t
    Prototype_Num = '0' # Prototype_Num
    Prototype_Class = 'Positive' # Positive/Negative

    Porotype_path = './positive_prototype_img_path' # positive_prototype_img_path/negative_prototype_img_path
    Model_Path = 'trained_models_path/'  # trained_model_file

    get_local_prototype(Local_Porotype_path=Porotype_path, Feature_Region_t=Feature_Region_t,
                        Prototype_Num=Prototype_Num, Model_Path=Model_Path, Prototype_Class=Prototype_Class)