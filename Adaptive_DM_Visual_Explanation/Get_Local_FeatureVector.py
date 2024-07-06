import os
from feature_vector_adaptive_dm import FeatureVector_CL

def get_local_feature_vector(Local_Porotype_path, Model_Path):
    Analysis_Path = os.path.join(Local_Porotype_path)
    Analysis_Path_Name = os.listdir(Analysis_Path)[0]
    Complete_Analysis_Positive_Path = os.path.join(Local_Porotype_path, Analysis_Path_Name)
    load_model_path = Model_Path

    print("The saliency map is being generated, please wait.")
    FeatureVector_CL(model_path=load_model_path, Feature_vector_path=Complete_Analysis_Positive_Path)

if __name__ == '__main__':

    FeatureVector_path = './input_image_path'  # The path of the image to be visualized by Adaptive-DM
    Model_Path = './trained_models_path/'  # trained_model_file

    get_local_feature_vector(Local_Porotype_path=FeatureVector_path, Model_Path=Model_Path)