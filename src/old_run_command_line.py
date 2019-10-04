'''
Created on Feb 13, 2018

@author: dani
'''

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import timeit
import os
import sys
#sys.path.insert(0, '../src')
import run
from util import convert_database_to_files
from evaluation.evaluation import evaluation
from util import parameter_estimation
import glob
from cnn import inception_pre_trained_feature_extraction

def run_evaluation():

    #Dani lap pro
    #path_database = '/Users/dani/Dropbox/prog/python/romfla/cells/'
    #path_cnn_trained = '/Users/dani/Dropbox/prog/python/romfla/cells/features/model.ckpt'
    #path_cnn_trained = '/Users/dani/Dropbox/prog/python/romfla/cells/output/inception_resnet_v2.ckpt'
    #path_output = path_database + 'output/'

    path_database = '/Users/dani/Dropbox/prog/python/romfla/cells/'
    path_cnn_trained = '/Users/dani/Dropbox/prog/python/romfla/cells/output/inception_resnet_v2.ckpt'
    path_output = path_database + 'output/'

    preprocessing_method = 'None'
    distance = 'ed' #euclidean
    searching_method = 'lsh'
    percent_database = 1
    percent_query = 1
    k_accuracy = 5

    #glcm
    #feature_extraction_method = 'glcm'
    #list_of_parameters = ['2', '256']#learning rate, number_of_epochs
    #lbp
    #feature_extraction_method = 'lbp'
    #list_of_parameters = ["2"]#radio
    #hog
    #feature_extraction_method = 'hog'
    #list_of_parameters = ["2", "12"]#number of cells and lenght blocks
    #cnn trained
    #feature_extraction_method = 'cnn_training'
    #list_of_parameters = ['0.01', '0']
    #inception
    feature_extraction_method = 'cnn'
    list_of_parameters = ['0','0']
    #fotf and inception
    #feature_extraction_method = 'fotf'
    #list_of_parameters = []
    #daisy
    #feature_extraction_method = 'daisy'
    #list_of_parameters = ['4', '3', '2', '8']

    start = timeit.default_timer()

    name_images_database, labels_database = convert_database_to_files.get_name_labels(path_database)
    MAP, ACCURACY, fig = evaluation(name_images_database, labels_database, name_images_database, labels_database,path_output,feature_extraction_method,distance,list_of_parameters,preprocessing_method,searching_method,path_cnn_trained=path_cnn_trained,percent_query=percent_query,percent_database=percent_database,k_accuracy=k_accuracy)

    print('Map for the database = ', np.mean(MAP))

    for i in range(len(MAP)):
        print('Map for class ', i, MAP[i])

    print('Accuracy for the database = ', np.mean(ACCURACY))

    for i in range(len(MAP)):
        print('Accuracy for class ', i, ACCURACY[i])

    stop = timeit.default_timer()
    print("Time =", stop - start)

    fig.savefig(path_output + 'average_precision_' + feature_extraction_method + '_' + searching_method + '.png')

def run_get_time():

    #dani laptop
    #path_database = '/Users/dani/Desktop/fibers_small/'
    #path_cnn_trained = '/Users/dani/Desktop/fibers_small/features/model.ckpt'
    #path_cnn_trained = '/Users/dani/Desktop/fibers_small/features/inception_resnet_v2_2016_08_30.ckpt'
    #path_retrieval = '/Users/dani/Desktop/fibers_small/query/'
    #path_output = path_database + 'features/'

    #cnn
    path_database = '/home/users/dani/databases/hexemer_split/hexemer_test/'
    path_retrieval = '/home/users/dani/databases/hexemer_split/hexemer_train/query/'
    path_cnn_trained = '/home/users/dani/databases/hexemer_split/hexemer_train/features/model.ckpt'
    #path_cnn_trained = '/home/users/dani/databases/models/inception_resnet_v2_2016_08_30.ckpt'
    path_output = path_database + 'features/'

    distance = 'ed'
    searching_method_all = ['lsh']
    preprocessing_method = 'None'
    percent_database = 0.6

    #fotf
    feature_extraction_method = 'cnn_training'
    list_of_parameters = ['0','0','0']

    name_images_database, labels_database, name_images_query, labels_query = convert_database_to_files.get_name_labels(path_database,path_retrieval)

    time_of_extraction_features, time_of_retrieval_images = evaluation.get_time_first_second_execution(name_images_database, labels_database, name_images_query, labels_query,path_cnn_trained,path_output,feature_extraction_method,distance,list_of_parameters,preprocessing_method,searching_method_all,percent_database=percent_database)

    print('Time for extraction of features = ', time_of_extraction_features[0,0])
    cont = 0
    for searching_method in searching_method_all:
        print('Time for the first execution using ', searching_method, '= ', time_of_retrieval_images[cont,0])
        print('Time for the second execution using ', searching_method, '= ', time_of_retrieval_images[cont,1])
        cont+=1

def run_parameter_estimation():

    #dani laptop
    path_database = '/Users/dani/Desktop/fibers_small/'
    path_output = path_database + 'features/'

    #cnn
    #path_database = '/home/users/dani/databases/cells/cells_train/'
    #path_output = path_database + 'features/'

    distance = 'ed'
    searching_method = 'kd'
    preprocessing_method = 'simple'
    percent_evaluate = 0.1

    ############################ EXTRACTION METHOD ################################
    #glcm
    feature_extraction_method = 'glcm'
    parameter1 = ['2','3','4','5','6','7','8','9','10','11','12','13','14','15']#distance from 1 to 12
    parameter2 = ['2', '3', '4', '5','6','7','8']#graylevel
    parameter3 = ['8']#graylevel
    parameters_variation = [parameter1, parameter2, parameter3]

    #lbp
    #feature_extraction_method = 'lbp'
    #parameter1 = ['2', '3', '4', '5', '6', '7', '8', '9', '10','11','12','13','14','15']#radio
    #parameters_variation = [parameter1]

    #hog #number of cells and blocks
    #feature_extraction_method = 'hog'
    #parameter1 = ['2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25']#number of cells
    #parameter2 = ['2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25']#number of blocks
    #parameters_variation = [parameter1, parameter2]

    #Daisy
    #feature_extraction_method = 'daisy'
    #parameter1 = ['1','2','3','4','5','6','7','8','9','10']
    #parameter2 = ['1','2','3','4','5','6','7','8','9','10']
    #parameter3 = ['1','2','3','4','5','6','7','8','9','10']
    #parameter4 = ['1','2','3','4','5','6','7','8','9','10']
    #parameters_variation = [parameter1, parameter2, parameter3, parameter4]

    start = timeit.default_timer()

    #get the list of names and labels
    name_images_database, labels_database = convert_database_to_files.get_name_labels(path_database)

    best_Map, best_parameter,list_parameters_map = parameter_estimation.parameter_estimation(name_images_database, labels_database, name_images_database, labels_database, path_output, feature_extraction_method, distance, parameters_variation, searching_method,preprocessing_method,percent_evaluate)

    for i in list_parameters_map:
        print('MAP =', i[4], '; p1 = ',i[0], '; p2 = ',i[1], '; p3 = ',i[2], '; p4 = ',i[3])

    print('Best MAP for',feature_extraction_method,'=', np.mean(best_Map))
    print('p1 =',best_parameter[0], '; p2 = ',best_parameter[1], '; p3 = ',best_parameter[2], '; p4 = ',best_parameter[3])

    stop = timeit.default_timer()
    print("Time =", stop - start)

    np.save(path_output + feature_extraction_method + '_Map_parameter_estimation', np.asarray(list_parameters_map))

def run_retrieval_process_using_folders_structure():
    ############################# Paths ############################################
    #dani machine
    path_database = '/Users/dani/Desktop/site_cric/test_pequeno/'
    path_cnn_pre_trained = '/Users/dani/Desktop/celulas_6_classes/celulas_6_classes_train_png/tensor/files/inception_resnet_v2_2016_08_30.ckpt'
    #path_cnn_pre_trained = '/Users/dani/Desktop/site_cric/test_pequeno/output/model.ckpt' #Only to initializing the cnn

    #path_save_cnn = '/Users/dani/Desktop/celulas_6_classes/celulas_6_classes_train_png/tensor/files/inception_resnet_v2_2016_08_30.ckpt' #This variable will be used on the inception resnet pre trained
    path_save_cnn = '/Users/dani/Desktop/site_cric/test_pequeno/output/model.ckpt' #To save and use in the retrieval process
    path_retrieval = '/Users/dani/Desktop/site_cric/test_pequeno/query/'

    #path_database = '/Users/dani/Desktop/site/celulas299_train/'
    #path_cnn_trained = '/Users/dani/Desktop/cells_small/features/inception_resnet_v2_2016_08_30.ckpt'
    #path_cnn_trained = '/Users/dani/Desktop/site/celulas299_2_classes/output/model.ckpt-65043'
    #path_retrieval = '/Users/dani/Desktop/site/celulas299_train/query/'

    #path_output = path_database + 'output/'


    #cnn machine
    #path_database = '/home/users/dani/databases/cells/cells_train_augmentation/database/'
    #path_cnn_trained = '/home/users/dani/databases/cells/cells_train_augmentation/features/model.ckpt'
    #path_retrieval = '/home/users/dani/databases/cells/cells_train_augmentation/query'

    path_output = path_database + 'output/'

    ############################# Parameters #######################################
    feature_extraction_method = 'cnn_training_inception'
    preprocessing_method = 'None'
    searching_method = 'bf'
    distance = 'ed'
    number_of_images = 10
    #list_of_parameters = ['0.1','100']
    #list_of_parameters = ['2','8','8']#distance, number of bins
    #list_of_parameters = ['4', '3', '2', '8']
    #list_of_parameters = ['0.01','1']
    list_of_parameters = ['0.05','3']

    #get the list of names and labels
    name_images_database, labels_database, name_images_query, labels_query = convert_database_to_files.get_name_labels(path_database,path_retrieval)

    #inception_feature_extraction.features_extraction_new(name_images_database,labels_database,path_cnn_trained,feature_extraction_method)

    ############################ Calling function ##################################
    start = timeit.default_timer()

    _, train_time, _ = run.run_command_line(name_images_database,labels_database,name_images_query,labels_query,path_cnn_pre_trained,path_save_cnn,path_output,feature_extraction_method,distance,number_of_images,list_of_parameters,preprocessing_method,searching_method, False)

    stop = timeit.default_timer()

    print("Total time = ", stop - start)
    print("Train time = ", train_time)

#this function was made exclusively for the structure of the kyager_database
def run_retrieval_process_using_txt_file():
    #dani machine
    root = '/Users/dani/Desktop/kyager_data_raw'
    path_database_class0 = '/Users/dani/Desktop/kyager_data_raw/WAXS.txt'
    path_database_class1 = '/Users/dani/Desktop/kyager_data_raw/SAXS.txt'
    path_query_class0 = '/Users/dani/Desktop/kyager_data_raw/WASXS_query.txt'
    path_query_class1 = '/Users/dani/Desktop/kyager_data_raw/SAXS_query.txt'
    path_cnn_trained = '/Users/dani/Desktop/kyager_data_raw/model.ckpt'
    path_output = '/Users/dani/Desktop/kyager_data_raw/features/'

    #dani cnn
    #root = '/home/users/dani/databases/kyager_data'
    #path_database_class0 = '/home/users/dani/databases/kyager_data/ringisotropic2_database.txt'
    #path_database_class1 = '/home/users/dani/databases/kyager_data/ringtextured2_database.txt'
    #path_query_class0 = '/home/users/dani/databases/kyager_data/ringisotropic2_query.txt'
    #path_query_class1 = '/home/users/dani/databases/kyager_data/ringtextured2_query.txt'
    #path_cnn_trained = '/home/users/dani/databases/kyager_data/model.ckpt'
    #path_output = '/home/users/dani/databases/kyager_data/features/'


    feature_extraction_method = 'cnn_training'
    searching_method = 'kd'
    preprocessing_method = 'simple'
    distance = 'ed'
    number_of_images = 10
    list_of_parameters = ['0.01','50000']


    with open(path_database_class0) as f:
        name_database_class0 = f.read().splitlines()

    with open(path_database_class1) as f:
        name_database_class1 = f.read().splitlines()

    with open(path_query_class0) as f:
        name_query_class0 = f.read().splitlines()

    with open(path_query_class1) as f:
        name_query_class1 = f.read().splitlines()

    for i in range(len(name_database_class0)):
        name_database_class0[i] = root + name_database_class0[i]

    for i in range(len(name_database_class1)):
        name_database_class1[i] = root + name_database_class1[i]

    for i in range(len(name_query_class0)):
        name_query_class0[i] = root + name_query_class0[i]

    for i in range(len(name_query_class1)):
        name_query_class1[i] = root + name_query_class1[i]

    labels_database_class0 = np.zeros(len(name_database_class0),dtype=np.int)
    labels_database_class1 = np.ones(len(name_database_class1),dtype=np.int)
    labels_query_class0 = np.zeros(len(name_query_class0),dtype=np.int)
    labels_query_class1 = np.ones(len(name_query_class1),dtype=np.int)

    name_images_database = []
    name_images_query = []

    name_images_database.extend(name_database_class0)
    name_images_database.extend(name_database_class1)
    labels_database = np.concatenate((labels_database_class0,labels_database_class1))

    name_images_query.extend(name_query_class0)
    name_images_query.extend(name_query_class1)
    labels_query = np.concatenate((labels_query_class0,labels_query_class1))

    #preprocessing to remove files inexisting
    name_images_database, labels_database = convert_database_to_files.preprocessing(name_images_database, labels_database)
    name_images_query, labels_query = convert_database_to_files.preprocessing(name_images_query, labels_query)

    run.run_command_line(name_images_database,labels_database,name_images_query,labels_query,path_cnn_trained,path_output,feature_extraction_method,distance,number_of_images,list_of_parameters,preprocessing_method,searching_method, False)

#This function was made exclusively for the structure of the Saxsgen
def start_Saxsgen():
    #dani machine
    path_database = '/Users/dani/Desktop/Saxsgen/'
    path_cnn_trained = '/Users/dani/Desktop/Saxsgen_features/model.ckpt'
    path_output = '/Users/dani/Desktop/Saxsgen_features/'

    #dani dresden
    #path_database = '/home/users/dani/databases/Saxsgen/'
    #path_cnn_trained = '/home/users/dani/databases/Saxsgen_features/model.ckpt'
    #path_output = '/home/users/dani/databases/Saxsgen_features/'

    feature_extraction_method = 'cnn_training'
    searching_method = 'kd'
    preprocessing_method = 'log'
    distance = 'ed'
    number_of_images = 10
    list_of_parameters = ['0.01','50000']

    folders = glob.glob(path_database + '*/')

    #get the name and labels for the database
    cont = 0
    name_images_database = []
    labels_database = []
    name_images_query = []
    labels_query = []
    for folder in folders:
        name_images = glob.glob(folder + "/*.jpg")
        labels = np.zeros(len(name_images),dtype=np.int)
        labels[:] = cont

        name_images_database.extend(name_images)
        labels_database.extend(labels)

        name_query = name_images_database.pop()
        label_query = labels_database.pop()

        name_images_query.append(name_query)
        labels_query.append(label_query)

        cont+=1
    labels_database = np.asarray(labels_database)
    labels_query = np.asarray(labels_query)

    run.run_command_line(name_images_database,labels_database,name_images_query,labels_query,path_cnn_trained,path_output,feature_extraction_method,distance,number_of_images,list_of_parameters,preprocessing_method,searching_method, False)

#run_retrieval_process_using_txt_file()
#start_Saxsgen()
run_retrieval_process_using_folders_structure()
#run_parameter_estimation()
#run_get_time()
#run_evaluation()
