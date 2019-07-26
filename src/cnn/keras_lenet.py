'''
Created on Jun 7, 2019

@author: flavio
'''

#import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";  # The GPU id to use, usually either "0" or "1";
#os.environ["CUDA_VISIBLE_DEVICES"]="0";  # Do other imports now...
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout 
from keras.layers import MaxPooling2D, Activation
from keras import optimizers
from keras.utils import to_categorical
from keras.models import load_model
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from skimage.transform import resize
import skimage.io as io
from src.util import parallel


def create_model(shape_in, num_classes, dropout_value = 0.5):
    model = Sequential()

    #First convolutional layer
    model.add(Conv2D(64, kernel_size=5, strides=(1, 1), padding='same', use_bias=True,
                     input_shape=(shape_in[0],shape_in[1],shape_in[2]),
                    name = 'conv_1'))
    model.add(Activation("relu"))

    #First max-pool layer
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), 
                           name = 'pool_1', padding='same'))
    
    #model.add(BatchNormalization())

    #Second convolutional layer
    model.add(Conv2D(64, kernel_size=3, strides=(1, 1), use_bias=True,
                     padding='same', name = 'conv_2'))
    model.add(Activation("relu"))

    #model.add(BatchNormalization())
    
    #Second max-pool layer
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), 
                           name = 'pool_2', padding='same'))
    
    #Third convolutional layer
    model.add(Conv2D(48, kernel_size=3, strides=(1, 1), use_bias=True,
                     padding='same', name = 'conv_3'))
    model.add(Activation("relu"))
    #model.add(BatchNormalization())

    #Third max-pool layer
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), 
                           name = 'pool_3', padding='same'))
    
    #Dropout
    model.add(Dropout(dropout_value))
    
    model.add(Flatten(name = 'flatten'))

    #First Dense
    model.add(Dense(192, name = 'dense_1', activation='relu', use_bias=True))
    
    #Dropout
    model.add(Dropout(dropout_value))

    #Second Dense
    model.add(Dense(64, name = 'dense_last', activation='relu', use_bias=True))
    
    #Dropout
    model.add(Dropout(dropout_value))

    #Softmax
    model.add(Dense(num_classes, activation='softmax', name = 'classification'))
    model.summary()
    return model

def im_resize(im,size1,size2):
    
    resize_ = np.zeros((size1,size2,3),dtype=np.uint8)
    if(len(im.shape) == 2):
        r = resize(im[:,:], (size1, size2),preserve_range=True)
        resize_[:,:,0] = r
        resize_[:,:,1] = r
        resize_[:,:,2] = r
    else:  
        r = resize(im[:,:,0], (size1, size2),preserve_range=True)
        g = resize(im[:,:,1], (size1, size2),preserve_range=True)
        b = resize(im[:,:,2], (size1, size2),preserve_range=True)
        resize_[:,:,0] = r
        resize_[:,:,1] = g
        resize_[:,:,2] = b
    
    return resize_

def read_database_parallel(im, name, label, im_size1 = 0, im_size2 = 0, num_channels = 3):
    
    if(im_size1 != 0):
        im = im_resize(im,im_size1,im_size2)
    
    #sometimes there are gray level images together with rgb images
    if(num_channels > 1 and len(im.shape)==2):
        im2 = np.zeros((im.shape[0],im.shape[1],3),dtype=np.uint8)
        im2[:,:,0]=im[:,:]
        im2[:,:,1]=im[:,:]
        im2[:,:,2]=im[:,:]
        im = im2

    return im,name,label,im.shape

    
def read_database(parameters):
    collection = io.imread_collection(parameters.NAME_IMAGES)
    res = parallel.apply_parallel(collection, collection.files, parameters.LABELS, read_database_parallel, {'im_size1': parameters.NEW_IMAGE_SIZE1, 'im_size2': parameters.NEW_IMAGE_SIZE2, 'num_channels': parameters.NUM_CHANNELS} )
    
    vector_images = []
    files = []
    for cont,e in enumerate(res):
        vector_images.append(e[0])
        files.append(e[1])
        parameters.LABELS[cont] = e[2]   
    parameters.NAME_IMAGES = files
    parameters.IMAGE_SIZE1 = res[0][3][0]  
    parameters.IMAGE_SIZE2 = res[0][3][1]
    
    vector_images = np.asarray(vector_images)
    
    return vector_images, parameters.LABELS
    

def train_lenet(parameters):
    shape_in = (parameters.IMAGE_SIZE1,parameters.IMAGE_SIZE2,parameters.NUM_CHANNELS)
    model = create_model(shape_in, parameters.NUM_CLASSES, dropout_value = 0.5)
    
    try:
        if(parameters.PATH_CNN_PRE_TRAINED != ''):
            model = load_model(parameters.PATH_CNN_PRE_TRAINED)
            print("Model restored from " + parameters.PATH_CNN_PRE_TRAINED)
        else:
            print("Initializing model randomly!")
    except:
        print("Initializing model randomly!")
        pass
    
    sgd = optimizers.SGD(lr=parameters.LEARNING_RATE, decay=1e-3)
    #keras.optimizers.Adam(lr = parameters.LEARNING_RATE)
    model.compile(sgd,
              loss='categorical_crossentropy', metrics=['accuracy'])
    
    train_datagen = ImageDataGenerator(rescale=1./255)
    
    X_train, Y_train = read_database(parameters)
    
    Y_train = to_categorical(Y_train)
    
    history = model.fit_generator(
      train_datagen.flow(X_train,Y_train, batch_size=parameters.BATCH_SIZE),
      steps_per_epoch=len(Y_train)/parameters.BATCH_SIZE,
      epochs=parameters.NUM_EPOCHS,verbose=1)
    
    model.save(parameters.PATH_SAVE_CNN)
    
    
def features_extraction_lenet(parameters):
    #layer_name = 'dense_last'
    
    try:
        model = load_model(parameters.PATH_SAVE_CNN)
        print('Model restored to extract features from ' + parameters.PATH_SAVE_CNN)
    except:
        try:
            model = load_model(parameters.PATH_CNN_PRE_TRAINED)
            print('Model restored to extract features from ' + parameters.PATH_CNN_PRE_TRAINED)
        except:
            raise ValueError('Model not found or model not compatible from: ',parameters.PATH_SAVE_CNN,'\nor\n',parameters.PATH_CNN_PRE_TRAINED)

    X_test, Y_test = read_database(parameters)

    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.layers[-2].output)
    feature_vectors_database = intermediate_layer_model.predict(X_test)    

    #features = pd.DataFrame(data=features)
    #features['Arousal'] = labels_regression_test

    #features.to_csv(path_save_csv, index=False)
    
    probability_vector = np.zeros((len(feature_vectors_database),parameters.NUM_CLASSES))
    return feature_vectors_database, parameters.NAME_IMAGES, parameters.LABELS, probability_vector
    