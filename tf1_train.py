import random
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K
from prepare_data import load_dataset, resize_image, IMAGE_SIZE        #從tf1_03_prepare_data.py中載入方法

class Dataset:                                                                #create Dataset class
    def __init__(self, path_name):
        self.train_images = None        #-----定義訓練集
        self.train_labels = None
        self.valid_images = None        #-----驗證集
        self.valid_labels = None
        self.test_images = None         #-----測試集
        self.test_labels = None
        self.path_name = path_name      #所有資料集載入的路徑位置
        self.input_shape = None         #資料shape大小

    def load(self, img_rows, img_cols, img_channels, nb_classes):             #Data Pre-processing
        images, labels = load_dataset(self.path_name)
        #分割Data Set: Training Set, Validation Set, Testing Set.
        random.seed('foobar')                                                 #設定 random seed
        train_images, valid_images, train_labels, valid_labels = train_test_split(images,
                                                                                  labels,
                                                                                  test_size = 0.3,
                                                                                  random_state = random.randint(0, 100))
        _, test_images, _, test_labels = train_test_split(images,
                                                          labels,
                                                          test_size = 0.5,
                                                          random_state = random.randint(0, 100))
        #從K.image_dim_ordering()取得圖片資料順序，如果是'th'，代表輸入圖片的順序是：channels(RGB),rows,cols
        #如果不是'th'，則輸入順序為:rows,cols,channels
        if K.image_dim_ordering() == 'th':
            train_images = train_images.reshape(train_images.shape[0], img_channels, img_rows, img_cols)
            valid_images = valid_images.reshape(valid_images.shape[0], img_channels, img_rows, img_cols)
            test_images = test_images.reshape(test_images.shape[0], img_channels, img_rows, img_cols)
            self.input_shape = (img_channels, img_rows, img_cols)
        else:
            train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, img_channels)
            valid_images = valid_images.reshape(valid_images.shape[0], img_rows, img_cols, img_channels)
            test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, img_channels)
            self.input_shape = (img_rows, img_cols, img_channels)
            #show the shape
            print(train_images.shape[0], 'train samples')
            print(valid_images.shape[0], 'valid samples')
            print(test_images.shape[0], 'test samples')

            #模型使用categorical_crossentropy作為loss function
            #將label以one hot encoding方式編碼(呼叫np_utils.to_categorical方法)
            train_labels = np_utils.to_categorical(train_labels, nb_classes)
            valid_labels = np_utils.to_categorical(valid_labels, nb_classes)
            test_labels = np_utils.to_categorical(test_labels, nb_classes)

            #將圖片資料數位化
            train_images = train_images.astype('float32')
            valid_images = valid_images.astype('float32')
            test_images = test_images.astype('float32')

            #將圖片資料標準化，數值落在0~1之間
            train_images /= 255
            valid_images /= 255
            test_images /= 255
            
            #設定最後要進入模型的資料
            self.train_images = train_images
            self.valid_images = valid_images
            self.test_images  = test_images
            self.train_labels = train_labels
            self.valid_labels = valid_labels
            self.test_labels  = test_labels

class Model:                                                                  #create CNN model class
    def __init__(self):
        self.model = None
    def build_model(self, dataset, nb_classes):                               #建立CNN模型
        self.model = Sequential()                                           #使用Keras建立一個空的線性堆疊模型
        #-----依序在模型中增加CNN需要的各層與設定-----
        self.model.add(Convolution2D(32, 3, 3, border_mode='same', 
                                     input_shape = dataset.input_shape))    #1-使用32個3X3的filter matrix
        self.model.add(Activation('relu'))                                  #2-使用RELU啟用
        self.model.add(Convolution2D(32, 3, 3))                             #3-使用32個3X3的filter matrix                             
        self.model.add(Activation('relu'))                                  #4-使用RELU啟用
        self.model.add(MaxPooling2D(pool_size=(2, 2)))                      #5-使用2x2的matrix做出池化層
        #self.model.add(Dropout(0.25))                                       #6-Dropout層
        self.model.add(Convolution2D(64, 3, 3, border_mode='same'))         #7-使用64個3X3的filter matrix
        self.model.add(Activation('relu'))                                  #8-使用RELU啟用
        self.model.add(Convolution2D(64, 3, 3))                             #9-使用64個3X3的filter matrix
        self.model.add(Activation('relu'))                                  #10-使用RELU啟用
        self.model.add(MaxPooling2D(pool_size=(2, 2)))                      #11-使用2x2的matrix做出池化層
        self.model.add(Dropout(0.25))                                       #12-Dropout層
        #-------進入全連結層-------
        self.model.add(Flatten())                                           #13-Flatten層
        self.model.add(Dense(512))                                          #14-512個節點的hidden layer
        self.model.add(Activation('relu'))                                  #15-使用RELU做激勵函數   
        self.model.add(Dropout(0.3))                                        #16-Dropout層
        self.model.add(Dense(nb_classes))                                   #17-加入輸出層
        self.model.add(Activation('softmax'))                               #18-使用softmax輸出one hot encoding的pattern
        self.model.summary()                                                #輸出模型全部資訊

    def train(self, dataset, batch_size, nb_epoch, data_augmentation):        #訓練模型
        sgd = SGD(lr = 0.01, decay = 1e-6, 
                  momentum = 0.9, nesterov = True)                            #使用SGD+momentum的演算法進行模型訓練  
        self.model.compile(loss = 'categorical_crossentropy',
                           optimizer = sgd,                                   #調整學習率的其中一種演算法(SGD)
                           metrics = ['accuracy'])                            #至此完成所以訓練模型的設定動作
        #data_augmentation參數:是否要將訓練資料透過平移、旋轉或加入雜訊等方式來處理
        #Data Augmentation有助於提升訓練資料樣本數，但也會同時增加訓練的時間
        if not data_augmentation:                                             #假如不做data_augmentation
            self.model.fit(dataset.train_images,
                           dataset.train_labels,
                           batch_size = batch_size,
                           nb_epoch = nb_epoch,
                           validation_data = (dataset.valid_images, dataset.valid_labels),
                           shuffle = True)

        #----------進行data_augmentation----------
        else:
            #建立datagen物件，datagen物件可以用來生成資料
            datagen = ImageDataGenerator(
                featurewise_center = False,                                  #是否使輸入資料去中心化（均值為0），
                samplewise_center  = False,                                  #是否使輸入資料的每個樣本均值為0
                featurewise_std_normalization = False,                       #是否將資料標準化（輸入資料除以資料集的標準差）
                samplewise_std_normalization  = False,                       #是否將每個樣本資料除以自身的標準差
                zca_whitening = False,                                       #是否對輸入資料施以ZCA白化
                rotation_range = 20,                                         #產生資料時圖片隨機轉動的角度(範圍為0～180)
                width_shift_range  = 0.2,                                    #產生資料時圖片水平偏移的幅度（單位為圖片寬度的佔比，0~1之間的浮點數）
                height_shift_range = 0.2,                                    #同上，只不過這裡是垂直
                horizontal_flip = True,                                      #是否進行隨機水平翻轉
                vertical_flip = False)                                       #是否進行隨機垂直翻轉
            datagen.fit(dataset.train_images)                                #計算全部續練樣本數
            #利用datagen物件生成資料並開始訓練
            self.model.fit_generator(datagen.flow(dataset.train_images, dataset.train_labels,
                                                   batch_size = batch_size),
                                     samples_per_epoch = dataset.train_images.shape[0],
                                     nb_epoch = nb_epoch,
                                     validation_data = (dataset.valid_images, dataset.valid_labels))

    def save_model(self, MODEL_PATH):                                         #儲存訓練好參數的模型
         self.model.save(MODEL_PATH)

    def load_model(self, MODEL_PATH):                                         #載入已儲存訓練好參數的模型
         self.model = load_model(MODEL_PATH)

    def evaluate(self, dataset):                                              #測試評估模型
         score = self.model.evaluate(dataset.test_images, dataset.test_labels, verbose = 1) #verbose = 1 為輸出進度條記錄
         print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))

    def face_predict(self, image):                                            #使用模型進行人臉辨識
        #從K.image_dim_ordering()取得圖片資料順序，如果是'th'，代表輸入圖片的順序是：channels(RGB),rows,cols
        #如果不是'th'，則輸入順序為:rows,cols,channels
        if K.image_dim_ordering() == 'th' and image.shape != (1, 3, IMAGE_SIZE, IMAGE_SIZE):
            image = resize_image(image)                                       #尺寸必須與訓練集一致都應該是IMAGE_SIZE x IMAGE_SIZE
            image = image.reshape((1, 3, IMAGE_SIZE, IMAGE_SIZE))             #與訓練模型不同，這次只是針對1張圖片進行預測
        elif K.image_dim_ordering() == 'tf' and image.shape != (1, IMAGE_SIZE, IMAGE_SIZE, 3):
            image = resize_image(image)
            image = image.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))
        #-----標準化資料-----
        image = image.astype('float32')                                       #astype()強制類型轉換
        image /= 255
        result = self.model.predict_proba(image)                              #輸出預測結果(one hot encoding type)
        print('face_predict_result_one-hot:', result)
        result = self.model.predict_classes(image)                            #轉換成實際預測結果0,1,2....其中一個，就看目前有幾個類別要預測
        print('face_predict_result_label_class:', result)
        return result[0]                                                      #回傳預測結果

path_name = './data/'
img_rows, img_cols, img_channels = IMAGE_SIZE, IMAGE_SIZE, 3
nb_classes = 3                                                                #label幾類
batch_size = 15
nb_epoch = 12
data_augmentation = True                                                      #是否資料擴增
MODEL_PATH = './model/three_user_face_model.h5'                               #完成後模型存檔位置

if __name__ == '__main__':                                                    #開始進行訓練
    dataset_face = Dataset(path_name)                                         #載入Dataset類別，創dataset_face物件
    CNN_model = Model()                                                       #載入Model類別，創CNN_model物件
    #----------------------------------------
    dataset_face.load(img_rows, img_cols, img_channels, nb_classes)
    CNN_model.build_model(dataset_face, nb_classes)
    CNN_model.train(dataset_face, batch_size, nb_epoch, data_augmentation)
    CNN_model.save_model(MODEL_PATH)

if __name__ == '__main__':                                                    #進行測試
    dataset_face = Dataset(path_name)
    CNN_model = Model()
    #----------------------------------------
    dataset_face.load(img_rows, img_cols, img_channels, nb_classes)
    CNN_model.load_model(MODEL_PATH)
    CNN_model.evaluate(dataset_face)