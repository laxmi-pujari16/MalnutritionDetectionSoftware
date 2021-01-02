from PyQt4 import QtGui
from PyQt4 import QtCore
from PyQt4.QtGui import QFileDialog
import sys
from PyQt4.QtCore import *
from PyQt4.QtGui import *

import time 

#import login
import home
import add
import error_log
import err_img
# import MySQLdb

import numpy as np
import cv2
import os

from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Convolution2D, Dropout, Dense, Flatten, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator, load_img
from numpy import array
from keras import regularizers
import matplotlib.pyplot as plt
#import mock
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array

from keras import preprocessing
from keras import backend as K
import tensorflow as tf



fname=""
fname1=""

# db = MySQLdb.connect("localhost","root","root","malnutrition")
# cursor = db.cursor()

# class Login(QtGui.QMainWindow, login.Ui_UserLogin):
#     def __init__(self):
#         super(self.__class__, self).__init__()
#         self.setupUi(self) 
#         self.pushButton.clicked.connect(self.log)
#         self.pushButton_2.clicked.connect(self.can)
#         self.pushButton_3.clicked.connect(self.addNew1)
        
#     def log(self):
#         i=0
#         a=self.lineEdit.text()
#         b=self.lineEdit_2.text()
#         sql = "SELECT * FROM login WHERE username='%s' and password='%s'" % (a,b)
#         try:
#             cursor.execute(sql)
#             results = cursor.fetchall()
#             for row in results:
#                 i=i+1
#         except Exception as e:
#            print(e)
#         if i>=0:
#             print("login success")
#             self.hide()
#             self.home=home()
#             self.home.show()
            
#         else:
#             print("login failed")
#             self.errlog=errlog()
#             self.errlog.show()
                    
#         db.close()
        
#     def can(self):
#         sys.exit()

#     def addNew1(self):
#         self.addNew=addNew()
#         self.addNew.show()

# class addNew(QtGui.QMainWindow, add.Ui_AdNewAdvertizer):
#     def __init__(self):
#         super(self.__class__, self).__init__()
#         self.setupUi(self)
#         self.pushButton.clicked.connect(self.save1)
#         self.pushButton_3.clicked.connect(self.can2)

#     def can2(self):
#         sys.exit()
        
#     def save1(self):
#         name=self.lineEdit.text()
#         email=self.lineEdit_2.text()
#         contact=self.lineEdit_3.text()
#         uname=self.lineEdit_4.text()
#         pwd=self.lineEdit_5.text()
#         sql = "INSERT INTO user(name, email, contact, username, password) VALUES ('%s', '%s', '%s', '%s', '%s' )" % (name,email,contact,uname,pwd)
#         try:
#                 cursor1.execute(sql)
#                 self.hide()
#                 db1.commit()
#         except:
#                 db1.rollback()
#                 self.erradd=erradd()
#                 self.erradd.show()
            

#         db1.close()
       

class home(QtGui.QMainWindow, home.Ui_Home):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.selimg)
        self.pushButton_2.clicked.connect(self.seldir)
        self.pushButton_3.clicked.connect(self.cnn)
        self.pushButton_4.clicked.connect(self.selimg2)
        self.pushButton_5.clicked.connect(self.ex)
        self.pushButton_6.clicked.connect(self.preproc)
        self.pushButton_7.clicked.connect(self.pred)
        

    def selimg(self):
        global fname
        self.QFileDialog = QtGui.QFileDialog(self)
        #self.QFileDialog.show()
        fname = QFileDialog.getOpenFileName(self, 'Open file','c:\\',"Image files (*.jpg *.png)")
        print(fname)
        dim = (600, 600) 
        img_to_show=cv2.imread(str(fname))
        resized = cv2.resize(img_to_show, dim, interpolation = cv2.INTER_AREA)
        cv2.imwrite("resized.jpg",resized)
        label = QLabel(self.label_5)
        pixmap = QPixmap("resized.jpg")
        label.setPixmap(pixmap)
        label.resize(pixmap.width(),pixmap.height())
        label.show()

    def selimg2(self):
        global fname1
        self.QFileDialog = QtGui.QFileDialog(self)
        #self.QFileDialog.show()
        fname1 = QFileDialog.getOpenFileName(self, 'Open file','c:\\',"Image files (*.jpg *.png)")
        print(fname1)
        dim = (600, 600) 
        img_to_show=cv2.imread(str(fname1))
        resized1 = cv2.resize(img_to_show, dim, interpolation = cv2.INTER_AREA)
        cv2.imwrite("resized1.jpg",resized1)
        label = QLabel(self.label_9)
        pixmap = QPixmap("resized1.jpg")
        label.setPixmap(pixmap)
        label.resize(pixmap.width(),pixmap.height())
        label.show()

        

    
    def seldir(self):
        self.QFileDialog = QtGui.QFileDialog(self)
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        print(folder)
        

    def preproc(self):
        global fname
        if fname=="":
            self.errimg=errimg()
            self.errimg.show()
        else:
            filename = fname
            print("file for processing",filename)
            image =cv2.imread(str(filename))
            #print type(image)
            cv2.imshow("Original Image", image)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv2.imshow("Grayscale Conversion", gray)
            gray = cv2.bilateralFilter(gray, 11, 17, 17)
            cv2.imshow("Bilateral Filter", gray)
            edged = cv2.Canny(gray, 27, 40)
            cv2.imshow("Canny Edges", edged)
            global fname1
            filename1 = fname1
            print("file for processing",filename1)
            image1 =cv2.imread(str(filename1))
            #print type(image)
            cv2.imshow("Original Image", image1)
            gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            cv2.imshow("Grayscale Conversion", gray1)
            gray1 = cv2.bilateralFilter(gray1, 11, 17, 17)
            cv2.imshow("Bilateral Filter", gray1)
            edged1 = cv2.Canny(gray1, 27, 40)
            cv2.imshow("Canny Edges", edged1)

    def cnn(self):
        #init the model
        model= Sequential()

        #add conv layers and pooling layers 
        model.add(Convolution2D(32,3,3, input_shape=(400,400,1),activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Convolution2D(32,3,3, input_shape=(400,400,1),activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Dropout(0.5)) #to reduce overfitting

        model.add(Flatten())

        #Now two hidden(dense) layers:
        model.add(Dense(output_dim = 150, activation = 'relu',
                        kernel_regularizer=regularizers.l2(0.01)))

        model.add(Dropout(0.5))#again for regularization

        model.add(Dense(output_dim = 150, activation = 'relu',
                        kernel_regularizer=regularizers.l2(0.01)))


        model.add(Dropout(0.5))#last one lol

        model.add(Dense(output_dim = 150, activation = 'relu',
                        kernel_regularizer=regularizers.l2(0.01)))

        #output layer
        model.add(Dense(output_dim = 4, activation = 'sigmoid'))


        #Now copile it
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


        #Now generate training and test sets from folders

        train_datagen=ImageDataGenerator(
                                           rescale = 1./255,
                                           shear_range = 0.2,
                                           zoom_range = 0.,
                                           horizontal_flip = False
                                         )

        test_datagen=ImageDataGenerator(rescale=1./255)

        training_set=train_datagen.flow_from_directory("Datasets/training_set",
                                                       target_size = (400,400),
                                                       color_mode='grayscale',
                                                       batch_size=10,
                                                       class_mode='categorical')

        test_set=test_datagen.flow_from_directory("Datasets/test_set",
                                                       target_size = (400,400),
                                                       color_mode='grayscale',
                                                       batch_size=10,
                                                       class_mode='categorical')






        #finally, start training
        hiss=model.fit_generator(training_set,
                                 samples_per_epoch = 520,
                                 nb_epoch = 10,
                                 validation_data = test_set,
                                 nb_val_samples = 320)



        plt.figure(figsize=(15,7))

        plt.subplot(1,2,1)
        plt.plot(hiss.history['acc'], label='train')
        plt.plot(hiss.history['val_acc'], label='validation')
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1,2,2)
        plt.plot(hiss.history['loss'], label='train')
        plt.plot(hiss.history['val_loss'], label='validation')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.show()

        #saving the weights
        model.summary()

        model.save_weights("weights.hdf5",overwrite=True)

        #saving the model itself in json format:
        model_json = model.to_json()
        with open("model.json", "w") as model_file:
            model_file.write(model_json)
        print("Model has been saved.")

        

    def pred(self):
        global fname
        global fname1
        age= self.lineEdit.text()
        gender= self.lineEdit_4.text()
        weight= self.lineEdit_2.text()
        height= self.lineEdit_3.text()
        skin_pic= str(fname1)
        nails_pic= str(fname)
        skin=""
        nail=""
        skin1=""
        nail1=""
        nutriw=""
        nutrih=""
        histarray={'healthy_nails':0, 'unhealthy_nails':0, 'healthy_skin': 0, 'unhealthy_skin': 0}

        def load_model(path):
            try:
                print("Here we will load model  ")
                json_file = open(os.path.join(path,'model.json'), 'r')
                loaded_model_json = json_file.read()
                json_file.close()
                model = model_from_json(loaded_model_json)
                model.load_weights(os.path.join(path,"weights.hdf5"))
                print("Model successfully loaded from disk.")

                #compile again
                model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
                return model
            except Exception as e:
                print(e)
                print("""Model not found. Please train the CNN by running the script """)
                return None    

        def update(histarray2):
            global histarray
            histarray=histarray2



        #def realtime(pic):
            #classes=['healthy_nails', 'unhealthy_nails', 'healthy_skin', 'unhealthy_skin']
            
            #frame=cv2.imread(pic)
            #frame = cv2.resize(frame, (400,400))
            #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            #frame=frame.reshape((1,)+frame.shape)
            #frame=frame.reshape(frame.shape+(1,))
            #test_datagen = ImageDataGenerator(rescale=1./255)
            #m=test_datagen.flow(frame,batch_size=1)
            #y_pred=model.predict_generator(m,1)
            #histarray2={'healthy_nails': y_pred[0][0], 'unhealthy_nails': y_pred[0][1], 'healthy_skin': y_pred[0][2], 'unhealthy_skin': y_pred[0][3]}
            #update(histarray2)
            #pred= classes[list(y_pred[0]).index(y_pred[0].max())]
            #return pred

        def predict(model, file, resolution=(300,300), needCanny=True):
            frame=cv2.imread(file)
            frame = cv2.resize(frame, resolution)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            if needCanny:
                frame = cv2.Canny(frame, 27, 40)
            cv2.imwrite('test.jpg', frame)
            
            image = tf.keras.preprocessing.image.load_img('test.jpg', color_mode="grayscale")
            input_arr = preprocessing.image.img_to_array(image)

            input_arr = np.array([input_arr])  # Convert single image to a batch.
            input_arr = input_arr/255.0
            predictions = model.predict(input_arr)
            # histarray2={'healthy_nails': y_pred[0][0], 'unhealthy_nails': y_pred[0][1], 'healthy_skin': y_pred[0][2], 'unhealthy_skin': y_pred[0][3]}
            # update(histarray2)
            return predictions[0][0] > 0.5
    

        dirname = os.getcwd()
        nailfoldername = os.path.join(dirname, 'nail_weight')
        nail_model = load_model(nailfoldername)
        
        skinfoldername = os.path.join(dirname, 'skin_weight')
        skin_model = load_model(skinfoldername)

        print("skin Pic=",skin_pic)
        print("Nails Pic=",nails_pic)
        
        skin=predict(skin_model, skin_pic, resolution=(400,400), needCanny=False)
        nails=predict(nail_model, nails_pic)
               
        self.textEdit.setText("")

        wt1=float(weight)
        ht1=float(height)/100
        bmi = wt1 / (ht1 * ht1)

        if int(age) >=18:
            if bmi < 18.5:
                nutriw= f"Weight is less for a {gender} adult"
                self.textEdit.append(nutriw)
                nutrih="Please increase your weight to match your height"
                self.textEdit.append(nutrih)
      
            if bmi >= 18.5 and bmi < 25:
                nutriw= f"Weight is normal for a {gender} adult."
                self.textEdit.append(nutriw)
                nutrih="Your height is just perfect"
                self.textEdit.append(nutrih)
            if bmi >= 25 and bmi < 30:
                nutriw= f"You are overweight for a {gender} adult."
                self.textEdit.append(nutriw)
                nutrih="Your weight should be reduced as your height is short as per your weight"
                self.textEdit.append(nutrih)
            if bmi >= 30:
                nutriw= f"You are obese for an {gender} adult."
                self.textEdit.append(nutriw)
                nutrih="Your weight should be reduced as your height is short as per your weight"
                self.textEdit.append(nutrih)
        if int(age) < 18:
            if bmi < 18.5:
                nutriw= f"Weight is less for a {gender} child"
                self.textEdit.append(nutriw)
                nutrih="Probably height is increasing at a faster rate"
                self.textEdit.append(nutrih)
            if bmi >= 18.5 and bmi < 25:
                nutriw= f"Weight is normal for a {gender} child."
                self.textEdit.append(nutriw)
                nutrih="height is perfect from bmi perspective"
                self.textEdit.append(nutrih)
            if bmi >= 25 and bmi < 30:
                nutriw= f"You are overweight for a {gender} child."
                self.textEdit.append(nutriw)
                nutrih="Try height increasing exercises which will bring bmi to normal"
                self.textEdit.append(nutrih)
            if bmi >= 30:
                nutriw= f"You are obese for an {gender} child."
                self.textEdit.append(nutriw)
                nutrih="Either decrease your weight or increase you height by exercise"
                self.textEdit.append(nutrih)

        if skin:
            skin1="Need Skin care, possible vitamin deficiency"
            self.textEdit.append(skin1)
        else:
            skin1="Skin condition is good"
            self.textEdit.append(skin1)

        if nails:
            nail1="Nails are in unhealthy condition possible calcium or protien nutrition deficiency"
            self.textEdit.append(nail1)
        else:
            nail1="Nails and nutrition seems good"
            self.textEdit.append(nail1)
            
    def ex(self):
        sys.exit()
        

class errlog(QtGui.QMainWindow, error_log.Ui_Error):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.ok1)
    def ok1(self):
        self.hide()

class errimg(QtGui.QMainWindow, err_img.Ui_Error):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.ok1)
    def ok1(self):
        self.hide()



# def main():
    # app = QtGui.QApplication(sys.argv)  
    # form = Login()                 
    # form.show()                         
    # app.exec_()                         
def main():
    app = QtGui.QApplication(sys.argv)  
    form = home()                 
    form.show()                         
    app.exec_()                         


if __name__ == '__main__':              
    main()                             
