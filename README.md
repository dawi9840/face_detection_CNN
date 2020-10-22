# face_detection_CNN
Use CNN model to classify the users face and combine with Arduino.

First one, we must to create 2 empty folders, named "data" and "model", which will be saved parameter files and used by the codes.

Second, use the codes to follow the next steps:

  step01: Run createdb.py to create a database.
  
  step02:  Use catch_pic_db.py to catch user face pictures and register into  step01 database.
  
  step03: Use showdb.py to make sure the database you have to load in.
  
  step04: Modify the prepare_data.py file, which in load_dataset() function with many labels(e.g. if you have 5 users data, you can modify the 5 labels in label_dict{[ labels ]}).
  
  step05: Run the tf1_train.py, remember to modify parameter about nb_classes, batch_size, and nb_epoch, also can choose data_augmentation true or false.
  
  step06: Remember prepare a Arduino to set the led then create a empty folder named "led_test" and put the led_test.ino in it.
  
  step07: Run the face_predict_gui.py and execute the GUI buttons.
  
  step08: Finally, use showdb.py code to make sure the database to show users have checked in successfully.
