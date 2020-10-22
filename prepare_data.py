import cv2
import sys
import shutil, os
import numpy as np

IMAGE_SIZE = 64                                                          #依照IMAGE_SIZE設定調整所有圖片大小

def resize_image(image, height = IMAGE_SIZE, width = IMAGE_SIZE):
    top, bottom, left, right = (0, 0, 0, 0)    
    h, w, _ = image.shape                   #取得圖片目前尺寸
    longest_edge = max(h, w)                #找到圖片最長的一邊
    if h < longest_edge:                    #計算短邊與長邊的差，補齊後再縮圖
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass
    BLACK = [0, 0, 0]                       #RGB顏色
    #cv2.copyMakeBorder(src, top, bottom, left, right , borderType, value)
    #cv2.BORDER_CONSTANT:表示邊界顏色由value的值決定
    #補足圖片不足寬度，讓長與寬相等
    constant = cv2.copyMakeBorder(image, top , bottom, left, right, cv2.BORDER_CONSTANT, value = BLACK)
    return cv2.resize(constant, (height, width))                         #調整圖片大小

def read_path(path_name):                                                #讀取訓練資料
    for dir_item in os.listdir(path_name):
        full_path = os.path.abspath(os.path.join(path_name, dir_item))   #組合成完整路徑
        if os.path.isdir(full_path):                                     #如果是資料夾，則一層層呼叫
            read_path(full_path)
        else:                                                            #取出檔案
            if dir_item.endswith('.jpg'):
                image = cv2.imread(full_path)
                image = resize_image(image, IMAGE_SIZE, IMAGE_SIZE)
                #如果要看到處理後圖片，可以將其儲存
                #cv2.imwrite('1.jpg', image)
                images.append(image)
                labels.append(path_name)
    return images,labels

def load_dataset(path_name):                                             #從指定位置讀入訓練資料
    images, labels = read_path(path_name)
    #計算要處理資料的大小，大小應該是(圖片數量*IMAGE_SIZE*IMAGE_SIZE*3)
    #2人假設共200張圖片，IMAGE_SIZE為64，那資料大小就是200*64*64*3
    #因為一張圖片為64 * 64畫素,一個畫素3個顏色值(BGR)
    images = np.array(images)
    print(images.shape)
    '''
    #label，'people_face1'資料夾的圖片全部都指定為0，另外一個資料夾下全部指定為1(預設只有兩類)
    #labels = np.array([0 if label.endswith('people_face1') else 1 for label in labels])
    '''
    lbl=[]
    #----給定資料夾做label，讓labels[]陣列裡填入[0, 1, 2, 3...]等答案--------
    '''
    #寫法1
    for label in labels:
        if label.endswith('user1'):
            lbl.append(0)
        elif label.endswith('user2'):
            lbl.append(1)
        elif label.endswith('user3'):
            lbl.append(2)
        elif label.endswith('user4'):
            lbl.append(3)
        else:
            lbl.append(4)
    labels = lbl
    return images, labels
    '''
    #寫法2
    label_dict = {'a':"user01", 'b':"user02", 'c':"user03"}
    for label in labels:
        if label == label_dict['a']:
            lbl.append(0)
        elif label == label_dict['b']:
            lbl.append(1)
        else:
            lbl.append(2)
    labels = lbl
    return images, labels

path_name = "data"
images = []
labels = []

if __name__ == '__main__':
    if len(sys.argv) != 1:
        print("Usage:%s path_name\r\n" % (sys.argv[0]))                  #Usage : 字串，主要是會顯示來告知使用者說應該怎麼使用你寫的
    else:
        images, labels = load_dataset(path_name)
