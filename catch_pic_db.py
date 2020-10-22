#整合SQLite資料庫與攝影機進行註冊
import cv2
import sys
import shutil, os
from time import sleep
from PIL import Image
import sqlite3

def emptydir(dirname):                                          #清空資料夾
    if os.path.isdir(dirname):                                  #資料夾存在就刪除
        shutil.rmtree(dirname)
        sleep(2)
    os.makedirs(dirname)                                        #建立資料夾

def CatchPICFromVideo(window_name, camera_idx, catch_pic_num):
    conn = sqlite3.connect('face_db_v1.sqlite')                    #建立或開啟資料庫
    cursor = conn.cursor()                                      #使用cursor物件執行SQL指令
    sqlstr = "select * from member"                             #To select from a member table in SQL
    cursor.execute(sqlstr)                                      #execute() 執行SQL的命令，舉凡插入、查詢、刪除等都是要靠它。
    rows = cursor.fetchall()                                    #fetchall():接收全部的返回結果行。
    member=[]                                                   #儲存會員字典結構資料
    for row in rows:
        member.append(row[1])                                   #將memberid存入member結構中用以做辨識是否重複？
    while True:                                                 #開始進行註冊
        memberdata = input('輸入帳號/姓名(按enter結束)')
        memberdata = memberdata + "/x"
        split_input = memberdata.split("/", 2)                  #split():找字串中"/"，並且分割2次
        if split_input[0]=='':                                  #沒有任何輸入則結束程式
            break
        elif split_input[0] in member:
            print('帳號已經存在，不可以重複建立')
        else:                                                   #---建立帳號與擷取影像---
            memberid = split_input[0]                           #使用者帳號
            name = split_input[1]                               #使用者姓名
            emptydir('data/' + memberid)                        #建立檔案儲存目錄
            #----擷取影像----
            flipCode = 1                                        #1:水平翻轉，0:垂直翻轉，-1:水平+垂直同時翻轉
            cv2.namedWindow(window_name)    
            cap = cv2.VideoCapture(camera_idx)
            classfier = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")#載入OpenCV的人臉偵測模型
            color = (0, 255, 0)                                 #設定人臉位置框的顏色
            num = 0                                             #圖片數目，拿來跟catch_pic_num做判斷用
            while cap.isOpened():                               #確認cap是啟動的，就開始擷取畫面並偵測人臉位置    
                ok, frame = cap.read()                          #讀取目前影像frame
                if not ok:                                      #如果讀取產生錯誤
                    break
                frame = cv2.flip(frame, flipCode)               #使frame水平翻轉
                grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #將目前影像frame轉換成灰階影像
                #開始進行人臉位置偵測，scaleFactor:圖片縮放比例，minNeighbors:每個候選矩形應該有多少個鄰居有把它保留下來。
                faceRects = classfier.detectMultiScale(grey, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))
                if len(faceRects) > 0:                          #大於0表示有偵測到人臉                                   
                    for faceRect in faceRects:                  #框選出每一張人臉
                        x, y, w, h = faceRect
                        img_name = '%s/%d.jpg'%('data/' + memberid, num) #儲存目前畫面為圖片
                        image = frame[y - 10: y + h + 10, x - 10: x + w + 10] #人臉框位移一下
                        cv2.imwrite(img_name, image)            #寫入圖檔，imwrite(檔名, 檔案)
                        num += 1
                        if num > (catch_pic_num):               #已經到達擷取設定數目
                            break
                        #---------框選出每一張人臉----------------
                        cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)
                        #rectangle(影像, 頂點座標(x,y), 斜對向頂點座標(x,y), 顏色, 線條寬度)
                        font = cv2.FONT_HERSHEY_SIMPLEX         #字體字型
                        cv2.putText(frame,'num:%d' % (num),(x + 30, y + 30), font, 1, (255,0,255),4)
                        #putText(影像, 文字, 座標(x,y), 字體字型, 文字大小比例, 顏色, 線條寬度)
                if num > (catch_pic_num): break                 #假如到達擷取設定數目，則break
                cv2.imshow(window_name, frame)                  #顯示框選後畫面
                c = cv2.waitKey(10)
                if c & 0xFF == ord('q'):  break                 #按q鍵可以離開
            #---------將資料使用資料庫存檔---------
            sqlstr="insert into member (memberid,name) values ('{}','{}')".format(memberid,name)  #SQL指令
            cursor.execute(sqlstr)                              #新增資料到資料庫內，execute():執行SQL的命令
            conn.commit()                                       #確認執行SQL指令
            print('註冊成功')
            conn.close()                                        #關閉資料庫
            #---------解除使用攝影機與關閉顯示視窗---------
            cap.release()
            cv2.destroyAllWindows()
            break

window_name = "Catch Face"
camera_idx = 0                                                  #camera_idx:設定影像來源，通常使用0表示預設鏡頭
catch_pic_num = 100                                              #設定擷取圖片數目

if __name__ == '__main__':
    if len(sys.argv) != 1:
        print("Usage:%s camera_id face_num_max path_name\r\n" % (sys.argv[0]))
    else:
        CatchPICFromVideo(window_name, camera_idx, catch_pic_num)
        print("finished!")

