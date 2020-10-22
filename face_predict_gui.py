import os
import cv2
import sys
import gc
from tf1_train import Model
from tkinter import *                                         #載入GUI需要的套件
from PIL import Image,ImageTk
import requests                                               #for Line Notify
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import sqlite3
from datetime import datetime
import tkinter as tk
import serial

class Client:
    CLIENT_ID = os.environ.get('LINE_NOTIFY_CLIENT_ID')
    CLIENT_SECRET = os.environ.get('LINE_NOTIFY_CLIENT_SECRET')
    REDIRECT_URI = os.environ.get('LINE_NOTIFY_REDIRECT_URI')
    def __init__(self,
                 client_id=None,
                 client_secret=None,
                 redirect_uri=None,
                 bot_origin=None,
                 api_origin=None,
                 *args, **kwargs):
        super(Client, self).__init__(*args, **kwargs)
        self.client_id = client_id or self.CLIENT_ID
        self.client_secret = client_secret or self.CLIENT_SECRET
        self.redirect_uri = redirect_uri or self.REDIRECT_URI

        self.bot_origin = bot_origin or "https://notify-bot.line.me"
        self.api_origin = api_origin or "https://notify-api.line.me"

    def get_auth_link(self, state):
        query_string = {
            'scope': 'notify',
            'response_type': 'code',
            'client_id': self.client_id,
            'redirect_uri': self.redirect_uri,
            'state': state
        }
        return '{url}/oauth/authorize?{query_string}'.format(
            url=self.bot_origin, query_string=urlencode(query_string))

    def get_access_token(self, code):
        response = self._post(
            url='{url}/oauth/token'.format(url=self.bot_origin),
            headers={
                'Content-Type': 'application/x-www-form-urlencoded',
            }, data={
                'grant_type': 'authorization_code',
                'code': code,
                'redirect_uri': self.redirect_uri,
                'client_id': self.client_id,
                'client_secret': self.client_secret
            })
        return response.json().get('access_token')

    def status(self, access_token):
        response = self._get(
            url='{url}/api/status'.format(url=self.api_origin),
            headers={
                'Content-Type': 'application/x-www-form-urlencoded',
                'Authorization': 'Bearer {token}'.format(token=access_token)
            })
        return response.json()

    def send_message(self, access_token, message, notification_disabled=False):
        params = {'message': message}
        if notification_disabled:
            params.update({'notificationDisabled': notification_disabled})

        response = self._post(
            url='{url}/api/notify'.format(url=self.api_origin),
            data=params,
            headers={
                'Authorization': 'Bearer {token}'.format(token=access_token)
            })
        return response.json()

    def send_message_with_sticker(
            self,
            access_token,
            message,
            sticker_id,
            sticker_package_id,
            notification_disabled=False):
        params = {
            'message': message,
            'stickerId': sticker_id,
            'stickerPackageId': sticker_package_id
        }
        if notification_disabled:
            params.update({'notificationDisabled': notification_disabled})

        response = self._post(
            url='{url}/api/notify'.format(url=self.api_origin),
            data=params,
            headers={
                'Authorization': 'Bearer {token}'.format(token=access_token)
            })
        return response.json()

    def send_message_with_image_url(
            self,
            access_token,
            message,
            image_thumbnail,
            image_fullsize,
            notification_disabled=False):

        params = {
            'message': message,
            'imageFullsize': image_fullsize,
            'imageThumbnail': image_thumbnail
        }
        if notification_disabled:
            params.update({'notificationDisabled': notification_disabled})

        response = self._post(
            url='{url}/api/notify'.format(url=self.api_origin),
            data=params,
            headers={
                'Authorization': 'Bearer {token}'.format(token=access_token)
            })
        return response.json()

    def send_message_with_image_file(
            self,
            access_token,
            message,
            file,
            notification_disabled=False):
        params = {'message': message}

        if notification_disabled:
            params.update({'notificationDisabled': notification_disabled})

        response = self._post(
            url='{url}/api/notify'.format(url=self.api_origin),
            data=params,
            files={'imageFile': file},
            headers={
                'Authorization': 'Bearer {token}'.format(token=access_token)
            })
        return response.json()

    def revoke(self, access_token):
        response = self._post(
            url='{url}/api/revoke'.format(url=self.api_origin),
            headers={
                'Content-Type': 'application/x-www-form-urlencoded',
                'Authorization': 'Bearer {token}'.format(token=access_token)
            })
        return response.json()

    def _get(self, url, headers=None, timeout=None):
        response = requests.get(
            url, headers=headers, timeout=timeout
        )

        self.__check_error(response)
        return response

    def _post(self, url, data=None, headers=None, files=None, timeout=None):
        response = requests.post(
            url, headers=headers, data=data, files=files, timeout=timeout
        )
        self.__check_error(response)
        return response

    @staticmethod
    def __check_error(response):
        if 200 <= response.status_code < 300:
            pass
        else:
            raise ValueError(response.json())

def send_message_with_sticker(
    self,
    access_token,
    message,
    sticker_id,
    sticker_package_id,
    notification_disabled=False):
    params = {
        'message': message,
        'stickerId': sticker_id,
        'stickerPackageId': sticker_package_id
    }
    if notification_disabled:
        params.update({'notificationDisabled': notification_disabled})
        response = self._post(
            url='{url}/api/notify'.format(url=self.api_origin),
            data=params,
            headers={
                'Authorization': 'Bearer {token}'.format(token=access_token)
            })
        return response.json()

def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype(
        "font/simsun.ttc", textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontText)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def video_loop():
    global user
    flipCode = 1
    success, frame = cap.read()                                       #從USB Cam取得影像
    if success:
        frame = cv2.flip(frame, flipCode)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)          #將影像灰階化
        cascade = cv2.CascadeClassifier(cascade_path)                 #載入人臉位置偵測模型
        #---------開始進行人臉偵測與辨識---------
        faceRects = cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
        if len(faceRects) > 0:                                        #如果偵測到人臉位置
            for faceRect in faceRects:
                x, y, w, h = faceRect
                image = frame[y-10: y+h+10, x-10: x+w+10]             #解取人臉偵測出來的區域，再放入辨識模型中辨識
                faceID = model.face_predict(image)
                if faceID == 0:
                    user = 'Harry Potter'
                    cv2.rectangle(frame, (x-10, y-10), (x+w+10, y+h+10), color, thickness=1.5)#(影像, 頂點座標, 對向頂點座標, 顏色, 線條寬度)
                    #frame = cv2ImgAddText(frame, user, (x+30, y+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2, cv2.LINE_AA) #cv2ImgAddText(影像, 文字, 座標, 字型, 大小, 顏色, 線條寬度, 線條種類)，cv2ImgAddText可顯示中文
                    frame = cv2.putText(frame, user, (x+30, y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,255), 2, cv2.LINE_AA)
                    sp.write(b'1')  # turn on LED
                elif faceID == 1:
                    user = 'Peng Yuyan'
                    cv2.rectangle(frame, (x-10, y-10), (x+w+10, y+h+10), color, thickness=2)
                    frame = cv2.putText(frame, user, (x+30, y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,255), 2, cv2.LINE_AA) #putText(影像, 文字, 座標, 字型, 大小, 顏色, 線條寬度, 線條種類)，putText顯示不出中文
                    sp.write(b'1')  # turn on LED
                elif faceID == 2:
                    user = '帥氣的 Dawi'
                    cv2.rectangle(frame, (x-10, y-10), (x+w+10, y+h+10), color, thickness=2)
                    frame = cv2ImgAddText(frame, user, x+10, y+10, (255, 0, 255), textSize=20)
                    #frame = cv2.putText(frame, user, (x+30, y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,255), 2, cv2.LINE_AA)
                    sp.write(b'1')  # turn on LED
                else:
                    pass
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)             #影像從RGB轉換成RGBA
        current_image = Image.fromarray(cv2image)                      #將影像轉換成Image物件
        imgtk = ImageTk.PhotoImage(image=current_image)
        panel.imgtk = imgtk
        panel.config(image=imgtk)
        k = cv2.waitKey(1)                                             #暫停毫秒數
        root.after(1, video_loop)                                      #重新抓取畫面到視窗上

def checkin_2line_2arduino():
    #加入報到資料到資料庫
    conn=sqlite3.connect('face_db_v1.sqlite')                            #建立或開啟資料庫
    cursor=conn.cursor()
    savetime = str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))         #取得目前報到時間
    sqlstr="insert into login values ('{}','{}')".format(user,savetime)  #SQL指令
    cursor.execute(sqlstr)                                               #新增資料到資料庫內
    conn.commit()                                                        #確認執行SQL指令
    conn.close()

    time = datetime.now().strftime("%Y/%m/%d %H:%M:%S\n")
    message = '[公告]：' + time + user + '已報到!'
    #token = 'Nv58Jq8MwsBjSJl9YGktd1s5MwVY3VchEOdmcHiYe9C'
    print(message)
    lab2.config(text=message)
    #lineNotifyMessage(token, message)

def turnon_led():                                             #開啟Arduino LED    
    sp.write(b'1')  # turn on LED
    string = sp.readline()
    lab2.configure(text=string)
    pass

def turnoff_led():                                            #關閉Arduino LED
    sp.write(b'0')  # turn on LED
    string = sp.readline()
    lab2.configure(text=string)
    pass

def show_now_time():
    time = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    lab2.config(text="現在時間：" + time)

def line_notify():
    client = Client()
    message_txt = "[公告]：" + user + "已報到!"
    message_with_time = datetime.now().strftime("%Y/%m/%d %H:%M:%S\n") + message_txt
    token = 'Nv58Jq8MwsBjSJl9YGktd1s5MwVY3VchEOdmcHiYe9C'
    response = client.send_message_with_sticker(
                access_token = token,
                message = message_with_time,
                sticker_id = 171,                             #https://devdocs.line.me/files/sticker_list.pdf
                sticker_package_id = 2)
    lab1.config(text=message_txt)
    #print(response)

def print_test():
    conn = sqlite3.connect('face_db_v1.sqlite')  #建立或開啟資料庫
    cursor = conn.cursor()
    sqlstr = "select * from login"               #SQL指令
    cursor.execute(sqlstr)
    rows = cursor.fetchall()
    for row in rows:
        message_txt = '登入者編號：{} / 登入時間：{}'.format(row[0],row[1])
        lab2.config(text=message_txt)
        print(message_txt)
    conn.close()

if __name__ == '__main__':
    if len(sys.argv) != 1:
        print("Usage:%s camera_id\r\n" % (sys.argv[0]))
        sys.exit(0)

    model = Model()
    MODEL_PATH = './model/three_user_face_model.h5'
    model.load_model(MODEL_PATH)                              #載入訓練完成的人臉辨識模型
    color = (0, 255, 0)                                       #設定框選人臉的矩形框顏色
    cap = cv2.VideoCapture(0)                                 #啟動USBCam擷取影像
    cascade_path = "haarcascade_frontalface_alt2.xml"         #設定讀取人臉偵測模型位置與檔案
    #-----------------加入GUI設定---------------------
    root = Tk()
    root.title("opencv + tkinter")                            #視窗標題
    root.resizable(1,1)                                       #(x,y): 1 可動調
    root.geometry('720x600')

    panel = Label(root)                                       #設定panel物件
    panel.grid(column=0, row=0, sticky=tk.W+tk.E)
    root.config(cursor="arrow")

    btn0 = tk.Button(root, text="報到",  width=10, command=line_notify)
    btn1 = tk.Button(root, text="時間",  width=10, command=show_now_time)
    btn2 = tk.Button(root, text="測試",  width=10, command=checkin_2line_2arduino)
    btn0.grid(column=1, row=1)
    btn1.grid(column=1, row=2)
    btn2.grid(column=1, row=3)

    lab1 = tk.Label(root, text="123", bg='green', fg='yellow', font=('Calibri', 15))
    lab2 = tk.Label(root, text="32111", bg='blue', fg='yellow', font=('Calibri', 15))
    lab1.grid(column=0, row=2)
    lab2.grid(column=0, row=3)
    #-----------------Arduino控制-------------------------
    sp = serial.Serial()
    sp.port = 'COM3'                                           #指定通訊埠名稱
    sp.baudrate = 9600                                         #設定傳輸速率
    sp.timeout = 5
    sp.open()
    #-------------------------------------------------------
    video_loop()
    root.mainloop()
    cap.release()
    cv2.destroyAllWindows()
