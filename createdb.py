#建立空白資料庫
import sqlite3

conn = sqlite3.connect('face_db_v1.sqlite')   #建立或開啟資料庫
cursor = conn.cursor()                     #使用cursor物件執行SQL指令
#建立資料表
sqlstr = "create table if not exists member (ID INTEGER PRIMARY KEY AUTOINCREMENT,'memberid' TEXT,'name' TEXT)"
                                           #ID:自動編號，用作訓練模型時的答案label
                                           #AUTOINCREMENT:是一個關鍵字，用於表中的字段值自動遞增。
                                           #memberid:使用者輸入的帳號
cursor.execute(sqlstr)                     #execute() 執行SQL的命令，舉凡插入、查詢、刪除等都是要靠它。
sqlstr="create table if not exists login ('memberid' TEXT,'login_time' TEXT)" # SQL指令
cursor.execute(sqlstr)
conn.commit()                              #確認執行SQL指令
conn.close()                               #關閉資料庫
print("created database finished!")
