#顯示報到紀錄資料
import sqlite3

conn=sqlite3.connect('face_db_v1.sqlite')  #建立或開啟資料庫
cursor=conn.cursor()
sqlstr="select * from login"               #SQL指令
cursor.execute(sqlstr)
rows=cursor.fetchall()
for row in rows:
    print('登入者編號：{} / 登入時間：{}'.format(row[0],row[1]))

conn.close()
