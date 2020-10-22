import sqlite3

conn=sqlite3.connect('face_db_v1.sqlite')       #建立或開啟資料庫
cursor=conn.cursor()                         #使用cursor物件執行SQL指令

#建立資料表
sqlstr = "select * from member"              #SQL指令
cursor.execute(sqlstr)                       #execute() 執行SQL的命令。
rows = cursor.fetchall()                     #fetchall():接收全部的返回結果行。

#顯示目前資料
for row in rows:
    print('id={},uiserid={},name={}'.format(row[0],row[1],row[2]))
