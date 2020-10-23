Environment:

    USB Cam
    
    Python 3.7.X
    
    Tensorflow 1.15.0
    
    Opencv 3.4.2
    
    keras 2.2.4
    
    sklearn 
-------------------

Install-Package:

    conda install -c anaconda pillow

    conda install opencv==3.4.2

    conda install keras==2.2.4  

    pip install sklearn 

    pip install line-bot-sdk

    pip install pyserial

----------------------------------

Env command:

    conda create –n [your_env] python=3.7
    # 新增環境，[your_env] 是你的環境名稱
    # python=X.X 是指定環境python版本，較常用3.6c或3.7版本

    conda activate [your_env] 
    # 進入環境

    conda env list
    # 如果忘記名稱，可用list查詢

    conda deactivate 
    # 退出環境

    conda env remove --name <myenv>  
    # 刪除環境
        
