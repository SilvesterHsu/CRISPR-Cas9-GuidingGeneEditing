import os
COUNT_NUM = 3


count = COUNT_NUM
while count:
    try:
        import flask
        print ('flask already installed!')
        break
    except:
        print ('flask unstalled, ready to install')
        os.system('pip install flask')
        count -= 1
        continue

count = COUNT_NUM
while count:
    try:
        import configparser
        print ('configparser already installed!')
        break
    except:
        print ('configparser unstalled, ready to install')
        os.system('pip install configparser')
        count -= 1
        continue


count = COUNT_NUM
while count:
    try:
        import sqlalchemy
        print ('sqlalchemy already installed!')
        break
    except:
        print ('sqlalchemy unstalled, ready to install')
        os.system('pip install sqlalchemy')
        count -= 1
        continue


count = COUNT_NUM
while count:
    try:
        import pymysql
        print ('pymysql already installed!')
        break
    except:
        print ('pymysql unstalled, ready to install')
        os.system('pip install pymysql')
        count -= 1
        continue


count = COUNT_NUM
while count:
    try:
        import pandas
        print ('pandas already installed!')
        break
    except:
        print ('pandas unstalled, ready to install')
        os.system('pip install pandas')
        count -= 1
        continue


count = COUNT_NUM
while count:
    try:
        import numpy
        print ('numpy already installed!')
        break
    except:
        print ('numpy unstalled, ready to install')
        os.system('pip install numpy')
        count -= 1
        continue

