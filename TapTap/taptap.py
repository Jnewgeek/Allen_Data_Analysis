# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 17:11:01 2019

@author: Administrator
"""

import requests
import pandas as pd
import time
import random
#import threading, threadpool
#lock = threading.Lock()
from requests.packages import urllib3
urllib3.disable_warnings()
import os
os.chdir(os.getcwd())
import re

import threading
from time import ctime

class MyThread(threading.Thread):

	def __init__(self,func,args,name='',prints=False):
		threading.Thread.__init__(self)
		self.name=name
		self.func=func
		self.args=args
		self.prints=prints

	def getResult(self):
		return self.res

	def run(self):
		if self.prints:print('Starting < %s > at: %s\n'%(self.name,ctime()))
		self.res=self.func(*self.args)
		if self.prints:print('< %s > finished at: %s\n'%(self.name,ctime()))

# 多线程
class TapTapgame:
    
    def __init__(self,filename):
        '''初始化参数.'''
        s = requests.session()  # 建立爬虫的回话,相当于浏览器的一个页面,在回话里共享header和cookie等数据
        s.verify = False
        s.trust_env = False
        s.headers = {
                    'Host': 'api.taptapdada.com',
                    'Connection': 'Keep-Alive',
                    'Accept-Encoding': 'gzip',
                    'User-Agent': 'okhttp/3.10.0'}
        self.s = s
        self.proxies = self.Abuyun_proxy()
        self.count = 0  # 统计变更代理服务器次数计数
        self.sucessfull = 0
        self.filename=filename
        #self.page=0
        self.base_url='https://api.taptapdada.com/review/v1/by-app?limit=10&app_id=31597' \
              '&X-UA=V%3D1%26PN%3DTapTap%26VN_CODE%3D551%26LOC%3DCN%26LANG%3Dzh_CN%26CH%3Dtencent%26' \
              'UID%3Dda4b99bf-5e2b-4204-a92f-235474b32c4c&from='

    # 阿布云代理
    def Abuyun_proxy(self):
        proxyHost = "http-dyn.abuyun.com"
        proxyPort = "9020"
        proxyUser = ""
        proxyPass = ""
        proxyMeta = "http://%(user)s:%(pass)s@%(host)s:%(port)s" % {
            "host": proxyHost,
            "port": proxyPort,
            "user": proxyUser,
            "pass": proxyPass,
        }
        self.proxies = {
            "http": proxyMeta,
            "https": proxyMeta,
        }

    def spider(self, url):
        """
        爬虫主体,防止IP被封,需要设置while 1
        :param url: 只要传入爬取的url,就能给与相应
        :return: response url返回HTML
        """
        while 1:
            try:
                response = self.s.get(url,proxies=self.proxies)
                if response.status_code == 200:
                    break
            except:# Exception as e:
                #print(e)
                self.count += 1
                time.sleep(random.random())
                # get_proxies在上一篇文章获取ip代理池中的一个方法,那里有详细说明和源码
                # 可以直接拿来用,这里不再赘述,感兴趣的小伙伴可以子看看
                self.Abuyun_proxy()  # 如果出问题就改变代理ip
                print('第%d次变更代理ip'%self.count)
                continue
            # 重试次数超过1000次，停止程序运行
            if self.count>1000:
                break
        self.sucessfull+=1
        print("成功获取%d个用户."%self.sucessfull)
        return response.json()
    
    # 获取用户基本信息和id_
    def parse_info(self,url):
        '''获取review中每个用户的基本信息和id号.'''
        response=self.spider(url)     # 获取用户数据json格式
        datas = response.get('data').get('list')
        if datas:
            for data in datas:
                # 评论人
                name = data.get('author').get('name')
                # 性别
                gender = data.get('author').get('gender')
                # id 
                id_ = data.get('author').get('id')
                # 设备
                device = data.get('device')
                # 评分
                score = data.get('score')
                # 游戏时长
                played_tips = data.get('played_tips')
                # 评论内容
                contents = data.get('contents').get('text')
    
                # 声明一个字典储存数据
                data_dict = {}
                data_dict['name'] = name
                data_dict['gender'] = gender
                data_dict['id'] = id_
                data_dict['device'] = device
                data_dict['score'] = score
                data_dict['played_tips'] = played_tips
                data_dict['contents'] = contents.replace('<br />', '')
                
                url_="https://api.taptapdada.com/user-app/v2/by-user?limit=10&X-UA=V%3D1%26PN%3DTapTap%26VN_CODE%3D557%26LOC%3DCN%26LANG%3Dzh_CN_%23Hans%26CH%3Dtencent%26UID%3Da589ebaa-9acf-47f9-bfb7-802750ec183d&from=0&sort=updated&user_id={}".format(id_)
                response=self.spider(url_)  # 获取游戏列表
                datas = response.get('data').get('list')
                #print(url_,datas)
                if datas:
                    title = '|'.join(data.get('app').get('title') for data in datas)
                    #num=len(title.split('|'))
                else:
                    title = ''
                    #num=0
                # 游戏列表
                data_dict['gameTitle']=title
                #data_dict['num']=num
                
                #data_list.append(data_dict)
                data_list=pd.DataFrame(data_dict,index=[0],columns=['id','name','gender','device','score','played_tips','contents','gameTitle'])
                data_list.to_csv(self.filename,index=False,header=False,mode='a+',encoding='utf-8')
        else:
            return False
    
    def main(self,length=200):
         '''主函数.'''
         threads=[]
         for i in range(0,(length//10+1)*10+1,10):
             t=MyThread(self.parse_info,(self.base_url+str(i),))
             threads.append(t)
         
         for i in range(len(threads)):
             threads[i].start()
            
         for i in range(len(threads)):
             threads[i].join()
             

if __name__ == '__main__':
    
    t1 = time.time()
    # 声明一个列表存储字典
    # 保存数据
    filename='%s.csv'%('碧蓝航线')
    data_list = pd.DataFrame(dict(zip(['name','gender','id','gameTitle','device','score','played_tips','contents'],
                                  ['name','gender','id','gameTitle','device','score','played_tips','contents'])),index=[0],
    columns=['id','name','gender','device','score','played_tips','contents','gameTitle'])
    data_list.to_csv(filename,index=False,header=False,encoding='utf-8')
    
    # 初始化类
    s=TapTapgame(filename)
    s.main(25434)
    # 转存xls文件
    data=pd.read_csv(filename,engine='python',encoding='utf-8')
    # 清洗数据
    def cleantime(x):
        '''清洗时间：游戏时长XX小时XX分钟.'''
        #print(x)
        if str(x)!='nan':
            if '小时' in str(x):
                pattern=re.compile(r'(\d*)小时(\d*).*')
                x=pattern.findall(x)[0]
                x=(int(x[0])*60 if x[0] else 0)+(int(x[1]) if x[1] else 0)  # 游戏时长(分钟)
            else:
                pattern=re.compile(r'(\d*)分钟')
                x=pattern.findall(x)[0]
                x=int(x[0])  # 游戏时长(分钟
        else:
            pass
        return x
    data['played_tips']=data['played_tips'].apply(cleantime)
    # 清洗评论信息
    data['contents']=data['contents'].str.replace('\s+','')
    
    data.to_excel(filename.replace('csv','xls'),index=False)
    t2 = time.time()
    print('finished! 总用时:%.2fs'%(t2 - t1))

    
