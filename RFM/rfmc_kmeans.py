# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 15:52:43 2019

@author: Administrator

基于RFMC的聚类分析
R: 最后一次下单距观察窗口的月份
F: 观察期间的下单量
M: 观察期间的总GMV
C: 观察期间的平均折扣率
"""
import pandas as pd
import time
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.externals import joblib
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000 # the default is 0
import numpy as np
import seaborn as sns
sns.set_style("white")
sns.set_context("notebook")

plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置-黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
sns.set(font='SimHei')  # 解决Seaborn中文显示问题
import os
os.chdir(os.getcwd())

# 自动创建文件夹
for i in ['./img','./tmp']:
    if not os.path.exists(i):
        os.mkdir(i)
        

# 处理数据
def programmer_1():
    datafile='data/customer.csv'
    resultfile = 'tmp/explore.xls'

    data = pd.read_csv(datafile, encoding='utf-8')
    # 包括对数据的基本描述，percentiles参数是指定计算多少的分位数表（如1/4分位数、中位数等）；T是转置，转置后更方便查阅
    explore = data.describe(percentiles=[], include='all').T
    # describe()函数自动计算非空值数，需要手动计算空值数
    explore['null'] = len(data) - explore['count']

    explore = explore[['null', 'max', 'min']]
    explore.columns = ['空值数', '最大值', '最小值']
    '''这里只选取部分探索结果。
    describe()函数自动计算的字段有count（非空值数）、unique（唯一值数）、top（频数最高者）、freq（最高频数）、mean（平均值）、std（方差）、min（最小值）、50%（中位数）、max（最大值）'''

    explore.to_excel(resultfile)
    
    return explore

# 筛选数据
def programmer_2():
    
    datafile = 'data/customer.csv'
    cleanedfile = 'tmp/customer_cleaned.csv'

    data = pd.read_csv(datafile, encoding='utf-8')

    data.fillna(0,inplace=True)
    # 去除 小于 0 的数据记录
    data=data[data['R']>=0]

    data.to_csv(cleanedfile,index=False,encoding='utf-8')
    
# 标准化
def programmer_3():
    
    time1=time.time()

    datafile = 'tmp/customer_cleaned.csv'
    zscoredfile = 'tmp/customer_zscore.csv'
    
    data = pd.read_csv(datafile,encoding='utf-8')  # 只保留属性值
    # 核心语句，实现标准化变换，类似地可以实现任何想要的变换。
    data = (data - data.mean(axis=0)) / (data.std(axis=0))
    data.columns = ['Z' + i for i in data.columns]
    data.fillna(0,inplace=True)

    data.to_csv(zscoredfile, index=False)
    
    time2=time.time()
    print('finished,用时%.2fs'%(time2-time1))
    
#保存模型
def save_model(model, filepath):
    joblib.dump(model, filename=filepath)

def load_model(filepath):
    model = joblib.load(filepath)
    return model

def programmer_4(k=5,load=False):
    
    time1=time.time()
    
    inputfile = 'tmp/customer_zscore.csv'
    data = pd.read_csv(inputfile,encoding='utf-8')

    if os.path.exists('kmeans.m') and load:  # 导入本地模型
        kmodel=load_model('kmeans.m')
    else:
        kmodel = KMeans(n_clusters=k, n_jobs=4)
    kmodel.fit(data)

    center=pd.DataFrame(kmodel.cluster_centers_,columns=['ZR','ZF','ZM','ZC'])
    labels=pd.Series(kmodel.labels_).value_counts()
    labels.name='cluster_num'
    data=center.join(labels)
    data['cluster_names']=['customer%s'%i for i in data.index]
    data=data[['cluster_names','cluster_num','ZR','ZF','ZM','ZC']]
    print(data)
    
    # 保存聚类中心数据
    data.to_csv('tmp/kmeans_result.csv',index=False,encoding='utf-8')
    
    # 为每一个email添加标签
    resultfile = 'tmp/customer_cleaned.csv'
    result = pd.read_csv(resultfile,encoding='utf-8')
    result['cluster']=kmodel.labels_
    
    # 保存聚类结果文件
    result.to_csv('tmp/customer_result.csv',index=False,encoding='utf-8')
    
    # 保存模型
    save_model(kmodel,'kmeans.m')
    
    time2=time.time()
    print('finished,用时%.2fs'%(time2-time1))
    
    return data,kmodel.labels_

# 绘制雷达图
def plot_radar(data,title=''):
    '''
    the first column of the data is the cluster name;
    the second column is the number of each cluster;
    the last are those to describe the center of each cluster.
    '''
    time1=time.time()
    kinds = data.iloc[:, 0]
    labels = data.iloc[:, 2:].columns
    centers = pd.concat([data.iloc[:, 2:], data.iloc[:,2]], axis=1)
    centers = np.array(centers)
    n = len(labels)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))
    
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True) # 设置坐标为极坐标
    
    # 画若干个五边形
    floor = np.floor(centers.min())     # 大于最小值的最大整数
    ceil = np.ceil(centers.max())       # 小于最大值的最小整数
    jg=(ceil-floor)/5
    for i in np.arange(floor, ceil + jg, jg):
        ax.plot(angles, [i] * (n + 1), '--', lw=0.5 , color='black',alpha=0.8)
    
    # 画不同客户群的分割线
    for i in range(n):
        ax.plot([angles[i], angles[i]], [floor, ceil], '--', lw=0.5, color='black',alpha=0.8)
    
    # 画不同的客户群所占的大小
    for i in range(len(kinds)):
        ax.plot(angles, centers[i], lw=2, label=kinds[i])
        #ax.fill(angles, centers[i])
    
    ax.set_thetagrids(angles * 180 / np.pi, labels) # 设置显示的角度，将弧度转换为角度
    plt.legend(loc='lower right', bbox_to_anchor=(1.5, 0.0)) # 设置图例的位置，在画布外
    
    ax.set_theta_zero_location('N')        # 设置极坐标的起点（即0°）在正北方向，即相当于坐标轴逆时针旋转90°
    ax.spines['polar'].set_visible(False)  # 不显示极坐标最外圈的圆
    ax.grid(False)                         # 不显示默认的分割线
    ax.set_yticks([])                      # 不显示坐标间隔
    
    plt.savefig('img/聚类结果属性雷达图%s.png'%title,dpi=200)
    
    time2=time.time()
    print('finished,用时%.2fs'%(time2-time1))
    
# 属性直方图
def cluster_density(labels,k=5,title=''):
    '''绘制每一个聚类结果各个属性的分布图.'''
    time1=time.time()
    datafile = 'tmp/customer_cleaned.csv'
    data=pd.read_csv(datafile,encoding='utf-8')
    r=pd.Series(labels)
    r.name='cluster_names'
    #data['cluster']=labels
    def density_plot(data,title):
        k=len(data.columns)
        p = data.plot(kind='kde', linewidth=2, subplots=True, sharex=False,grid=True)
        [p[i].set_ylabel('密度') for i in range(k)]
        p[0].set_title(title)
        [p[i].grid(linestyle='--',alpha=0.8) for i in range(k)]
        plt.legend()
        plt.tight_layout()
        return plt
    
    # 保存概率密度图
    for i in range(k):
        density_plot(data[r == i],'分群_%s密度图'%(i+1)).savefig('./img/分群_%i%s.png' % (i+1,title),dpi=200)
        
    time2=time.time()
    print('finished,用时%.2fs'%(time2-time1))
        
# 可视化聚类结果        
def programmer_5(r,k=5,title=''):
    # 进行数据降维
    time1=time.time()
    datafile='tmp/customer_zscore.csv'
    data_zs=pd.read_csv(datafile,encoding='utf-8')
    tsne = TSNE()
    tsne.fit_transform(data_zs)
    tsne = pd.DataFrame(tsne.embedding_, index=data_zs.index)

    plt.figure()
    # 不同类别用不同颜色和样式绘图
    for i in range(k):
        d = tsne[r == i]
        plt.plot(d[0], d[1])
#    d = tsne[r == 1]
#    plt.plot(d[0], d[1], 'go')
#    d = tsne[r == 2]
#    plt.plot(d[0], d[1], 'b*')
    plt.title('聚类效果图')
    plt.savefig('img/聚类效果图%s.png'%title,dpi=200)
    
    time2=time.time()
    print('finished,用时%.2fs'%(time2-time1))
    
def programmer_6(center,labels,threshold=2,k=5,annotate_=True,title=''):
    """
    k：聚类中心数
    threshold：离散点阈值
    iteration：聚类最大循环次数
    """
    time1=time.time()
    datafile='tmp/customer_zscore.csv'
    data_zs=pd.read_csv(datafile,encoding='utf-8')
    data_zs['cluster']=labels

    norm = []
    for i in range(k):  # 逐一处理
        norm_tmp = data_zs[['ZR', 'ZF','ZM','ZC']][data_zs['cluster'] == i] - center.loc[i,['ZR', 'ZF','ZM','ZC']]
        norm_tmp = norm_tmp.apply(np.linalg.norm, axis=1)
        # 求相对距离并添加
        norm.append(norm_tmp / norm_tmp.median())

    norm = pd.concat(norm)
    
    plt.figure()
    # 正常点
    norm[norm <= threshold].plot(style='go')
    # 离群点
    discrete_points = norm[norm > threshold]
    discrete_points.plot(style='rx')
    # 标记离群点
    if annotate_:
        for i in range(len(discrete_points)):
            _id = discrete_points.index[i]
            n = discrete_points.iloc[i]
            plt.annotate('(%s, %0.2f)' % (_id, n), xy=(_id, n), xytext=(_id, n))
    else:
        pass

    plt.xlabel('编号')
    plt.ylabel('相对距离')
    plt.title('离群点标记(%d倍标准差)'%threshold)
    plt.grid(linestyle='--',alpha=0.8)
    plt.savefig('img/离群点标记%s.png'%title,dpi=200)
    
    time2=time.time()
    
    # 导出文件清除了离群值后的数据记录
    #data_zs[['ZL', 'ZR', 'ZF','ZM','ZC']][norm <= threshold].to_excel('tmp/zscoreddata_01_%d.xls'%threshold,index=False)
    index=norm[norm <= threshold].index
    print('finished,用时%.2fs,数据量为%d个,正常数据占比%.2f%%'%(time2-time1,len(norm),len(index)/len(norm)*100))
    
    return norm[norm <= threshold].index

# 主函数
def main(begin=True,k=4,title=''):
    '''begin==True时默认从头开始,否则仅为去除离群值后的结果.'''
    if begin:
        print('>> 数据探索...')
        print(programmer_1())    # 数据探索
        # 筛选数据
        print('>> 数据筛选...')
        programmer_2()
    else:
        pass
    print('>> 数据标准化...')
    programmer_3()
    print('>> 聚类分析...')
    center,labels=programmer_4(k,True)    # 聚类
    print('>> 属性雷达图...')
    plot_radar(center,title=title)              # 绘制雷达图
    print('>> 属性直方图...')
    cluster_density(labels,k=k,title=title)
#    print('>> 聚类效果图...')
#    programmer_5(labels,k=k,title=title)
    print('>> 离群点...')
    norm_index=programmer_6(center,labels,threshold=3,k=k,annotate_=False,title=title)
    return norm_index
    
if __name__=="__main__":
    #首次调用主函数
    norm_index=main()      # 返回去除离群值后的索引
    # 重新筛选数据
    print('去除离群值影响.')
    data=pd.read_csv('tmp/customer_cleaned.csv',encoding='utf-8').loc[norm_index,:] 
    data.to_csv('tmp/customer_cleaned.csv',index=False)  # 去除离群值的影响
    norm_index=main(False,title='_去除离群值')
    
    
    
    
    
    
    
    