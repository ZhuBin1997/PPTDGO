# -*-coding:utf-8 -*-
import xlwt
import numpy as np
import matplotlib.pyplot as plt

# 1、数据产生
def normal_worker(avg,sigma,K,M,excel=True):
    # avg 均值; sigma^2 是方差; K workers数量; M objects数量
    # avg和sigma是长为M的list
    # excel表示是否生成excel文件
    workers=np.zeros((K,M),dtype=int)
    for k in range(K):
        for m in range(M):
            t=np.random.normal(avg[m],sigma[m])
            while t<0 or t>2*avg[m]:    
                t=np.random.normal(avg[m],sigma[m])
            workers[k,m]=t
    # print('normal workers:')
    # print(workers)
    if excel:
        wbk = xlwt.Workbook()
        sheet = wbk.add_sheet('sheet 1')
        for k in range(K):
            for m in range(M):
                sheet.write(k,m,int(workers[k,m]))
        wbk.save('normal_workers.xls')
    return workers

def lazy_worker(avg,K,M,excel=True):
    workers=np.zeros((K,M),dtype=int)
    for m in range(M):
        workers[:,m]=np.random.uniform(0,2*avg[m],K)
    # print('lazy workers:')
    # print(workers)
    if excel:
        wbk = xlwt.Workbook()
        sheet = wbk.add_sheet('sheet 1')
        for k in range(K):
            for m in range(M):
                sheet.write(k,m,int(workers[k,m]))
        wbk.save('lazy_workers.xls')
    return workers

def data_create(avg,sigma,K,M,excel=True):
    # 返回两个大小为(K,M)的矩阵
    normal=normal_worker(avg,sigma,K,M,excel)
    lazy=lazy_worker(avg,K,M,excel)
    return normal,lazy


# 2、Accuracy&Convergence

# 一次处理
def Truth_Discovery_Base(xi,ti,K,M):
    # t:M  x:K,M   
    x=xi[0:K,0:M]
    t=ti[0:M]
    tm=t
    dxt=np.power((x-tm),2)
    d_sum=dxt.sum(axis=1)
    sum=d_sum.sum()
    wk=np.log(sum/d_sum)
    wk_sum=wk.sum()
    tm2=(np.transpose(x)*wk).sum(axis=1)/wk_sum
    return tm2,wk


# 多次循环，e代表两次迭代间最大误差(貌似没用到)
def Truth_Discovery(xi,ti,K,M,e=0.01):
    # t:M  x:K,M   
    x=xi[0:K,0:M]
    t=ti[0:M]
    tm=t
    tm2,wk=Truth_Discovery_Base(x,tm,K,M)
    while np.sum(np.abs(tm-tm2)>e)>0:
        tm=tm2
        tm2,wk=Truth_Discovery_Base(x,tm,K,M)
        # print('emmmm')
    # return (tm2+0.5).astype(int)
    return tm2,wk

# 均方根误差
def RMSE(avg,tm,M):
    a=np.array(avg[0:M])
    t=np.array(tm[0:M])
    return np.sqrt(np.sum(np.power(a-t,2))/(M-1))


def Convergence(x,ti,K,M,N):
    # N为迭代次数
    tm=ti
    tm2,wk=Truth_Discovery_Base(x,tm,K,M)
    con=[]
    for i in range(N):
        con.append(np.linalg.norm(tm-tm2))
        tm=tm2
        tm2,wk=Truth_Discovery_Base(x,tm,K,M)
    return con

def ini_avg(M):
    def_range_min = 30
    def_range_max = 10000
    t_v = np.random.randint(def_range_min,def_range_max,size=M)
    return t_v

def ini_ti(avg,M,delta):
    t=[]
    for x in range(M):
        t.append(np.random.randint(avg[x]-delta,avg[x]+delta))
    return t

def function2():
    M=20
    avg = ini_avg(M)
    #avg=[100,1000,500,600,140,100,1000,500,600,140,100,1000,500,600,104,100,1000,500,600,100]
    sigma=[4]*M
    rmse=[]
    K=100
    K1 =0
    # 生成数据
    normal,lazy=data_create(avg,sigma,K,M,False)

    # 图1
    ti = ini_ti(avg,20,3)
    a =0
    #ti=[5,1005,40,70,15,5,1005,40,70,15,5,1005,40,70,15,5,1005,40,70,15]
    for i in range(6):
        rmse_temp =[]

        for k in [10,20,30,40,50,60]:
            K1 = a
            workers = np.zeros((k + K1, M))
            workers[0:k] = normal[0:k]
            workers[k:k + K1] = lazy[0:K1]
            tm=np.array(ti)
            # 应用一次算法
            tm2,wk=Truth_Discovery_Base(workers,tm,k+K1,M)
        
            # tm2=Truth_Discovery(normal,tm,K,M,0.001)
            # print('aaa',tm2)
            rmse_temp.append(RMSE(avg,tm2,M))
        a += 1
        rmse.append(rmse_temp)
    #print(rmse)

    font1 = {'family':"Times New Roman",
             'weight' : 'normal',
             'size' : 14,}
    font2 = {'family': "Times New Roman",
             'weight': 'normal',
             'size': 16,}

    ax=plt.figure()
    x_rmse=[i for i in [10,20,30,40,50,60]]
    plt.plot(x_rmse,rmse[0],'-ro',linewidth = 4,markerfacecolor='white',markeredgecolor ='r',markeredgewidth=4,markersize = 11, label = '0 Lazy Worker')
    plt.plot(x_rmse, rmse[1],'-b^',linewidth = 4,markerfacecolor='white',markeredgecolor ='b',markeredgewidth=4,markersize = 10, label = '1 Lazy Worker')
    plt.plot(x_rmse, rmse[2], '-gs', linewidth=4, markerfacecolor='white', markeredgecolor='g', markeredgewidth=4,
             markersize=10, label='2 Lazy Workers')
    plt.plot(x_rmse, rmse[3], '-cp', linewidth=4, markerfacecolor='white', markeredgecolor='c', markeredgewidth=4,
             markersize=10, label='3 Lazy Workers')
    plt.plot(x_rmse, rmse[4], '-kD', linewidth=4, markerfacecolor='white', markeredgecolor='k', markeredgewidth=4,
             markersize=10, label='4 Lazy Workers')
    plt.plot(x_rmse, rmse[5], '-yd', linewidth=4, markerfacecolor='white', markeredgecolor='y', markeredgewidth=4,
             markersize=10, label='5 Lazy Workers')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.xlabel('Number of Workers',font1)

    plt.ylabel('RMSE',font1)
    plt.legend(numpoints=1)
    # plt.grid()

    for i in range(0,6):
        print("rmse[i]",rmse[i])

    # 图2
    t1 = ini_ti(avg, 20, 3)
    # t1=[5,1005,40,70,15,5,1005,40,70,15,5,1005,40,70,15,5,1005,40,70,15]
    a = 0
    # ti=[5,1005,40,70,15,5,1005,40,70,15,5,1005,40,70,15,5,1005,40,70,15]
    N = 10
    con =[]
    k =100
    for i in range(6):
        con_temp = []


        K1 = a
        workers = np.zeros((k + K1, M))
        workers[0:k] = normal[0:k]
        workers[k:k + K1] = lazy[0:K1]
        tm = np.array(ti)
            # 应用一次算法
        con_temp.append(Convergence(workers, t1, K, M, N))

            # tm2=Truth_Discovery(normal,tm,K,M,0.001)
            # print('aaa',tm2)
           # rmse_temp.append(RMSE(avg, tm2, M))
        a += 1
        con.append(Convergence(workers, t1, k+K1, M, N))

    plt.figure()
    x_rmse = [i for i in range(1,11)]
    plt.plot(x_rmse,con[0], '-ro', linewidth=4, markerfacecolor='white', markeredgecolor='r', markeredgewidth=4,
             markersize=11, label='0 Lazy Worker')
    plt.plot(x_rmse, con[1], '-b^', linewidth=4, markerfacecolor='white', markeredgecolor='b', markeredgewidth=4,
             markersize=10, label='1 Lazy Worker')
    plt.plot(x_rmse, con[2], '-gs', linewidth=4, markerfacecolor='white', markeredgecolor='g', markeredgewidth=4,
             markersize=10, label='2 Lazy Workers')
    plt.plot(x_rmse, con[3], '-cp', linewidth=4, markerfacecolor='white', markeredgecolor='c', markeredgewidth=4,
             markersize=10, label='3 Lazy Workers')
    plt.plot(x_rmse, con[4], '-kD', linewidth=4, markerfacecolor='white', markeredgecolor='k', markeredgewidth=4,
             markersize=10, label='4 Lazy Workers')
    plt.plot(x_rmse, con[5], '-yd', linewidth=4, markerfacecolor='white', markeredgecolor='y', markeredgewidth=4,
             markersize=10, label='5 Lazy Workers')

 #   plt.title('Convergence')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel(r'Iteration $i$',font2)
    plt.ylabel('Convergence Value',font1)
    plt.legend(numpoints=1)
    # plt.grid()
    plt.show()

    for i in range(0,6):
        print("con[",i,"]",con[i])


# 5 （normal和lazy各自单独统计，结果随机性很大）
def function5():
    # M:5,40  K=10
    K=10
    M=40
    avg=[100,1000,500,600,140,100,1000,500,600,140,100,1000,500,600,104,100,1000,500,600,100,100,1000,500,600,140,100,1000,500,600,140,100,1000,500,600,104,100,1000,500,600,100]
    tm=[100,1000,500,600,140,100,1000,500,600,140,100,1000,500,600,104,100,1000,500,600,100,100,1000,500,600,140,100,1000,500,600,140,100,1000,500,600,104,100,1000,500,600,100]
    # tm=[5,1005,40,70,15,5,1005,40,70,15,5,1005,40,70,15,5,1005,40,70,15,5,1005,40,70,15,5,1005,40,70,15,5,1005,40,70,15,5,1005,40,70,15]
    sigma=[4]*M
    K=10
    normal,lazy=data_create(avg,sigma,K,M,False)
    normal_reward=np.zeros((K,M+1))
    lazy_reward=np.zeros((K,M+1))
    x=[i for i in range(5,M+1)]
    for m in range(5,M+1):
        tm2,wk=Truth_Discovery(normal,tm,K,m,0.001)
        normal_reward[:,m]=wk/wk.sum()
        print('111',tm2)
        tm2,wk=Truth_Discovery(lazy,tm,K,m,0.001)
        lazy_reward[:,m]=wk/wk.sum()
        print('222',tm2)
    # print(normal_reward)
    # print(normal_reward.shape)
    plt.figure()

    plt.plot(x,normal_reward[6][5:],'g', label='normal 1')
    plt.plot(x,lazy_reward[6][5:],'r', label='lazy 1')

    # plt.title('Convergence')
    # plt.xlabel('Iteration t')
    # plt.ylabel('Convergence value')
    plt.legend(numpoints=1)
    plt.show()

# （等量的normal和lazy一起统计，Normal权重高于Lazy，效果很好）
def function6():
    # M:5,40  K=10
    K=40
    M=20

    avg= ini_avg(M)
    tm=ini_ti(avg,M,3)
    # tm=[5,1005,40,70,15,5,1005,40,70,15,5,1005,40,70,15,5,1005,40,70,15,5,1005,40,70,15,5,1005,40,70,15,5,1005,40,70,15,5,1005,40,70,15]
    sigma=[4]*M
    K1= int(K/10)
    normal,lazy=data_create(avg,sigma,K,M,False)
    workers=np.zeros((K+K1,M))
    workers[0:K]=normal
    workers[K:K+K1]=lazy[0:K1]
    reward = 0
  #  reward=np.zeros((K+K1,M+1))

    reward_normal = []
    reward_lazy = []
    for i in range(7):
        normal = 0.0
        r = [1,5,10,15,20,25,30]
        lazy =  0.0
        for j in range(r[i]):
            tm2,wk=Truth_Discovery(workers,tm,K+K1,M,0.001)
            w = wk.sum()
            Price = np.random.randint(50,100)
            normal += wk[0:K].sum()*Price/(K*w)
            lazy_set = wk[K:K+K1]
            lazy += lazy_set.sum()*Price/(K1*w)
        reward_normal.append(normal)
        reward_lazy.append(lazy)
    print('\n tm2:',tm2)
    # print(normal_reward)
    # print(normal_reward.shape)
    plt.figure()
    x =[1,5,10,15,20,25,30]
    font1 = {'family': "Time New Roman",
             'weight': 'normal',
             'size': 14,}

    print('\n reward_normal:',reward_normal)
    print('\n reward_lazy:',reward_lazy)
    
    plt.plot(x,reward_normal,'-g^',linewidth = 4,markerfacecolor='white',markeredgecolor ='g',markeredgewidth=4,markersize = 10, label='Normal Worker')
    plt.plot(x, reward_lazy, '-ro', linewidth=4, markerfacecolor='white', markeredgecolor='r', markeredgewidth=4,
             markersize=10, label='Lazy Worker')
  #  k=6
   # plt.plot(x,reward[k][5:],'g',marker='s', label='normal 6', mfc='w',ms=5)
    #plt.plot(x,reward[k+K][5:],'r', marker='s',label='lazy 6', mfc='w',ms=5)


    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.xlabel('Number of Tasks',font1)
    plt.ylabel('Average Rewards',font1)



    plt.legend(numpoints=1)
    plt.show()


if __name__ == '__main__':
    function2()

    function6()


# 密码学ppt，密码学算法，密码学库