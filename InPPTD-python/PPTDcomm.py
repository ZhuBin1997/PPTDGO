from phe import paillier
import numpy as np
import time
import random
# import matplotlib.pyplot as plt

K = 50
M_range = [20, 40, 60,80,100]
# M_range = [10, 20, 30, 40, 50,60,70,80,90,100]
Ite_Number = 5

size_C = 4096
size_P = 64

def toMB_ite_all(comm):
    return round(comm*Ite_Number/(1024*1024*8),3)

def toKB(comm):
    return round(comm/(1024*8),3)

# 初始化
for M in [10, 50, 100,500,1000]:
    L2PPTD = 2*M*size_P
    NPPTD = 2*M*size_P
    RPTD = 2*(M+1)*size_C
    PETD = 2*M*size_P
    InPPTD = 2*(M+1)*size_P
    print(M," & ",toKB(L2PPTD)," & ",toKB(NPPTD)," & ",toKB(RPTD)," & ",toKB(PETD)," & ",toKB(InPPTD),"\\\\")
    

# 迭代
for M in M_range:
    PPTD = (2*K+K+K*M)*size_C+M*64
    L2PPTD = (K+M+1)*size_C
    RPTD = (3*K+M+1)*size_P+(2*K+2*M+1)*size_C
    SDH = 4*M*size_C
    PETD = (6*K+5*M+1)*size_C+K*SDH
    InPPTD = (2*K+2*M+1)*size_C+(M+1)*size_P
    print(M," & ",toMB_ite_all(L2PPTD)," & ",toMB_ite_all(RPTD)," & ",toMB_ite_all(PETD)," & ",toMB_ite_all(InPPTD),"\\\\")
    
    
M = 50
K_range = [10, 50, 100, 150, 200]
# K_range = [10, 25, 50, 75, 100, 125, 150, 175, 200]
for K in K_range:
    PPTD = (2*K+K+K*M)*size_C+M*64
    L2PPTD = (K+M+1)*size_C
    RPTD = (3*K+M+1)*size_P+(2*K+2*M+1)*size_C
    SDH = 4*M*size_C
    PETD = (6*K+5*M+1)*size_C+K*SDH
    InPPTD = (2*K+2*M+1)*size_C+(M+1)*size_P
    print(K," & ",toMB_ite_all(L2PPTD)," & ",toMB_ite_all(RPTD)," & ",toMB_ite_all(PETD)," & ",toMB_ite_all(InPPTD),"\\\\")
    