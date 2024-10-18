from phe import paillier
import copy
from math import log
import random
import time


import csv


def write_csv_one_line(file_path, file_name, data_list):
    f = open(file_path + file_name, "a+", encoding="utf-8", newline="")
    csv_writer = csv.writer(f)
    csv_writer.writerow(data_list)
    f.close()


class PETD:
    K = 0
    M = 0
    sk = None
    pk = None
    E_ak = list()
    E_dk = list()
    bitlength = 32
    lam = 0.5
    E_X = None
    E_Y = None

    def __init__(self, K, M, bitlegth):
        self.K = K
        self.M = M
        self.bitlength = bitlegth
        public_key, private_key = paillier.generate_paillier_keypair(n_length=2048)
        self.sk = private_key
        self.pk = public_key
        self.E_X = self.pk.encrypt(0)
        self.E_Y = self.pk.encrypt(0)
        self.E_ak = [self.pk.encrypt(0) for k in range(self.K)]

    def get_user_data(self):
        x_mt_k_2 = list()
        x_mt_k_1 = list()
        E_x_m_k = list()
        for m in range(self.M):
            # truth = random.randint(0,1000)
            truth = 20
            x_mt_k_2.append([random.randint(0, int(1e6)) for k in range(self.K)])
            x_mt_k_1.append(
                [
                    truth
                    + 0.5 * random.randint(-int(0.5 * truth), int(0.5 * truth))
                    - x_mt_k_2[m][k]
                    for k in range(self.K)
                ]
            )
            E_x_m_k.append([self.pk.encrypt(truth) for k in range(self.K)])
        return x_mt_k_1, x_mt_k_2, E_x_m_k

    def get_d(self, x_m_1, x_m_2, E_x_m):
        E_d = None

        for m in range(self.M):
            r_temp = random.randint(0, int(1e6))
            E_u = E_x_m[m].__mul__(-1)
            E_u = E_u._add_scalar(x_m_1[m] + r_temp)

            u = self.sk.decrypt(E_u)
            z = u + x_m_2[m]

            # X = self.pk.encrypt(z)
            # Y = self.pk.encrypt(z*z)
            self.E_X.__mul__(0)
            self.E_Y.__mul__(0)
            X = self.E_X._add_scalar(z)
            Y = self.E_Y._add_scalar(z * z)

            deta1 = X._add_scalar(-r_temp)
            deta2 = Y._add_encrypted(deta1.__mul__(-2 * r_temp))._add_scalar(
                -r_temp * r_temp
            )
            if m == 0:
                E_d = deta2
            else:
                E_d = E_d._add_encrypted(deta2)
        return E_d

    def weigth_update(self, x_mt_k_1, x_mt_k_2, E_x_m):
        E_d_k_ = list()
        E_sum_dk = None
        r_dk = list()
        R_sum_dk = 0
        c1_cost = 0
        for k in range(self.K):
            # 获取用户数据
            x_m_1 = list()
            x_m_2 = list()
            for m in range(self.M):
                x_m_1.append(x_mt_k_1[m][k])
                x_m_2.append(x_mt_k_2[m][k])
            E_d_temp = self.get_d(x_m_1, x_m_2, E_x_m)
            self.E_ak[k] = self.E_ak[k].__mul__(self.lam)._add_encrypted(E_d_temp)
            E_d_k_.append(self.E_ak[k])
            if k == 0:
                E_sum_dk = E_d_k_[k]
            else:
                E_sum_dk = E_sum_dk._add_encrypted(E_d_k_[k])
            r_d = random.randint(0, int(1e6))
            r_dk.append(r_d)
            # C1
            start_time = time.time()
            E_d_k_[k] = E_d_k_[k].__mul__(r_d)
            elapsed_time = time.time() - start_time
            c1_cost += elapsed_time
            
        # C1
        start_time = time.time()
        
        R_sum_dk = random.randint(0, int(1e6))
        E_sum_dk = E_sum_dk.__mul__(R_sum_dk)
        E_wk_ = list()
        for k in range(self.K):
            dk_ = self.sk.decrypt(E_d_k_[k])
            wk_ = log(R_sum_dk, 2) - log(dk_, 2)
            E_wk_.append(E_sum_dk._add_scalar(wk_ - R_sum_dk))

        E_wk = list()
        for k in range(self.K):
            E_wk.append(E_wk_[k]._add_scalar(log(R_sum_dk / r_dk[k], 2)))
        
        elapsed_time = time.time() - start_time
        c1_cost += elapsed_time
        return E_wk, c1_cost

    def truth_update(self, E_wk, x_mt_k_1, x_mt_k_2):
        c1_cost = 0
        c2_cost = 0
        # C1
        start_time = time.time()
        X_m_1_list = list()
        for m in range(self.M):
            X_m_1_list.append(E_wk[0].__mul__(x_mt_k_1[m][0]))
            for k in range(1, self.K):
                X_m_1_list[m] = X_m_1_list[m]._add_encrypted(
                    E_wk[k].__mul__(x_mt_k_1[m][k])
                )
        elapsed_time = time.time() - start_time
        c1_cost += elapsed_time

        # C2
        start_time = time.time()
        X_m_2_list = list()
        for m in range(self.M):
            X_m_2_list.append(E_wk[0].__mul__(x_mt_k_2[m][0]))
            for k in range(1, self.K):
                X_m_2_list[m] = X_m_2_list[m]._add_encrypted(
                    E_wk[k].__mul__(x_mt_k_2[m][k])
                )
        elapsed_time = time.time() - start_time
        c2_cost += elapsed_time

        # C1
        start_time = time.time()
        E_W = E_wk[0]
        for k in range(1, self.K):
            E_W = E_W._add_encrypted(E_wk[k])

        X_m_list = list()
        Y_m_list = list()
        r_m_list = list()
        E_W_list = list()
        for m in range(self.M):
            X_m_list.append(X_m_1_list[m]._add_encrypted(X_m_2_list[m]))
            r_m = random.randint(0, int(1e6))
            Y_m_list.append(X_m_list[m].__mul__(r_m * r_m))
            r_m_list.append(r_m)
            E_W_list.append(E_W.__mul__(r_m))
        elapsed_time = time.time() - start_time
        c1_cost += elapsed_time

        # C2
        start_time = time.time()
        Z_m_list = list()
        for m in range(self.M):
            x_m = self.sk.decrypt(Y_m_list[m]) / self.sk.decrypt(E_W_list[m])
            Z_m_list.append(self.pk.encrypt(x_m))
        elapsed_time = time.time() - start_time
        c2_cost += elapsed_time

        # C1
        start_time = time.time()
        for m in range(self.M):
            E_x_m = Z_m_list[m].__mul__(-1 * r_m_list[m])
        elapsed_time = time.time() - start_time
        c1_cost += elapsed_time

        return E_x_m, c1_cost, c2_cost


def var_K(filename, direct_path, K, M):
    K_list = list()
    all_time_weight_list = list()
    all_time_truth_list = list()

    K_list.append("M=%d,K=" % (M))
    all_time_truth_list.append("all_time_truth")
    all_time_weight_list.append("all_time_weight")

    for i in range(8):
        K = K + 10
        # M = M + 5

        all_time_weight = 0
        all_time_truth = 0

        petd = PETD(K, M, 32 - 6)

        x_mt_k_1, x_mt_k_2, E_x_m_k = petd.get_user_data()

        for i in range(3):
            E_x_m = [petd.pk.encrypt(10, precision=1) for i in range(M)]

            start_time = time.perf_counter_ns()
            E_wk = petd.weigth_update(x_mt_k_1, x_mt_k_2, E_x_m)
            end_time = time.perf_counter_ns()
            all_time_weight += (end_time - start_time) / 1000000

            start_time = time.perf_counter_ns()
            E_x_m = petd.truth_update(E_wk, x_mt_k_1, x_mt_k_2)
            end_time = time.perf_counter_ns()
            all_time_truth += (end_time - start_time) / 1000000

        K_list.append(K)
        all_time_weight_list.append(all_time_weight / 3)
        all_time_truth_list.append(all_time_truth / 3)

    print("K_list", K_list)
    print("all_time_weight_list", all_time_weight_list)
    print("all_time_truth_list", all_time_truth_list)
    write_csv_one_line(direct_path, filename, K_list)
    write_csv_one_line(direct_path, filename, all_time_weight_list)
    write_csv_one_line(direct_path, filename, all_time_truth_list)


def var_M(filename, direct_path, K, M):
    M_list = list()
    all_time_weight_list = list()
    all_time_truth_list = list()

    M_list.append("K=%d,M=" % (K))
    all_time_truth_list.append("all_time_truth")
    all_time_weight_list.append("all_time_weight")

    for i in range(8):
        # K = K + 10
        M = M + 5

        all_time_weight = 0
        all_time_truth = 0

        petd = PETD(K, M, 32 - 6)

        x_mt_k_1, x_mt_k_2, E_x_m_k = petd.get_user_data()

        for i in range(3):
            E_x_m = [petd.pk.encrypt(10, precision=1) for i in range(M)]

            start_time = time.perf_counter_ns()
            E_wk = petd.weigth_update(x_mt_k_1, x_mt_k_2, E_x_m)
            end_time = time.perf_counter_ns()
            all_time_weight += (end_time - start_time) / 1000000

            start_time = time.perf_counter_ns()
            E_x_m = petd.truth_update(E_wk, x_mt_k_1, x_mt_k_2)
            end_time = time.perf_counter_ns()
            all_time_truth += (end_time - start_time) / 1000000

        M_list.append(M)
        all_time_weight_list.append(all_time_weight / 3)
        all_time_truth_list.append(all_time_truth / 3)

    print("M_list", M_list)
    print("all_time_weight_list", all_time_weight_list)
    print("all_time_truth_list", all_time_truth_list)
    write_csv_one_line(direct_path, filename, M_list)
    write_csv_one_line(direct_path, filename, all_time_weight_list)
    write_csv_one_line(direct_path, filename, all_time_truth_list)


# K = 20
# M = 0
# filename = "New_RMSE_Fig1.csv"
# direct_path = "D:\workPlace/researchCode/PPTD/ExperimentalData/"
# var_K(filename,direct_path,20,10)
# var_M(filename,direct_path,100,0)


# n = 10
# K = 5
# M = 5
# bitlength = 32
# petd = PETD(K,M,bitlength)
# petd.get_user_data()
# x_mt_k_1,x_mt_k_2,E_x_m_k = petd.get_user_data()
# E_x_m = [petd.pk.encrypt(10,precision=1) for i in range(M)]
# E_wk = petd.weigth_update(x_mt_k_1,x_mt_k_2,E_x_m)
# E_x_m = petd.truth_update(E_wk,x_mt_k_1,x_mt_k_2)

# for i in range(10):
#     x = [random.randint(0,2**bitlength) for i in range(n)]
#     x_m_1 = [random.randint(0,2**bitlength) for i in range(n)]
#     x_m_2 = [x[i] - x_m_1[i] for i in range(n)]
#     x_m = [random.randint(0,2**bitlength) for i in range(n)]
#     E_x_m = [petd.pk.encrypt(x_m[i]) for i in range(n)]

#     # print("d",sum([x[i] - x_m[i] for i in range(n)]))
#     start_time = time.perf_counter_ns()
#     E_d = petd.get_d(x_m_1,x_m_2,E_x_m)
#     end_time = time.perf_counter_ns()
#     print("get_d_times:",(end_time-start_time)/1000000)
