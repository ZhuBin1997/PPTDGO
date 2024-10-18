from phe import paillier
import numpy as np
import time
import random
# import matplotlib.pyplot as plt
from PETD import *

# -------------------------------------------------------------------------------------------------
# benchmarking operations
low = 0
high = 1e6
batch = 100
a = np.random.randint(low, high, batch).tolist()
b = np.random.randint(low, high, batch).tolist()
pk, sk = paillier.generate_paillier_keypair(n_length=2048)
cta = [pk.encrypt(x) for x in a]
ctb = [pk.encrypt(x) for x in b]


def test_encryption_time():
    secret_list = np.random.randint(low, high, batch).tolist()
    start_time = time.time()
    cts = [pk.encrypt(x) for x in secret_list]
    elapsed_time = time.time() - start_time
    return elapsed_time


def test_HAdd_time():
    start_time = time.time()
    ctc = [cta[i] + ctb[i] for i in range(batch)]
    elapsed_time = time.time() - start_time
    return elapsed_time


def test_HMultScalar_time():
    start_time = time.time()
    ctc = [cta[i] * b[i] for i in range(batch)]
    elapsed_time = time.time() - start_time
    return elapsed_time


def test_decryption_time():
    start_time = time.time()
    decrypted_list = [sk.decrypt(x) for x in cta]
    elapsed_time = time.time() - start_time
    return elapsed_time


# ---------------------------------------------------------------------------------------
# calculating the cost of each protocol
def RPTD_iterations(
    encryption_cost, HAdd_cost, HMultScalar_cost, decryption_cost, M, K
):
    # FN：$2M$ Enc + $KM+4K$ HAdd + $KM+K+1$HMultScalar。
    # CSP：$2M+K$ Dec + $K$ Enc。

    fn_cost = (
        2 * M * encryption_cost
        + (K * M + 4 * K) * HAdd_cost
        + (3 * K * M + K + 1) * HMultScalar_cost
    )
    csp_cost = (2 * M + K) * decryption_cost + K * encryption_cost
    return fn_cost, csp_cost


def RPTD_data_collection():
    pass


def L2PPTD_iterations(
    encryption_cost, HAdd_cost, HMultScalar_cost, decryption_cost, M, K
):
    # SA: $MK$ (Enc + HMultScalar) + $K(M-1)$ HAdd + $M$ Dec
    # SB：$M$ Enc + $KM$ HMultScalar + $M(K-1)$ HAdd + $K$ Dec

    sa_cost = (
        M * K * (encryption_cost + HMultScalar_cost)
        + (K * (M - 1)) * HAdd_cost
        + M * decryption_cost
    )
    sb_cost = (
        M * encryption_cost
        + M * K * HMultScalar_cost
        + (M * (K - 1) + M) * HAdd_cost
        + K * decryption_cost
    )
    return sa_cost, sb_cost


def L2PPTD_data_collection(
    encryption_cost, HAdd_cost, HMultScalar_cost, decryption_cost, M, K
):
    sa_cost = M * K * encryption_cost
    sb_cost = M * K * encryption_cost
    return sa_cost, sb_cost


def InPPTD_data_collection(
    encryption_cost, HAdd_cost, HMultScalar_cost, decryption_cost, M, K
):
    sp_cost = K * (M + 1) * encryption_cost
    cp_cost = K * (M + 1) * encryption_cost + K * (M + 1) * HAdd_cost
    return sp_cost, cp_cost


def InPPTD_iterations(
    encryption_cost, HAdd_cost, HMultScalar_cost, decryption_cost, M, K
):
    # SP: K+M Enc + K+M+1 Dec
    # CP: 2M Enc + 4KM+K+M-1 HAdd + 3KM+K+1 HMultScalar
    sp_cost = (K + M) * encryption_cost + (K + M + 1) * decryption_cost
    cp_cost = (
        2 * M * encryption_cost
        + (4 * K * M + K + M - 1) * HAdd_cost
        + (3 * K * M + K + 1) * HMultScalar_cost
    )
    return sp_cost, cp_cost


def InPPTD_single_incentive(
    encryption_cost, HAdd_cost, HMultScalar_cost, decryption_cost, n, K
):
    cp_cost = K * encryption_cost + K * HAdd_cost + K * HMultScalar_cost
    return cp_cost


def InPPTD_aggregation_incentive(
    encryption_cost, HAdd_cost, HMultScalar_cost, decryption_cost, n, K
):
    cp_cost = (n - 1) * K * HAdd_cost
    sp_cost = K * decryption_cost
    return sp_cost, cp_cost

# implementation of Zhao
# def PETD_iterations_Zhao(M,K):
#     bitlen = 32
#     c1_cost = 0
#     c2_cost = 0
#     petd = PETD(K, M, bitlen)
#     x_mt_k_1, x_mt_k_2, E_x_m_k = petd.get_user_data()
#     E_x_m = [petd.pk.encrypt(10, precision=1) for i in range(M)]

#     res = petd.weigth_update(x_mt_k_1, x_mt_k_2, E_x_m)
#     E_wk = res[0]
#     c1_cost += res[1]

#     res = petd.truth_update(E_wk, x_mt_k_1, x_mt_k_2)
#     c1_cost += res[1]
#     c2_cost += res[2]
#     return c1_cost, c2_cost


# Complexity results
# 总结：
# - 计算开销：SDH
# - - C1: $3M$ Enc + $4M$ HMult + $4M$ Add + $M$ PD1
# - - C2: $2M$ Enc + $M$ PD2
# - 计算开销：迭代
# - - C1: $K$ Enc + $MK+3(M+K)$ HMult + $MK+3K-2$ HAdd + $2K+2M$ PD1 + $K$ SDH
# - - C2: $K+M$ Enc + $MK$ HMult + $M(K-1)$ HAdd + $2K+2M$ PD2 + $K$ SDH
# - 通信开销：$(6K+5M+1) S_c$ + $K$ SDH
def PETD_preparation_Complexity(
    encryption_cost, HAdd_cost, HMultScalar_cost, decryption_cost, M, K
):
    c1_cost = 0
    c1_cost += M * encryption_cost
    return c1_cost


def PETD_SDH_Complexity(encryption_cost, HAdd_cost, HMultScalar_cost, decryption_cost, M, K):
    PD1_cost = decryption_cost
    PD2_cost = decryption_cost + HAdd_cost

    c1_cost = (
        3 * M * encryption_cost
        + 4 * M * HMultScalar_cost
        + 4 * M * HAdd_cost
        + M * PD1_cost
    )
    c2_cost = 2 * M * encryption_cost + M * PD2_cost
    return c1_cost, c2_cost


# def PETD_iterations_Complexity(
#     encryption_cost, HAdd_cost, HMultScalar_cost, decryption_cost, M, K
# ):
#     PD1_cost = decryption_cost
#     PD2_cost = decryption_cost + HAdd_cost

#     sdh = K * PETD_SDH_Complexity(
#         encryption_cost, HAdd_cost, HMultScalar_cost, decryption_cost, M, K
#     )
#     c1_cost = (
#         sdh[0]
#         + K * encryption_cost
#         + (M * K + 3 * (M + K)) * HMultScalar_cost
#         + (M * K + 3 * K - 2) * HAdd_cost
#         + (2 * K + 2 * M) * PD1_cost
#     )
#     c2_cost = (
#         sdh[1]
#         + (K + M) * encryption_cost
#         + (M * K) * HMultScalar_cost
#         + (M * (K - 1)) * HAdd_cost
#         + (2 * K + 2 * M) * PD2_cost
#     )
#     return c1_cost, c2_cost


def PETD_iterations_Complexity(
    encryption_cost, HAdd_cost, HMultScalar_cost, decryption_cost, M, K
):
    PD1_cost = decryption_cost
    PD2_cost = decryption_cost + HAdd_cost
    sdh = PETD_SDH_Complexity(
        encryption_cost, HAdd_cost, HMultScalar_cost, decryption_cost, M, K
    )
    
    c1_cost = (
        sdh[0]*K
        + K * encryption_cost
        + (M * K + 3 * (M + K)) * HMultScalar_cost
        + (M * K + 3 * K - 2) * HAdd_cost
        + (2 * K + 2 * M) * PD1_cost
    )
    c2_cost = (
        sdh[1]*K
        + (K + M) * encryption_cost
        + (M * K) * HMultScalar_cost
        + (M * (K - 1)) * HAdd_cost
        + (2 * K + 2 * M) * PD2_cost
    )
    return c1_cost, c2_cost

if __name__ == "__main__":
    encryption_cost = test_encryption_time() / batch
    print(
        f"Single Paillier encryption with key size 2048-bit takes {encryption_cost:.6f} seconds"
    )
    HAdd_cost = test_HAdd_time() / batch
    print(
        f"Single Paillier addition with key size 2048-bit takes {HAdd_cost:.6f} seconds"
    )
    HMultScalar_cost = test_HMultScalar_time() / batch
    print(
        f"Single Paillier Multiplication via a scalar with key size 2048-bit takes {HMultScalar_cost:.6f} seconds"
    )
    decryption_cost = test_decryption_time() / batch
    print(
        f"Single Paillier decryption with key size 2048-bit takes {decryption_cost:.6f} seconds"
    )

    M = 50
    K_range = [10, 25, 50, 75, 100, 125, 150, 175, 200]
    # RPTD results
    fn_cost_list = []
    csp_cost_list = []
    for k in K_range:
        fn_cost, csp_cost = RPTD_iterations(
            encryption_cost, HAdd_cost, HMultScalar_cost, decryption_cost, M, k
        )
        fn_cost_list.append(fn_cost)
        csp_cost_list.append(csp_cost)
    np.savetxt("data/RPTD_Iterations_FN_cost.csv", fn_cost_list, delimiter=",")
    np.savetxt("data/RPTD_Iterations_CSP_cost.csv", csp_cost_list, delimiter=",")

    # L2PPTD results
    sa_iteration_cost_list = []
    sb_iteration_cost_list = []
    sa_data_collection_cost_list = []
    sb_data_collection_cost_list = []
    for k in K_range:
        sa_iteration_cost, sb_iteration_cost = L2PPTD_iterations(
            encryption_cost, HAdd_cost, HMultScalar_cost, decryption_cost, M, k
        )
        sa_iteration_cost_list.append(sa_iteration_cost)
        sb_iteration_cost_list.append(sb_iteration_cost)
        sa_data_collection_cost, sb_data_collection_cost = L2PPTD_data_collection(
            encryption_cost, HAdd_cost, HMultScalar_cost, decryption_cost, M, k
        )
        sa_data_collection_cost_list.append(sa_data_collection_cost)
        sb_data_collection_cost_list.append(sb_data_collection_cost)
    np.savetxt("data/L2PPTD_Iterations_SA_cost.csv", sa_iteration_cost_list, delimiter=",")
    np.savetxt("data/L2PPTD_Iterations_SB_cost.csv", sb_iteration_cost_list, delimiter=",")
    np.savetxt(
        "data/L2PPTD_DataCollection_SA_cost.csv", sa_data_collection_cost_list, delimiter=","
    )
    np.savetxt(
        "data/L2PPTD_DataCollection_SB_cost.csv", sb_data_collection_cost_list, delimiter=","
    )

    # InPPTD results
    sp_data_collection_cost_list = []
    cp_data_collection_cost_list = []

    sp_iteration_cost_list = []
    cp_iteration_cost_list = []

    cp_single_incentive_cost_list = []

    sp_aggregation_incentive_cost_list = []
    cp_aggregation_incentive_cost_list = []

    for k in K_range:
        sp_data_collection_cost, cp_data_collection_cost = InPPTD_data_collection(
            encryption_cost, HAdd_cost, HMultScalar_cost, decryption_cost, M, k
        )
        sp_data_collection_cost_list.append(sp_data_collection_cost)
        cp_data_collection_cost_list.append(cp_data_collection_cost)

        sp_iteration_cost, cp_iteration_cost = InPPTD_iterations(
            encryption_cost, HAdd_cost, HMultScalar_cost, decryption_cost, M, k
        )
        sp_iteration_cost_list.append(sp_iteration_cost)
        cp_iteration_cost_list.append(cp_iteration_cost)

        cp_single_incentive_cost = InPPTD_single_incentive(
            encryption_cost, HAdd_cost, HMultScalar_cost, decryption_cost, M, k
        )
        cp_single_incentive_cost_list.append(cp_single_incentive_cost)

        sp_aggregation_incentive_cost, cp_aggregation_incentive_cost = (
            InPPTD_aggregation_incentive(
                encryption_cost, HAdd_cost, HMultScalar_cost, decryption_cost, M, k
            )
        )
        sp_aggregation_incentive_cost_list.append(sp_aggregation_incentive_cost)
        cp_aggregation_incentive_cost_list.append(cp_aggregation_incentive_cost)

    np.savetxt(
        "data/InPPTD_DataCollection_SP_cost.csv", sp_data_collection_cost_list, delimiter=","
    )
    np.savetxt(
        "data/InPPTD_DataCollection_CP_cost.csv", cp_data_collection_cost_list, delimiter=","
    )
    np.savetxt("data/InPPTD_Iterations_SP_cost.csv", sp_iteration_cost_list, delimiter=",")
    np.savetxt("data/InPPTD_Iterations_CP_cost.csv", cp_iteration_cost_list, delimiter=",")

    np.savetxt(
        "data/InPPTD_Single_Incentive_CP_cost.csv", cp_single_incentive_cost_list, delimiter=","
    )

    np.savetxt(
        "data/InPPTD_Aggregation_Incentive_SP_cost.csv",
        sp_aggregation_incentive_cost_list,
        delimiter=",",
    )
    np.savetxt(
        "data/InPPTD_Aggregation_Incentive_CP_cost.csv",
        cp_aggregation_incentive_cost_list,
        delimiter=",",
    )

    # PETD result: by implementation of Zhao
    # c1_cost_list = []
    # c2_cost_list = []
    # for k in K_range:
    #     c1_cost, c2_cost = PETD_iterations(M, k)
    #     c1_cost_list.append(c1_cost)
    #     c2_cost_list.append(c2_cost)
    # np.savetxt("PETD_Iterations_C1.csv", c1_cost_list, delimiter=",")
    # np.savetxt("PETD_Iterations_C2.csv", c2_cost_list, delimiter=",")

    # PETD result: by complexity analysis
    c1_iteration_cost_list = []
    c2_iteration_cost_list = []

    c1_preparation_cost_list = []
    for k in K_range:
        c1_cost, c2_cost = PETD_iterations_Complexity(
             encryption_cost, HAdd_cost, HMultScalar_cost, decryption_cost, M, k
        )
        c1_iteration_cost_list.append(c1_cost)
        c2_iteration_cost_list.append(c2_cost)

        c1_preparation_cost_list.append(PETD_preparation_Complexity(encryption_cost, HAdd_cost, HMultScalar_cost, decryption_cost, M, k))
    np.savetxt("data/PETD_Iterations_C1_cost_complexity.csv", c1_iteration_cost_list, delimiter=",")
    np.savetxt("data/PETD_Iterations_C2_cost_complexity.csv", c2_iteration_cost_list, delimiter=",")
    np.savetxt("data/PETD_Preparation_C1_cost_complexity.csv", c1_preparation_cost_list, delimiter=",")
