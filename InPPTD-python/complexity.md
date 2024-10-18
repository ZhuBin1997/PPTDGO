# RPTD 计算代价
CSP具有密钥对，FN不具备。

Preparation phase:
- 第一步：每个用户$u_k$对每个观测值，共$K$个，生成一个随机数$\alpha_m^k$。并且加密观测值$E(x_m^k)$，随机数$E(\alpha_m^k)$，以及随机数的平方和$E(\sum (\alpha_m^k)^2)$，共$2M+1$次加密。
- 第二步：用户生成MAC，不考虑；
- 第三步：FN验证用户合法性与数据完整性。
- 第四步：FN将所有$E(\alpha_m^k)$的密文传送给CSP。

Iteration phase：
- 第一步：FN操作。对每个用户，需要进行$M$次HMultScalar，$M+1$次HAdd，共$KM$次HMultScalar + $KM$次HAdd。
其次进行$K$次HAdd，并进行一次HMultScalar，再进行$K$次HMultScalar。总计$(KM+2K)$ HAdd + $(KM+K+1)$ HMultScalar。
- 第二步：CSP操作。对每一个用户，进行一次解密，共$K$次解密。然后在进行一次解密。之后在明文上求$\log$操作。最后，对每个用户，进行一次加密保护权重信息，共$K$次加密。
- 第三步：FN操作。首先进行$2M$次加密，对$\beta$进行加密，以及$2KM$次HMultScalar+HAdd。
- 第四步：CSP操作。对每个object，CSP进行两次解密，共计$2M$次解密。


总结：

- FN：$2M$ Enc + $(KM+4K)$ HAdd + $(KM+K+1)$ HMultScalar。
- CSP：$(2M+K)$ Dec + $K$ Enc。

# L $^2$ PPTD计算代价


两个服务器SA与SB各自具有一对密钥对

Initialization阶段：
- 第一步：每个用户为自己每个观测值生成一个$\alpha_m^k$与$\tilde{x}_m^k = x_m^k-\alpha_m^k$。
- 第二步：将$\alpha_m^k$上传给SB，$\tilde{x}_m^k$上传给SA。
- 第三步：SA加密生成$E_A(\tilde{x}_m^k)$，需要$MK$次加密。
- 第四步：SB加密生成$E_B(\alpha_m^k)$，需要$MK$次加密。

Iteration阶段：
- 第一步：SA操作。$MK$ (Enc + HMultScalar) + $K(M-1)$ HAdd。
- 第二步：SB操作。进行$K$次解密。
- 第三步：SB操作。$KM$ HMultScalar + $M(K-1) + M$ HAdd + $M$ Enc。
- 第四步：SA进行$M$次解密。

总结：

- SA：$MK$ (Enc + HMultScalar) + $K(M-1)$ HAdd + $M$ Dec
- SB：$M$ Enc + $KM$ HMultScalar + $M(K-1)$ HAdd + $K$ Dec


# RPTD 通信代价

Preparation阶段：

- 第一步：每个用户对每个object，生成两个密文，$E(x_m^k)$和$E(\alpha_m^k)$，共$M$个密文。进一步生成$E(\sum (\alpha_m^k)^2)$一个密文。无通信。
- 第二步：每个用户对数据进行AES加密，得到$AES(\tilde{x}_m^k), AES(E(x_m^k)),AES(E(\alpha_m^k))$ 以及 $AES(E(\sum (\alpha_m^k)^2)),H,MAC$两组数据。前者是$3M$个AES密文，后者是一个AES密文，一个哈希链和一个MAC。全都发送给FN。FN总共收到$KM$个普通AES密文，$2MK+K$个paillier密文的AES加密密文，$K$个哈希链与MAC。
- 第三步：验证步骤，无通信。
- 第四步：FN将$MK$个paillier密文发送给CSP。

Iteration阶段：

- 第一步：FN操作。向CSP发送$(K+1) S_p$与$(K+1) S_c$。
- 第二步：CSP操作。向FN发送$K(S_p+S_c)$。
- 第三步：FS操作。向CSP发送$2M S_c$。
- 第四步：CSP操作。向FN发送$(K+M)S_p$。
- 第五步：FN操作。计算步骤，无通信。

总结：

- $(3K+M+1)S_p+(2K+2M+1)S_c$。

# L2PPTD 通信代价
Initialization阶段：

- 第一步：用户操作。将扰动后的数据传送给SA，对每个用户是$M S_p$。对SA而言，是$KM S_p$。
- 第二步：用户操作，将随机数传送给SB。对每个用户是$M S_p$。对SB而言，是$KM S_p$。
- 第三步：SA操作，传送$MK S_c$给SB。
- 第四步：SB操作，传送$MK S_c$给SB。

Iteration阶段：

- 第一步：SA操作，发送$K S_c$给SB。
- 第二步：SB操作，计算步骤，无通信。
- 第三步：SB操作，发送$(M+1) S_c$。
- 第四步：SA操作，计算步骤，无通信。

总结：

- $(K+M+1)S_c$


# PETD计算代价
密钥：两对密钥


准备阶段：
仅C1需要需要初始化真值，并对其进行加密，$M$ Enc。


sub-protocol：SDH
- 第一步：C1需要$M$(Enc+HMult+HAdd+PD1)，传送$2MS_c$。
- 第二步：C2需要$M$(PD2+2Enc)，传送$2MS_c$。
- 第三步：C1需要$M$(2Enc+3HMult+3HAdd)。

Iteration阶段：
- weight estimation阶段：
- - 第一步：C1与C2运行$K$次SDH，生成$K$个密文，由C1持有，代价为$K$ SDH；
- - 第二步：C1操作。$K$ HMult，$K$ PD1，$K-1$HAdd，$K$ HMult，$K$ PD1。传送$4K S_c$给C2；
- - 第三步：C2操作。$K$ PD2，$K$ PD2，$K$ 明文$\log$，$K$ Enc。传送$K S_c$给C1。
- - 第四步：C1操作。$K$ Enc，$K$ HMult，$K$ HAdd。无通信

- truth estimation阶段：
- - 第一步：C1操作。$MK$ HMult，$M(K-1)$ HAdd。传送$K S_c$给C2。
- - 第二步：C2操作。$MK$ HMult，$M(K-1)$ HAdd。传送$M S_c$给C1。
- - 第三步：C1操作。$K-1$ HAdd，$M$ HAdd，$M$ (2 HMult)，$M$ (2PD1)，传送$(3M+1)S_c$给C2。
- - 第四步：C2操作。$2M$ PD2，$M$ Enc。传送$M S_c$给C1。
- - 第五步：C1操作。$M$ HMult。无通信。

总结：
- 计算开销：SDH
- - C1: $3M$ Enc + $4M$ HMult + $4M$ Add + $M$ PD1 
- - C2: $2M$ Enc + $M$ PD2
- 计算开销：迭代
- - C1: $K$ Enc + $MK+3(M+K)$ HMult + $MK+3K-2$ HAdd + $2K+2M$ PD1 + $K$ SDH
- - C2: $K+M$ Enc + $MK$ HMult + $M(K-1)$ HAdd + $2K+2M$ PD2 + $K$ SDH
- 通信开销：
- - SDH：$4M S_c$
- - 迭代：$(6K+5M+1) S_c$ + $K$ SDH
