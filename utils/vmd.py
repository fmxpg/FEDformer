import os

import matplotlib.pyplot as plt
import pandas
from vmdpy import VMD


def vmd(root_path='../dataset/illness', data_path='national_illness.csv',
           target='OT', alpha=2000, tau=0, k=8, dc=0, init=1, tol=1e-6):
    """
    调用vmdpy实现时间信号的vmd分解。
    :param root_path: 时间信号文件存储的位置。
    :param data_path: 时间信号文件的名称，需要是csv文件。
    :param target: 需要进行vmd分解的列的名称。默认为"OT"。
    :param alpha: vmd分解带宽限制，经验取值为数据长度的1.5到2倍。默认为2000。
    :param tau: [可选]噪声容限。
    :param k: vmd分解分量的个数，过少会丢失信息，过多会引入噪声。默认为8。
    :param dc: [可选]信号是否包含直流分量。
    :param init: [可选]初始化w值。默认为1，不建议修改。
    :param tol: [可选]控制误差大小常量。
    :return: 返回包含时间和各分量的DataFrame。
    """

    # 读取原始文件并获得时间和目标信息
    if data_path.endswith('.csv'):
        data_path = data_path[:-4]
    df = pandas.read_csv(os.path.join(root_path, data_path + '.csv'))
    df_time, df_data = df['date'], df[target]

    # 绘制目标时间序列图
    # t = range(len(df_data))
    # plt.figure(1)
    # plt.plot(t, df_data, label='data')
    # plt.title('original data')

    # 进行vmd分解，并绘制vmd分解后各分量的时间序列图
    u, u_hat, omega = VMD(df_data.values, alpha, tau, k, dc, init, tol)
    # plt.figure(2)
    # for i in range(k):
    #     plt.subplot(k, 1, i+1)
    #     plt.plot(t, u[i].T, label='u')

    # 将各分量求和，绘制原始时间序列和重构的时间序列对比图
    # df_rebuild = sum(u)
    # plt.figure(3)
    # plt.plot(t, df_data, label='original data')
    # plt.plot(t, df_rebuild, label='rebuilt data')
    # plt.show()

    # 将vmd分解的结果保存到文件
    df_new = pandas.DataFrame(df_time)
    for i in range(k):
        df_new[target + '_' + str(i)] = u[i]
    # print(df_new)
    df_new.to_csv(os.path.join(root_path, data_path + '_' + target + '_vmd.csv'), index=False)

    return df_new


if __name__ == "__main__":
    vmd()
