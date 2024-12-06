import os

import numpy as np

from utils.metrics import metric


class Exp_Vmd:
    def __init__(self, exp0, exps):
        self.original_exp = exp0
        self.exps = exps

    def test(self, setting0, settings, vmd_setting):
        _, trues_ori = self.original_exp.test(setting0, load=True, inverse=True)
        preds_ori = []
        for exp, setting in zip(self.exps, settings):
            pred, _ = exp.test(setting, load=True, inverse=True)
            preds_ori.append(pred)
        preds_ori = sum(preds_ori)

        preds_ori = np.array(preds_ori)
        trues_ori = np.array(trues_ori)
        print('test shape:', preds_ori.shape, trues_ori.shape)
        preds = preds_ori.reshape(-1, preds_ori.shape[-2], preds_ori.shape[-1])
        trues = trues_ori.reshape(-1, trues_ori.shape[-2], trues_ori.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + vmd_setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result.txt", 'a')
        f.write(vmd_setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
