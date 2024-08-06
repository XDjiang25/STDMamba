import random

from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, CorrVisual, AvgVisual, ResVisual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'DLinear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        elif 'Mamba' in self.args.model:
                            outputs,outTrendSeason = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            f_dim = -1 if self.args.features == 'MS' else 0
                            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                            weightTrend = torch.tanh(batch_y)**2
                            weightSeason = (1 - torch.abs(torch.tanh(batch_y))) **2
                            loss1 = 0#criterion(torch.mul(outTrend,weightTrend), torch.mul(batch_y,weightTrend))
                            loss2 = 0#criterion(torch.mul(outSeason,weightSeason), torch.mul(batch_y,weightSeason)) 
                        elif 'TCN' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                                loss1 = 0
                                loss2 = 0
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark) 
                                loss1 = 0
                                loss2 = 0       
                else:
                    if 'DLinear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    elif 'Mamba' in self.args.model:
                        outputs,outTrendandSeason = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        f_dim = -1 if self.args.features == 'MS' else 0
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        weightTrend = torch.tanh(batch_y)**2
                        weightSeason = (1 - torch.abs(torch.tanh(batch_y))) **2
                        loss1 = 0 #criterion(torch.mul(outTrend,weightTrend), torch.mul(batch_y,weightTrend))
                        loss2 = 0 #criterion(torch.mul(outSeason,weightSeason), torch.mul(batch_y,weightSeason))
                    elif 'TCN' in self.args.model:
                        outputs = self.model(batch_x)
                        loss1 = 0
                        loss2 = 0
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            loss1 = 0
                            loss2 = 0
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            loss1 = 0
                            loss2 = 0                                                                      
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'DLinear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        elif 'Mamba' in self.args.model:
                            
                            outputs,main0,enc_out = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        #print("output",outputs.size())
                        #print("outTrend",outTrend.size())
                        #print("outSeason",outSeason.size())
                            f_dim = -1 if self.args.features == 'MS' else 0
                            batch_x = batch_x[:, -self.args.d_model:, f_dim:].to(self.device)
                            #weightTrend = torch.tanh(batch_y)**2
                            #weightSeason = (1 - torch.abs(torch.tanh(batch_y)))**2
                        #print("weightTrend",weightTrend.size())
                        #print("weightSeason",weightSeason.size())

                            loss1 = 0 #criterion(torch.mul(outTrend,weightSeason), torch.mul(batch_y,weightSeason))
                            loss2 = criterion(main0+enc_out,batch_x)                            
                        elif 'TCN' in self.args.model:
                            outputs = self.model(batch_x)
                            loss1 = 0
                            loss2 = 0
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                                loss1 = 0
                                loss2 = 0
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                                loss1 = 0
                                loss2 = 0
                           
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss0 = criterion(outputs, batch_y) ## MSE(y,yhat)
                    
                        l_w = max(-1 * np.exp(epoch/90) + 1, 0.2)
                        loss =  l_w * (loss1+loss2) + loss0
                        
                        
                        train_loss.append(loss.item())
                else:
                    if 'DLinear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                        loss1 = 0
                        loss2 = 0
                    elif 'Mamba' in self.args.model:
                        outputs,outTrendandSeason = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        #print("output",outputs.size())
                        #print("outTrend和outSean加起来",outTrendandSeason.size())
                        #print("outSeason",outSeason.size())
                        f_dim = -1 if self.args.features == 'MS' else 0
                        #print("batch_y",batch_y.size())
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        #print("batch_y",batch_y.size())
                        weightTrend = torch.tanh(batch_y)**2
                        weightSeason = (1 - torch.abs(torch.tanh(batch_y))) **2
                        #print("weightTrend",weightTrend.size())
                        #print("weightSeason",weightSeason.size())
                        

                        loss1 = 0
                        loss2 = 0 #criterion(outTrendandSeason,batch_y)
                    elif 'TCN' in self.args.model:
                        outputs = self.model(batch_x)
                        loss1 = 0
                        loss2 = 0    
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            loss1 = 0
                            loss2 = 0
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            loss1 = 0
                            loss2 = 0


                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss0 = criterion(outputs, batch_y) ## MSE(y,yhat)
                    
                    l_w = max(-1 * np.exp(epoch/45) + 2, 0.15)
                    loss =  l_w * (loss1+loss2) + loss0
                    
                    
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:                   
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    #total_params = sum(p.numel() for p in self.model.parameters())
                    #print("Number of parameter: %.2fM" % (total_params/1e6))
                    # print(speed)
                    # allocated_memory = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)
                    # cached_memory = torch.cuda.memory_cached() / (1024 * 1024 * 1024)
                    # total = allocated_memory + cached_memory
                    # print('allocated_memory:', allocated_memory)
                    # print('cached_memory:', cached_memory)
                    # print('total:', total)
                    #print()
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

            # get_cka(self.args, setting, self.model, train_loader, self.device, epoch)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                # time_points = random.sample(range(batch_x.size()[1]), 5)
                # 假设您有一个包含每个变量标准差的张量stds，形状为(321,)
                # 定义扰动强度
                # epsilon = 1
                # # 创建一个与原始张量形状相同的张量来存储扰动
                # perturbed_tensor = torch.zeros_like(batch_x)
                # # 对每个选定的时间点添加扰动
                # for time_point in time_points:
                #     # 生成与tensor在该时间点形状相同的随机噪声
                #     noise = torch.randn(1, 321) * epsilon
                #     perturbed_tensor[:, time_point, :] += noise.float().to(self.device)
                # batch_x += perturbed_tensor
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'DLinear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        elif 'Mamba' in self.args.model:
                            outputs,outTrendandoutSeason = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        elif 'TCN' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)                
                else:
                    if 'DLinear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    elif 'Mamba' in self.args.model:
                        outputs,outTrendoutSeason = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    elif 'TCN' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)                     
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    #gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    #pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    gt = true[0, : ,-1]
                    pd = pred[0, : ,-1]
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        
        p = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        t = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        
        p = p[0, : ,-1]
        t = t[0, : ,-1]
        avgvalue = np.mean(p)
        
        
        expanded_average = np.full_like(t,avgvalue)
        
        AvgVisual(expanded_average, p, os.path.join(folder_path,'avg.pdf'))
        ResVisual(expanded_average, p, os.path.join(folder_path,'res.pdf'))        
        # gt = trues[0, :, -1]
        # pd = preds[0, :, -1]
        
        # CorrVisual(gt,pd,os.path.join(folder_path,'Cor.pdf'))
        
        
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('rmse:{}, mae:{}, mape:{}'.format(rmse, mae, mape))
        f = open("result_SST_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('rmse:{}, mae:{}, mape:{}'.format(rmse, mae, mape))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
    def get_input(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        inputs = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            input = batch_x.detach().cpu().numpy()
            inputs.append((input))
        folder_path = './results/' + setting + '/'
        np.save(folder_path + 'input.npy', inputs)

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs.detach().cpu().numpy()
                if pred_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = pred_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                preds.append(outputs)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return