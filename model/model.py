# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

# model
from model.change_detction_dataset import ChangeDetectionDataset

# Other
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm as tqdm
from IPython import display
from math import ceil

L = 1024
N = 2


class ModelConfig(object):
    def __init__(
        self,
        n_epochs=50,
        gpu_enabled=False,
    ):
        self.n_epochs = n_epochs
        self.gpu_enabled = gpu_enabled


class Model:
    def __init__(
        self,
        model,
        config: ModelConfig,
        train_dataset: ChangeDetectionDataset,
        train_loader: DataLoader,
        criterion: nn.NLLLoss,
        model_name,
    ):
        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.train_loader = train_loader
        self.criterion = criterion
        self.model_name = model_name

    def train(self, save=True):
        t = np.linspace(1, self.config.n_epochs, self.config.n_epochs)

        epoch_train_loss = 0 * t
        epoch_train_accuracy = 0 * t
        epoch_train_change_accuracy = 0 * t
        epoch_train_nochange_accuracy = 0 * t
        epoch_train_precision = 0 * t
        epoch_train_recall = 0 * t
        epoch_train_Fmeasure = 0 * t
        epoch_test_loss = 0 * t
        epoch_test_accuracy = 0 * t
        epoch_test_change_accuracy = 0 * t
        epoch_test_nochange_accuracy = 0 * t
        epoch_test_precision = 0 * t
        epoch_test_recall = 0 * t
        epoch_test_Fmeasure = 0 * t

        fm = 0
        best_fm = 0

        lss = 1000
        best_lss = 1000

        plt.figure(num=1)
        plt.figure(num=2)
        plt.figure(num=3)

        optimizer = torch.optim.Adam(self.model.parameters(), weight_decay=1e-4)

        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
        
        for epoch_index in tqdm(range(self.config.n_epochs),position=0):
            self.model.train()
            train_loader_len = len(self.train_loader)
            count = 1
            print("Epoch: " + str(epoch_index + 1) + " of " + str(self.config.n_epochs))
            for batch in tqdm(self.train_loader,position=1, leave=False):
                print(f"Training: {count}/ {train_loader_len}")
                I1 = Variable(batch["I1"].float())
                I2 = Variable(batch["I2"].float())
                label = torch.squeeze(Variable(batch["label"]))

                if self.config.gpu_enabled:
                    I1 = I1.cuda()
                    I2 = I2.cuda()
                    label = label.cuda()

                optimizer.zero_grad()
                output = self.model(I1, I2)
                loss = self.criterion(output, label.long())
                loss.backward()
                optimizer.step()
                count = count + 1

            scheduler.step()

            (
                epoch_train_loss[epoch_index],
                epoch_train_accuracy[epoch_index],
                cl_acc,
                pr_rec,
            ) = self.test(self.train_dataset)
            epoch_train_nochange_accuracy[epoch_index] = cl_acc[0]
            epoch_train_change_accuracy[epoch_index] = cl_acc[1]
            epoch_train_precision[epoch_index] = pr_rec[0]
            epoch_train_recall[epoch_index] = pr_rec[1]
            epoch_train_Fmeasure[epoch_index] = pr_rec[2]
            (
                epoch_test_loss[epoch_index],
                epoch_test_accuracy[epoch_index],
                cl_acc,
                pr_rec,
            ) = self.test(self.train_dataset)
            epoch_test_nochange_accuracy[epoch_index] = cl_acc[0]
            epoch_test_change_accuracy[epoch_index] = cl_acc[1]
            epoch_test_precision[epoch_index] = pr_rec[0]
            epoch_test_recall[epoch_index] = pr_rec[1]
            epoch_test_Fmeasure[epoch_index] = pr_rec[2]

            plt.figure(num=1)
            plt.clf()
            (l1_1,) = plt.plot(
                t[: epoch_index + 1],
                epoch_train_loss[: epoch_index + 1],
                label="Train loss",
            )
            (l1_2,) = plt.plot(
                t[: epoch_index + 1],
                epoch_test_loss[: epoch_index + 1],
                label="Test loss",
            )
            plt.legend(handles=[l1_1, l1_2])
            plt.grid()
            #         plt.gcf().gca().set_ylim(bottom = 0)
            plt.gcf().gca().set_xlim(left=0)
            plt.title("Loss")
            display.clear_output(wait=True)
            display.display(plt.gcf())

            plt.figure(num=2)
            plt.clf()
            (l2_1,) = plt.plot(
                t[: epoch_index + 1],
                epoch_train_accuracy[: epoch_index + 1],
                label="Train accuracy",
            )
            (l2_2,) = plt.plot(
                t[: epoch_index + 1],
                epoch_test_accuracy[: epoch_index + 1],
                label="Test accuracy",
            )
            plt.legend(handles=[l2_1, l2_2])
            plt.grid()
            plt.gcf().gca().set_ylim(0, 100)
            #         plt.gcf().gca().set_ylim(bottom = 0)
            #         plt.gcf().gca().set_xlim(left = 0)
            plt.title("Accuracy")
            display.clear_output(wait=True)
            display.display(plt.gcf())

            plt.figure(num=3)
            plt.clf()
            (l3_1,) = plt.plot(
                t[: epoch_index + 1],
                epoch_train_nochange_accuracy[: epoch_index + 1],
                label="Train accuracy: no change",
            )
            (l3_2,) = plt.plot(
                t[: epoch_index + 1],
                epoch_train_change_accuracy[: epoch_index + 1],
                label="Train accuracy: change",
            )
            (l3_3,) = plt.plot(
                t[: epoch_index + 1],
                epoch_test_nochange_accuracy[: epoch_index + 1],
                label="Test accuracy: no change",
            )
            (l3_4,) = plt.plot(
                t[: epoch_index + 1],
                epoch_test_change_accuracy[: epoch_index + 1],
                label="Test accuracy: change",
            )
            plt.legend(handles=[l3_1, l3_2, l3_3, l3_4])
            plt.grid()
            plt.gcf().gca().set_ylim(0, 100)
            #         plt.gcf().gca().set_ylim(bottom = 0)
            #         plt.gcf().gca().set_xlim(left = 0)
            plt.title("Accuracy per class")
            display.clear_output(wait=True)
            display.display(plt.gcf())

            plt.figure(num=4)
            plt.clf()
            (l4_1,) = plt.plot(
                t[: epoch_index + 1],
                epoch_train_precision[: epoch_index + 1],
                label="Train precision",
            )
            (l4_2,) = plt.plot(
                t[: epoch_index + 1],
                epoch_train_recall[: epoch_index + 1],
                label="Train recall",
            )
            (l4_3,) = plt.plot(
                t[: epoch_index + 1],
                epoch_train_Fmeasure[: epoch_index + 1],
                label="Train Dice/F1",
            )
            (l4_4,) = plt.plot(
                t[: epoch_index + 1],
                epoch_test_precision[: epoch_index + 1],
                label="Test precision",
            )
            (l4_5,) = plt.plot(
                t[: epoch_index + 1],
                epoch_test_recall[: epoch_index + 1],
                label="Test recall",
            )
            (l4_6,) = plt.plot(
                t[: epoch_index + 1],
                epoch_test_Fmeasure[: epoch_index + 1],
                label="Test Dice/F1",
            )
            plt.legend(handles=[l4_1, l4_2, l4_3, l4_4, l4_5, l4_6])
            plt.grid()
            plt.gcf().gca().set_ylim(0, 1)
            plt.title("Precision, Recall and F-measure")
            display.clear_output(wait=True)
            display.display(plt.gcf())
            fm = epoch_train_Fmeasure[epoch_index]
            if fm > best_fm:
                best_fm = fm
                save_str = (
                    "net-best_epoch-"
                    + str(epoch_index + 1)
                    + "_fm-"
                    + str(fm)
                    + ".pth.tar"
                )
                torch.save(self.model.state_dict(), save_str)

            lss = epoch_train_loss[epoch_index]
            if lss < best_lss:
                best_lss = lss
                save_str = (
                    "net-best_epoch-"
                    + str(epoch_index + 1)
                    + "_loss-"
                    + str(lss)
                    + ".pth.tar"
                )
                torch.save(self.model.state_dict(), save_str)

            if save:
                im_format = "png"

                plt.figure(num=1)
                plt.savefig(self.model_name + "-01-loss." + im_format)

                plt.figure(num=2)
                plt.savefig(self.model_name + "-02-accuracy." + im_format)

                plt.figure(num=3)
                plt.savefig(self.model_name + "-03-accuracy-per-class." + im_format)

                plt.figure(num=4)
                plt.savefig(self.model_name + "-04-prec-rec-fmeas." + im_format)

        out = {
            "train_loss": epoch_train_loss[-1],
            "train_accuracy": epoch_train_accuracy[-1],
            "train_nochange_accuracy": epoch_train_nochange_accuracy[-1],
            "train_change_accuracy": epoch_train_change_accuracy[-1],
            "test_loss": epoch_test_loss[-1],
            "test_accuracy": epoch_test_accuracy[-1],
            "test_nochange_accuracy": epoch_test_nochange_accuracy[-1],
            "test_change_accuracy": epoch_test_change_accuracy[-1],
        }

        print("pr_c, rec_c, f_meas, pr_nc, rec_nc")
        print(pr_rec)

        return out

    def test(self, dset):
        self.model.eval()
        tot_loss = 0
        tot_count = 0

        n = 2
        class_correct = list(0.0 for i in range(n))
        class_total = list(0.0 for i in range(n))
        class_accuracy = list(0.0 for i in range(n))

        tp = 0
        tn = 0
        fp = 0
        fn = 0

        for idx, img_index in dset.names.iterrows():
            if idx == 0:
                continue
            img_index = img_index[0]
            I1_full, I2_full, cm_full = dset.get_img(img_index)

            s = cm_full.shape

            steps0 = np.arange(0, s[0], ceil(s[0] / N))
            steps1 = np.arange(0, s[1], ceil(s[1] / N))
            for ii in range(N):
                for jj in range(N):
                    xmin = steps0[ii]
                    if ii == N - 1:
                        xmax = s[0]
                    else:
                        xmax = steps0[ii + 1]
                        ymin = jj
                    if jj == N - 1:
                        ymax = s[1]
                    else:
                        ymax = steps1[jj + 1]
                    I1 = I1_full[:, xmin:xmax, ymin:ymax]
                    I2 = I2_full[:, xmin:xmax, ymin:ymax]
                    cm = cm_full[xmin:xmax, ymin:ymax]
                    I1 = Variable(torch.unsqueeze(I1, 0).float())
                    I2 = Variable(torch.unsqueeze(I2, 0).float())
                    cm = Variable(
                        torch.unsqueeze(torch.from_numpy(1.0 * cm), 0).float()
                    )

                    if self.config.gpu_enabled:
                        I1 = I1.cuda()
                        I2 = I2.cuda()
                        cm = cm.cuda()

                    output = self.model(I1, I2)
                    loss = self.criterion(output, cm.long())
                    #         print(loss)
                    tot_loss += loss.data * np.prod(cm.size())
                    tot_count += np.prod(cm.size())

                    _, predicted = torch.max(output.data, 1)

                    c = predicted.int() == cm.data.int()
                    for i in range(c.size(1)):
                        for j in range(c.size(2)):
                            l = int(cm.data[0, i, j])
                            class_correct[l] += c[0, i, j]
                            class_total[l] += 1

                    pr = (predicted.int() > 0).cpu().numpy()
                    gt = (cm.data.int() > 0).cpu().numpy()

                    tp += np.logical_and(pr, gt).sum()
                    tn += np.logical_and(np.logical_not(pr), np.logical_not(gt)).sum()
                    fp += np.logical_and(pr, np.logical_not(gt)).sum()
                    fn += np.logical_and(np.logical_not(pr), gt).sum()
        net_loss = tot_loss / tot_count
        net_accuracy = 100 * (tp + tn) / tot_count

        for i in range(n):
            class_accuracy[i] = 100 * class_correct[i] / max(class_total[i], 0.00001)

        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f_meas = 2 * prec * rec / (prec + rec)
        prec_nc = tn / (tn + fn)
        rec_nc = tn / (tn + fp)

        pr_rec = [prec, rec, f_meas, prec_nc, rec_nc]

        return net_loss, net_accuracy, class_accuracy, pr_rec

    def kappa(tp, tn, fp, fn):
        N = tp + tn + fp + fn
        p0 = (tp + tn) / N
        pe = ((tp+fp)*(tp+fn) + (tn+fp)*(tn+fn)) / (N * N)
        
        return (p0 - pe) / (1 - pe)

    def evaluate(self,dset):
        self.model.eval()
        tot_loss = 0
        tot_count = 0

        n = 2
        class_correct = list(0. for i in range(n))
        class_total = list(0. for i in range(n))
        class_accuracy = list(0. for i in range(n))
        
        tp = 0
        tn = 0
        fp = 0
        fn = 0
    
        for idx, img_index in tqdm(dset.names.iterrows()):
                img_index = img_index[0]
                I1_full, I2_full, cm_full = dset.get_img(img_index)
                
                s = cm_full.shape
        
                for ii in range(ceil(s[0]/L)):
                        for jj in range(ceil(s[1]/L)):
                                xmin = L*ii
                                xmax = min(L*(ii+1),s[1])
                                ymin = L*jj
                                ymax = min(L*(jj+1),s[1])
                                I1 = I1_full[:, xmin:xmax, ymin:ymax]
                                I2 = I2_full[:, xmin:xmax, ymin:ymax]
                                cm = cm_full[xmin:xmax, ymin:ymax]
                                I1 = Variable(torch.unsqueeze(I1, 0).float())
                                I2 = Variable(torch.unsqueeze(I2, 0).float())
                                cm = Variable(torch.unsqueeze(torch.from_numpy(1.0*cm),0).float())

                                if self.config.gpu_enabled:
                                        I1 = I1.cuda()
                                        I2 = I2.cuda()
                                        cm = cm.cuda()

                                output = self.model(I1, I2)
                                
                                loss = self.criterion(output, cm.long())
                                tot_loss += loss.data * np.prod(cm.size())
                                tot_count += np.prod(cm.size())

                                _, predicted = torch.max(output.data, 1)

                                c = (predicted.int() == cm.data.int())
                                for i in range(c.size(1)):
                                        for j in range(c.size(2)):
                                                l = int(cm.data[0, i, j])
                                                class_correct[l] += c[0, i, j]
                                                class_total[l] += 1
                        
                                pr = (predicted.int() > 0).cpu().numpy()
                                gt = (cm.data.int() > 0).cpu().numpy()
                                
                                tp += np.logical_and(pr, gt).sum()
                                tn += np.logical_and(np.logical_not(pr), np.logical_not(gt)).sum()
                                fp += np.logical_and(pr, np.logical_not(gt)).sum()
                                fn += np.logical_and(np.logical_not(pr), gt).sum()
        
        net_loss = tot_loss/tot_count        
        net_loss = float(net_loss.cpu().numpy())
    
        net_accuracy = 100 * (tp + tn)/tot_count
    
        for i in range(n):
                class_accuracy[i] = 100 * class_correct[i] / max(class_total[i],0.00001)
                class_accuracy[i] =  float(class_accuracy[i].cpu().numpy())

        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        dice = 2 * prec * rec / (prec + rec)
        prec_nc = tn / (tn + fn)
        rec_nc = tn / (tn + fp)
        
        # pr_rec = [prec, rec, dice, prec_nc, rec_nc]
    
        k = self.kappa(tp, tn, fp, fn)
    
        return {'net_loss': net_loss, 
                'net_accuracy': net_accuracy, 
                'class_accuracy': class_accuracy, 
                'precision': prec, 
                'recall': rec, 
                'dice': dice, 
                'kappa': k}