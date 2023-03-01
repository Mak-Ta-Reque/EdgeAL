import numpy as np
import torch
# https://stats.stackexchange.com/questions/179835/how-to-build-a-confusion-matrix-for-a-multiclass-classifier

def calculate_miou(confusion_matrix):
    MIoU = np.divide(np.diag(confusion_matrix), (
        np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
        np.diag(confusion_matrix)))
    MIoU = np.nanmean(MIoU)
    return MIoU

def per_class_dice(y_pred, y_true, num_class):
   
    avg_dice = 0
    y_pred = y_pred.data.squeeze() #.cpu().numpy()
    y_true = y_true.data.squeeze() #.cpu().numpy()
    dice_all = np.zeros(num_class)
    for i in range(num_class):
        GT = y_true[:,:,i].view(-1)
        Pred = y_pred[:,:,i].view(-1)
        inter = (GT * Pred).sum() + 0.0001
        union = GT.sum()  + Pred.sum()  + 0.0001
        t = 2 * inter / union
        if t == 2.0:
            t = t-1.0
        avg_dice = avg_dice + (t / num_class)
        dice_all[i] = t
    return avg_dice, dice_all

class Evaluator(object):

    def __init__(self, num_class):
        np.seterr(divide='ignore', invalid='ignore')
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.iou = []
        self.ind_iou = []

    def Pixel_Accuracy(self):
        return np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()

    def Pixel_Accuracy_Class(self):
        Acc = np.divide(np.diag(self.confusion_matrix), self.confusion_matrix.sum(axis=1))
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.divide(np.diag(self.confusion_matrix), (
            np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
            np.diag(self.confusion_matrix)))
        MIoU = np.nanmean(MIoU)
        return MIoU
    def Intersection_over_Union(self):
        ind_iou = np.array(self.ind_iou)
        iou = np.array(self.iou)
        return np.nanmean(ind_iou, axis=0), np.nanmean(iou)

    def Mean_Intersection_over_Union_20(self):
        MIoU = 0
        if self.num_class > 20:
            subset_20 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 23, 27, 32, 33, 35, 38])
            confusion_matrix = self.confusion_matrix[subset_20[:, None], subset_20]
            MIoU = np.divide(np.diag(confusion_matrix), (
                np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
                np.diag(confusion_matrix)))
            MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.divide(np.diag(self.confusion_matrix), (
            np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
            np.diag(self.confusion_matrix)))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def _generate_all_dice(self, gt_image, pre_image):
       
        
        max_val, idx = torch.max(pre_image, 1)
        #print(gt_image.shape)
        idx =idx.cpu()
       
        for gt, pred in zip(gt_image, idx):
            gt = torch.from_numpy(gt).type(torch.int64)
            gt[gt==255] = 0
            #print(torch.max(pred))
            #print(torch.min(pred))
            label_oh = torch.nn.functional.one_hot(gt, num_classes=self.num_class)
            pred_oh = torch.nn.functional.one_hot(pred, num_classes=self.num_class)
            avgiou, iou = per_class_dice(pred_oh, label_oh, self.num_class)
            if np.min(iou) < 0.0: raise ValueError(f"Dice scire is negative for sample {np.argmin(iou)}")
            if np.max(iou) == 2.0 : raise ValueError(f"Dice scire is greter than one for sample {np.argmax(iou)}")

            self.ind_iou.append(iou.tolist())
            self.iou.append(avgiou.tolist())


    def add_batch(self, gt_image, pre_image, max_pred = None, return_miou=False):
        assert gt_image.shape == pre_image.shape
        confusion_matrix = self._generate_matrix(gt_image, pre_image)
        self.confusion_matrix += confusion_matrix
        self._generate_all_dice(gt_image,  max_pred)
        if return_miou:
            return calculate_miou(confusion_matrix)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.ind_iou = []
        self.iou = []

    def dump_matrix(self, path):
        np.save(path, self.confusion_matrix)
