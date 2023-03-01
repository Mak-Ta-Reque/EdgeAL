import numpy as np
import torch
import random
from torchvision import transforms
from scipy.ndimage import gaussian_filter
import constants

class TransformOCTMaskAdjustment(object):
    """
    Adjust OCT 2015 Mask
    from: classes [0,1,2,3,4,5,6,7,8,9], where 9 is fluid, 0 is empty space above and 8 empty space below
    to: class 0: not class, classes 1-7: are retinal layers, class 8: fluid
    """
    """
    Adjust AROI Mask
    from: classes [0,1,2,3,4,5,6,7], where  0 is empty space above and 4 empty space below
    to: class 0: not class, classes 1-7: are retinal layers,
    """
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        #mask[mask == 8] = 0 #For duke
        #mask[mask == 9] = 8 # For duke
        
        #mask[mask == 4] = 0 #For AROI
        #mask[mask == 7] = 4 # For AROI

        #if torch.max(mask) > 7: raise ValueError("Classes in AROI is higher than 7")
        #if torch.min(mask) < 0 : raise ValueError("Classes in AROI is lower than 7")
        return {'image': img,
                'label': mask}


class TransformStandardization(object):
    """
    Standardizaton / z-score: (x-mean)/std
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img -= self.mean
        img /= self.std

        return {'image': img,
                'label': mask}
    def __repr__(self):
        return self.__class__.__name__ + f": mean {self.mean}, std {self.std}"

class TransformOCTBilinear(object):
    def __init__(self, img_size=(128,128)):
        self.img_size = img_size
        self.trans = transforms.Resize(img_size)
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = self.trans(img)
        mask = self.trans(mask).int().squeeze(0)
    
        return {'image': img,
                'label': mask}
    def __repr__(self):
        return self.__class__.__name__ + f": size {self.img_size}"

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """

    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = img.astype(np.float32)
        mask = mask.astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {'image': img,
                'label': mask}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        mask = sample['label']
        if len(img.shape) == 3:
            img = img.transpose((2, 0, 1))

        #mask = mask.astype(np.float32)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).reshape(1, *mask.shape)
        #mask = torch.Tensor(mask).reshape(1, 1, *mask.shape).int()
        return {'image': img,
                'label': mask}


class RandomGaussianBlur(object):

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = gaussian_filter(img, sigma=random.random())

        return {'image': img,
                'label': mask}


class RandomHorizontalFlip(object):

    def __call__(self, sample):

        img = sample['image']
        mask = sample['label']

        if random.random() < 0.5:
            img = np.fliplr(img)
            mask = np.fliplr(mask)

        return {'image': img,
                'label': mask}


class FixScaleCrop(object):

    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        w, h = img.shape[1], img.shape[0]

        if w > h:
            oh = self.crop_size[0]
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size[1]
            oh = int(1.0 * h * ow / w)

        img = np.array(Image.fromarray(img).resize(ow, oh)) 
        mask = np.array(Image.fromarray(mask).resize(ow, oh), PIL.Image.NEAREST)

        # center crop
        w, h = img.shape[1], img.shape[0]
        x1 = int(round((w - self.crop_size[1]) / 2.))
        y1 = int(round((h - self.crop_size[0]) / 2.))
        img = img[y1: y1 + self.crop_size[0], x1: x1 + self.crop_size[1], :]
        mask = mask[y1: y1 + self.crop_size[0], x1: x1 + self.crop_size[1]]

        return {'image': img,
                'label': mask}


def transform_training_sample(image, target, base_size):
    if constants.IN_CHANNELS == 1:
        mean = (46.3758)
        std = (53.9434)
    elif constants.IN_CHANNELS == 3 :
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
    else:
        raise NotImplemented('Normalizer values are not define')
    composed_transforms = transforms.Compose([
        ToTensor(),
        TransformOCTBilinear(base_size),
        TransformOCTMaskAdjustment(),
        TransformStandardization(mean=mean, std=std),
    ])

    return composed_transforms({'image': image, 'label': target})


def transform_validation_sample(image, target, base_size=None):
    if constants.IN_CHANNELS == 1:
        mean = (46.3758)
        std = (53.9434)
    elif constants.IN_CHANNELS == 3 :
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
    composed_transforms = transforms.Compose([
        ToTensor(),
        TransformOCTBilinear(base_size),
        TransformOCTMaskAdjustment(),
        TransformStandardization(mean=mean, std=std),
        
    ])

    return composed_transforms({'image': image, 'label': target})
