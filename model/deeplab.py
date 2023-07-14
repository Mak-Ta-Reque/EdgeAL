import torch
import torch.nn as nn
import torch.nn.functional as F
from model.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from model.aspp import build_aspp
from model.decoder import build_decoder
from model.backbone import build_backbone
from collections import OrderedDict
from model.ffc import FFC_BN_ACT, ConcatTupleLayer


class DeepLab(nn.Module):

    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=False, freeze_bn=False, mc_dropout=False):
        print(backbone, output_stride, num_classes, sync_bn, freeze_bn, mc_dropout )
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm, mc_dropout=mc_dropout)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)
        self.return_features = False

        if freeze_bn:
            self.freeze_bn()

    def set_return_features(self, return_features):
        self.return_features = return_features

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x, features = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        if self.return_features:
            return x, features
        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

class UNet(nn.Module):

    def __init__(self, in_channels=3, num_classes=1, init_features=32, useNonLin=True):
        super(UNet, self).__init__()
        # Modfying model for badge selection methon active learning
        self.embSize = 128
        
        self.useNonLin = useNonLin
        # OLd code
 
        out_channels = num_classes
        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.lm1 = UNet._block(features * 8, self.embSize,  name="emb")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        
        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        
        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )
        self.softmax = nn.Softmax2d()

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))
        

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        if self.useNonLin: emb = F.relu(self.lm1(enc4))
        else: emb = self.lm1(enc4)
        return self.softmax(self.conv(dec1))#, emb
    def get_embedding_dim(self):
        return self.embSize
    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


class YNet_general(nn.Module):

    def __init__(self, in_channels=3, num_classes=1, init_features=32, ratio_in=0.5, ffc=True, skip_ffc=False,
                 cat_merge=True):
        super(YNet_general, self).__init__()
        out_channels = num_classes
        self.ffc = ffc
        self.skip_ffc = skip_ffc
        self.ratio_in = ratio_in
        self.cat_merge = cat_merge

        features = init_features
        ############### Regular ##################################
        self.encoder1 = YNet_general._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = YNet_general._block(features, features * 2, name="enc2")  # was 1,2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = YNet_general._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = YNet_general._block(features * 4, features * 4, name="enc4")  # was 8
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.return_features = False

        if ffc:
            ################ FFC #######################################
            self.encoder1_f = FFC_BN_ACT(in_channels, features, kernel_size=1, ratio_gin=0, ratio_gout=ratio_in)
            self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_f = FFC_BN_ACT(features, features * 2, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)  # was 1,2
            self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_f = FFC_BN_ACT(features * 2, features * 4, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)
            self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_f = FFC_BN_ACT(features * 4, features * 4, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)  # was 8
            self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)

        else:
            ############### Regular ##################################
            self.encoder1_f = YNet_general._block(in_channels, features, name="enc1_2")
            self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_f = YNet_general._block(features, features * 2, name="enc2_2")  # was 1,2
            self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_f = YNet_general._block(features * 2, features * 4, name="enc3_2")  #
            self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_f = YNet_general._block(features * 4, features * 4, name="enc4_2")  # was 8
            self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = YNet_general._block(features * 8, features * 16, name="bottleneck")  # 8, 16

        if skip_ffc and not ffc:
            self.upconv4 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_general._block((features * 8) * 2, features * 8, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_general._block((features * 6) * 2, features * 4, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 4, features * 2, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_general._block((features * 3) * 2, features * 2, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_general._block(features * 3, features, name="dec1")  # 2,3

        elif skip_ffc and ffc:
            self.upconv4 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_general._block((features * 8) * 2, features * 8, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_general._block((features * 6) * 2, features * 4, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 4, features * 2, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_general._block((features * 3) * 2, features * 2, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_general._block(features * 3, features, name="dec1")  # 2,3

        else:
            self.upconv4 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_general._block((features * 6) * 2, features * 8, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_general._block((features * 4) * 2, features * 4, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 4, features * 2, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_general._block((features * 2) * 2, features * 2, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_general._block(features * 2, features, name="dec1")  # 2,3

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )
        self.softmax = nn.Softmax2d()
        self.catLayer = ConcatTupleLayer()
    def set_return_features(self, return_features):
        self.return_features = return_features
    def apply_fft(self, inp, batch):
        ffted = torch.fft.fftn(inp)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)

        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        return ffted

    def forward(self, x):
        batch = x.shape[0]
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))

        enc3 = self.encoder3(self.pool2(enc2))

        enc4 = self.encoder4(self.pool3(enc3))
        enc4_2 = self.pool4(enc4)

        if self.ffc:
            enc1_f = self.encoder1_f(x)
            enc1_l, enc1_g = enc1_f
            if self.ratio_in == 0:
                enc2_f = self.encoder2_f((self.pool1_f(enc1_l), enc1_g))
            elif self.ratio_in == 1:
                enc2_f = self.encoder2_f((enc1_l, self.pool1_f(enc1_g)))
            else:

                enc2_f = self.encoder2_f((self.pool1_f(enc1_l), self.pool1_f(enc1_g)))

            enc2_l, enc2_g = enc2_f
            if self.ratio_in == 0:
                enc3_f = self.encoder3_f((self.pool2_f(enc2_l), enc2_g))
            elif self.ratio_in == 1:
                enc3_f = self.encoder3_f((enc2_l, self.pool2_f(enc2_g)))
            else:
                enc3_f = self.encoder3_f((self.pool2_f(enc2_l), self.pool2_f(enc2_g)))

            enc3_l, enc3_g = enc3_f
            if self.ratio_in == 0:
                enc4_f = self.encoder4_f((self.pool3_f(enc3_l), enc3_g))
            elif self.ratio_in == 1:
                enc4_f = self.encoder4_f((enc3_l, self.pool3_f(enc3_g)))
            else:
                enc4_f = self.encoder4_f((self.pool3_f(enc3_l), self.pool3_f(enc3_g)))

            enc4_l, enc4_g = enc4_f
            if self.ratio_in == 0:
                enc4_f2 = self.pool1_f(enc4_l)
            elif self.ratio_in == 1:
                enc4_f2 = self.pool1_f(enc4_g)
            else:
                enc4_f2 = self.catLayer((self.pool4_f(enc4_l), self.pool4_f(enc4_g)))

        else:
            enc1_f = self.encoder1_f(x)
            enc2_f = self.encoder2_f(self.pool1_f(enc1_f))
            enc3_f = self.encoder3_f(self.pool2_f(enc2_f))
            enc4_f = self.encoder4_f(self.pool3_f(enc3_f))
            enc4_f2 = self.pool4(enc4_f)

        if self.cat_merge:
            a = torch.zeros_like(enc4_2)
            b = torch.zeros_like(enc4_f2)

            enc4_2 = enc4_2.view(torch.numel(enc4_2), 1)
            enc4_f2 = enc4_f2.view(torch.numel(enc4_f2), 1)

            bottleneck = torch.cat((enc4_2, enc4_f2), 1)
            bottleneck = bottleneck.view_as(torch.cat((a, b), 1))

        else:
            bottleneck = torch.cat((enc4_2, enc4_f2), 1)

        bottleneck = self.bottleneck(bottleneck)

        dec4 = self.upconv4(bottleneck)

        if self.ffc and self.skip_ffc:
            enc4_in = torch.cat((enc4, self.catLayer((enc4_f[0], enc4_f[1]))), dim=1)

            dec4 = torch.cat((dec4, enc4_in), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)

            enc3_in = torch.cat((enc3, self.catLayer((enc3_f[0], enc3_f[1]))), dim=1)
            dec3 = torch.cat((dec3, enc3_in), dim=1)
            dec3 = self.decoder3(dec3)

            dec2 = self.upconv2(dec3)
            enc2_in = torch.cat((enc2, self.catLayer((enc2_f[0], enc2_f[1]))), dim=1)
            dec2 = torch.cat((dec2, enc2_in), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            enc1_in = torch.cat((enc1, self.catLayer((enc1_f[0], enc1_f[1]))), dim=1)
            dec1 = torch.cat((dec1, enc1_in), dim=1)

        elif self.skip_ffc:
            enc4_in = torch.cat((enc4, enc4_f), dim=1)

            dec4 = torch.cat((dec4, enc4_in), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)

            enc3_in = torch.cat((enc3, enc3_f), dim=1)
            dec3 = torch.cat((dec3, enc3_in), dim=1)
            dec3 = self.decoder3(dec3)

            dec2 = self.upconv2(dec3)
            enc2_in = torch.cat((enc2, enc2_f), dim=1)
            dec2 = torch.cat((dec2, enc2_in), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            enc1_in = torch.cat((enc1, enc1_f), dim=1)
            dec1 = torch.cat((dec1, enc1_in), dim=1)

        else:
            dec4 = torch.cat((dec4, enc4), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)
            dec3 = torch.cat((dec3, enc3), dim=1)
            dec3 = self.decoder3(dec3)
            dec2 = self.upconv2(dec3)
            dec2 = torch.cat((dec2, enc2), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            dec1 = torch.cat((dec1, enc1), dim=1)

        dec1 = self.decoder1(dec1)
        if self.return_features:
            return self.softmax(self.conv(dec1)), enc1
       

        return self.softmax(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


def get_model(name="deeplab", **kwargs):
    if name == "deeplab":
        return DeepLab(**kwargs)
    if name == "unet":
        model = UNet(in_channels = kwargs["in_channels"], num_classes = kwargs["num_classes"])
        return model
    elif name == "y_net_gen":
        model = YNet_general(in_channels = kwargs["in_channels"], num_classes = kwargs["num_classes"], ffc=False)
        return model
    elif name == "y_net_gen_ffc":
        model = YNet_general(in_channels = kwargs["in_channels"], num_classes = kwargs["num_classes"], ffc=True, ratio_in=kwargs["ratio"])
        return model
    else:
        print("Model name not found")
        assert False

if __name__ == "__main__":
    model = DeepLab(backbone='mobilenet', output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())


