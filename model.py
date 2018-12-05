import  torch
from    torch import nn
from    torch.nn import functional as F
import  math
from    utils import Reshape, Flatten, ResBlk



class Encoder(nn.Module):

    def __init__(self, imgsz, ch):
        """

        :param imgsz:
        :param ch:
        """
        super(Encoder, self).__init__()


        layers = [
            nn.Conv2d(3, ch, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, stride=None, padding=0),
        ]

        # 128 => 64
        mapsz = imgsz // 2
        ch_next = ch * 2
        block = 1

        while mapsz > 4:
            # add resblk
            layers.extend([
                ResBlk([3, 3], [ch, ch_next, ch_next]),
                nn.AvgPool2d(kernel_size=2, stride=None)
            ])
            block += 1
            mapsz = mapsz // 2
            ch = ch_next
            ch_next = ch_next * 2 if ch_next <= 256 else 512 # set max ch=512

        layers.extend([
            ResBlk([3, 3], [ch_next, ch_next, ch_next]),
            nn.AvgPool2d(kernel_size=2, stride=None),
            Flatten()
        ])

        self.net = nn.Sequential(*layers)


    def forward(self, x):
        """

        :param x:
        :return:
        """
        return self.net(x)




class Decoder(nn.Module):


    def __init__(self, imgsz, ch):
        """

        :param imgsz:
        :param ch:
        """
        super(Decoder, self).__init__()

        layers = [
            nn.Linear(512, 512 * 4 * 4),
            nn.ReLU(inplace=True),
            Reshape(512, 4, 4),
            ResBlk([3, 3], [512, 512, 512])
        ]

        mapsz = 4
        upsamples = int(math.log2(imgsz) - 2)
        block = 2
        ch_next = 512

        for i in range(upsamples-6):
            layers.extend([
                nn.Upsample(scale_factor=2),
                ResBlk([3, 3], [ch_next, ch_next, ch_next])
            ])

            mapsz = mapsz * 2

        while mapsz < imgsz:
            ch_next = ch_next // 2 if ch_next >=32 else 16 # set mininum ch=16
            layers.extend([
                nn.Upsample(scale_factor=2),
                ResBlk([3, 3], [ch_next * 2, ch_next, ch_next])
            ])
            mapsz = mapsz * 2

        layers.extend([
            nn.Conv2d(ch_next, 3, kernel_size=5, stride=1, padding=0),
            nn.Sigmoid()
        ])

        self.net = nn.Sequential(*layers)


    def forward(self, x):
        """

        :param x:
        :return:
        """
        return self.net(x)





class IntroVAE(nn.Module):


    def __init__(self, z_dim):
        super(IntroVAE, self).__init__()


        self.encoder = Encoder(1024, 32)
        self.decoder = Decoder(1024, 32)

        self.mu_net = nn.Linear(2*z_dim, z_dim)
        self.log_sigma2_net = nn.Linear(2 * z_dim, z_dim)

        self.alpha = 1
        self.beta = 1
        self.margin = 125

    def reparametrization(self, z_):
        """

        :param z_:
        :return:
        """
        mu, log_sigma2 = self.mu_net(z_), self.log_sigma2_net(z_)
        eps = torch.randn_like(log_sigma2)
        # reparametrization trick
        # TODO
        z = mu + torch.exp(log_sigma2 / 2) * eps

        return z, mu, log_sigma2

    def kld(self, mu, log_sigma2):
        """
        compute the kl divergence between N(mu, sigma^2) and N(0, 1)
        :param mu:
        :param log_sigma2:
        :return:
        """
        # TODO
        kl = 0.5 * (1 + log_sigma2 - torch.power(mu, 2) - torch.exp(log_sigma2))

        return kl


    def forward(self, x):
        """

        :param x: [b, 3, 1024, 1024]
        :return:
        """
        # x => z_ae => z_ae, mu_ae, log_sigma^2_ae => x_r_ae
        # get reconstructed z
        z_ae_ = self.encoder(x)
        # sample from z_r_
        z_ae, mu_ae, log_sigma2_ae = self.reparametrization(z_ae_)
        # sample from normal dist by shape of z_r
        z_p = torch.randn_like(z_ae)
        # get reconstructed x
        x_ae = self.decoder(z_ae)


        # z_r => x_r => z_r_ => z_r_hat, mu_r_hat, log_sigma2_r_hat
        z_r = z_ae.detach()
        x_r = self.decoder(z_r)
        z_r_ = self.encoder(x_r)
        z_r_hat, mu_r_hat, log_sigma2_r_hat = self.reparametrization(z_r_)

        # z_r => x_r.detach() => z_r_ng_ => z_r_ng, mu_r_ng, log_sigma2_r_ng
        z_r_ng_ = self.encoder(x_r.detach())
        z_r_ng, mu_r_ng, log_sigma2_r_ng = self.reparametrization(z_r_ng_)

        # z_p => x_p => z_p_ => z_p_hat, mu_p_hat, log_sigma^2_p_hat
        x_p = self.decoder(z_p)
        z_p_ = self.encoder(x_p)
        z_p_hat, mu_p_hat, log_sigma2_p_hat = self.reparametrization(z_p_)
        # z_p => x_p.detach() => z_p_ng_ => z_p_ng, mu_p_ng, log_sigma2_p_ng
        z_p_ng_ = self.encoder(x_p.detach())
        z_p_ng, mu_p_ng, log_sigma2_p_ng = self.reparametrization(z_p_ng_)


        reg_ae = self.kld(mu_ae, log_sigma2_ae)
        reg_r_ng = self.kld(mu_r_ng, log_sigma2_r_ng)
        reg_p_ng = self.kld(mu_p_ng, log_sigma2_p_ng)
        reg_r_hat = self.kld(mu_r_hat, log_sigma2_r_hat)
        reg_p_hat = self.kld(mu_p_hat, log_sigma2_p_hat)
        loss_ae = F.mse_loss(x_ae, x)
        loss_ae2 = F.mse_loss(x_r, x)


        # by Eq.11, the 2nd term of loss
        # max(0, margin - l)
        encoder_l2 = torch.clamp(self.margin - reg_p_ng, min=0) + torch.clamp(125 - reg_r_ng, min=0)
        encoder_loss = reg_ae + self.alpha * encoder_l2 + self.beta * loss_ae

        # by Eq.12, the 1st term of loss
        decoder_l1 = reg_r_hat + reg_p_hat
        decoder_loss = self.alpha * decoder_l1 + self.beta * loss_ae2


        return encoder_loss, decoder_loss


















def main():
    pass





if __name__ == '__main__':
    main()