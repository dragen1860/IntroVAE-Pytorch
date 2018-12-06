import  torch
from    torch import nn, optim
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

        x = torch.randn(2, 3, imgsz, imgsz)
        print('Encoder:', list(x.shape), end='=>')

        layers = [
            nn.Conv2d(3, ch, kernel_size=5, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, stride=None, padding=0),
        ]

        out = nn.Sequential(*layers)(x)
        print(list(out.shape), end='=>')

        # 128 => 64
        mapsz = imgsz // 2
        ch_cur = ch
        ch_next = ch_cur * 2

        while mapsz > 4:
            # add resblk
            layers.extend([
                ResBlk([3, 3], [ch_cur, ch_next, ch_next]),
                nn.AvgPool2d(kernel_size=2, stride=None)
            ])
            mapsz = mapsz // 2
            ch_cur = ch_next
            ch_next = ch_next * 2 if ch_next < 512 else 512 # set max ch=512

            out = nn.Sequential(*layers)(x)
            print(list(out.shape), end='=>')

        layers.extend([
            ResBlk([3, 3], [ch_cur, ch_next, ch_next]),
            nn.AvgPool2d(kernel_size=2, stride=None),
            Flatten()
        ])

        self.net = nn.Sequential(*layers)


        out = nn.Sequential(*layers)(x)
        print(list(out.shape))


    def forward(self, x):
        """

        :param x:
        :return:
        """
        return self.net(x)




class Decoder(nn.Module):


    def __init__(self, imgsz, z_dim):
        """

        :param imgsz: 1024
        :param z_dim: 512
        """
        super(Decoder, self).__init__()

        mapsz = 4
        upsamples = int(math.log2(imgsz) - 2) # 8
        ch_next = z_dim
        # print('z:', [2, z_dim], 'upsamples:', upsamples, ', recon:', [2, 3, imgsz, imgsz])
        print('Decoder:', [z_dim], '=>', [ch_next, mapsz, mapsz], end='=>')

        # z: [b, z_dim] => [b, z_dim, 4, 4]
        layers = [
            # z_dim => z_dim * 4 * 4 => [z_dim, 4, 4] => [z_dim, 4, 4]
            nn.Linear(z_dim, z_dim * mapsz * mapsz),
            nn.ReLU(inplace=True),
            Reshape(z_dim, mapsz, mapsz),
            ResBlk([3, 3], [z_dim, z_dim, z_dim])
        ]


        # scale imgsz up while keeping channel untouched
        # [b, z_dim, 4, 4] => [b, z_dim, 8, 8] => [b, z_dim, 16, 16]
        for i in range(upsamples-6): # if upsamples > 6
            layers.extend([
                nn.Upsample(scale_factor=2),
                ResBlk([3, 3], [ch_next, ch_next, ch_next])
            ])
            mapsz = mapsz * 2

            tmp = torch.randn(2, z_dim)
            net = nn.Sequential(*layers)
            out = net(tmp)
            print(list(out.shape), end='.=>')
            del net

        # scale imgsz up and scale imgc down
        # mapsz: 4, ch_next: 512
        # x: [b, ch_next, mapsz, mapsz]
        # [b, z_dim, 16, 16] => [z_dim//2, 32, 32] => [z_dim//4, 64, 64] => [z_dim//8, 128, 128]
        # => [z_dim//16, 256, 256] => [z_dim//32, 512, 512] => [z_dim//64, 1024, 1024]

        while mapsz < imgsz:
            ch_cur = ch_next
            ch_next = ch_next // 2 if ch_next >=32 else ch_next # set mininum ch=16
            layers.extend([
                # [2, 32, 32, 32] => [2, 32, 64, 64]
                nn.Upsample(scale_factor=2),
                # => [2, 16, 64, 64]
                ResBlk([3, 3], [ch_cur, ch_next, ch_next])
            ])
            mapsz = mapsz * 2

            tmp = torch.randn(2, z_dim)
            net = nn.Sequential(*layers)
            out = net(tmp)
            print(list(out.shape), end='=>')
            del net


        # [b, ch_next, 1024, 1024] => [b, 3, 1024, 1024]
        layers.extend([
            nn.Conv2d(ch_next, 3, kernel_size=5, stride=1, padding=2),
            nn.Sigmoid()
        ])

        self.net = nn.Sequential(*layers)

        tmp = torch.randn(2, z_dim)
        out = self.net(tmp)
        print(list(out.shape))

    def forward(self, x):
        """

        :param x: [b, z_dim]
        :return:
        """
        # print('before forward:', x.shape)
        x =  self.net(x)
        # print('after forward:', x.shape)
        return x





class IntroVAE(nn.Module):


    def __init__(self, imgsz, z_dim):
        """

        :param imgsz:
        :param z_dim: h_dim is the output dim of encoder, and we use mu_net/sigma net to convert it from
        h_dim to z_dim
        """
        super(IntroVAE, self).__init__()


        self.encoder = Encoder(imgsz, 32)

        # get z_dim
        x = torch.randn(2, 3, imgsz, imgsz)
        z_ = self.encoder(x)
        h_dim = z_.size(1)

        # create mu net and sigm net
        self.mu_net = nn.Linear(h_dim, z_dim)
        self.log_sigma2_net = nn.Linear(h_dim, z_dim)

        # sample
        z, mu, log_sigma = self.reparametrization(z_)

        # create decoder by z_dim
        self.decoder = Decoder(imgsz, z_dim)
        out = self.decoder(z)

        # print
        print('x:', list(x.shape), 'z_:', list(z_.shape), 'z:', list(z.shape), 'out:', list(out.shape))
        print('z_dim:', z_dim)


        self.alpha = 1
        self.beta = 1
        self.margin = 125
        self.z_dim = z_dim
        self.h_dim = h_dim

        self.optim_encoder = optim.Adam(self.encoder.parameters(), lr=1e-3)
        self.optim_decoder = optim.Adam(self.decoder.parameters(), lr=1e-3)



    def reparametrization(self, z_):
        """

        :param z_: [b, 2*z_dim]
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
        # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
        kl = - 0.5 * (1 + log_sigma2 - torch.pow(mu, 2) - torch.exp(log_sigma2))
        kl = kl.sum()

        return kl


    def forward(self, x):
        """

        :param x: [b, 3, 1024, 1024]
        :return:
        """
        # x => z_ae_ => z_ae, mu_ae, log_sigma^2_ae => x_ae
        # get reconstructed z
        z_ae_ = self.encoder(x)
        # sample from z_r_
        z_ae, mu_ae, log_sigma2_ae = self.reparametrization(z_ae_)
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
        # sample from normal dist by shape of z_r
        z_p = torch.randn_like(z_ae)
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


        self.optim_encoder.zero_grad()
        encoder_loss.backward()
        self.optim_encoder.step()

        self.optim_decoder.zero_grad()
        decoder_loss.backward()
        self.optim_decoder.step()



        print(encoder_loss.item(), decoder_loss.item(), loss_ae.item(), loss_ae2.item())


















def main():
    pass





if __name__ == '__main__':
    main()