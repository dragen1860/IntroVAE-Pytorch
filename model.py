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
            nn.Conv2d(3, ch, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(ch),
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
                ResBlk([1, 3, 3], [ch_cur, ch_next, ch_next, ch_next]),
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
            ResBlk([3, 3], [ch_next, ch_next, ch_next]),
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
            nn.BatchNorm1d(z_dim * mapsz * mapsz),
            nn.ReLU(inplace=True),
            Reshape(z_dim, mapsz, mapsz),
            ResBlk([3, 3], [z_dim, z_dim, z_dim])
        ]


        # scale imgsz up while keeping channel untouched
        # [b, z_dim, 4, 4] => [b, z_dim, 8, 8] => [b, z_dim, 16, 16]
        for i in range(2): # if upsamples > 6
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

        while mapsz < imgsz//2:
            ch_cur = ch_next
            ch_next = ch_next // 2 if ch_next >=32 else ch_next # set mininum ch=16
            layers.extend([
                # [2, 32, 32, 32] => [2, 32, 64, 64]
                nn.Upsample(scale_factor=2),
                # => [2, 16, 64, 64]
                ResBlk([1, 3, 3], [ch_cur, ch_next, ch_next, ch_next])
            ])
            mapsz = mapsz * 2

            tmp = torch.randn(2, z_dim)
            net = nn.Sequential(*layers)
            out = net(tmp)
            print(list(out.shape), end='=>')
            del net


        # [b, ch_next, 1024, 1024] => [b, 3, 1024, 1024]
        layers.extend([
            nn.Upsample(scale_factor=2),
            ResBlk([3, 3], [ch_next, ch_next, ch_next]),
            nn.Conv2d(ch_next, 3, kernel_size=5, stride=1, padding=2),
            # sigmoid / tanh
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


    def __init__(self, args):
        """

        :param imgsz:
        :param z_dim: h_dim is the output dim of encoder, and we use mu_net/sigma net to convert it from
        h_dim to z_dim
        """
        super(IntroVAE, self).__init__()

        imgsz = args.imgsz
        z_dim = args.z_dim


        self.encoder = Encoder(imgsz, 16)

        # get z_dim
        x = torch.randn(2, 3, imgsz, imgsz)
        z_ = self.encoder(x)
        h_dim = z_.size(1)

        # create mu net and sigm net
        # self.mu_net = nn.Linear(h_dim, z_dim)
        # self.log_sigma2_net = nn.Linear(h_dim, z_dim)
        self.z_net = nn.Linear(h_dim, 2 * z_dim)

        # sample
        z, mu, log_sigma = self.reparametrization(z_)

        # create decoder by z_dim
        self.decoder = Decoder(imgsz, z_dim)
        out = self.decoder(z)

        # print
        print('x:', list(x.shape), 'z_:', list(z_.shape), 'z:', list(z.shape), 'out:', list(out.shape))
        print('z_dim:', z_dim)


        self.alpha = args.alpha
        self.beta = args.beta
        self.gamma = args.gamma
        self.margin = args.margin
        self.z_dim = z_dim
        self.h_dim = h_dim

        self.optim_encoder = optim.Adam(self.encoder.parameters(), lr=args.lr)
        self.optim_decoder = optim.Adam(self.decoder.parameters(), lr=args.lr)


    def set_alph_beta_gamma(self, alpha, beta, gamma):
        """
        this func is for pre-training, to set alpha=0 to transfer to vilina vae.
        :param alpha: for adversarial loss
        :param beta: for reconstruction loss
        :param gamma: for variational loss
        :return:
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def reparametrization(self, z_):
        """

        :param z_: [b, 2*z_dim]
        :return:
        """
        # mu, log_sigma2 = self.mu_net(z_), self.log_sigma2_net(z_)
        # [b, 1024] => [b, 512], [b, 512]
        mu, log_sigma2 = self.z_net(z_).chunk(2, dim=1)
        eps = torch.randn_like(log_sigma2)
        # reparametrization trick
        # mean + sigma * eps
        z = mu + torch.exp(log_sigma2).sqrt() * eps

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

    def output_activation(self, x):
        """

        :param x:
        :return:
        """
        return torch.tanh(x)

    def forward(self, x):
        """
        The notation used here all come from Algorithm 1, page 6 of official paper.
        can refer to Figure7 in page 15 as well.
        :param x: [b, 3, 1024, 1024]
        :return:
        """

        # 1. update encoder
        z_ = self.encoder(x)
        z, mu, log_sigma2 = self.reparametrization(z_)
        xr = self.output_activation(self.decoder(z))
        zp = torch.randn_like(z)
        xp = self.output_activation(self.decoder(zp))

        loss_ae = F.mse_loss(xr, x)
        reg_ae = self.kld(mu, log_sigma2)

        zr_ng_ = self.encoder(xr.detach())
        zr_ng, mur_ng, log_sigma2r_ng =  self.reparametrization(zr_ng_)
        regr_ng = self.kld(mur_ng, log_sigma2r_ng)
        # max(0, margin - l)
        regr_ng = torch.clamp(self.margin - regr_ng, min=0)
        zpp_ng_ = self.encoder(xp.detach())
        zpp_ng, mupp_ng, log_sigma2pp_ng = self.reparametrization(zpp_ng_)
        regpp_ng = self.kld(mupp_ng, log_sigma2pp_ng)
        # max(0, margin - l)
        regpp_ng = torch.clamp(self.margin - regpp_ng, min=0)


        encoder_adv = regr_ng + regpp_ng
        encoder_loss = self.gamma * reg_ae + self.alpha * encoder_adv + self.beta * loss_ae
        self.optim_encoder.zero_grad()
        encoder_loss.backward()
        self.optim_encoder.step()


        # 2. update decoder
        z_ = self.encoder(x)
        z, mu, log_sigma2 = self.reparametrization(z_)
        xr = self.output_activation(self.decoder(z))
        zp = torch.randn_like(z)
        xp = self.output_activation(self.decoder(zp))

        loss_ae = F.mse_loss(xr, x)

        zr_ = self.encoder(xr)
        zr, mur, log_sigma2r = self.reparametrization(zr_)
        regr = self.kld(mur, log_sigma2r)
        zpp_ = self.encoder(xp)
        zpp, mupp, log_sigma2pp = self.reparametrization(zpp_)
        regpp = self.kld(mupp, log_sigma2pp)

        # by Eq.12, the 1st term of loss
        decoder_adv = regr + regpp
        decoder_loss = self.alpha * decoder_adv + self.beta * loss_ae
        self.optim_decoder.zero_grad()
        decoder_loss.backward()
        self.optim_decoder.step()



        return encoder_loss, decoder_loss, reg_ae, encoder_adv, decoder_adv, loss_ae, xr, xp


















def main():
    pass





if __name__ == '__main__':
    main()