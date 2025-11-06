import argparse
import numpy as np
class get_config():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='CT img Recon')
        self.parser.add_argument("--use_gpu", type=bool, default=True, help='use GPU or not')
        self.parser.add_argument('--gpu_idx', type=int, default=0, help='idx of gpu')
        self.parser.add_argument('--data_type', default='IMA', help='dcm, IMA')
        self.parser.add_argument('--resume', default=True, help='resume training')  #False
        self.parser.add_argument('--manualSeed', type=int, default=205,help='manual seed')
        self.parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
        #self.parser.add_argument("--step_size", type=int, default=5, help="When to decay learning rate")
        self.parser.add_argument("--milestone", type=int, default=[55,70,80,90], nargs='+',
                                 help="When to decay learning rate")
        self.parser.add_argument('--log_dir', default='./result_1k_60/logs/', help='tensorboard logs')

        # Training Parameters
        self.parser.add_argument('--epoch', type=int, default=100, help='#epoch ')
        self.parser.add_argument('--tr_batch', type=int, default=1, help='batch size')
        self.parser.add_argument('--vl_batch', type=int, default=1, help='val batch size')
        self.parser.add_argument('--ts_batch', type=int, default=1, help='batch size')
        self.parser.add_argument('--test_model', default='epoch99.pth', help='dcm, png')

        self.parser.add_argument('--deep', type=int, default=17, help='depth')
        self.parser.add_argument('--img_size', default=[512,512], help='image size')
        self.parser.add_argument('--sino_size', nargs='*', default=[360,800], help='sino size')
        self.parser.add_argument('--poiss_level',default=5e6, help='Poisson noise level')
        self.parser.add_argument('--gauss_level',default=[0.05], help='Gaussian noise level')
        self.parser.add_argument('--test_gauss_level', default=[0.05], help='Poisson noise level')
        self.parser.add_argument('--test_poiss_level', default=5e6, help='Poisson noise level')
        self.parser.add_argument('--S', type=int, default=1, help='the number of total iterative stages')
        self.parser.add_argument('--eta1', type=float, default=0.001, help='initialization for stepsize eta1')
        self.parser.add_argument('-opts', type=str, help='Path to option YAML file.',
                            default='./options/train/SMID.yml')

        self.parser.add_argument('--train_ps', type=int, default=512, help='patch size of training sample')
        self.parser.add_argument('--embed_dim', type=int, default=32, help='dim of emdeding features')
        self.parser.add_argument('--win_s_projection', type=str, default='linear', help='linear/conv token projection')
        self.parser.add_argument('--tokenize', type=int, default=4, help='window size of self-attention')
        self.parser.add_argument('--token_mlp', type=str, default='leff', help='ffn/leff token mlp')
        self.parser.add_argument('--win_size', type=int, default=4, help='window size of self-attention')
        self.parser.add_argument('--token_projection', type=str, default='linear', help='linear/conv token projection')

        self.parser.add_argument("--out_dir", type=str, dest="out_dir", help="Results dir path", default='15')

        self.parser.add_argument('--sigma', type=int, default=80, help='sigma')
        self.parser.add_argument('--depth', type=str, default=8, help='depth')
        # self.parser.add_argument('--beta', type=int, default=0.001, help='window size of self-attention')
        self.parser.add_argument('--beta', type=int, default=0.001, help='window size of self-attention')
        self.parser.add_argument('--gamma', type=str, default=1.02, help='ffn/leff token mlp')



        self.parser.parse_args(namespace=self)


        self.mode= 'sparse'   #sparse,limited
        self.phase = 'test'   #'tr test'
        self.imagenum_train = '1000'
        self.imagenum_val = '100'
        self.imagenum_test = '500'


        # Result saving locations
        self.info = self.mode
        self.img_dir = './result_1k_60/' + self.info + '/img/'
        self.model_dir = './result_1k_60/' + self.info + '/ckp/'


        if self.phase == 'tr':
            print('!!!!!!训练步骤!!!!!!')
            print('This is mode %s' % self.mode)

            self.tr_dir = r'./aapm_npytry/60/train'
            self.vl_dir = r'./aapm_npytry/60/val'
            self.test_img_dir = r'./aapm_npytry/60/test'


            if self.resume == True:
                self.resume_ckp_dir = self.model_dir +'epoch29.pth'
                self.resume_ckp_resume = '29'
        elif self.phase == 'test':
            self.test_ckp_dir = self.model_dir + self.test_model
            print('!!!!!!测试步骤!!!!!!')
            print('This is mode %s' %self.mode)
            print(self.test_ckp_dir)
            self.test_save_img = True
            self.test_img_dir =  r'./aapm_npytry/60/test'



