import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from model import DD_Net
from data_loader.dataset_1k import Phantom2dDatasettrain1, Phantom2dDatasetval1, Phantom2dDatasettest
from trainer.train_1k_noprompt import Trainer
from trainer.test_1k_noprompt import Tester
from head_HM import *
from config_1k_60 import get_config
# from model.In_DAMP import DAMP1
from Model import network_sct



if __name__ == '__main__':
    args = get_config()
    log(args) #加载参数
    # model
    # model = DD_Net()   #后处理
    model = network_sct(args)

    # main
    if args.phase == 'tr':
        test_dir = args.test_img_dir
        tr_dataset = Phantom2dDatasettrain1(args, phase='tr', datadir=args.tr_dir, length=int(args.imagenum_train), angle=[60])
        vl_dataset = Phantom2dDatasetval1(args, phase='vl', datadir=args.vl_dir, length=int(args.imagenum_val),angle=[60])
        test_dset = Phantom2dDatasettest(args, phase='test', datadir=test_dir, length=int(args.imagenum_test),
                                         angle=[60])
        train = Trainer(args, model, tr_dset=tr_dataset, vl_dset=vl_dataset,test_dset=test_dset)
        train.tr()

    elif args.phase == 'test':
        test_dir = args.test_img_dir
        test_dset = Phantom2dDatasettest(args, phase='test', datadir=test_dir, length=int(args.imagenum_test), angle=[60])
        test = Tester(args, model, test_dset=test_dset)
        test.test()
    print('[*] Finish!')







