import resource
import os
import random

import torch
from torch.autograd import Variable
import torch.utils.data as Data

from config import opt
from utils import non_model
from make_dataset import test_Dataset
from net import model_TransSR

import numpy as np

from tqdm import tqdm
import SimpleITK as sitk
import warnings
warnings.filterwarnings("ignore")

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2000, rlimit[1]))


def test(**kwargs):
    # stage 1
    kwargs, data_info_dict = non_model.read_kwargs(kwargs)
    opt.load_config(kwargs['config_path'])
    config_dict = opt._spec(kwargs)

    # stage 2
    save_model_folder = '../model/%s/%s/' % (opt.path_key, str(opt.net_idx))
    save_output_folder = '../test_output/%s/%s/' % (
        opt.path_key, str(opt.net_idx))
    os.makedirs(save_output_folder, exist_ok=True)

    # stage 3
    save_model_list = sorted(os.listdir(save_model_folder))
    use_model = [each for each in save_model_list if each.endswith('pkl')][-1]
    print('use model:', use_model)
    use_model_path = save_model_folder + use_model
    config_dict = non_model.update_kwargs(use_model_path, kwargs)
    opt._spec(config_dict)
    print('load config done')

    # stage 4 Dataloader Setting
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    GLOBAL_WORKER_ID = None

    def worker_init_fn(worker_id):
        global GLOBAL_WORKER_ID
        GLOBAL_WORKER_ID = worker_id
        set_seed(GLOBAL_SEED + worker_id)

    GLOBAL_SEED = 2022
    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)
    torch.cuda.manual_seed(GLOBAL_SEED)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    ###### GPU ######
    data_gpu = opt.gpu_idx
    torch.cuda.set_device(data_gpu)

    save_model_path = save_model_folder + use_model
    save_dict = torch.load(save_model_path, map_location=torch.device('cpu'))
    config_dict = save_dict['config_dict']
    config_dict.pop('path_img')
    config_dict['mode'] = 'test'
    opt._spec(config_dict)

    # test set
    test_list = [each.split('.')[0] for each in sorted(os.listdir(opt.path_img + 'test/1mm/'))]
    test_set = test_Dataset(test_list)
    test_data_num = len(test_set.img_list)
    test_batch = Data.DataLoader(dataset=test_set, batch_size=opt.val_bs, shuffle=False,
                                num_workers=opt.test_num_workers, worker_init_fn=worker_init_fn)
    print('load test data done, num =', test_data_num)

    load_net = save_dict['net']
    load_model_dict = load_net.state_dict()

    net = model_TransSR.TVSRN()
    net.load_state_dict(load_model_dict, strict=False)

    del save_dict
    net = net.cuda()
    net = net.eval()

    with torch.no_grad():
        pid_list = []
        traning_psnr_list = []
        testing_psnr_list = []
        training_ssim_list = []
        testing_ssim_list = []

        for i, return_list in tqdm(enumerate(test_batch)):
            case_name, x, y, pos_list = return_list
            case_name = case_name[0]

            pid_list.append(case_name)

            x = x.squeeze().data.numpy()
            y = y.squeeze().data.numpy()

            y_pre = np.zeros_like(y)
            pos_list = pos_list.data.numpy()[0]

            for pos_idx, pos in enumerate(pos_list):
                tmp_x = x[pos_idx]
                tmp_pos_z, tmp_pos_y, tmp_pos_x = pos

                tmp_x = torch.from_numpy(tmp_x)
                tmp_x = tmp_x.unsqueeze(0).unsqueeze(0)
                im = Variable(tmp_x.type(torch.FloatTensor).cuda())
                tmp_y_pre = net(im)
                tmp_y_pre = torch.clamp(tmp_y_pre, 0, 1)
                y_for_psnr = tmp_y_pre.data.squeeze().cpu().numpy()

                D = y_for_psnr.shape[0]
                # pos_z_s = 5 * tmp_pos_z + 3
                # pos_y_s = tmp_pos_y
                # pos_x_s = tmp_pos_x
                front_padding = opt.scale // 2 + opt.scale % 2
                pos_z_s = opt.scale * tmp_pos_z + front_padding
                pos_y_s = tmp_pos_y
                pos_x_s = tmp_pos_x

                y_pre[pos_z_s: pos_z_s+D, pos_y_s:pos_y_s +
                      opt.vc_y, pos_x_s:pos_x_s+opt.vc_x] = y_for_psnr

            del tmp_y_pre, im
        
            front_padding = opt.scale // 2 + opt.scale % 2
            back_padding = opt.scale // 2 + 1
            y_pre = y_pre[front_padding:-back_padding]
            y = y[front_padding:-back_padding]
            
            # y_pre = y_pre[5:-5]
            # y = y[5:-5]

            n = y_pre.shape[0]
            training_index = np.arange(opt.scale - front_padding, n, opt.scale)
            testing_index = np.setdiff1d(np.arange(n), training_index)

            save_name_pre = save_output_folder + '%s_pre.nii.gz' % case_name
            output_pre = sitk.GetImageFromArray(y_pre)
            sitk.WriteImage(output_pre, save_name_pre)

            training_psnr = non_model.cal_psnr(y_pre[training_index], y[training_index])
            testing_psnr = non_model.cal_psnr(y_pre[testing_index], y[testing_index])
            traning_psnr_list.append(training_psnr)
            testing_psnr_list.append(testing_psnr)

            print('pid:', case_name)
            print('training psnr:', training_psnr)
            print('testing psnr:', testing_psnr)

            # pid_ssim_list = []
            pid_training_ssim_list = []
            pid_testing_ssim_list = []
            for z_idx, z_layer in enumerate(y_pre):
                mask_layer = y[z_idx]

                tmp_ssim = non_model.cal_ssim(
                    mask_layer, z_layer, cuda_use=data_gpu)
                # pid_ssim_list.append(tmp_ssim)
                if z_idx in training_index:
                    pid_training_ssim_list.append(tmp_ssim)
                else:
                    pid_testing_ssim_list.append(tmp_ssim)

            training_ssim_list.append(np.mean(pid_training_ssim_list))
            testing_ssim_list.append(np.mean(pid_testing_ssim_list))
            
            print('training ssim:', np.mean(pid_training_ssim_list))
            print('testing ssim:', np.mean(pid_testing_ssim_list))

        # print(np.mean(psnr_list))
        # print(np.mean(ssim_list))
        print('training psnr:', np.mean(traning_psnr_list))
        print('testing psnr:', np.mean(testing_psnr_list))


        with open(save_output_folder + f'{use_model}.txt', 'w') as f:
            for i, j, k, l, m in zip(pid_list, traning_psnr_list, testing_psnr_list, training_ssim_list, testing_ssim_list):
                f.write('%s, training psnr:%.4f, testing psnr:%.4f, training ssim:%.4f, testing ssim:%.4f\n' % (i, j, k, l, m))


if __name__ == '__main__':
    import fire

    fire.Fire()
