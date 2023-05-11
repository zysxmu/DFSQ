import torch
import os
from tqdm import tqdm
import data_new
import utility
from model.setq.edsr_setq import SetQ_EDSR
from model.setq.rdn_setq import SetQ_RDN
from model.setq.bnsrresnet_setq import SetQ_SRResNet
from option import args
from model.setq_conv_quant_ops import SetQConv2d
import matplotlib.pyplot as plt
import pickle
plt.switch_backend('agg')

torch.manual_seed(args.seed)
ckp = utility.checkpoint(args)
device = torch.device('cpu' if args.cpu else f'cuda:{args.gpu_id}')

def prepare(*arg):
    def _prepare(tensor):
        if args.precision == 'half':
            tensor = tensor.half()
        return tensor.cuda()

    return [_prepare(a) for a in arg]


def test(model, loader_test, scale, is_teacher=False, save = False):
    torch.set_grad_enabled(False)
    ckp.write_log('\nEvaluation:')
    ckp.add_log(
        torch.zeros(1, len(loader_test), len(scale))
    )

    model.eval()
    timer_test = utility.timer()

    if args.save_results:
        ckp.begin_background()
    savesau = {}
    savesal = {}

    for idx_data, d in enumerate(loader_test):
        for idx_scale, scale in enumerate(args.scale):
            d.dataset.set_scale(idx_scale)
            i = 0
            for lr, hr, filename in tqdm(d, ncols=80):
                i += 1
                lr, hr = prepare(lr, hr)
                lr, hr = lr.to(device), hr.to(device)
                sr, s_res = model(lr)
                sr = utility.quantize(sr, args.rgb_range)
                save_list = [sr]
                cur_psnr = utility.calc_psnr(
                    sr, hr, scale, args.rgb_range, dataset=d
                )
                ckp.log[-1, idx_data, idx_scale] += cur_psnr
                if args.save_gt:
                    save_list.extend([lr, hr])

                if args.save_results and save is True:
                    save_name = f'{args.a_bits}bit_{filename[0]}'
                    ckp.save_results(d, save_name, save_list, scale, experiment=args.test_name)
            # pdb.set_trace()
            ckp.log[-1, idx_data, idx_scale] /= len(d)
            ckp.write_log(
                '[{} x{}] PSNR: {:.3f}'.format(
                    d.dataset.name,
                    scale,
                    ckp.log[-1, idx_data, idx_scale],
                )
            )
    if args.save_results:
        ckp.end_background()

    ckp.write_log(
        'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
    )

    torch.set_grad_enabled(True)


def conv_forward_hook(module, input, output):
    if module.raw_input is None:
        module.raw_input = []
    if module.raw_output is None:
        module.raw_output = []
    module.raw_input.append(input[0].cpu())
    module.raw_output.append(output.cpu())



class Calibrator():
    def __init__(self, net, calib_loader, test_loader):
        self.net = net
        self.calib_loader = calib_loader
        self.test_loader = test_loader
        self.calibrated = False

    def quant_calib(self):
        with torch.no_grad():
            for name, module in self.net.named_modules():
                if type(module) == SetQConv2d:
                    qps_ = []
                    print("======name:{},w_bits:{},a_bits:{}=====".format(name, module.w_bits, module.a_bits))

                    file_name = "{}/{}/{}_maxv.pkl".format(
                         "result", args.quant_file,name.replace('.', '_'))
                    
                    with open(file_name,'rb') as f:
                        dc = pickle.load(f)
                        for c in range(module.in_channels):
                            temp = dc[c]
                            qps = temp['qps']
                            qps = torch.Tensor(list(qps.squeeze().tolist()))
                            qps_.append(qps)
                        qps_ = torch.stack(qps_)
                        module.qps = qps_.cuda()
                        module.calibration_step()
                        torch.cuda.empty_cache()

        print("cabliration finish...")
        with torch.no_grad():
            for name, module in self.net.named_modules():
                if isinstance(module, SetQConv2d):
                    module.mode = "quant"
            self.net.eval()
            test(self.net, self.test_loader, scale=args.scale, save=True)
   
    def save(self):
        hooks = []
        for name, module in self.net.named_modules():
            if type(module) == SetQConv2d:
                module.mode = "raw"
                hooks.append(module.register_forward_hook(conv_forward_hook))

        with torch.no_grad():
            for i, (lr, hr, _,) in enumerate(self.calib_loader):
                lr, hr = prepare(lr, hr)
                sr, s_res = self.net(lr)
                if i == args.calib_round:
                    break

        for hook in hooks:
            hook.remove()

        for name, module in self.net.named_modules():
            if type(module) == SetQConv2d:
                module.raw_input = torch.cat(module.raw_input, dim=0)
                module.raw_output = torch.cat(module.raw_output, dim=0)

                folder_path = "data/{}".format(args.quant_file)
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                
                file_path = "{}/{}.pkl".format(folder_path, name.replace('.', '_'))
                if os.path.exists(file_path) == False:
                    d = {'name':name,
                        'raw_input':module.raw_input.detach().numpy(),
                        'postReLU':module.postReLU,
                        'bit':module.a_bits
                        }
                    print(d['raw_input'].shape)
                    with open(file_path,'wb') as f:
                        pickle.dump(d,f)

def main():
    loader = data_new.Data(args)
    calib_loader=loader.loader_train
    test_loader=loader.loader_test
    if args.model.lower() == 'edsr':
        model = SetQ_EDSR(args,bias=True).to(device)
    elif args.model.lower() == 'rdn':
        model = SetQ_RDN(args).to(device)
    elif args.model.lower() == 'bnsrresnet':
        model = SetQ_SRResNet(args).to(device)
    else:
        raise ValueError('not expected model = {}'.format(args.model))

    checkpoint = torch.load(args.pre_train)
    checkpoint = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    model.load_state_dict(checkpoint,strict = False)
    
    print("check fp model accuracy:")
    test(model,test_loader,scale=args.scale,save = False)
    

    calib=Calibrator(model,calib_loader,test_loader)
    if args.calib is False:
        calib.save()
    else:
        calib.quant_calib()

    # test
    test(model,test_loader,scale=args.scale)

if __name__ == '__main__':
    main()
