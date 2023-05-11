# Distribution-Flexible Subset Quantization for Post-Quantizing Super-Resolution Networks

## Dependence
* Python 3.8
* PyTorch >= 1.7.0

## Datasets
Please download DIV2K datasets.

Then, create a directory 'datasets' and re-organise the downloaded dataset directory as follows:

```
option.py
main_setq.py
datasets
  benchmark
  DIV2K
```
Additionally,you need to create some directory:'data'、'log'、'result'  as follows:
```
option.py
main_setq.py
data
log
result
datasets
  benchmark
  DIV2K
```
## Usage

### 1: train full-precision models:
An example:
```
python main_ori.py --scale 4 \
--model edsr
--save edsr_baseline_x4 \
--patch_size 192 \
--epochs 300 \
--decay 200 \
--gclip 0 \
--dir_data ./datasets
```
### 2: get activation map for cluster:

An example:
```
python main_setq.py --scale 4 \
--model edsr \
--pre_train path/fp_model --patch_size 192 \
--w_bits 4 --a_bits 4 \
--quant_file "edsr_4x_4bit" \
--data_test "Set14+Set5+B100+Urban100" \

```
The result will be saved in data/edsr_4x_4bit/

### 3: cluster for quantization paramters:
```
python cluster_process.py 
```

### 4: test quantized models

An example:
```
python main_setq.py --scale 4 \
--w_bits 4 --a_bits 4 \
--model edsr \
--pre_train path/fp_model --patch_size 192 \
--data_test "Set14+Set5+B100+Urban100" \
--quant_file "edsr_4x_4bit" --calib \
```


### calculate PSNR/SSIM

After saving the images, modify path in`metrics/calculate_PSNR_SSIM.m` to generate results.

```
matlab -nodesktop -nosplash -r "calculate_PSNR_SSIM('$dataset',$scale,$bit);quit"
```

refer to `metrics/run.sh` for more details.

##  Trained FP models and quantized models' cluster results(quantization paramters): 
[here](https://drive.google.com/drive/folders/1pI3EBmX6aa59Hj0IHTin1530y5CrH9Ap?usp=share_link)
Download these model. Then use the commands above to obtain the reported results of the paper.


##  Acknowledgments

Code is implemented based on [DDTB](https://github.com/zysxmu/DDTB) and [PAMS](https://github.com/colorjam/PAMS)
