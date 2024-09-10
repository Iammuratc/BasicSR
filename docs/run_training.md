You can follow the steps on the WS to start training on the HIDE dataset. The HIDE dataset was already unzipped and the patches were created at `/mnt/2tb-1/image_enhancement/HIDE_dataset/patches`.

After setting up a virtual environment, training script can be initiated by the following commands on the WS by using 1 (i.e., single training) or 4 GPUs (i.e., distributed training):

For training on a single GPU:
Change the following lines in `basicsr/utils/options.py`.

Uncomment `parser.add_argument('--local-rank', type=int, default=0)`

Comment  `parser.add_argument('--local-rank', default=os.environ['LOCAL_RANK'])`

```  python3 basicsr/train.py --opt /mnt/2tb-1/image_enhancement/experiments/Deblurring_Restormer/Deblurring_Restormer.yml ```

For distributed training:

``` bash ./scripts/dist_train.sh 4 /mnt/2tb-1/image_enhancement/experiments/Deblurring_Restormer/Deblurring_Restormer.yml ```

Make sure you pass the correct data paths for `dataroot_gt` and `dataroot_lq`. 
