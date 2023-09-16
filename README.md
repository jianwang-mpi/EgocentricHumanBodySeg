# Egocentric Human Body Segmentation

This repo is based on [HRNet-Semantic-Segmentation](https://arxiv.org/abs/1904.04514) 
and further used in [ScenEgo](https://github.com/yt4766269/SceneEgo) project for egocentric human body pose estimation.


### Installation
1. Install pytorch >= v1.11.0 following [official instruction](https://pytorch.org/).
2. run: `pip install -r requirements.txt`
3. Download [pretrained models](https://nextcloud.mpi-klsb.mpg.de/index.php/s/jHCDJ4wDTerPmqk) and put them under `checkpoints/` folder.

### Run the demo
```shell
python demo.py --cfg experiments/egocentric/egocentric_seg_test.yaml --img_dir data/example_data/imgs --model_path checkpoints/ego_human_seg.pth.tar
```
The result will be saved under `data/example_data/segs`.
### Citation
If you find this work or code is helpful in your research, please cite:
````
@inproceedings{wang2023scene,
  title={Scene-aware Egocentric 3D Human Pose Estimation},
  author={Wang, Jian and Luvizon, Diogo and Xu, Weipeng and Liu, Lingjie and Sarkar, Kripasindhu and Theobalt, Christian},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={13031--13040},
  year={2023}
}
````
