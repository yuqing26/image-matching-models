# Image-Matching-Models

A repository to easily try 19 different image matching models.

Some results with SIFT-LightGlue (respectively outdoor, indoor, satellite, painting and false positive)
<p float="left">
  <img src="https://github.com/gmberton/image-matching-models/blob/29b6c2ba42e3d7b235029a7bf75ddf7a1780cafa/assets/example_sift-lg/output_3.jpg" height="150" />
  <img src="https://github.com/gmberton/image-matching-models/blob/29b6c2ba42e3d7b235029a7bf75ddf7a1780cafa/assets/example_sift-lg/output_2.jpg" height="150" />
  <img src="https://github.com/gmberton/image-matching-models/blob/29b6c2ba42e3d7b235029a7bf75ddf7a1780cafa/assets/example_sift-lg/output_4.jpg" height="150" />
  <img src="https://github.com/gmberton/image-matching-models/blob/29b6c2ba42e3d7b235029a7bf75ddf7a1780cafa/assets/example_sift-lg/output_1.jpg" height="150" />
  <img src="https://github.com/gmberton/image-matching-models/blob/29b6c2ba42e3d7b235029a7bf75ddf7a1780cafa/assets/example_sift-lg/output_0.jpg" height="150" />
</p>

Some results with LoFTR
<p float="left">
  <img src="https://github.com/gmberton/image-matching-models/blob/29b6c2ba42e3d7b235029a7bf75ddf7a1780cafa/assets/examples_loftr/output_3.jpg" height="150" />
  <img src="https://github.com/gmberton/image-matching-models/blob/29b6c2ba42e3d7b235029a7bf75ddf7a1780cafa/assets/examples_loftr/output_2.jpg" height="150" />
  <img src="https://github.com/gmberton/image-matching-models/blob/29b6c2ba42e3d7b235029a7bf75ddf7a1780cafa/assets/examples_loftr/output_4.jpg" height="150" />
  <img src="https://github.com/gmberton/image-matching-models/blob/29b6c2ba42e3d7b235029a7bf75ddf7a1780cafa/assets/examples_loftr/output_1.jpg" height="150" />
  <img src="https://github.com/gmberton/image-matching-models/blob/29b6c2ba42e3d7b235029a7bf75ddf7a1780cafa/assets/examples_loftr/output_0.jpg" height="150" />
</p>


## Use

To use this repo simply run

```bash
git clone --recursive https://github.com/gmberton/image-matching-models
```
You can install this package for use in other scripts/notebooks with the following
```bash
cd image-matching-models
python -m pip install -e .
```
You can then use any of the matchers with 

```python
from matching import get_matcher

matcher = get_matcher('your matcher name here!')
img_size = 560
device = 'cuda' # 'cpu'

img0 = matcher.image_loader('path/to/img0.png', resize=img_size)
img1 = matcher.image_loader('path/to/img1.png', resize=img_size)

num_inliers, H, mkpts0, mkpts1 = matcher(img0, img1, device=device)
```

You can also run as a standalone script, which will perform inference on the the examples inside `./assets`. It is possible to specify also resolution and num_keypoints. This will take a few seconds also on a laptop's CPU, and will produce the same images that you see above.

```
python main.py --matcher sift-lg --device cpu --log_dir output_sift-lg
```

Where `sift-lg` will use `SIFT + LightGlue`.

**You can choose any of the following methods:
loftr, sift-lg, superpoint-lg, disk-lg, aliked-lg, doghardnet-lg, roma, dedode, steerers, sift-nn, orb-nn, patch2pix, patch2pix_superglue, superglue, r2d2, d2net, duster, doghardnet-nn, xfeat**

The script will generate an image with the matching keypoints for each pair, under `./output_sift-lg`.

All the matchers can run on GPU, and most of them can run both on GPU or CPU. A few can't run on CPU.


### Use on your own images

To use on your images you have two options:
1. create a directory with sub-directories, with two images per sub-directory, just like `./assets/example_pairs`. Then use as `python main.py --input path/to/dir`
2. create a file with pairs of paths, separate by a space, just like `assets/example_pairs_paths.txt`. Then use as `python main.py --input path/to/file.txt`
3. import the matcher package into a script/notebook and use from there

## Models

| Model | Code | Paper |
|-------|------|-------|
| XFeat (CVPR '24) | [Official](https://github.com/verlab/accelerated_features) | [arxiv](https://arxiv.org/abs/2404.19174) |
| GIM (ICLR '24) | [Official](https://github.com/xuelunshen/gim?tab=readme-ov-file) | [arxiv](https://arxiv.org/abs/2402.11095) |
| RoMa (CVPR '24) | [Official](https://github.com/Parskatt/RoMa) | [arxiv](https://arxiv.org/abs/2305.15404) |
| Dust3r (CVPR '24) | [Official](https://github.com/naver/dust3r) | [arxiv](https://arxiv.org/abs/2312.14132) |

## TODO

- [ ] Add a table to the README with the source for each model (code source and paper)
- [ ] Add parameter for RANSAC threshold
- [ ] It might be useful to return other outputs (e.g. `kpts0, kpts1`) (for the methods that have them)
- [x] Add DeDoDe + LightGlue from kornia
- [ ] Add CVNet
- [ ] Add TransVPR
- [ ] Add Patch-NetVLAD
- [ ] Add SelaVPR
- [x] Add xFeat
- [ ] Add any other local features method

PRs are very much welcomed :-)


### Adding a new method

To add a new method simply add it to `./matching`. If the method requires external modules, you can add them to `./third_party` with `git submodule add`: for example, I've used this command to add the LightGlue module which is automatically downloaded when using `--recursive`

```
git submodule add https://github.com/cvg/LightGlue third_party/LightGlue
```

This command automatically modifies `.gitmodules` (and modifying it manually doesn't work).


## Note
This repo is not optimized for speed, but for usability. The idea is to use this repo to find the matcher that best suits your needs, and then use the original code to get the best out of it.

## Cite

This repo was created as part of the EarthMatch paper

```
@InProceedings{Berton_2024_EarthMatch,
    author    = {Gabriele Berton, Gabriele Goletto, Gabriele Trivigno, Alex Stoken, Barbara Caputo, Carlo Masone},
    title     = {EarthMatch: Iterative Coregistration for Fine-grained Localization of Astronaut Photography},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2024},
}
```
