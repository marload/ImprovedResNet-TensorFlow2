# Improved Residual Networks
This is a repository that implements ["Improved Residual Networks for Image and Video Recognition"](https://arxiv.org/pdf/2004.04989.pdf) using TensorFlow2. The author's official repository is [pytorch-iresnet](https://github.com/iduta/iresnet). Most of them were referred to on [pytorch-iresnet](https://github.com/iduta/iresnet).

The accuracy on ImageNet (Data in the official repository):

| Network | 50-layers  | 101-layers | 152-layers | 200-layers |
| :-----: | :--------: | :--------: | :--------: | :--------: |
| ResNet  |   76.12%   |   78.00%   |   78.45%   |   77.55%   |
| iResnet | **77.31**% | **78.64**% | **79.34**% | **79.48**% |


### Requirements

* tensorflow > 2.1

### Usage

```bash
$ python src/train \
--nets={iresnet18, iresnet50, etc...}
--batch_size={default=64}
--epochs={default=100}
```