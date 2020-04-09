---
date: 2020-02-15
title: "Dive into Faster RCNN"
toc: true
tags: ["object detection", "rcnn"]
---

Last year, I had a chance to be involved in an Advanced Computer Vision class held by a non-profit organization. During the class, object detection is one of the fields that I found myself interested in the most. This motivated me to write a series of blogs in order to understand better some famous approaches that has been applied in the field. Though, the idea has been postponed until recently :v. The first part of this series is about Faster RCN, one of the state-of-the-art method used for object detection. In this blog post, I will walk you through the detail of Faster RCNN. Hopefully, at the end of this blog, you would figure out the way Faster RCNN works.

## Outline
* [Object detection](#intro)
* [Faster RCNN](#faster-rcnn)
    - [VGG Shared Network]()
    - [Region Proposal Network]()
    - [Region-based CNN]()
* [Loss Function](#loss-function)
    - [Regression Loss]()
    - [Classification Loss]()
* [Training](#training)
    - [Augment data]()
    - [Create anchor generator]()
* [Detection](#detection)

## A little bit of object detection
In object detection, we received an image with bounding boxes indicating a various type of objects. There are many approaches these days.

![](https://i.imgur.com/agk4axh.jpg)

<a id="faster-rcnn"></a>
## Faster RCNN architecture
Faster RCNN is the last detection model in the RCNN trilogy (RCNN - Fast RCNN - Faster RCNN), which relies on proposed regions to detect objects. Though, unlike its predecessors  which use selective search to find out the best regions, Faster RCNN makes use of neural network and "learn" to propose regions directly. These proposed regions then is fed into another neural network to be refined once again.
<!-- This is the main reason that makes Faster RCNN faster and better than its predecessors. -->

First, let take a look at the overall architecture of Faster RCNN. It comprises of $2$ modules

* The region proposal module takes feature map from a feature network and proposes regions
* The Fast RCNN detector module takes those regions to predict the classes that the object belongs to.

![](https://i.imgur.com/Zsu3nEn.png)

The feature network, which is VGG in the context of this blog, is shared between both model.

To easily keep track of the story, let's follow a specific example in which we are given an image of shape $300\times400\times3$.
### Feature Shared Network

[TODO]Pretrained models

We use VGG as a feature network. The VGG receives an input image and produce a feature map with reduced sizes. The size of the feature map is determined by the net structure. For example, in case we use VGG, the feature map's shape is $18 \times 25 \times 512$.

![](https://i.imgur.com/dut8uoM.png)

### Region Proposal Network (RPN)
The goal of RPN is to propose regions that highly contain object. In order to do that, RPN does
* generate a predefined number of fixed-size anchors
* predict the objectness of each of these anchors
* refine their coordinates

#### Predefined anchors
<!-- RPN accepts VGG feature map as input. -->

For each pixel spatial location on the VGG feature map, we generate a predefined number of fixed size anchors. The shape of these anchor boxes are determined by a combination of predefined scales and edge ratios. In our example, if we use $3$ scales $64$, $128$, $256$ and $3$ edge ratios $1:1$, $1:2$, $2:1$, there will be $3*3=9$ type of anchors at each pixel location and a total of $18 * 25 * 9 = 4050$ anchors to be generated as a result.
![](https://i.imgur.com/BxG5M0Z.png)

It is important to note that even though anchor boxes are created based on the feature map's spatial location, they reference to the original input image, in which anchor boxes generated from the same feature map pixel location are centered at the same point on the original input, as illustrated in this figure below.
<img src="https://i.imgur.com/BNTidcL.png" width="400"/>

<!-- ![](https://i.imgur.com/3D1N77A.png) -->
<!-- ![](https://i.imgur.com/scAnbm9.png) -->

#### RPN architecture
<!-- It consists of 3 convolution layers: one convolutional layer with 512 filters of size 3x3 followed by two sibling 1x1 convolutional layers - one with $K$ filters acting as a classifier and the other with $4K$ filters acting as a regressor. -->

The RPN is then designed to predict objectness of each anchor (classification) and its coordinates (regression). It consists of $3$ layers: one convolutional layer with $512$ filters of size $3 \times 3$ followed by two sibling $1 \times 1$ convolutional layers. These two sibling layers - one with $K$ filters and the other with $4K$ filters - allow for classification and regression, respectively.

<img                    src="https://i.imgur.com/o1pTYG2.png" width="500"
/>

In our example, after passing the VGG feature map through RPN, it produces a classification output with shape of $18 \times 25 \times K$ and a regression output with shape of $18 \times 25 \times 4K$, where $K$ denotes the number of generated anchors at each feature map location.
<!-- ![](https://i.imgur.com/o1pTYG2.png) -->

```python
def rpn(base_layers, num_anchors):
    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(base_layers)

    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)

    return [x_class, x_regr, base_layers]
```

#### Create labeled data for training RPN
Now, we need labeled data to train the RPN.
##### Label for classification

For training classification task, each anchor box is labeled as
* positive - containing object
* negative - background
* ignored - being ignored when training

based on the overlap with its nearest ground-truth bounding box.

![](https://i.imgur.com/3m5ITZD.png)

We use IoU to measure these overlaps. Let $p$ denotes the IoU between current anchor box and its nearest ground-truth bounding box. The rule is detailed as follows
* If $p > \text{max_rpn}$, label it positive
* If $p < \text{min_rpn}$, label it negative
* If $\text{min_rpn} < p < \text{max_rpn}$, ignore it when training

``` python
# overlaps for RPN
cfg.rpn_min_overlap = 0.3
cfg.rpn_max_overlap = 0.7
```

##### Label for regression

The anchor box refinement is modeled as a regression problem, in which we predict the delta $(\color{red}{t_x, t_y, t_w, t_h})$ for each anchor box. This delta denotes the change needed to refine our predefined anchor boxes, as illustrated in this figure below

<!-- ![](https://i.imgur.com/x7kGAvI.png) -->

<img src="https://i.imgur.com/7h3T6TK.png" width="400"/>

Formally, we have

<div>
$$\begin{align}
\color{blue}{x} & = x_a + \color{red}{t_x}*w_a \\
\color{blue}{y} & = y_a + \color{red}{t_y}*h_a \\
\color{blue}{w} & = w_a * e^{\color{red}{t_w}} \\
\color{blue}{h} & = h_a * e^{\color{red}{t_h}}
\end{align}
$$
</div>

or

<div>
$$\begin{align}
\color{red}{t_x} & = (\color{blue}{x} - x_a) / w_a \\
\color{red}{t_y} & = (\color{blue}{y} - y_a) / h_a \\
\color{red}{t_w} & = log(\color{blue}{w}/w_a) \\
\color{red}{t_h} & = log(\color{blue}{h}/h_a)
\end{align}
$$
</div>

where $(x_a, y_a, w_a, h_a)$ denotes the anchor box's coordinates and $(\color{blue}{x, y, w, h})$ denotes the refined box's coordinates.

To create data for anchor regression training, we calculate the "ground-truth" delta $(\color{red}{t_x^*, t_y^*, t_w^*, t_h^*})$ based on each anchor box's coordinates $(x_a, y_a, w_a, h_a)$ and its nearest ground-truth bounding box's coordinates $(\color{blue}{x^*, y^*, w^*, h^*})$.

<div>
$$\begin{align}
\color{red}{t_x^*} & = (\color{blue}{x^*} - x_a) / w_a \\
\color{red}{t_y^*} & = (\color{blue}{y^*} - y_a) / h_a \\
\color{red}{t_w^*} & = log(\color{blue}{w^*}/w_a) \\
\color{red}{t_h^*} & = log(\color{blue}{h^*}/h_a)
\end{align}
$$
</div>

Among those generated anchor boxes, the positive anchors are probably outnumbered by the negative ones. Thus, to avoid imbalanced classification, we only use some anchor boxes for training. Specifically, only $256$ anchor boxes is chosen for training the RPN.

For example, with $4050$ anchor boxes generated, assume that we have $4000$ anchor boxes labeled as "positive", $50$ anchor boxes labeled as "negative"

#### RPN losses
##### 1. Regression Loss

The smooth L1 loss is used for regression training. Its formulation is as below

$$smooth_{L1}(x) =
\begin{cases}
0.5x^2 & \mbox{if} \;  \lvert x \rvert < 1, \\
\lvert x \rvert - 0.5 & \mbox{otherwise}.
\end{cases}
$$

where $x$ denotes the difference between prediction and ground truth $t  - \color{blue}{t^*}$.

<img src="https://i.imgur.com/HKcpwC2.png" width="300"/>

The reason smooth L1 loss is preferred to L1 and L2 loss is because it can handle the problem of these two losses. Being quadratic for small values ($\lvert x \rvert < 1$) and linear for large values ($\lvert x \rvert \geq 1$), smooth L1 loss is now less sensitive to outliers than L2 loss and also does not suffer from the problem of L1 loss, which is not differentiable around zero.

```python
# regression loss for rpn
def rpn_loss_regr(cfg):
    def rpn_loss_regr_fixed_num(y_true, y_pred):
        x = y_true[:, :, :, 4 * cfg.num_anchors:] - y_pred
        x_abs = K.abs(x)
        x_bool = K.cast(K.less_equal(x_abs, 1.0), tf.float32)

        return cfg.lambda_rpn_regr * K.sum(
            y_true[:, :, :, :4 * cfg.num_anchors] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(cfg.epsilon + y_true[:, :, :, :4 * cfg.num_anchors])

    return rpn_loss_regr_fixed_num
```
##### 2. Classification Loss
```python
def rpn_loss_cls(cfg):
    def rpn_loss_cls_fixed_num(y_true, y_pred):
        return cfg.lambda_rpn_class * K.sum(y_true[:, :, :, :cfg.num_anchors] * K.binary_crossentropy(y_pred[:, :, :, :], y_true[:, :, :, cfg.num_anchors:])) / K.sum(cfg.epsilon + y_true[:, :, :, :cfg.num_anchors])

    return rpn_loss_cls_fixed_num
```

#### Use RPN to propose regions
##### RPN prediction

<img src="https://i.imgur.com/uNnQDrw.png" width="400"/>

After training, we use RPN to predict the bounding box coordinates at each feature map location.

<!-- $$\begin{align}
\color{blue}{x} & = x_a + \color{red}{t_x}*w_a \\
\color{blue}{y} & = y_a + \color{red}{t_y}*h_a \\
\color{blue}{w} & = w_a * e^{\color{red}{t_w}} \\
\color{blue}{h} & = h_a * e^\color{red}{t_h}
\end{align}$$ -->

In our example, there is $4050$ anchor boxes in total. Assume that the RPN predict $3000$ positive bounding boxes - meaning that they are containing object.

##### Non-max suppression

<img src="https://i.imgur.com/dn7grUV.png" width="400"/>

It is very likely that there are many bounding boxes, among those are predicted by RPN, referring to the same object. This leads to redundant proposals, which can be eliminated by an algorithm known as non max suppression. The idea of non max suppression is to filter out all but the box with highest confidence score for each highly-overlapped bounding box cluster, making sure that a particular object is identified only once.

The algorithm can be summarized as follows:
* Given a list of proposals along with their confidence score, and a predefined overlap threshold
    * Initialize a list $L$ to contain bounding boxes.
    * Sort the list, denoted by $S$,  by confidence score in descending order
    * Iterate through $S$, at each iteration
        * Compute the overlap between the current bounding box and the remain bounding boxes in $S$
        * Suppress all bounding boxes that have the computed overlap above the predefined threshold hold from $S$
        * Discard the current box from $S$, then move it to $L$
    * Return $L$

<img src="https://i.imgur.com/Mh1L9XC.png" width="400"/>

### Region-based Convolutional Neural Network
Now we have feature map patches as regions ready for the next phase. Now one notable problem arises here is that those proposed regions are not in the same shape, which makes it difficult for neural network training. This is where we need RoI pooling layer to help construct fixed-size feature maps from those arbitrary-size regions.

#### RoI Pooling
To understand RoI pooling, let begin with a 2D example.
##### A 2D example
<!--  Given an input slice of arbitrary size, -->

No matter what the shape of the input slice is, a $2 \times 2$ RoI pooling layer always transform the input to the output of size $2 \times 2$ by
* Split the input into a $2 \times 2$ matrix of roughly equal regions
* Do max pooling on each region

like this figure below (given input of shape $4 \times 4$ or $5 \times 5$).


<!-- ![](https://i.imgur.com/oSbRTQf.png) -->

<img src="https://i.imgur.com/0Z6wlit.png" width="400"/>


##### RoI used in Faster RCNN

<img src="https://i.imgur.com/Clu7DyN.png" width="400"/>

#### Detection Network
<!-- Those fixed-size feature maps from RoI pooling are subsequently fed into the final classifier. -->

Those fixed-size feature maps from RoI pooling are then flattened and subsequently fed into a fully connected network for final detection. The net consists of $2$ fully connected layers of $4096$ neurons, followed by other $2$ sibling fully connected layers - one has $N$ neurons for classifying proposals and the other has $4*(N - 1)$ neurons for bounding box regression, where $N$ denotes the number of classes, including the background. Note that when a bounding box is classified as background, regression is unneeded. Hence, it makes sense that we only need $4*(N - 1)$ neurons for regression in total.

<img src="https://i.imgur.com/o05O9LM.png" width="500"/>

In our example, each $7\times7\times512$ feature map is fed to the detection net to produce the classification output has size of $4$, and the regression output has size of $12$.

#### Labeled data for FCNN
<!-- After non max suppresion step, we get
For each proposed region, the RPN predict  -->

##### Label for classification

Similar to the RPN, we make use of IoU metric to label data. Let $p$ denotes the overlap between a refined anchor box produced by RPN and its nearest ground-truth anchor box. For each anchor box we label as follows
* if $p < \text{min_cls}$, ignore it when training.
* if $\text{min_cls} \leq p < \text{max_cls}$, label it as background.
* if $p \geq \text{max_cls}$, label it as the class to which its nearest ground-truth box belongs.

```python
cfg.classifier_min_overlap = 0.1
cfg.classifier_max_overlap = 0.5
```

##### Label for bounding box regression

For regression, we also calculate the "ground-truth" deltas $(\color{red}{t_x^*, t_y^*, t_w^*, t_h^*})$ in the same fashion as those in RPN, but now based on each refined anchor box's coordinates from the RPN $(x_r, y_r, w_r, h_r)$ and its nearest ground-truth bounding box's coordinates $(\color{blue}{x^*, y^*, w^*, h^*})$.

<!-- <a id="loss-function"></a> -->
#### RCNN losses

##### 1. Regression Loss
```python
# regresssion loss for detection network
def class_loss_regr(num_classes, cfg):
    def class_loss_regr_fixed_num(y_true, y_pred):
        x = y_true[:, :, 4*num_classes:] - y_pred
        x_abs = K.abs(x)
        x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')
        return cfg.lambda_cls_regr * K.sum(y_true[:, :, :4*num_classes] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(cfg.epsilon + y_true[:, :, :4*num_classes])
    return class_loss_regr_fixed_num
```

##### 2. Classification Loss
```python
def class_loss_cls(cfg):
    def class_loss_cls_fixed_num(y_true, y_pred):
        return cfg.lambda_cls_class * K.mean(categorical_crossentropy(y_true[0, :, :], y_pred[0, :, :]))
    return class_loss_cls_fixed_num
```

<!-- <a id="training"></a>
## Training

<a id="detection"></a>
## Detection -->

__References__
1. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks ([arxiv](https://arxiv.org/abs/1506.01497))
