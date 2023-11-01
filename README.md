# semi-supervised-large-size-dataset

We had presented a methodology to harness unlabeled data when few annotated dataset is available. The framework described in [Semi-supervised Classification](https://github.com/ekchacon/semi-supervised-regular-size-datasets.git) is tested with regular size benchmark datasets such as the MNIST and Fashion.

In this project, we extend our study by applying the previous developed machine learning method for large datasets, specifically using the Quickdraw dataset.

# Quickdraw dataset

The Quickdraw bitmap dataset comprises grayscale images with dimensions of 28x28 pixels, encompassing 345 distinct drawing classes and a total of 50 million examples. For our experiments, we have selected a subset of 10 classes, containing 700,000 training examples and 25,000 test examples.

# the aim of the project

<!-- This content will not appear in the rendered Markdown -->

<!-- The main objective of this project is to defeat the challenges faced with data pre-processing, model feeding and the training time when working with large datasets, for example the Quickdraw dataset. In order to successfully do this we employed the tools TensorFlow library offers. -->

This project primarily aims to address challenges related to data pre-processing and training time, particularly in the context of large datasets like the Quickdraw dataset. To achieve this goal, we have leveraged the capabilities provided by the TensorFlow library.

<!-- For data pre-processing, the *data* TensorFlow module has a *Dataset class* that represents a potentially huge dataset. This class does not need to load the full dataset into memory but processes data in a streaming way avoiding ran out of GPU memory. We also used the TensorFlow Datasets to download and create a specific \textit{Dataset} instance (e.g. Quickdraw). -->

For data pre-processing, we utilized the *data* TensorFlow module, which includes a *Dataset class* designed to handle potentially large datasets efficiently. This class utilizes a streaming approach, mitigating GPU memory limitations by avoiding the need to load the entire dataset into memory. Additionally, we employed TensorFlow *Datasets* to facilitate the download and creation of specific Dataset instances, such as the Quickdraw dataset.

<!-- To accelerate the training process, we utilised two tools the Multi-worker startegy of tensorFlow and the linear-epoch gradual-warmup epoch. The former tool acelerate the training process by leveraging more than one GPU in different servers and the latter tool use the GPU memory afficiently by using larger batch sizes. -->

To enhance training efficiency, we employed two techniques: TensorFlow's Multi-worker strategy and the linear-epoch gradual-warmup epoch method. The former accelerates the training process by harnessing multiple GPUs across different servers, while the latter optimizes GPU memory efficiency through the use of larger batch sizes.

<!-- should i put the info of setup each method? put references -->

# Dataset configuration for experiments

# Experiment design

# Results for Quickdraw Bitmap dataset
