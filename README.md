# Semi-supervised large-size dataset

We had presented a methodology to harness unlabeled data when few annotated dataset is available. The framework described in [Semi-supervised Classification](https://github.com/ekchacon/semi-supervised-regular-size-datasets.git) is tested with regular size benchmark datasets such as the MNIST and Fashion.

In this project, we extend our study by applying the previous developed machine learning method for large datasets, specifically using the Quickdraw dataset.

# Quickdraw dataset

The Quickdraw bitmap dataset comprises grayscale images with dimensions of 28x28 pixels, encompassing 345 distinct drawing classes and a total of 50 million examples. For our experiments, we have selected a subset of 10 classes, containing 700,000 training examples and 25,000 test examples.

# The aim of the project

<!-- This content will not appear in the rendered Markdown -->

<!-- The main objective of this project is to defeat the challenges faced with data pre-processing, model feeding and the training time when working with large datasets, for example the Quickdraw dataset. In order to successfully do this we employed the tools TensorFlow library offers. -->

This project primarily aims to address challenges related to data pre-processing and training time, particularly in the context of large datasets like the Quickdraw dataset. To achieve this goal, we have leveraged the capabilities provided by the TensorFlow library and other methodologies.

<!-- For data pre-processing, the *data* TensorFlow module has a *Dataset class* that represents a potentially huge dataset. This class does not need to load the full dataset into memory but processes data in a streaming way avoiding ran out of GPU memory. We also used the TensorFlow Datasets to download and create a specific \textit{Dataset} instance (e.g. Quickdraw). -->

For data pre-processing, we utilized the *data* TensorFlow module, which includes a *Dataset class* designed to handle potentially large datasets efficiently. This class utilizes a streaming approach, mitigating GPU memory limitations by avoiding the need to load the entire dataset into memory. Additionally, we employed TensorFlow *Datasets* to facilitate the download and creation of specific Dataset instances, such as the Quickdraw dataset.

<!-- To accelerate the training process, we utilised two tools the Multi-worker startegy of tensorFlow and the linear-epoch gradual-warmup epoch. The former tool acelerate the training process by leveraging more than one GPU in different servers and the latter tool use the GPU memory afficiently by using larger batch sizes. -->

To enhance training efficiency, we employed two techniques: [TensorFlow's Multi-worker strategy](https://www.tensorflow.org/tutorials/distribute/multi_worker_with_ctl) and [the linear-epoch gradual-warmup epoch method](https://arxiv.org/pdf/1901.08256.pdf). The former accelerates the training process by harnessing multiple GPUs across different servers, while the latter optimizes GPU memory efficiency through the use of larger batch sizes.

<!-- should i put the info of setup each method? put references -->

# Dataset configuration for experiments

The original subsets were transformed into our subsets to facilitate experimentation. In the context of limited labeled examples, the number of training labeled examples was significantly reduced from 700,000 to 116,690 (16.67%). The remaining examples were allocated for pre-training or used as unlabeled data, amounting to 583,310 examples. The test subset remained unchanged.

| Dataset                          | Original subsets | Our subsets          |
| :------------------------------- | :--------------- | :------------------- |
| Quickdraw bitmap                 | 700k Training    | 583,310 Pre-training |
| (725,000 full)                   |                  | 116,690 Training     |
|                                  | 25k Test         | 25k Test             |

# Experiment design

<!-- In order to test our proposed method and other alternative approaches, we designed a set of experiments based on the 16.67% training examples, which is still gradually decresed to create different scenarios of few labeled training examples until reaching small amount of the 0.33% of the original training subset. Specifically, the amount of labeled examples start with 116,690 and end with 2310. -->

To evaluate our proposed method and alternative approaches, we conducted a series of experiments utilizing the 16.67% training examples. This percentage was systematically reduced to generate various scenarios involving a diminishing number of labeled training examples, ultimately reaching as low as 0.33% of the original training subset. The experiment commenced with 116,690 labeled examples and concluded with 2,310 labeled examples.

| Quickdraw | \%    |
|-----------|-------|
| 116690    | 16.67 |
| 105000    | 15.00 |
| 93310     | 13.33 |
| 81690     | 11.67 |
| 70000     | 10.00 |
| 58310     | 8.33  |
| 46690     | 6.67  |
| 35000     | 5.00  |
| 23310     | 3.33  |
| 11690     | 1.67  |
| 10500     | 1.50  |
| 9310      | 1.33  |
| 8190      | 1.17  |
| 7000      | 1.00  |
| 5810      | 0.83  |
| 4690      | 0.67  |
| 3500      | 0.50  |
| 2310      | 0.33  |

# Results for Quickdraw Bitmap dataset

The figure below provides a performance analysis across an increasing number of few labeled examples. Supervised learning consistently exhibits the lowest performance among these examples. A competition arises between semi-supervised layer-wise and self-training methods, with the former outperforming in the first half of few labeled examples, and the latter surpassing it in the the second half. Notably, our proposed method, self-training layer-wise, consistently outperforms all other methods throughout.

Our proposed method consistently outperforms the other methods in the range of 0.50% to 5.00%, which corresponds to scenarios with the smallest percentages of available examples. In the 0.33% category, it exhibits only a marginal lead over the semi-supervised layer-wise method. In the broader range of 6.67% to 16.67%, the method maintains solid performance. Overall, it achieves accuracy just below 80% at 0.33% and reaches approximately 90% at 16.67%.

<!-- ![image](https://github.com/ekchacon/semi-supervised-large-size-dataset/assets/46211304/e7ab53f6-fe14-4d47-8996-ce191d628e45) -->

<img width="770" alt="image" src="https://github.com/ekchacon/semi-supervised-large-size-dataset/assets/46211304/e7ab53f6-fe14-4d47-8996-ce191d628e45">

# Discussion

Real-world datasets, including seismic data  (Pratama, et al., [2022](https://recil.ensinolusofona.pt/bitstream/10437/9972/1/Television%20reshaped%20by%20big%20data.pdf)), medical records (Tiago, et al., [2022](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9893790)), and text corpora (Yang, et al., [2018](https://ieeexplore.ieee.org/abstract/document/8456138)), often comprise substantial amounts of unlabeled data alongside a limited number of labeled examples. To address the challenges associated with training neural networks on such datasets, we employed the extensive Quickdraw Bitmap dataset. A dataset is deemed large or massive when it encompasses approximately one million examples.

In conjunction with the utilization of the multi-worker strategy to accelarate the training process across multiple GPUs, we implemented the LEGW method to optimize memory utilization by employing larger batch sizes. The application of the LEGW method necessitated specific configurations for pre-training and fine-tuning across the four learning techniques under evaluation.

Our method demonstrates superior performance when applied to a large dataset across various dataset sizes. Notably, alternative methods fail to exhibit superior behavior compared to our method at any percentage of labeled dataset of Quickdraw. This consistent performance of the proposed method can be attributed to the extensive use of unlabeled data of the Quickdraw dataset during the pre-training stage, leading to enhanced generalization.

