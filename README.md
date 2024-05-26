## Code for "When does compositional structure yield compositional generalization? A kernel theory."

NOTE: Due to upload limitations on OpenReview, this repository does not contain the collated files used to generate the images. To access the repository including those files, please use the anonymous Google Drive link provided here: https://drive.google.com/drive/folders/1sXCxfKETordLgCxzKdwO3qhzKxtX31Bc?usp=sharing

### Experiments

We specify the conda environment in 'environment.yml'.

The main script is given by ``train.py``, where the following aspects of the experiment can be specified separately:
- The task structure
- The embedding, i.e. how should inputs be presented?
- The model that is trained
- The analysis that is run on the model.

We provide a configuration file for each experiment presented in the paper. Running these configuration files results in a multirun folder, which we collated into a set of files using the ``collate.py`` script. We provide the corresponding scripts but note that the folders would have different names upon replication and the scripts would have to be changed accordingly. Note that the script needs to be run as

``python train.py --config-path configs --config-name <config_file_name> -m``

In the following, sorted after task, we'll first specify the folder in which the corresponding data can be found, together with a short description. We'll the list the config files necessary to reproduce the data.

- Symbolic addition
  - '19_addition_mnist': Behavior of ConvNets trained on MNIST version of symbolic addition (plotted in Fig. 4 and Fig. 9).
    - addition-mnist-new.yaml
    - addition-mnist-new-2.yaml
    - addition-mnist-dispersed.yaml
  - 'C18_addition_resnet': Behavior of ResNets trained on CIFAR-10 version of symbolic addition (Fig. 4 and Fig. 10)
    - addition-cifar-resnet-improved.yaml
    - addition-cifar-resnet-improved-2.yaml
  - 'C16_vit_cifar_improved': Behavior of Vision Transformers trained on CIFAR-10 version of symbolic addition (Fig. 4 and Fig. 10)
    - addition-cifar-vit-improved-2.yaml
  - 'D1_addition_dispersed': Behavior of kernel models on randomly sampled dispersed training sets (Fig. 7)
    - addition-kernel-dispersed-2.yaml
  - '18_addition_rich': Behavior of rich networks on symbolic addition (Fig. 8)
    - addition-rich.yaml
  - 'C13_addition_deep': Behavior of deep networks on symbolic addition (Fig. 8)
    - addition-deep.yaml
- Context dependence
  - '12_cdm_kernel': Behavior of kernel models on CD-1, CD-2, and CD-3 (as plotted in Fig. 3).
    - cdm-kernel-1.yaml
    - cdm-kernel-2.yaml
    - cdm-kernel-3.yaml
  - '22_cdm_mnist': Behavior of ConvNets trained on MNIST version of CD-1, CD-2, and CD-3 (as plotted in Fig. 4 and Fig. 12)
    - mnist-cdm-1.yaml
    - mnist-cdm-2.yaml
    - mnist-cdm-3.yaml
  - 'C15_cdm_cifar_resnet_improved': Behavior of ResNets trained on CIFAR-10 version of context dependence (plotted in Fig. 4 and Fig. 13)
    - cifar-cdm-resnet-improved-1.yaml
    - cifar-cdm-resnet-improved-2.yaml
    - cifar-cdm-resnet-improved-3.yaml
  - 'C19_cdm_cifar_vit': Behavior of Vision Transformers trained on CIFAR-10 version of context dependence (plotted in Fig. 4 and Fig. 13)
    - cifar-cdm-vit-improved-1.yaml
    - cifar-cdm-vit-improved-2.yaml
    - cifar-cdm-vit-improved-3.yaml
  - 'C14_cdm_deep': Behavior of deep networks on context dependence (plotted in Fig. 11)
    - cdm-deep-1.yaml
    - cdm-deep-2.yaml
    - cdm-deep-3.yaml
  - '25_cdm_rich': Behavior of rich networks on context dependence
    - cdm-rich-1.yaml
    - cdm-rich-2.yaml
    - cdm-rich-3.yaml
- Transitive equivalence
  - '23_te_rich': Behavior of rich and lazy ReLU networks on transitive equivalence (as plotted in Fig. 5).
    - transitive-equivalence.yaml
  - '24_te_mnist': Behavior of ConvNets trained on MNIST version of transitive equivalence (as plotted in Fig. 5).
    - equivalence-mnist.yaml
  - 'C17_equivalence_deep': Behavior of deep networks on transitive equivalence (plotted in Fig. 14)

### Figures

We then create the figures using the `.Rmd` files in 'analysis'. We post-processed some of them for their final presentation in the paper.
