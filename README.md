# Diffusion Experiments
This is the repo for implementation of our diffusion-based study.


# Reproduciblity

Experiments were done on the Google Colab, and ".py" files were organized according to to run with Google Colab. Our ".ipynb" and ".py" files can be found in related directories. Commands for running .py files can be found in ".ipynb" files. Config files can be found in the "configs" directories. For experiments of GCN/GAT, config files were separated into training, thresholding, and tuning run modes.

You can use the %cd command on your Google Colab notebook to go to the experiment folder.

%cd /content/MY_PATH_DIFFUSION_EXPERIMENT_FOLDER

You can use the %run command on your Google Colab notebook to run .py files with config files. The name of the config file must be located after the name of the ".py" file.

For example, for Text GCN based experiments, This command is for experiments with method T-DIFF NORM REG on Interest dataset.  

%run textgcn-t-methods-experiments.py configs/config-interest-t-diff-norm-reg.ini

Another example for GCN/GAT based experiments, This command is for experiments with method DIFF NORM GCN on the Interest dataset. 

%run gcn-gat-experiments.py configs/training-configs/config-interest-diff-norm-gcn-training.ini
