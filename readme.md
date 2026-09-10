### This Code is based on the work of Jahn, T., Chemseddine, J., Hagemann, P., Wald, C., & Steidl, G. (2025). Trajectory generator matching for time series. arXiv preprint arXiv:2505.23215.

### How to run
Run `conda env create -f environment.yml && conda activate tgm-env` to install dependencies.

Rename the root project folder to `tgm`. (Otherwise, the absolute package style import logic will not work.) 

Now, all scripts `script_*.py` (located in the subfolder _scripts_) should be executable and train the model specified in the respective `config.model` file on the dataset specified in the scripts name. 
The scripts differ mainly with respect to the data set which is loaded/created. Note that stock data creation might take a while (especially on cpu) and no messages are printed until it is completed. 
In contrast `script_spiral.py` creates a simple toy dataset and should start training (and printing output) almost immediately.

If `cuda` is not available, change the `device` option in the respective `config.model` and `config.train` file to `cpu`.

When adding new scripts: a) run new scripts as modules or b) import `_pathfix.py` at the top of new scripts to add the parent folder of the project root `tgm` to the path. (Otherwise, the absolute package style import logic will not work.)

### On the code structure

The interesting code lies in the subfolders _models_ and _train_, implementing the proposed models, training and generation procedure.
For consistency with the previous implementation, the model parameter eta from the paper has been replaced by a parameter sigma=eta² throughout the code.

### The config files

Each config file consists of three parts: `config.data` specifies the data set, `config.model` specifies the model, `config.train` specifies training parameters. Unfortunately, this clean separation was not kept entirely.
Therefore, modifications of `config.data` currently require corresponding modifications in `config.model` and `config.train` (and the other way round). A list of these dependecies, and how they should be resolved follows:

- Parameter `config.model.data_dim` needs to align with and should be infered from `config.data`.

- Parameters `config.train.val_trajectory_length/x_start/t_start/t_end` need to align with and should be infered from `config.data`.

- Parameter `config.train.no_bridges` needs to align with and should be infered from subsampling rate `config.data.T_sub`. 

- There are two different keyword arguments `device`, one in `config.model` and another in `config.train`, which have to coincide.

### Warning

The code is only partially commented, and some parts are rather messy. In particular, all the `eval_*.py` files have been written almost entirely by ChatGPT with a focus on fast results. 
Luckily they can be ignored entirely unless grid searches need to be evaluated systematically. For training the models and playing around with some hyperparameters, the `script_*.py` scripts should be sufficient.

Note that in these scripts some plotting options are still commented out, since they caused trouble on the cluster.

Some options may seem confusing since they were designed with the extension to real world data in mind, where no common time grid is available anymore. 
E.g. the booleans `config.train.regular_val_set_provided` and `config.train.irregular_val_set_provided` can be ignored for now (the first is always `True`, the second is ignored).

### TODOs

- load only train, val and test split with dataset specific method, and make methods `subsample` and `make_loader` dataset independent
- Reorganize the config files.
- Add comments and proper whitespace.
