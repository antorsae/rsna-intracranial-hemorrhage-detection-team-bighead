## Before cloning this repo

Make sure you have git-lfs installed:

```
sudo apt install git-lfs
git lfs install
```

## Start here

Directory tree:

```
.
├── data
│   ├── unzip
│   │   ├── stage_2_test_images
│   │   └── stage_2_train_images
│   ├── predictions
├── env
└── models
```

Set up conda env with:

```
conda env create -n ihd -f=env/tfgpu.yml
conda activate ihd
```

Then run `jupyter-notebook` from the repo's root dir:

```
jupyter notebook --no-browser --NotebookApp.iopub_msg_rate_limit=10000000000
```

## Steps to reproduce submission:

1. Start with NBs:
 1. `0-preprocess-generate_csvs.ipynb`
 2. `1-preprocess-brain_norm.ipynb`
 3. `2-preprocess-pickle.ipynb`

... to pregenerate dcm metadata + diagnosis pivot tables + various pickles. 

For convenience we've already included these in the git repository so, altenatively, you can skip to step 2.

2. Train (level 1) L1 models:

a. fastai v1 library: `3a-L1-train-and-generate-predictions-fastai_v1.ipynb` to train the following architectures:
* `resnet18`
* `resnet50`
* `resnet34`
* `resnet101`
* `densenet121`

For each architecture we need to train 5 models (each model for each of 5 different folds). 

All the variables must be set in cell #4, e.g.

```
model_fn = None
SZ = 512
arch = 'resnet34'
fold=0
n_folds=5
n_epochs = 4
lr = 1e-3
n_tta = 10

#model_fn = 'resnet34_sz512_cv0.0821_weighted_loss_fold1_of_5'

if model_fn is not None:
    model_fn_fold = int(model_fn[-6])-1
    assert model_fn_fold == fold
```
  
b. fastai v2 library to train subdural-focused models: same instructions as a) but use file `3b-L1-train-and-generate-predictions-fastai_v2.ipynb`
* `resnet18`
* `resnet34`
* `resnet101`
* `resnext50_32x4d`
* `densenet121`

To train models from scratch and generate test and OOF predictions, you need to:
- Set arch to each of the archs above and train the model for each fold (set `FOLD` variable from 0 to 4 to train each fold. You NEED to train all 5 folds).
- Comment the second `model_fn` instance (this is used if you need to fine-tune an existing model)
- Execute all code except for the final section which builds CSV to send to submit single-model predictions to Kaggle (which we do NOT want to do at this stage).

The code for fastai v1 allocates batch size and finds LR dynamically, but in the fastai v2 version you need to specify your GPU memory in cell #4 as well.

For convenience: since training models takes a long time, we are providing trained models and test and OOF predictions so, altenatively, you can skip to step 3.

3. Train (level 2) L2 models and generate submission: With all the models (5 models per arch) trained and predictions in `./data/predictions`, run `4-L2-train-and-submit.ipynb` to generate the final predictions/submission.

## Resources

* Dataset visualizer: https://rsna.md.ai/annotator/project/G9qOnN0m/workspace
