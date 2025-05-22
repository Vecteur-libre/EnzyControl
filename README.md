# EnzyControl

EnzyControl is a method that enables functional and substrate-specific control in enzyme backbone generation. This repo is built based on [protein-frame-flow](https://github.com/microsoft/protein-frame-flow).

## Installation

Set up your conda environment.

```bash
# Conda environment with dependencies.
conda env create -f EnzyControl.yml

# Activate environment
conda activate EnzyControl

# Manually need to install torch-scatter.
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu117.html

# Install local package.
# Current directory should be protein-frame-flow/
pip install -e .
```



## Dataset

Our dataset is available at [this](https://zenodo.org/records/15462173). You can download and put it in the folder `"/metadata/"`.



## Inference

To run sampling, we specify the settings in `config/inference_enzyme.yaml`. You need download our [checkpoint](https://drive.google.com/file/d/1loybOsNeAHSXVB4jCt8UqLJkYq_yLdME/view?usp=drive_link), and put it in the folder `"/weights/"`, then run the following command.

```python
# Single GPU
python -W ignore experiments/inference_se3_flows.py -cn inference_enzyme

# Multiple GPU
python -W ignore experiments/inference_se3_flows.py -cn inference_enzyme inference.num_gpus=2
```





## Training

All training flags are in `configs/base.yaml`. Below is explanation-by-example of the main flags to change. Note you can combine multiple flags in the command.

If you want to retrain our model, you need to download our [dataset](https://zenodo.org/records/15462173), preprocess it into the same format as the example data, and run the following command. And you also need to download the pretrained model from [here](https://drive.google.com/file/d/1wUdzM9Yn1M1PIAuxPwGpsbNk8FlQMfcQ/view?usp=drive_link), putting it in the folder `"/weights/"`

```python
# Train on EnzyBind
python -W ignore experiments/train_se3_flows.py data.dataset=enzyme data.task=inpainting

# Training with larger batches. Depends on GPU memory
python -W ignore experiments/train_se3_flows.py data.dataset=enzyme data.task=inpainting data.sampler.max_num_res_squared=600_000

# Training with more GPUs
python -W ignore experiments/train_se3_flows.py data.dataset=enzyme data.task=inpainting experiment.num_devices=4
```

