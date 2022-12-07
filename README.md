# future-object-detection
This is the official repository for ["Future Object Detection with Spatiotemporal Transformers"](https://arxiv.org/abs/2204.10321)

To get started, please check out the [Colab notebook](https://colab.research.google.com/drive/1Gt3jCvO6t7_7HsPYIOCkTr9kwVszcQZN)

## Pretrained Models

We provide pre-trained models on NuImages and NuScenes for a few prediction horizons. The AP50 scores are computed for **future object detection**, meaning that the model predicts all the future bounding boxes, and the score is the AP50 of the predicted bounding boxes against the ground truth of the future image. The exact training configurations can be found under `runs/`.

| Model                | Horizon  | Dataset  | AP50 Car | AP50 Pedestrian | Link                                                                              |
| -------------------- | -------- | -------- | -------- | --------------- | --------------------------------------------------------------------------------- |
| Spatiotemporal + IMU | 500ms    | NuImages | 43.5     | 15.5            | [google drive](https://drive.google.com/file/d/1BkKvCfrJYORvRtPRAr5Uonltc4Nf4IGa) |
| Spatiotemporal + IMU | 500ms    | NuScenes | 54.0     | 21.0            | [google drive](https://drive.google.com/file/d/10sEHjsEJZfT0-02ED4MG-Eki4s1D_zCb) |
| Spatiotemporal + IMU | 250ms    | NuScenes | 66.3     | 37.7            | [google drive](https://drive.google.com/file/d/1DzxE34NAWZzdM5L-Ru3Yyf6Fp0yvCDV9) |
| Spatiotemporal + IMU | 50/100ms | NuScenes | 71.1     | 44.9            | [google drive](https://drive.google.com/file/d/11JHOyPBaugXkNSags71xgaORlS1sIk20) |

## Installation and Setup

1. Clone the repository

2. Initialize the submodules:

    `git submodule update --init --recursive`

2. Install the requirements. We recommend using docker, and provide a dockerfile in the root directory.
    - To build the docker image, run `docker build -t future-od .`
    - But if you want to use your own environment, you can of course replicate the dockerfile in a conda environment.
    - We provide a requirements.txt file for your convenience, but note that it has not been used to generate the weights/results in the paper.

3. Download the dataset(s), and do one of the following:
    - Place the data directly under `./data`, e.g. `./data/nuimages/`.
    - Or create a symlink to the data directory, e.g. `ln -s /path/to/nuimages ./data/nuimages`.
    - Or bind the data directory to the container, e.g. `docker run -v /path/to/nuimages:/workspace/data/nuimages ...`
    - Or update the `config.py` file to point to the desired data directory.

4. (NuScenes only) Perform additional NuScenes setup
    - Download and extract the CAN bus data from [here](https://www.nuscenes.org/download). It should end up under ./data/nuscenes/can_bus
    - Generate 2d projections by running [export_2d_annotations_as_json.py](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/scripts/export_2d_annotations_as_json.py), e.g. `python export_2d_annotations_as_json.py --version=v1.0-mini --dataroot=./data/nuscenes`

5. (optional) Download a pretrained model, and place it under `./checkpoints`. Links are provided above.

6. (optional) Get familiar with the model by running the demo notebook

    `jupyter notebook demo.ipynb`

## Training / Evaluation

There are a number of pre-defined training scripts under `./runs`. You can run them with the following command:

    PYTHONPATH=.:./ConditionalDETR python runs/<train_script>

There are also a corresponding set of evaluation scripts, if you only want to evaluate an existing model (e.g. one of our pretrained models):

    PYTHONPATH=.:./ConditionalDETR python runs/eval/<eval_script>

We support multi-gpu training with `torch.distributed.launch`. For example, to train a model on 8 gpus, run:

    python3 -u -m torch.distributed.launch --nproc_per_node=8 --master_port=$RANDOM <scrip_name> <script_args> --distributed

Note on Weights & Biases. By default we enable logging to W&B, but this requires that you have an account and expose your API key as an environment variable. If you don't want to use W&B, you can disable it by setting `--disable_wandb` in the training/evaluation scripts.

## Singularity + Slurm -- highly specific and optional :)

Furthermore, we provide an example script for training with 4 gpu:s in a slurm+singularity setting. You can run it with:

    ./slurm.sh <script_name> <script_args>

 Note that you need to have a .sif file in the current directory, which is a singularity image containing the code and all dependencies. You can create one with the following command:

    singularity build future-od.sif docker://future-od:latest
