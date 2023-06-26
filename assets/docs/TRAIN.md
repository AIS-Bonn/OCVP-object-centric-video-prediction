# Training

We provide our entire pipeline for training a SAVi model for object-centric video decomposition, as well as for training our object-centric video predictor modules.


## Train SAVi Video Decomposition Model

**1.** Create a new experiment using the `src/01_create_experiment.py` script. This will create a new experiments folder in the `/experiments` directory.

```
usage 01_create_experiment.py [-h] -d EXP_DIRECTORY [--name NAME] [--config CONFIG]

optional arguments:
  -d EXP_DIRECTORY, --exp_directory EXP_DIRECTORY Directory where the experiment folder will be created
  --name NAME           Name to give to the experiment
```



**2.** Modify the experiment parameters located in `experiments/YOUR_EXP_DIR/YOUR_EXP_NAME/experiment_params.json` to adapt to your dataset and training needs.
You provide two examples for training SAVi on the [Obj3D](https://github.com/AIS-Bonn/OCVP-object-centric-video-prediction/blob/master/experiments/Obj3D/experiment_params.json) and [MOVi-A](https://github.com/AIS-Bonn/OCVP-object-centric-video-prediction/blob/master/experiments/MOViA/experiment_params.json) datasets.



**3.** Train SAVi given the specified experiment parameters:

```
usage: 02_train_savi.py [-h] -d EXP_DIRECTORY [--checkpoint CHECKPOINT] [--resume_training]

optional arguments:
  -d EXP_DIRECTORY, --exp_directory EXP_DIRECTORY
                        Path to the experiment directory
  --checkpoint CHECKPOINT
                        Checkpoint with pretrained parameters to load
  --resume_training     For resuming training
```


#### Example: SAVi Training

Below we provide an example of how to train a new SAVi model:

```
python src/01_create_experiment.py -d new_exps --name my_exp
python src/02_train_savi.py -d experiments/new_exps/my_exp
```


## Train an Object-Centric Video Prediction Model

Training an object-centric video prediction requires having a pretrained SAVi model. You can use either our provided pretrained models, or you can train your own SAVi video decomposition models.


**1.** Create a new predictor experiment using the `src/src/01_create_predictor_experiment.py` script. This will create a new predictor folder in the specified experiment directory.

```
usage: 01_create_predictor_experiment.py [-h] -d EXP_DIRECTORY --name NAME --predictor_name PREDICTOR_NAME

optional arguments:
  -d EXP_DIRECTORY, --exp_directory EXP_DIRECTORY
                        Directory where the predictor experimentwill be created
  --name NAME           Name to give to the predictor experiment
  --predictor_name PREDICTOR_NAME
                        Name of the predictor module to use: ['LSTM', 'Transformer', 'OCVP-Seq', 'OCVP-Par']
```


**2.** Modify the experiment parameters located in `experiments/YOUR_EXP_DIR/YOUR_EXP_NAME/YOUR_PREDICTOR_NAME/experiment_params.json` to adapt the predictor training parameters to your dataset and training needs.
We provide examples for each predictor module on the Obj3D and MOVi-A datasets. For instance:
 - [LSTM on Obj3D](https://github.com/AIS-Bonn/OCVP-object-centric-video-prediction/blob/master/experiments/Obj3D/Predictor_LSTM/experiment_params.json)
 - [Transformer on Obj3D](https://github.com/AIS-Bonn/OCVP-object-centric-video-prediction/blob/master/experiments/Obj3D/Predictor_Transformer/experiment_params.json)
 - [OCVP-Par on MOVi-A](https://github.com/AIS-Bonn/OCVP-object-centric-video-prediction/blob/master/experiments/MOViA/Predictor_OCVPPar/experiment_params.json)


 **3.** Train your predictor  given the specified experiment parameters and a pretrained SAVi model:

 ```
 usage: 04_train_predictor.py [-h] -d EXP_DIRECTORY [--checkpoint CHECKPOINT] [--resume_training] -m SAVI_MODEL --name_predictor_experiment
                             NAME_PREDICTOR_EXPERIMENT

optional arguments:
  -h, --help            show this help message and exit
  -d EXP_DIRECTORY, --exp_directory EXP_DIRECTORY
                        Path to the father exp. directory
  --checkpoint CHECKPOINT
                        Checkpoint with predictor pretrained parameters to load
  --resume_training     Resuming training
  -m SAVI_MODEL, --savi_model SAVI_MODEL
                        Path to SAVi checkpoint to be used during in training or validation, from inside the experiments directory
  --name_predictor_experiment NAME_PREDICTOR_EXPERIMENT
                        Name to the directory inside the exp_directory corresponding to a predictor experiment.
 ```

#### Example: Predictor Training

Below we provide an example of how to train an object-centric predictor given a pretrained SAVi model. This example continues the example above

```
python src/01_create_predictor_experiment.py \
  -d new_exps/my_exp \
  --name my_OCVPSeq_model \
  --predictor_name OCVP-Seq

python src/04_train_predictor.py \
  -d experiments/new_exps/my_exp
  --savi_model checkpoint_epoch_final.pth
  --name_predictor_experiment my_OCVPSeq_model
```

## Training Time

The table below summarizes the amout of epochs, hours and iterations (batches) required to train both the SAVi scene parsing module, as well as our OCVP-Seq predictor.
These values correspond to experiments trained with an NVIDIA A6000 with 48Gb.

| Dataset | Model | Iters.  | Epochs | Time |
| --- | --- | --- | --- | --- |
| Ojb3D | SAVi | 100k | 2000 | 26h |
|Obj3D | OCVP-Seq | 70k | 1500 | 34h |
|MOVi-A |SAVi | 150k | 2000| 120h |
|MOVi-A | OCVP-Seq | 50k | 300 | 18h |


## Further Comments

 - You can find examples of some experiment directories under the `experiments` directory.

 - The training can be monitored using Tensorboard.
   To launch tensorboard,
```
tensorboard --logdir experiments/EXP_DIR/EXP_NAME --port 8888
```

 - In case of questions, do not hesitate to open an issue or contact the authors at `villar@ais.uni-bonn.de`
