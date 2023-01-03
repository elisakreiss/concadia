# Context-compatible ResNet-LSTM and DenseNet-LSTM

Proposed in [Xu et al., 2015](https://arxiv.org/abs/1502.03044), ResNet-LSTM generates a label token-by-token, using an LSTM decoder. The input to the LSTM is a concatenated representation of the image using a pretrained ResNet, and the context using BERT embeddings. 

To incorporate context, we took inspiration from the importance of attention mechanisms on the input image as used in state-of-the-art “captioning” models ([Xu et al., 2015](https://arxiv.org/abs/1502.03044); [Lu et al., 2017](https://arxiv.org/abs/1612.01887); [Anderson et al., 2018](https://arxiv.org/abs/1707.07998)) and extended them to the context input. To ensure the same number of trainable parameters between the with-context vs. no-context models, the no-context models received all-ones vectors in place of context embeddings.

DenseNet-LSTM follows the same structure as the ResNet-LSTM but leverages pretrained DenseNet features instead.

## Dev setup

First, set up a (free) account on https://wandb.ai/. We use this tool to visualize training progress and manage checkpoints.

We have included 2 virtual environment setup files, depending on the hardware. For older GPU types (e.g., NVIDIA T4), use `environment.yml`; for newer GPU types (e.g., NVIDIA A10G), use `g5_environment.yml`. In our experience, the former is compatible with AWS g4dn instances, and the latter is compatible with AWS g5 instances.

After setting up and activating the virtual env, execute the following commands:
```bash
pip install git+https://github.com/Maluuba/nlg-eval.git@master
conda install h5py
nlg-eval --setup
```
And you should be good to go!

## Model experiments

### Download raw data

First, follow the instructions [here](https://github.com/elisakreiss/concadia/blob/master/dataset/README.md#download-concadia) to download the two files that make up the raw dataset: `wiki_split.json` and `resized.zip`. (The instructions page also contains more details on what these files contain.) In addition,
- download `wiki_split_randomfilename.json` [here](https://drive.google.com/file/d/1i4bcYLcjWSVRuVr5zMoLOl47RUTw1aoo/view?usp=sharing): this file is identical to `wiki_split.json`, except the image corresponding to each text label is scrambled
- download `wiki_split_randomcontext.json` [here](https://drive.google.com/file/d/15hhpaciXoNy_EpYOFfYngriL4ygnd5Hs/view?usp=sharing): this file is identical to `wiki_split.json`, except the nearby paragraph corresponding to each image is scrambled.

Store them in the following structure:
- `{DATA_DIR}`
    - `wiki_split.json`
    - `wiki_split_randomfilename.json`
    - `wiki_split_randomcontext.json`
    - `{IMAGE_SUBDIR}`
        - `resized/` (output of decompressing `resized.zip`)

### Experiment 1: Description generation vs. caption generation using image alone
The argument structure follows the pattern (1) text type to generate (e.g., description), (2) text type that functions as additional context (e.g., caption), (3) if something should be randomized for a baseline condition (e.g., none). 

To train a model to generate **description**, run the following command after `cd code/`:
```
# data generation
python create_input_files.py description caption --randomized none \
--root_dir {DATA_DIR} \
--image_subdir {IMAGE_SUBDIR}

# model training
python train.py description caption none --blank_context \
--data_dir $DATA_DIR \
--output_dir $OUTPUT_DIR
```
To train a model to generate **caption**, use
```
# data generation
python create_input_files.py caption description --randomized none \
--root_dir {DATA_DIR} \
--image_subdir {IMAGE_SUBDIR}

# model training
python train.py caption description none --blank_context
--data_dir $DATA_DIR \
--output_dir $OUTPUT_DIR
```
`OUTPUT_DIR` is a directory of your choice, where eval results on the dev set and intermediate checkpoints are saved.

Note that you still provide a context in the command (i.e., caption and description, respectively) but the context will be set to blank due to the `--blank_context` flag. This ensures that the number of parameters is not a confound in the with vs. without context comparison.

By default, the image encoder is a ResNet. If you would like to use a DenseNet instead, append `--image_encoder_type densenet` to the command.

After each epoch, evaluation on the dev set is performed using greedy decoding.

#### Sub-experiment discussed in Section 4.5

We find that the model does better on the description generation task than the caption generation task. There are two potential explanations:
1. Information from the image alone has a closer connection to the description than to the caption.
2. Descriptions have certain image-independent linguistic properties that make them easier to learn (see Section 4.5 for details of these properties). 

If the latter were true, then the model would be better at description generation even if image-label mappings are shuffled. Thus, we conduct an experiment where we train models on a version of the dataset with shuffled image-label pairings. We find that when image-label pairings are shuffled, the model achieves similar performance on description generation vs. caption generation, hence invalidating the 2nd hypothesis.

To reproduce this experiment, use the commands detailed above, while replacing `none` with `img` (to indicate that `img` is shuffled)

#### A note on hardware
We used the same hyperparameters across description generation experiments and caption generation experiments. We noticed that caption generation experiments tend to be more memory-intensive. In our experience, with our choice of hyperparameters, the description models could be trained on an AWS g4dn instance (with 16G GPU memory), whereas the caption models had to be trained on an AWS g5 instance (with 24G GPU memory).

### Experiment 2: Generation using image and additional text as context

To train a model to generate **description** using _caption_ as context, run the following command
```
# data has been generated in Exp. 1
python train.py description caption none \
--data_dir $DATA_DIR \
--output_dir $OUTPUT_DIR
```
To train a model to generate **description** using _the nearby paragraph_ as context, run the following command
```
# data generation
python create_input_files.py description context --randomized none \
--root_dir {DATA_DIR} \
--image_subdir {IMAGE_SUBDIR}

# model training
python train.py description context none \
--data_dir $DATA_DIR \
--output_dir $OUTPUT_DIR
```
To train a model to generate **caption** using _description_ as context, use
```
# data has been generated in Exp. 1
python train.py caption description none \
--data_dir $DATA_DIR \
--output_dir $OUTPUT_DIR
```
To train a model to generate **caption** using _the nearby paragraph_ as context, use
```
# data generation
python create_input_files.py caption context --randomized none \
--root_dir {DATA_DIR} \
--image_subdir {IMAGE_SUBDIR}

# model training
python train.py caption context none \
--data_dir $DATA_DIR \
--output_dir $OUTPUT_DIR
```

Like before, you can swap out the image encoder. The context encoder is a BERT model by default. If you would like to use a RoBERTa model instead, append `--context_encoder_type roberta` to the command.

Additionally, to ensure that not just language in general accounts for potential performance gains, we provide control conditions where the model receives paragraphs from other images. To train a model with scrambled paragraphs, run the following:
```
# data generation
python create_input_files.py {description/caption} context --randomized context \
--root_dir {DATA_DIR} \
--image_subdir {IMAGE_SUBDIR}

# model training
python train.py {description/caption} context context \
--data_dir $DATA_DIR \
--output_dir $OUTPUT_DIR
```

### Evaluating on the test set
After training all models, use the following command to run beam search (with beam size of 5) on the test set to generate sequences for the best checkpoint from each run:
```
python test_eval_beamsearch.py \
--data_dir $DATA_DIR \
--runs_dir $OUTPUT_DIR \
--wandb_project ${WANDB_USERNAME}/concadia
```
All generated sequences will be saved in a directory called `paper_runs_perf/`. This repo contains the sequences obtained from our models.

To compute metrics based on the generated sequences:
```
cd paper_runs_perf/
python beamsearch_perf.py
```
For the results from our models, see `paper_runs_perf/full_performance_data.csv`.
