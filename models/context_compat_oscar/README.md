# Context-compatible OSCAR(VinVL)

OSCAR(VinVL) is a pretrained Transformer-based vision-language model that achieves state-of-the-art results on many tasks via finetuning ([Zhang et al., 2021](https://arxiv.org/abs/2101.00529)). For each image, the VinVL pretrained visual feature extractor provides visual features, i.e., vector embeddings of object-enclosing regions, and object tags in text. To obtain a unified representation of the image–text pair, the OSCAR model concatenates the BERT embeddings of the text and object tags with the visual features ([Li et al., 2020](https://arxiv.org/abs/2004.06165)). 

We supply context by appending it to the object tags before the BERT encoding.

This implementation is based on [microsoft/Oscar](https://github.com/microsoft/Oscar).

## Dev setup
Check out [INSTALL.md](INSTALL.md) for setup instructions.

## Data

Download OSCAR(VinVL)-compatible Concadia [here](https://drive.google.com/file/d/1oUulBH1smxEOrZwwJj3aOED654htTVJk/view?usp=sharing).

### Generation process
In case you would like to use a different dataset for experimentation, feel free to work off of [this bash script](https://github.com/feifang24/scene_graph_benchmark/blob/main/extract_features.sh), which we used to preprocess Concadia into OSCAR(VinVL)-compatible train/dev/test data. Note that this is a compute-intensive process, since preprocessing requires running object detection inference on all images in Concadia. 

## Pretrained checkpoint

We use the publicly released pretrained Oscar-VinVL model as starting checkpoint to train our description generation models and caption generation models. Download the **"pretrained model checkpoint"** from the Oscar repo [here](https://github.com/microsoft/Oscar/blob/master/VinVL_MODEL_ZOO.md#image-captioning-on-coco). 

Checkpoints that have been finetuned on the image captioning task have also been released, but we did not opt to use those since they are likely biased towards caption generation.

Our experiments are conducted using the base checkpoint rather than the large checkpoint. 

## Model experiments

Use the following command (which we will refer to as "main command") to finetune a description/caption generation model. In the following subsections, we will explain how to set the arguments.
```
python oscar/run_captioning.py $LABEL_COND $CONTEXT_COND $RANDOMIZED \
    --data_dir $DATA_DIR \
    --model_name_or_path $PRETRAINED_MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --do_train \
    --do_test \
    --do_lower_case \
    --add_od_labels \
    --learning_rate 3e-5 \
    --tie_weights \
    --freeze_embedding \
    --label_smoothing 0.1 \
    --drop_worst_ratio 0.2 \
    --drop_worst_after 20000 \
    --num_workers 1 \
    --num_train_epochs 20
```
- `DATA_DIR`: where OSCAR(VinVL)-compatible Concadia is stored.
- `PRETRAINED_MODEL_PATH`: the folder where pretrained Oscar-VinVL checkpoint is stored.
  - By default, the downloaded checkpoint contains one nested folder after unzipping. Set this to the innermost folder. In other words, this folder should contain files such as `config.json` and `pytorch_model.bin`. 
- `OUTPUT_DIR`: The output directory to save intermediate checkpoints, as well as results on the validation set and test set.

### Experiment 1: Description generation vs. caption generation using image alone

The argument structure follows the pattern (1) text type to generate (e.g., description), (2) text type that functions as additional context (e.g., caption), (3) if something should be randomized for a baseline condition (e.g., none).

Since we do not provide additional context in this experiment,
- To train a model to generate **description**, run the command below, then run the main command:
```
export LABEL_COND=description
export CONTEXT_COND=none
export RANDOMIZED=none
```
- To train a model to generate **caption**, run the command below, then run the main command:
```
export LABEL_COND=caption
export CONTEXT_COND=none
export RANDOMIZED=none
```

After each epoch, evaluation on the dev set is performed using greedy decoding.

At the end of training, evaluation on the test set is performed (with beam size of 5) using the best checkpoint during training (based on performance on dev set).


### Experiment 2: Generation using image and additional text as context

To train a model to generate **description** using _caption_ as context:
```
export LABEL_COND=description
export CONTEXT_COND=caption
export RANDOMIZED=none
```

To train a model to generate **description** using _the nearby paragraph_ as context:
```
export LABEL_COND=description
export CONTEXT_COND=context
export RANDOMIZED=none
```

To train a model to generate **caption** using _description_ as context, use
```
export LABEL_COND=caption
export CONTEXT_COND=description
export RANDOMIZED=none
```

To train a model to generate **caption** using _the nearby paragraph_ as context, use
```
export LABEL_COND=caption
export CONTEXT_COND=context
export RANDOMIZED=none
```

Additionally, to ensure that not just language in general accounts for potential performance gains, we provide control conditions where the model receives paragraphs from other images. To train a model with scrambled paragraphs, use the following args:
```
export LABEL_COND={caption/description}
export CONTEXT_COND=context
export RANDOMIZED=context
```

### Test performance

As mentioned earlier, evaluation on test set is automatically performed at the end of training. This outputs not only the text sequences generated by the model for each test example, as well as the overall metrics. 

To aggregate the performance across runs, use the following command:
```
python aggregate_test_performance.py $OUTPUT_DIR
```
This outputs a file named `full_performance_data.csv`, which details the test metrics from each run.

#### Performance of our models

Download [this .zip file](https://drive.google.com/file/d/1Y1bcLwGmMg7lYoYSiGfGb4J4LMCWw6Ig/view?usp=sharing) if you would like to inspect our model's predictions on the test set along with metrics. After unzipping, the directory is structured as follows:

- `concadia_oscar_runs`
  - experiment type in the format of `{LABEL_COND}_{CONTEXT_COND}_{RANDOMIZED}`
    - run id in the format of datetime
      - `test_eval`
        - `pred.vinvl_data.test.best.beam5.max52.odlabels.eval.json`: metrics
        - `pred.vinvl_data.test.best.beam5.max52.odlabels.debug.json`: generated sequence for each example in the test set
