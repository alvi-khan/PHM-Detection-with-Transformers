# An Evaluation of Transformer-Based Models in Personal Health Mention Detection

This repository contains the code used in ["An Evaluation of Transformer-Based Models in Personal Health Mention Detection"](https://ieeexplore.ieee.org/document/10054937), which has been accepted in [ICCIT 2022](https://iccit.org.bd/2022/).

## Prerequisites

- Python 3
- Anaconda
- CPU or NVIDIA GPU

## Environment Setup

To setup the environment, use the Anaconda Terminal to create a new conda environment and install the required Python libraries.

Conda Environment Setup:
```
conda create --name PHM python=3.9
```

Conda Environment Activation:
```
conda activate PHM
```

Library Installation:
```
pip install -r requirements.txt
```

## Dataset

The [Illness Dataset](https://github.com/p-karisani/illness-dataset) has been used to train this model. Place the `data.txt` file in the root folder of the project.

## Hyperparameters

The tunable hyperparameters can be found at the end of the `train.py` file.

### Models

The `params.pretrained_model` vairable specifies the model to use. The value should be specified as `params.models.MODEL_NAME`, e.g., `params.models.bert`. The models available are bert, roberta and xlnet.

## Training

To train the model, simply run the `train.py` file.
```
python train.py
```

## Evaluation

To evaluate a trained model seperately, run the `evaluate.py` file.
```
python evaluate.py
```
The tunable hyperparameters for the evaluation stage can be found at the bottom of the file.

Ensure that the filename for the model weights is `Model.bin` and place the file inside a folder with the same name as the pre-trained model for the weights. For example, the weights for the `BERT` model should be placed in a folder named `bert`.

The model weights used in the paper can be found under the [Releases](https://github.com/alvi-khan/PHM-Detection-with-Transformers/releases) section.

## Citation

If you find any of the code from this repository useful in your work, please consider citing the following paper:

```
@inproceedings{khan2022evaluation,
  title={An Evaluation of Transformer-Based Models in Personal Health Mention Detection},
  author={Khan, Alvi Aveen and Kamal, Fida and Nower, Nuzhat and Ahmed, Tasnim and Chowdhury, Tareque Mohmud},
  booktitle={2022 25th International Conference on Computer and Information Technology (ICCIT)},
  pages={1--6},
  year={2022},
  publisher={IEEE},
  url={https://ieeexplore.ieee.org/document/10054937}
}
```
