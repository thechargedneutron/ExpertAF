# ExpertAF: Expert Actionable Feedback from Video

Code implementation of CVPR 2025 paper 'ExpertAF: Expert Actionable Feedback from Video'.

[![arXiv](https://img.shields.io/badge/arXiv-2408.00672-00ff00.svg)](https://arxiv.org/pdf/2408.00672.pdf)  [![Website shields.io](https://img.shields.io/website-up-down-green-red/http/shields.io.svg)](https://vision.cs.utexas.edu/projects/ExpertAF/)

![Teaser](teaser.png)

## Dataset and path

We create the following weakly-supervised training and testing data for ExpertAF:

(insert table here)

Note that the data follows the format for LLaVA-style training (insert link).

The code also uses the following pre-extracted or trained features:



## Installation and usage

This code is based on LLaVA (insert link) (NeurIPS 2023). Please see installation instructions in (insert link) to setup the environment.

### Key architectural changes compared to LLaVA

- We create the training data in the same format as LLaVA, i.e., a list of samples with conversation-style format.
- Replaced `vision_tower` with an option to use pretrained pre-extracted features (helps reduce the parameter and avoids feature computation during train and test).
- Replaced `<image>` tag in LLaVA with `<pose1>` and `<pose2>` for learner and expert features (video and pose).

### Running the code

We use `slurm_hp_with_mover.sh` that runs the code in SLURM with hyper-parameter search. Set the variables correctly for the mode and run it. The script first copies the working directory to a destination path and then runs the code--ensuring a version of the code is saved along with the result.

The above code runs `pretrain_v15.sh` which can also be modified as per the dataset being tested and the input conditions.

Run the code with

```
sbatch -J name_of_the_run slurm_hp_with_mover.sh 
```


## Cite


