# JTNN_BO

## Required file:

* `jtnn/` contains JTNN model codes
* `vocab.txt` contains vocabulary list used for training JTNN
* `model.iter` contains your pretrained JTNN model
* `latent_features.txt` and  `targets.txt` contains the latent representations of your dataset and their corresponding target scores

## Files explained:

* `GPytorch_BO.py` is the main file performing Bayesian Optimization implemented using GPytorch, to run this code:
```
python GPytroch_BO.py -t 200000 -e 250 -v path_to_vocab -m path_to_model -l path_to_latent_features -s path_to_target_scores -o output_dir
```
the code will generate a list of valid molecules in a saved file.

