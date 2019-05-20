# Aspect-based Sentiment Detection using Deep Neural Networks and Transfer Learning

Majority of the implementation was done by us while some small modules were borrowed from a repository on Hierarchical Attention Models by [Matvey Ezhov.](https://github.com/ematvey/hierarchical-attention-networks)

## Project Requirements
* tensorflow==1.4.0
* python==3.5.2
* Install from requirements.txt.remote

## Embeddings
* Glove - https://nlp.stanford.edu/projects/glove/
    * http://nlp.stanford.edu/data/glove.840B.300d.zip
* FastText - https://fasttext.cc/docs/en/english-vectors.html
    * English : https://s3-us-west-1.amazonaws.com/fasttext-vectors/crawl-300d-2M-subword.zip
    * German : https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.de.vec
* Elmo - https://allennlp.org/elmo
    * Embeddings were computed using `allennlp` library. See the following links.
        * https://github.com/allenai/allennlp
        * https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md

## File Descriptions
* models
    * **model.py** : This is the main file where tensorflow model is defined. It also includes code for the calculation of loss, accuracy etc. Tensorboard scalars such as
loss, accuracy, f1_scores are also added in this file.
    * **model_elmo.py** : This file defines our model for using Elmo encoded words. Elmo encodings are precomputed in pre-processing and model gets reviews in the
in the encoding form.
    * **model_components.py** : This file contains implementation of model components like Attention, BiLSTM etc
    * **model_germeval17.py** : This is just a separate file with the same implementation of our model. This becomes handy to tweak the model in order to experiment
with germEval17 dataset.
    * **fc_progressive_nets.py** : This file defines the PNN model with FC lateral connections.
    * **cnn_progressive_nets.py** : This file defines the PNN model with CNN lateral connections.

* preprocessing
    * **semEval16_prepare.py** : Use this script for preparing SemEval2016 raw dataset to a format compatible with our model. It reads xml file and saves the formatted
data into .pickle file.
    * **semEval16_data_processing.py** : Use this script for processing the formatted data. It computes vocabulary and word frequency from the training set. Each word is assigned a
vocabulary code. Every word in the dataset is replaced by the correspoding vocabulary code. Some codes are reserved for unknown and padding words.
    * **elmo_data_processing.py** : This file is used to process semEval data and compute elmo embeddings. Final processed data is in the form of elmo embeddings
and not vocab codes.
    * **germEval17_prepare.py** : Use this script for preparing SemEval2016 raw dataset to a format compatible with our model. It reads xml file and saves the formatted
data into .pickle file. We are treating a review as a single sentence for this dataset.
    * **germEval16_data_processing.py** : Use this script for processing the formatted data. It computes vocabulary and word frequency from the training set. Each word is assigned a
vocabulary code. Every word in the dataset is replaced by the correspoding vocabulary code. Some codes are reserved for unknown and padding words.
    * **organic_prepare.py** : Use this script for preparing organic raw dataset to a format compatible with our model. It reads xml file and saves the formatted
data into .pickle file.
    * **organic_data_processing.py** : Use this script for processing the formatted data. It computes vocabulary and word frequency from the training set. Each word is assigned a
vocabulary code. Every word in the dataset is replaced by the correspoding vocabulary code. Some codes are reserved for unknown and padding words.
    * **organic_csv_to_xml.py** : Use this file for converting organic data from csv to xml format similar to semEval xml format.
    * **fasttext_de_db_prepare.py** : This file prepares a dataset for training fasttext embedding. It simply combines GermEval and DB tweets data.
    * **combined_data.py** : This file is used for combining train or val sets of restaurant,laptops,organic data.
    * **combined_data_processing** : This file is used for generating a combined multi-domain dataset.

* trainers
    * **trainer.py** : This is the main training executer. It loads batches and runs traning and evaluation sessions. You can also configure all model hyper parameters like
learning rate, batch size, word embedding size etc from this file.
    * **elmo_trainer.py** : Run this trainer if you want to train your model using Elmo embeddings.
    * **germeval17_trainer.py** : Run this trainer if you want to experiment with germEval17 dataset.
    * **fasttext_trainer.py** : Run this trainer if you want to train fasttext emebdding on db+germeval combined data.
    * **fc_progressive_net_trainer.py** : Run this trainer if you want to experiment with PNN having FC lateral connections.
    * **cnn_progressive_net_trainer.py** : Run this trainer if you want to experiment with PNN having CNN lateral connections.

* utils
    * **util.py** : This file contains a lot of helper methods. There are methods for padding data, creating label weights, creating batch,
computing f1_scores etc.
    * **data_util.py** : This file contains a lot of data manipulation related helper methods. There are methods for reading/writing files like xml, json etc. It
also has methods for loading train/test/val datasets.
    * **visualization_util.py** : This file contains helper methods for creating confusion matrix, plotting f1, accuracy curves etc.
    * **evaluation_utils.py** : This file contains helper methods for computing f1 scores and other evaluating metrics. 
    * **prediction_formatter.py**: This file is used for converting germEval predictions to an xml format so that germEval evaluation script can be applied.

* tests
    * **test.py** : Contains small test experiments.  

* config
    * **settings.py** : Different file path related settings can be controlled from this file.
    * **tensor_names.py** : Contains names of the tensors used in the model. These names helps in reloading a checkpoint and also in passing feed_dict to session run.
    * **operation_names.py** : Contains names of tensorflow operations used in the model. These are used for telling the session run which operations to execute.

* organic_data_splitting.ipynb : This file contains the logic that was used for splitting organic data into three sets - train,val,test. 

## Commands
* For preparing data in the correct format execute the following command from the project root directory. Make sure to set the correct `INPUT_FILE_PATH` and 
`OUTPUT_FILE_NAME`. You will have to run this script individually for training and test dataset. 
```
host@master-thesis-sumit$ python preprocessing/semEval16_prepare.py
```

* For processing the formatted file execute the following command from the project root directory. Make sure to set the correct `TYPE` and `FILE_NAME` variables.
```
host@master-thesis-sumit$ python preprocessing/semEval16_data_processing.py
```

* Start model training with default hyperparameters using the following command from the project root directory.
```
host@master-thesis-sumit$ python trainers/trainer.py
```

* Start model training by passing updated hyperparameters via command line params. For more details about hyperparameters see `trainer.py`
```
host@master-thesis-sumit$ python trainers/trainer.py --batch_size=128 --epochs=100
```

* Use the following command for generating Elmo embeddings. First for every sentence generate tokens, then save these space separated tokens in a text file. Do this
for every sentence in your dataset. Make sure each sentence is in a new line in the text file.
```
allennlp elmo all_sentences.txt elmo_embeddings.hdf5 --average --use-sentence-keys
```

## Additional Helpful Commands
### tmux
```
tmux new
tmux attach-session -t 0
tmux ls
Ctrl+b d
Ctrl+b [
```

### tensorboard
* Start tensorboard.
```
tensorboard --logdir <dir path>
```

* Connect to tensorboard server running on remote machine via ssh tunneling.
```
ssh -N -L 6006:localhost:6006 sdugar@social5.cm.in.tum.de
```

### jupyter
* Start jupyter notebook on remote machine.
```
jupyter notebook --no-browser --port=8080
```

* Connect to jupyter notebook server running on remote machine via ssh tunneling.
```
ssh -N -L 8080:localhost:8080 sdugar@social5.cm.in.tum.de
```

## Resources
* **tmux** : https://hackernoon.com/a-gentle-introduction-to-tmux-8d784c404340
* **spaCy** : https://spacy.io/usage/models
* **NLTK** : https://www.nltk.org/
* **gensim** : https://github.com/RaRe-Technologies/gensim
* **TextBlob** : https://textblob.readthedocs.io/en/dev/
* **ray** : https://ray.readthedocs.io/en/latest/tune.html

