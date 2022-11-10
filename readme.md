## small-language-models

Code for the paper 'Small Language Models for Tabular Data', which may be accessed on [arxiv](https://arxiv.org/abs/2211.02941).

### General Details

Note firstly that this is research code and is not productionized as-is.  In particular, there are non-optimal approaches taken in the data encoding steps (for example, the `data_formatter.Format` class iterates through all specified rows of the tabular data in question sequentially).

The `fcnet.py` module contains code for exploring model embeddings, whereas the `fcnet_original.py` is a better place to start for exploring the capability of direct character encoding.  `fcnet_original.py` also contains class methods that allow for the visualization of the gradient field for specified parameters, work that did not make it into the paper.

As one might expect, `fcnet_categorical.py` and `transformer_categorical.py` contain code for applying the structured sequence encoding method to datasets with categorical outputs such as the well-known [Titanic](https://www.kaggle.com/c/titanic) dataset, which is supplied for convenience.  Note that these modules are capable of encoding all ascii characters rather than numerical and certain special characters as for the others: if one wants this extended encoding to be applied to a continuous output, it should be relatively straightforward to combine the ascii-based character encoding to the models specified in `fcnet.py` and `transformer.py`.

### More Information

For more information on the theory behind the architectures used as well as more thorough implementation details, see this [technical blog](https://blbadger.github.io/neural-networks3.html), and for more details on input attribution see [here](https://blbadger.github.io/nn_interpretations.html).
