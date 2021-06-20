.. -*- mode: rst -*-

.. |github| image:: https://img.shields.io/github/workflow/status/alan-turing-institute/sktime/build-and-test?logo=github
.. _github: https://github.com/alan-turing-institute/sktime/actions?query=workflow%3Abuild-and-test

.. |appveyor| image:: https://img.shields.io/appveyor/ci/mloning/sktime/main?logo=appveyor
.. _appveyor: https://ci.appveyor.com/project/mloning/sktime

.. |pypi| image:: https://img.shields.io/pypi/v/sktime?color=orange
.. _pypi: https://pypi.org/project/sktime/

.. |conda| image:: https://img.shields.io/conda/vn/conda-forge/sktime
.. _conda: https://anaconda.org/conda-forge/sktime

.. |discord| image:: https://img.shields.io/static/v1?logo=discord&label=discord&message=chat&color=lightgreen
.. _discord: https://discord.com/invite/gqSab2K

.. |gitter| image:: https://img.shields.io/static/v1?logo=gitter&label=gitter&message=chat&color=lightgreen
.. _gitter: https://gitter.im/sktime/community

.. |binder| image:: https://mybinder.org/badge_logo.svg
.. _binder: https://mybinder.org/v2/gh/alan-turing-institute/sktime/main?filepath=examples

.. |zenodo| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3749000.svg
.. _zenodo: https://doi.org/10.5281/zenodo.3749000

.. |azure| image:: https://img.shields.io/azure-devops/build/mloning/30e41314-4c72-4751-9ffb-f7e8584fc7bd/1/main?logo=azure-pipelines
.. _azure: https://dev.azure.com/mloning/sktime/_build

.. |codecov| image:: https://img.shields.io/codecov/c/github/alan-turing-institute/sktime?label=codecov&logo=codecov
.. _codecov: https://codecov.io/gh/alan-turing-institute/sktime

.. |readthedocs| image:: https://readthedocs.org/projects/sktime/badge/?version=latest
.. _readthedocs: https://www.sktime.org/en/latest/?badge=latest

.. |twitter| image:: https://img.shields.io/twitter/follow/sktime_toolbox?label=%20Twitter&style=social
.. _twitter: https://twitter.com/sktime_toolbox

.. |python| image:: https://img.shields.io/pypi/pyversions/sktime
.. _python: https://www.python.org/

.. |codestyle| image:: https://img.shields.io/badge/code%20style-black-000000.svg
.. _codestyle: https://github.com/psf/black

.. |contributors| image:: https://img.shields.io/github/contributors/alan-turing-institute/sktime?color=pink&label=all-contributors
.. _contributors: https://github.com/alan-turing-institute/sktime/blob/main/CONTRIBUTORS.md

.. |tutorial| image:: https://img.shields.io/youtube/views/wqQKFu41FIw?label=watch&style=social
.. _tutorial: https://www.youtube.com/watch?v=wqQKFu41FIw&t=14s


Rand-TS: Randomized Trees for Time Series Representation Learning
=================

  A unified, fexible and interpretable representation learning framework for univariate and multivariate time-series (MTS)


Quickstart
----------

.. code-block:: python

    import numpy as np
    import pandas as pd
    from os import listdir
    from RandTS import *
   
    directory = r'Dataset_Univariate\\'
    datasets = listdir(directory)
    
    train,test,labels_train,labels_test,X_train,X_test = load_data_new(directory, datasets[0])
    
    classifier = RandTS()
    classifier.fit(X_train,labels_train)
    y_pred = classifier.predict(X_test,labels_test)
    classifier.test_accuracy
    >>> 0.7714285714285715

Documentation
-------------

Univariate
~~~~~~~~~~

.. code-block:: python

    RandTS(method = 'l',depth = 5, ntree=10, var=10, featureSelection = None)

RandTS object with specified paramters:

*  method = {'l','d','b'} : 'l' for 'level', 'd' for 'difference', 'b' for both
* depth (int) : Tree depth
* ntree (int) : Number of randomized trees
* featureSelection: {None, 'supervised', 'unsupervised'}
* var (int) : Works with unsupervised feature selection. Sepcifies variance threshold for elimination

.. code-block:: python

    RandTS.fit(train, train_labels)

Trains the RandTS model on the given data. It does not directly use train_labels but it stores it in order to make prediction in the following steps.

.. code-block:: python

    RandTS.predict(test, test_labels)
    
Predicts the test lables based on the provided test data and returns the predictions.

.. code-block:: python

    RandTS.selectParameters(train, train_labels, param_kwargs={'depth_cv':[3,5,10], 'ntree_cv':[100], 'rep_num':1, 'method_cv':['l','d','b']})
    
Applies parameter selection in a given set of parameters and replication number.

* param_kwargs (dict): parameter set that is considered in parameter selection.

    * 'depth_cv':[3,5,10]
    * 'ntree_cv':[100]
    * 'rep_num':1
    * 'method_cv':['l','d','b']

Multivariate
~~~~~~~~~~~~

.. code-block:: python

    RandTSMultivariate(method = 'l',depth = 5, ntree=10, var=10, featureSelection = None)

RandTS object with specified paramters:

*  method = {'l','d','b'} : 'l' for 'level', 'd' for 'difference', 'b' for both
* depth (int) : Tree depth
* ntree (int) : Number of randomized trees
* featureSelection: {None, 'supervised', 'unsupervised'}
* var (int) : Works with unsupervised feature selection. Sepcifies variance threshold for elimination

.. code-block:: python

    RandTSMultivariate.fit(train, train_labels, ndims)

Trains the RandTS model on the given data. It does not directly use train_labels but it stores it in order to make prediction in the following steps.

* ndims: Number of variables in the multivariate time series.

.. code-block:: python

    RandTSMultivariate.predict(test, test_labels)

Predicts the test lables based on the provided test data and returns the predictions.

.. code-block:: python

    RandTS.selectParameters(train, train_labels, param_kwargs={'depth_cv':[3,5,10], 'ntree_cv':[100], 'rep_num':1, 'method_cv':['l','d','b']})
    
Applies parameter selection in a given set of parameters and replication number.

* param_kwargs (dict): parameter set that is considered in parameter selection.

    * 'depth_cv':[3,5,10]
    * 'ntree_cv':[100]
    * 'rep_num':1
    * 'method_cv':['l','d','b']

How to cite Rand-TS
------------------

If you use Rand-TS in a scientific publication, we would appreciate citations to the following paper:

Berk Gorgulu, Mustafa Gokce Baydogan (2021): “Randomized Trees for Time Series Representation and Similarity” 

Bibtex entry:

.. code-block:: latex

    @article{gorgulu2021randomized,
      title={Randomized Trees for Time Series Representation and Similarity},
      author={G{\"o}rg{\"u}l{\"u}, Berk and Baydo{\u{g}}an, Mustafa G{\"o}k{\c{c}}e},
      journal={Pattern Recognition},
      pages={108097},
      year={2021},
      publisher={Elsevier}
    }
