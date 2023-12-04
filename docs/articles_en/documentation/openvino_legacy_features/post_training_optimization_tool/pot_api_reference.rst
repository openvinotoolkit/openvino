.. {#pot_compression_api_README}

API Reference
=============


Post-training Optimization Tool API provides a full set of interfaces and helpers that allow users to implement a custom optimization pipeline for various types of DL models including cascaded or compound models. Below is a full specification of this API:

DataLoader
++++++++++++++++++++

.. code-block:: sh

   class openvino.tools.pot.DataLoader(config)


The base class for all DataLoaders.

``DataLoader`` loads data from a dataset and applies pre-processing to them providing access to the pre-processed data 
by index. 

All subclasses should override the ``__len__()`` function, which should return the size of the dataset, and ``__getitem__()``, 
which supports integer indexing in the range of 0 to ``len(self)``. ``__getitem__()`` method can return data in one of the possible formats:

.. code-block:: sh

   (data, annotation)


or

.. code-block:: sh

   (data, annotation, metadata)


``data`` is the input that is passed to the model at inference so that it should be properly preprocessed. ``data`` can be either ``numpy.array`` object or dictionary where the key is the name of the model input and the value is ``numpy.array`` which corresponds to this input. The format of ``annotation`` should correspond to the expectations of the ``Metric`` class. ``metadata`` is an optional field that can be used to store additional information required for post-processing.

Metric
++++++++++++++++++++

.. code-block:: sh

   class openvino.tools.pot.Metric()


An abstract class representing an accuracy metric.

All instances should override the following properties:

- ``value`` - returns the accuracy metric value for the last model output in a format of ``Dict[str, numpy.array]``.
- ``avg_value`` - returns the average accuracy metric over collected model results in a format of ``Dict[str, numpy.array]``.
- ``higher_better`` should return ``True`` if a higher value of the metric corresponds to better performance, otherwise, returns ``False``. Default implementation returns ``True``.

and methods:

- ``update(output, annotation)`` - calculates and updates the accuracy metric value using the last model output and annotation. The model output and annotation should be passed in this method. It should also contain the model-specific post-processing in case the model returns the raw output.
- ``reset()`` - resets collected accuracy metric.
- ``get_attributes()`` - returns a dictionary of metric attributes:

  .. code-block:: sh

     {metric_name: {attribute_name: value}}


  Required attributes:

  - ``direction`` - (``higher-better`` or ``higher-worse``) a string parameter defining whether the metric value should be increased in accuracy-aware algorithms.
  - ``type`` - a string representation of metric type. For example, 'accuracy' or 'mean_iou'.

Engine
++++++++++++++++++++

.. code-block:: sh

   class openvino.tools.pot.Engine(config, data_loader=None, metric=None)

Base class for all Engines.

The engine provides model inference, statistics collection for activations and calculation of accuracy metrics for a dataset.

*Parameters*

- ``config`` - engine specific config.
- ``data_loader`` - ``DataLoader`` instance to iterate over dataset.
- ``metric`` - ``Metric`` instance to calculate the accuracy metric of the model.

All subclasses should override the following methods:

- ``set_model(model)`` - sets/resets a model.

  *Parameters*

  - ``model`` - `CompressedModel` instance for inference.

- `predict(stats_layout=None, sampler=None, metric_per_sample=False, print_progress=False)` - performs model inference on the specified subset of data.

  *Parameters*

  - `stats_layout` - dictionary of statistic collection functions. An optional parameter.

    .. code-block:: sh

       {
           'node_name': {
               'stat_name': fn
           }
       }

  - `sampler` - `Sampler` instance that provides a way to iterate over the dataset. (See details below).
  - `metric_per_sample` - if `Metric` is specified and this parameter is set to True, then the metric value should be 
    calculated for each data sample, otherwise for the whole dataset.
  - `print_progress` - print inference progress.

  *Returns*

  - a tuple of dictionaries of per-sample and overall metric values if ``metric_per_sample`` is True

    .. code-block:: sh

       (
           {
               'sample_id': sample_index,
               'metric_name': metric_name,
               'result': metric_value
           },
           {
               'metric_name': metric_value
           }
       )


    Otherwise, a dictionary of overall metrics.

    .. code-block:: sh

       { 'metric_name': metric_value }


- a dictionary of collected statistics

  .. code-block:: sh

     {
         'node_name': {
             'stat_name': [statistics]
         }
     }


Pipeline
++++++++++++++++++++

.. code-block:: sh

   class openvino.tools.pot.Pipeline(engine)


Pipeline class represents the optimization pipeline.

*Parameters*

- ``engine`` - instance of ``Engine`` class for model inference.

The pipeline can be applied to the DL model by calling ``run(model)`` method where ``model`` is the ``NXModel`` instance.

Create a pipeline
--------------------

The POT Python* API provides the utility function to create and configure the pipeline:

.. code-block:: sh

   openvino.tools.pot.create_pipeline(algo_config, engine)


*Parameters*

- ``algo_config`` - a list defining optimization algorithms and their parameters included in the optimization pipeline. 
  The order in which they are applied to the model in the optimization pipeline is determined by the order in the list.

  Example of the algorithm configuration of the pipeline:

  .. code-block:: sh

     algo_config = [
         {
             'name': 'DefaultQuantization',
             'params': {
                 'preset': 'performance',
                 'stat_subset_size': 500
             }
          },
         ...
     ]


- ``engine`` - instance of ``Engine`` class for model inference.

*Returns*

- instance of the ``Pipeline`` class.

Helpers and Internal Model Representation
#########################################

To simplify the implementation of optimization pipelines we provide a set of ready-to-use helpers. Here we also 
describe an internal representation of the DL model and how to work with it.

IEEngine
++++++++++++++++++++

.. code-block:: sh

   class openvino.tools.pot.IEEngine(config, data_loader=None, metric=None)

IEEngine is a helper which implements Engine class based on :doc:`OpenVINO&trade; Inference Engine Python API <api/ie_python_api/api>`.
This class support inference in synchronous and asynchronous modes and can be reused as-is in the custom pipeline or 
with some modifications, e.g. in case of custom post-processing of inference results.

The following methods can be overridden in subclasses:

- ``postprocess_output(outputs, metadata)`` - Processes model output data using the image metadata obtained during data loading.

  *Parameters*

  - ``outputs`` - dictionary of output data per output name.
  - ``metadata`` - information about the data used for inference.

  *Return*

  - list of the output data in an order expected by the accuracy metric if any is used

``IEEngine`` supports data returned by ``DataLoader`` in the format:

.. code-block:: sh

   (data, annotation)


or

.. code-block:: sh

   (data, annotation, metadata)


Metric values returned by a ``Metric`` instance are expected to be in the format:

- for ``value()``:

  .. code-block:: sh

     {metric_name: [metric_values_per_image]}

- for ``avg_value()``:

  .. code-block:: sh

     {metric_name: metric_value}


In order to implement a custom ``Engine`` class you may need to get familiar with the following interfaces:

CompressedModel
++++++++++++++++++++

The Python POT API provides the ``CompressedModel`` class as one interface for working with single and cascaded DL model. 
It is used to load, save and access the model, in case of the cascaded model, access each model of the cascaded model.

.. code-block:: sh

   class openvino.tools.pot.graph.nx_model.CompressedModel(**kwargs)

The CompressedModel class provides a representation of the DL model. A single model and cascaded model can be 
represented as an instance of this class. The cascaded model is stored as a list of models.

*Properties*

- ``models`` - list of models of the cascaded model.
- ``is_cascade`` - returns True if the loaded model is a cascaded model.

Read model from OpenVINO IR
++++++++++++++++++++++++++++++

The Python POT API provides the utility function to load the model from the OpenVINO&trade; Intermediate Representation (IR):

.. code-block:: sh

   openvino.tools.pot.load_model(model_config)

*Parameters*

- ``model_config`` - dictionary describing a model that includes the following attributes:
  - ``model_name`` - model name.
  - ``model`` - path to the network topology (.xml).
  - ``weights`` - path to the model weights (.bin).

  Example of ``model_config`` for a single model:

  .. code-block:: sh

     model_config = {
         'model_name': 'mobilenet_v2',
         'model': '<PATH_TO_MODEL>/mobilenet_v2.xml',
         'weights': '<PATH_TO_WEIGHTS>/mobilenet_v2.bin'
     }

  Example of ``model_config`` for a cascaded model:

  .. code-block:: sh

     model_config = {
         'model_name': 'mtcnn',
         'cascade': [
             {
                 'name': 'pnet',
                 "model": '<PATH_TO_MODEL>/pnet.xml',
                 'weights': '<PATH_TO_WEIGHTS>/pnet.bin'
             },
             {
                 'name': 'rnet',
                 'model': '<PATH_TO_MODEL>/rnet.xml',
                 'weights': '<PATH_TO_WEIGHTS>/rnet.bin'
             },
             {
                 'name': 'onet',
                 'model': '<PATH_TO_MODEL>/onet.xml',
                 'weights': '<PATH_TO_WEIGHTS>/onet.bin'
             }
         ]
     }


*Returns*

- ``CompressedModel`` instance

Save a model to IR
----------------------

The Python POT API provides the utility function to save a model in the OpenVINO&trade; Intermediate Representation (IR):

.. code-block:: sh

   openvino.tools.pot.save_model(model, save_path, model_name=None, for_stat_collection=False)


*Parameters*

- ``model`` - ``CompressedModel`` instance.
- ``save_path`` - path to save the model.
- ``model_name`` - name under which the model will be saved.
- ``for_stat_collection`` - whether the model is saved to be used for statistic collection or for inference (affects only cascaded models). If set to False, removes model prefixes from node names.

*Returns*

- list of dictionaries with paths:

  .. code-block:: sh

     [
         {
             'name': model name,
             'model': path to .xml,
             'weights': path to .bin
         },
         ...
     ]


Sampler
++++++++++++++++++++

.. code-block:: sh

   class openvino.tools.pot.samplers.Sampler(data_loader=None, batch_size=1, subset_indices=None)

Base class for all Samplers.

Sampler provides a way to iterate over the dataset.

All subclasses the ``__iter__()`` method, providing a way to iterate over the dataset, and a ``__len__()`` method 
that returns the length of the returned iterators.

*Parameters*

- ``data_loader`` - instance of ``DataLoader`` class to load data.
- ``batch_size`` - number of items in batch, default is 1.
- ``subset_indices`` - indices of samples to load. If ``subset_indices`` is set to None then the sampler will take elements from the whole dataset.

BatchSampler
++++++++++++

.. code-block:: sh

   class openvino.tools.pot.samplers.batch_sampler.BatchSampler(data_loader, batch_size=1, subset_indices=None):

Sampler provides an iterable over the dataset subset if ``subset_indices`` is specified 
or over the whole dataset with a given ``batch_size``. Returns a list of data items.


