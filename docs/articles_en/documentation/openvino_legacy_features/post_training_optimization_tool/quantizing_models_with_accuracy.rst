.. {#pot_accuracyaware_usage}

Quantizing Models with Accuracy Control
=======================================


.. toctree::
   :maxdepth: 1
   :hidden:

   AccuracyAwareQuantization Method <accuracy_aware_README>


The Accuracy-aware Quantization algorithm allows performing quantization while maintaining accuracy within a pre-defined range. Note that it should be used only if the :doc:`Default Quantization <pot_default_quantization_usage>` introduces a significant accuracy degradation. The reason for it not being the primary choice is its potential for performance degradation, due to some layers getting reverted to the original precision.

To proceed with this article, make sure you have read how to use :doc:`Default Quantization <pot_default_quantization_usage>`.

.. note::
   
   The Accuracy-aware Quantization algorithm's behavior is different for the GNA ``target_device``. In this case, it searches for the best configuration and selects between INT8 and INT16 precisions for the weights of each layer. The algorithm works for the ``performance`` preset only. It is not useful for the ``accuracy`` preset, since the whole model is already in INT16 precision.

A script for Accuracy-aware Quantization includes four steps:

1. Prepare data and dataset interface.
2. Define accuracy metric.
3. Select quantization parameters.
4. Define and run the quantization process.

Prepare data and dataset interface
##################################

This step is the same as in :doc:`Default Quantization <pot_default_quantization_usage>`. The only difference is that ``__getitem__()`` should return ``(data, annotation)`` or ``(data, annotation, metadata)``. The ``annotation`` is required and its format should correspond to the expectations of the ``Metric`` class. The ``metadata`` is an optional field that can be used to store additional information required for post-processing.

Define accuracy metric
######################

To control accuracy during optimization, the ``openvino.tools.pot.Metric`` interface should be implemented. Each implementation should override the following properties and methods:

**Properties**

- ``value`` - returns the accuracy metric value for the last model output in a format of ``Dict[str, numpy.array]``.
- ``avg_value`` - returns the average accuracy metric over collected model results in a format of ``Dict[str, numpy.array]``.
- ``higher_better`` if a higher value of the metric corresponds to better performance, returns ``True`` , otherwise, ``False``. The default implementation returns ``True``.

**Methods**

- ``update(output, annotation)`` - calculates and updates the accuracy metric value, using the last model output and annotation. The model output and annotation should be passed in this method. It should also contain the model-specific post-processing in case the model returns the raw output.
- ``reset()`` - resets collected accuracy metric. 
- ``get_attributes()`` - returns a dictionary of metric attributes:

  .. code-block:: console
     
     {metric_name: {attribute_name: value}}
  
  Required attributes: 
  
  - ``direction`` - (``higher-better`` or ``higher-worse``) a string parameter defining whether the metric value should be increased in accuracy-aware algorithms.
  - ``type`` - a string representation of a metric type. For example, "accuracy" or "mean_iou".


Below is an example of the accuracy top-1 metric implementation with POT API:
```python
from openvino.tools.pot import Metric

class Accuracy(Metric):

    # Required methods
    def __init__(self, top_k=1):
        super().__init__()
        self._top_k = top_k
        self._name = 'accuracy@top{}'.format(self._top_k)
        self._matches = [] # container of the results
    
    @property
    def value(self):
        """ Returns accuracy metric value for all model outputs. """
        return {self._name: self._matches[-1]}

    @property
    def avg_value(self):
        """ Returns accuracy metric value for all model outputs. """
        return {self._name: np.ravel(self._matches).mean()}

    def update(self, output, target):
        """ Updates prediction matches.
        :param output: model output
        :param target: annotations
        """
        if len(output) > 1:
            raise Exception('The accuracy metric cannot be calculated '
                            'for a model with multiple outputs')
        if isinstance(target, dict):
            target = list(target.values())
        predictions = np.argsort(output[0], axis=1)[:, -self._top_k:]
        match = [float(t in predictions[i]) for i, t in enumerate(target)]

        self._matches.append(match)

    def reset(self):
        """ Resets collected matches """
        self._matches = []

    def get_attributes(self):
        """
        Returns a dictionary of metric attributes {metric_name: {attribute_name: value}}.
        Required attributes: 'direction': 'higher-better' or 'higher-worse'
                             'type': metric type
        """
        return {self._name: {'direction': 'higher-better',
                             'type': 'accuracy'}}
```


An instance of the ``Metric`` implementation should be passed to ``IEEngine`` object responsible for model inference.

.. code-block:: python
   
   metric = Accuracy()
   engine = IEEngine(config=engine_config, data_loader=data_loader, metric=metric)

Select quantization parameters
##############################

Accuracy-aware Quantization uses the Default Quantization algorithm at the initialization step in such an order that all its parameters are also valid and can be specified. The only parameter required exclusively by Accuracy-aware Quantization is:

- ``"maximal_drop"`` - the maximum accuracy drop which has to be achieved after the quantization. The default value is ``0.01`` (1%).

Run quantization
################

The example code below shows a basic quantization workflow with accuracy control. ``UserDataLoader()`` is a placeholder for the implementation of ``DataLoader``.

.. code-block:: python
   
   from openvino.tools.pot import IEEngine
   from openvino.tools.pot import load_model, save_model
   from openvino.tools.pot import compress_model_weights
   from openvino.tools.pot import create_pipeline
   
   # Model config specifies the model name and paths to model .xml and .bin file
   model_config = Dict(
       {
           "model_name": "model",
           "model": path_to_xml,
           "weights": path_to_bin,
       }
   )
   
   # Engine config
   engine_config = Dict({"device": "CPU"})
   
   algorithms = [
       {
           "name": "AccuracyAwareQuantization",
           "params": {
               "target_device": "ANY", 
               "stat_subset_size": 300,
               'maximal_drop': 0.02
           },
       }
   ]
   
   # Step 1: Implement and create user's data loader.
   data_loader = UserDataLoader()
   
   # Step 2: Implement and create user's data loader.
   metric = Accuracy()
   
   # Step 3: Load the model.
   model = load_model(model_config=model_config)
   
   # Step 4: Initialize the engine for metric calculation and statistics collection.
   engine = IEEngine(config=engine_config, data_loader=data_loader, metric=metric)
   
   # Step 5: Create a pipeline of compression algorithms and run it.
   pipeline = create_pipeline(algorithms, engine)
   compressed_model = pipeline.run(model=model)
   
   # Step 6 (Optional): Compress model weights to quantized precision
   #                    to reduce the size of the final .bin file.
   compress_model_weights(compressed_model)
   
   # Step 7: Save the compressed model to the desired path.
   # Set save_path to the directory where the model should be saved.
   compressed_model_paths = save_model(
       model=compressed_model,
       save_path="optimized_model",
       model_name="optimized_model",
   )
   
   # Step 8 (Optional): Evaluate the compressed model. Print the results.
   metric_results = pipeline.evaluate(compressed_model)
   

It is worth noting that now the ``evaluate`` method that can compute accuracy on demand is also available in the ``Pipeline`` object.

In case when Accuracy-aware Quantization does not allow achieving the desired accuracy-performance trade-off, it is recommended to try Quantization-aware Training from :doc:`NNCF <tmo_introduction>`.

Examples
########

* `Quantization of Object Detection model with control of accuracy <https://github.com/openvinotoolkit/openvino/tree/master/tools/pot/openvino/tools/pot/api/samples/object_detection>`__


