# Quantizing Model with Accuracy Control{#pot_accuracyaware_usage}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   AccuracyAwareQuantization Method <pot_compression_algorithms_quantization_accuracy_aware_README>

@endsphinxdirective

## Introduction
In case when [DefaultQuantization](@ref pot_default_quantization_usage) alorithm introduces a significant accuracy degradation, AccuracyAwareQuantization algorithm can be used to stay within the pre-defined range defined by the user. This may cause a 
degradation in performance in comparison to [DefaultQuantization](@ref pot_default_quantization_usage) algorithm because some layers can be reverted back to the original precision.

> **NOTE**: In case of GNA `target_device`, POT moves INT8 weights to INT16 to stay in the pre-defined range of the accuracy drop. Thus, the algorithm works for the `performance` (INT8) preset only. For the `accuracy` preset, this algorithm is not helpful.

## Prepare data
This step is the same as in the case of [DefaultQuantization](@ref pot_default_quantization_usage). The only difference is that `__getitem__()` method should return `(data, annotation)` or `(data, annotation, metadata)` where `annotation` is required and corresponds to the expectations of the `Metric` class.

## Define accuracy metric
In order to control accuracy during the optimization a `openvino.tools.pot.Metric` interface should be implemented. Each implementaion should override the following properties:
- `value` - returns the accuracy metric value for the last model output in a format of `Dict[str, numpy.array]`.
- `avg_value` - returns the average accuracy metric over collected model results in a format of `Dict[str, numpy.array]`.
- `higher_better` should return `True` if the higher value of the metric corresponds to better performance, otherwise, returns `False`. Default implementation returns `True`.

and methods:
- `update(output, annotation)` - calculates and updates the accuracy metric value using last model output and annotation.
- `reset()` - resets collected accuracy metric. 
- `get_attributes()` - returns a dictionary of metric attributes:
   ```
   {metric_name: {attribute_name: value}}
   ```
   Required attributes: 
   - `direction` - (`higher-better` or `higher-worse`) a string parameter defining whether metric value 
    should be increased in accuracy-aware algorithms.
   - `type` - a string representation of metric type. For example, 'accuracy' or 'mean_iou'.

Below is an example of accuracy top-1 metric implementation with POT API:
```python
from openvino.tools.pot import metric

class Accuracy(Metric):

    # Required methods
    def __init__(self, top_k=1):
        super().__init__()
        self._top_k = top_k
        self._name = 'accuracy@top{}'.format(self._top_k)
        self._matches = []
    
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

An instance of the `Metric` implementation should be passed to `IEEngine` object responsible for model inference.

```
metric = UserMetric()
engine = IEEngine(config=engine_config, data_loader=data_loader, metric=metric)
```

## Quantization parameters
Since the DefaultQuantization algorithm is used as an initialization, all its parameters are also valid and can be specified. Here we
describe only AccuracyAwareQuantization required parameters:
- `"maximal_drop"` - maximum accuracy drop which has to be achieved after the quantization. Default value is `0.01` (1%).

## Run quantization

The code snippet below shows basic quantization workflow with accuracy control. `UserDataLoader` and `UserMetric` are placeholders for user's implementation of `DataLoader` and `Metric` APIs.

```
from openvino.tools.pot import IEEngine
from openvino.tools.pot load_model, save_model
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

# Step 1: implement and create user's data loader
data_loader = UserDataLoader(..)

# Step 2: implement and create user's data loader
metric = UserMetric(..)

# Step 3: load model
model = load_model(model_config=model_config)

# Step 4: Initialize the engine for metric calculation and statistics collection.
engine = IEEngine(config=engine_config, data_loader=data_loader, metric=metric)

# Step 5: Create a pipeline of compression algorithms and run it.
pipeline = create_pipeline(algorithms, engine)
compressed_model = pipeline.run(model=model)

# Step 6 (Optional): Compress model weights to quantized precision
#                    in order to reduce the size of the final .bin file.
compress_model_weights(compressed_model)

# Step 7: Save the compressed model to the desired path.
# Set save_path to the directory where the model should be saved
compressed_model_paths = save_model(
    model=compressed_model,
    save_path="optimized_model",
    model_name="optimized_model",
)

# Step 8 (Optional): Evaluate the compressed model. Print the results.
metric_results = pipeline.evaluate(compressed_model)
```

It is worth noting that now `evaluate` method that can compute accuracy on demand is also available in the `Pipeline` object.

## Examples

 * [Quantization of Object Detection model with control of accuracy](https://github.com/openvinotoolkit/openvino/tree/master/tools/pot/openvino/tools/pot/api/samples/object_detection)

