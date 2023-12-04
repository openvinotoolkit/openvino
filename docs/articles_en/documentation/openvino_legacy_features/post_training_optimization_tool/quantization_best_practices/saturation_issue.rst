.. {#pot_saturation_issue}

Saturation (overflow) Issue Workaround
======================================


Introduction
####################

8-bit instructions of older Intel CPU generations (based on SSE, AVX-2, and AVX-512 instruction sets) are prone to so-called saturation (overflow) of the intermediate buffer when calculating the dot product, which is an essential part of Convolutional or MatMul operations. This saturation can lead to a drop in accuracy when running inference of 8-bit quantized models on the mentioned architectures. Additionally, it is impossible to predict if the issue occurs in a given setup since most computations are executed in parallel during DL model inference, which makes this process non-deterministic. This is a common problem for models with non-ReLU activation functions and low level of redundancy (for example, optimized or efficient models). It can prevent deploying the model on legacy hardware or creating cross-platform applications. The problem does not occur on GPUs or CPUs with Intel Deep Learning Boost (VNNI) technology and further generations.

Saturation Problem Detection
############################

The only way to detect the saturation issue is to run inference on a CPU that allows it and then on one that does not (for example, a VNNI-based CPU). A significant difference in accuracy (more than 1%) will be the main indicator of the saturation issue impact.

Saturation Issue Workaround
###########################

While quantizing activations use the full range of 8-bit data types, there is a workaround using only 7 bits to represent weights (of Convolutional or Fully-Connected layers). Using this algorithm for the first layer can help mitigate the saturation issue for many models. However, this can lead to lower accuracy due to reduced representation of weights.

POT tool provides three options to deal with the saturation issue. The options can be enabled in the POT configuration file using the ``saturation_fix`` parameter:

* "First_layer" option -- (default) fix saturation issue for the first layer.
* "All" option -- apply for all layers in the model.
* "No" option -- do not apply saturation fix at all.

Below is an example of the section in the POT configuration file with the ``saturation_fix`` option:

.. code-block:: json

   "algorithms": [
       {
           "name": "DefaultQuantization",
           "params": {
               "preset": "performance",
               "stat_subset_size": 300,
               "saturation_fix": "all" // Apply the saturation fix to all the layers
           }
       }
   ]


If you observe the saturation issue, try the "all" option during model quantization. If the accuracy problem still occurs, try using `Quantization-aware training from NNCF <https://github.com/openvinotoolkit/nncf>`__ and fine-tuning the model.

Use the "no" option when leaving out legacy CPU HW. It might also lead to slightly better accuracy.

Additional Resources
####################

* `Lower Numerical Precision Deep Learning Inference and Training blogpost <https://www.intel.com/content/www/us/en/developer/articles/technical/lower-numerical-precision-deep-learning-inference-and-training.html>`__
* :doc:`Configuration file description <pot_configs_README>`

