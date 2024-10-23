OpenVINO™ Explainable AI Toolkit (2/3): Deep Dive
=================================================

.. warning::

   Important note: This notebook requires python >= 3.10. Please make
   sure that your environment fulfill to this requirement before running
   it

This is the **second notebook** in series of exploring `OpenVINO™
Explainable AI
(XAI) <https://github.com/openvinotoolkit/openvino_xai/>`__:

1. `OpenVINO™ Explainable AI Toolkit (1/3):
   Basic <explainable-ai-1-basic-with-output.html>`__
2. `OpenVINO™ Explainable AI Toolkit (2/3): Deep
   Dive <explainable-ai-2-deep-dive-with-output.html>`__
3. `OpenVINO™ Explainable AI Toolkit (3/3): Saliency map
   interpretation <explainable-ai-3-map-interpretation-with-output.html>`__

`OpenVINO™ Explainable AI
(XAI) <https://github.com/openvinotoolkit/openvino_xai/>`__ provides a
suite of XAI algorithms for visual explanation of
`OpenVINO™ <https://github.com/openvinotoolkit/openvino>`__ Intermediate
Representation (IR) models.

Using **OpenVINO XAI**, you can generate **saliency maps** that
highlight regions of interest in input images from the model’s
perspective. This helps users understand why complex AI models produce
specific responses.

This notebook shows an example of how to use OpenVINO XAI, exploring its
methods and functionality.

It displays a heatmap indicating areas of interest where a neural
network (for classification or detection) focuses before making a
decision.

Let’s imagine the case that our OpenVINO IR model is up and running on a
inference pipeline. While watching the outputs, we may want to analyze
the model’s behavior for debugging or understanding purposes.

By using the OpenVINO XAI ``Explainer``, we can visualize why the model
gives such responses, meaning on which areas it focused before
predicting a particular label.


**Table of contents:**


-  `Prerequisites <#prerequisites>`__

   -  `Install requirements <#install-requirements>`__
   -  `Imports <#imports>`__
   -  `Download IR model <#download-ir-model>`__
   -  `Load the Image <#load-the-image>`__
   -  `Preprocess image for
      MobileNet <#preprocess-image-for-mobilenet>`__

-  `Basic usage: Explainer in AUTO
   mode <#basic-usage-explainer-in-auto-mode>`__

   -  `Create Explainer object <#create-explainer-object>`__
   -  `Generate explanation <#generate-explanation>`__
   -  `Visualize saliency maps <#visualize-saliency-maps>`__
   -  `Save saliency maps <#save-saliency-maps>`__
   -  `Generate saliency maps for all
      classes <#generate-saliency-maps-for-all-classes>`__

-  `Pre- and post-process
   functions <#pre--and-post-process-functions>`__
-  `Visualization Parameters <#visualization-parameters>`__
-  `Explainer in WHITEBOX mode <#explainer-in-whitebox-mode>`__

   -  `ReciproCAM XAI method <#reciprocam-xai-method>`__
   -  `Insert XAI branch <#insert-xai-branch>`__
   -  `Insertion-related parameters <#insertion-related-parameters>`__

-  `Explainer in BLACKBOX mode <#explainer-in-blackbox-mode>`__
-  `Advanced <#advanced>`__

   -  `Import ImageNet label names and add them to saliency
      maps <#import-imagenet-label-names-and-add-them-to-saliency-maps>`__
   -  `Activation map XAI method <#activation-map-xai-method>`__

Installation Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~

This is a self-contained example that relies solely on its own code.

We recommend running the notebook in a virtual environment. You only
need a Jupyter server to start. For details, please refer to
`Installation
Guide <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide>`__.

Prerequisites
-------------



Install requirements
~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    %%capture
    
    import platform
    
    # Install openvino package
    %pip install -q "openvino>=2024.2.0" opencv-python tqdm scipy
    
    %pip install -q --no-deps "openvino-xai>=1.1.0"
    %pip install -q -U "numpy==1.*"
    %pip install -q scipy
    
    if platform.system() != "Windows":
        %pip install -q "matplotlib>=3.4"
    else:
        %pip install -q "matplotlib>=3.4,<3.7"

Imports
~~~~~~~



.. code:: ipython3

    from pathlib import Path
    
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    
    import openvino.runtime as ov
    from openvino.runtime.utils.data_helpers.wrappers import OVDict
    import openvino_xai as xai
    from openvino_xai.explainer import ExplainMode
    from openvino_xai.explainer.explanation import Explanation
    
    # Fetch `notebook_utils` module
    import requests
    
    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
    )
    
    open("notebook_utils.py", "w").write(r.text)
    
    from notebook_utils import download_file

Download IR model
~~~~~~~~~~~~~~~~~



In this notebook for demonstration purposes we’ll use an already
converted to IR model from OpenVINO storage.

.. code:: ipython3

    base_artifacts_dir = Path("./artifacts").expanduser()
    
    model_name = "v3-small_224_1.0_float"
    model_xml_name = f"{model_name}.xml"
    model_bin_name = f"{model_name}.bin"
    
    model_xml_path = base_artifacts_dir / model_xml_name
    
    base_url = "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/mobelinet-v3-tf/FP32/"
    
    if not model_xml_path.exists():
        download_file(base_url + model_xml_name, model_xml_name, base_artifacts_dir)
        download_file(base_url + model_bin_name, model_bin_name, base_artifacts_dir)
    else:
        print(f"{model_name} already downloaded to {base_artifacts_dir}")

.. code:: ipython3

    # Create ov.Model
    model = ov.Core().read_model(model_xml_path)

Load the Image
~~~~~~~~~~~~~~



.. code:: ipython3

    # Download the image from the openvino_notebooks storage
    image_filename = download_file(
        "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco.jpg",
        directory="data",
    )
    
    # The MobileNet model expects images in RGB format.
    image = cv2.cvtColor(cv2.imread(filename=str(image_filename)), code=cv2.COLOR_BGR2RGB)
    plt.imshow(image)


.. parsed-literal::

    'data/coco.jpg' already exists.
    



.. parsed-literal::

    <matplotlib.image.AxesImage at 0x7f0180958940>




.. image:: explainable-ai-2-deep-dive-with-output_files/explainable-ai-2-deep-dive-with-output_11_2.png


Preprocess image for MobileNet
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    # Resize to MobileNetV3 input image shape.
    preprocessed_image = cv2.resize(src=image, dsize=(224, 224))
    # Add batch dimension
    preprocessed_image = np.expand_dims(preprocessed_image, 0)

Basic usage: ``Explainer`` in ``AUTO`` mode
-------------------------------------------



The easiest way to generate saliency maps is to use ``Explainer`` in
``ExplainMode.AUTO`` mode (``AUTO`` mode is used by default).

Under the hood of ``AUTO`` mode, ``Explainer`` will first try to run the
``WHITEBOX`` mode. If ``WHITEBOX`` fails, it will then run the
``BLACKBOX`` mode as a fallback option. See more details about
`WHITEBOX <#explainer-in-whitebox-mode>`__ and
`BLACKBOX <#explainer-in-blackbox-mode>`__ modes below.

Generating saliency maps involves model inference. The explainer will
perform model inference, but to do so, it requires ``preprocess_fn`` and
``postprocess_fn``. We can avoid passing ``preprocess_fn`` by
preprocessing (e.g., resizing and adding a batch dimension as shown
above) the input data beforehand - by default, ``preprocess_fn`` is the
identity function. We expect that current example will successfully use
``WHITEBOX`` mode under the hood, therefore we don’t pass
``postprocess_fn`` (``postprocess_fn`` is not required for ``WHITEBOX``
mode, only for ``BLACKBOX``).

To learn more about pre- and post-process functions, refer to the `pre-
and post-process functions <#pre--and-post-process-functions>`__
section.

Create ``Explainer`` object
~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    explainer = xai.Explainer(
        model=model,
        task=xai.Task.CLASSIFICATION,
    )


.. parsed-literal::

    INFO:openvino_xai:Assigning preprocess_fn to identity function assumes that input images were already preprocessed by user before passing it to the model. Please define preprocessing function OR preprocess images beforehand.
    INFO:openvino_xai:Target insertion layer is not provided - trying to find it in auto mode.
    INFO:openvino_xai:Using ReciproCAM method (for CNNs).
    INFO:openvino_xai:Explaining the model in white-box mode.
    

Generate ``explanation``
~~~~~~~~~~~~~~~~~~~~~~~~



The predicted class for this model-image pair is
``flat-coated_retriever`` with class index ``206``. So here and further
we will check saliency maps for this index.

.. code:: ipython3

    # You can choose class(es) to generate saliency maps for.
    # In this notebook we will check maps for predicted class with index 206 - "flat-coated retriever"
    retriever_class_index = 206

.. code:: ipython3

    explanation = explainer(
        preprocessed_image,
        targets=retriever_class_index,  # can be a single target or a container of targets
        overlay=True,  # saliency map overlay over the original image, False by default, set to True for better visual inspection
    )

Visualize saliency maps
~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    explanation: Explanation
    # explanation.saliency_map: Dict[int: np.ndarray]  # where key - class id, value - processed saliency map (e.g. 354 x 500 x 3 shape)
    
    # Check saved saliency maps
    print(f"Saliency maps were generated for the following classes: {explanation.targets}")
    print(f"Saliency map size: {explanation.shape}")
    
    # Visualize generated saliency maps for each target class (.plot() supports plotting multiple saliency maps)
    explanation.plot()


.. parsed-literal::

    Saliency maps were generated for the following classes: [206]
    Saliency map size: (224, 224, 3)
    


.. image:: explainable-ai-2-deep-dive-with-output_files/explainable-ai-2-deep-dive-with-output_22_1.png


Save saliency maps
~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    # Save saliency map
    explanation.save(base_artifacts_dir, "explain_auto_")

.. code:: ipython3

    # Plot saved saliency map
    image_sal_map = cv2.imread(f"{base_artifacts_dir}/explain_auto_{retriever_class_index}.jpg")
    image_sal_map = cv2.cvtColor(image_sal_map, cv2.COLOR_BGR2RGB)
    plt.imshow(image_sal_map)




.. parsed-literal::

    <matplotlib.image.AxesImage at 0x7f011efc9090>




.. image:: explainable-ai-2-deep-dive-with-output_files/explainable-ai-2-deep-dive-with-output_25_1.png


Generate saliency maps for all classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



To obtain saliency maps for all classes, set ``targets`` to ``None`` or
``-1``.

.. code:: ipython3

    explanation = explainer(preprocessed_image, targets=-1)
    
    # Check saved saliency maps
    print(f"Saliency maps were generated for the following classes: {explanation.targets[:5]} ... {explanation.targets[-5:]}")
    print(f"Saliency map size: {explanation.shape}")


.. parsed-literal::

    Saliency maps were generated for the following classes: [0, 1, 2, 3, 4] ... [996, 997, 998, 999, 1000]
    Saliency map size: (224, 224, 3)
    

Pre- and post-process functions
-------------------------------



The explainer can apply pre-processing internally during model
inference, allowing you to provide a raw image as input to the
explainer.

To enable this, define ``preprocess_fn`` and provide it to the explainer
constructor. By default, ``preprocess_fn`` is an identity function that
passes the input without any changes, assuming it is preprocessed
beforehand.

In ``AUTO`` mode, the explainer tries to run the ``WHITEBOX`` mode
first. If it fails, the corresponding exception will be raised, and the
``BLACKBOX`` mode will be enabled as a fallback.

The ``BLACKBOX`` mode requires access to the output ``logits``
(activated or not). Therefore, in such cases, ``postprocess_fn`` is
required, which accepts the raw IR model output and returns ``logits``
(see below for a reference).

.. code:: ipython3

    def preprocess_fn(x: np.ndarray) -> np.ndarray:
        # Implementing pre-processing based on model's pipeline
        x = cv2.resize(src=x, dsize=(224, 224))
    
        # Add batch dimension
        x = np.expand_dims(x, 0)
        return x
    
    
    def postprocess_fn(x: OVDict):
        # Implementing post-processing function based on model's pipeline
        # Return "logits" model output
        return x[0]

.. code:: ipython3

    # Create explainer object
    explainer = xai.Explainer(
        model=model,
        task=xai.Task.CLASSIFICATION,
        preprocess_fn=preprocess_fn,
        postprocess_fn=postprocess_fn,
    )
    
    explanation = explainer(image, targets=retriever_class_index)


.. parsed-literal::

    INFO:openvino_xai:Target insertion layer is not provided - trying to find it in auto mode.
    INFO:openvino_xai:Using ReciproCAM method (for CNNs).
    INFO:openvino_xai:Explaining the model in white-box mode.
    

Visualization Parameters
------------------------



-  resize (True by default): If True, resize saliency map to the input
   image size.
-  colormap (True by default): If True, apply colormap to the grayscale
   saliency map.
-  overlay (False by default): If True, generate overlay of the saliency
   map over the input image.
-  original_input_image (None by default): Provide the original,
   unprocessed image to apply the overlay. This ensures the overlay is
   not applied to a preprocessed image, which may be resized or
   normalized and lose readability.
-  overlay_weight (0.5 by default): Weight of the saliency map when
   overlaying the input data with the saliency map.

.. code:: ipython3

    # Create explainer object
    explainer = xai.Explainer(model=model, task=xai.Task.CLASSIFICATION)
    
    # Generate overlayed saliency_map
    explanation = explainer(
        preprocessed_image,
        targets=[retriever_class_index],  # target can be a single label index, label name or a list of indices/names
        overlay=True,  # False by default
        original_input_image=image,  # to apply overlay on the original image instead of preprocessed one that was used for the explainer
    )
    
    explanation.plot()
    
    # Save saliency map
    explanation.save(base_artifacts_dir, "overlay_")


.. parsed-literal::

    INFO:openvino_xai:Assigning preprocess_fn to identity function assumes that input images were already preprocessed by user before passing it to the model. Please define preprocessing function OR preprocess images beforehand.
    INFO:openvino_xai:Target insertion layer is not provided - trying to find it in auto mode.
    INFO:openvino_xai:Using ReciproCAM method (for CNNs).
    INFO:openvino_xai:Explaining the model in white-box mode.
    


.. image:: explainable-ai-2-deep-dive-with-output_files/explainable-ai-2-deep-dive-with-output_34_1.png


.. code:: ipython3

    # Generate saliency map without overlay over original image
    explanation = explainer(
        preprocessed_image,
        targets=[retriever_class_index],  # target can be a single label index, label name or a list of indices/names
        overlay=False,  # False by default
    )
    
    explanation.plot()
    
    # Save saliency map
    explanation.save(base_artifacts_dir, "colormap_")



.. image:: explainable-ai-2-deep-dive-with-output_files/explainable-ai-2-deep-dive-with-output_35_0.png


.. code:: ipython3

    # Return low-resolution (raw) gray-scale saliency map
    explanation = explainer(
        preprocessed_image,
        targets=[retriever_class_index],  # target can be a single label index, label name or a list of indices/names
        resize=False,  # True by default
        colormap=False,  # True by default
    )
    
    explanation.plot()
    
    # Save saliency map
    explanation.save(base_artifacts_dir, "grayscale_")



.. image:: explainable-ai-2-deep-dive-with-output_files/explainable-ai-2-deep-dive-with-output_36_0.png


``Explainer`` in ``WHITEBOX`` mode
----------------------------------



``ReciproCAM`` XAI method
~~~~~~~~~~~~~~~~~~~~~~~~~



``Explainer`` in ``WHITEBOX`` mode treats the model as a white box and
performs its inner modifications. ``Explainer`` inserts extra XAI nodes
after the backbone to estimate which activations are important for model
prediction.

If a method is not specified, the XAI branch will be generated using the
`ReciproCAM <https://arxiv.org/abs/2209.14074>`__ method.

By default, the insertion of the XAI branch will be done automatically
by searching for the correct node - ``target_layer`` (``target_layer``
can be specified manually).

It works quickly and precisely, requiring only one model inference.

.. code:: ipython3

    # Create explainer object
    explainer = xai.Explainer(
        model=model,
        task=xai.Task.CLASSIFICATION,
        preprocess_fn=preprocess_fn,
        explain_mode=ExplainMode.WHITEBOX,  # defaults to ExplainMode.AUTO
        explain_method=xai.Method.RECIPROCAM,  # ReciproCAM is the default white-box method for CNNs
    )


.. parsed-literal::

    INFO:openvino_xai:Target insertion layer is not provided - trying to find it in auto mode.
    INFO:openvino_xai:Using ReciproCAM method (for CNNs).
    INFO:openvino_xai:Explaining the model in white-box mode.
    

Insert XAI branch
~~~~~~~~~~~~~~~~~



It’s possible to update the model with an XAI branch using the
``insert_xai`` functional API.

``insert_xai`` will return an OpenVINO model with the XAI branch
inserted and an additional ``saliency_map`` output.

This helps to avoid OpenVINO XAI dependency in the inference
environment.

**Note**: XAI branch introduce an additional computational overhead
(usually less than a single model forward pass).

.. code:: ipython3

    # insert XAI branch
    model_xai: ov.Model
    model_xai = xai.insert_xai(
        model,
        task=xai.Task.CLASSIFICATION,
        explain_method=xai.Method.RECIPROCAM,
        target_layer="MobilenetV3/Conv_1/Conv2D",  # optional, by default insert_xai will try to find target_layer automatically
        embed_scaling=True,
    )


.. parsed-literal::

    INFO:openvino_xai:Target insertion layer MobilenetV3/Conv_1/Conv2D is provided.
    INFO:openvino_xai:Using ReciproCAM method (for CNNs).
    INFO:openvino_xai:Insertion of the XAI branch into the model was successful.
    

**Note**: ``insert_xai`` supports both OpenVINO IR and PyTorch models.
See documentation for more details.

Insertion-related parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~



If automatic search for correct node fails, you can set up a correct
node manually with ``target_layer`` argument. For classification, it’s
the last backbone node with shape [1, num_channels, feature_map_height,
feature_map_width]. For example, for the used MobileNetV3 it will be
``MobilenetV3/Conv_1/Conv2D`` layer with [1, 576, 7, 7] output shape.

To find the right ``target_layer`` for your model, check the name of the
last convolutional node in the backbone using ``.XML`` file (optionally,
use some graph visualization tool, such as Netron).

``embed_scaling`` **default True** (for speed purposes), this parameter
ensures that saliency map scaling is embedded into the graph, which
results in being able to visualize saliency maps right away without
further postprocessing.

.. code:: ipython3

    # Create explainer object
    explainer = xai.Explainer(
        model=model,
        task=xai.Task.CLASSIFICATION,
        preprocess_fn=preprocess_fn,
        explain_mode=ExplainMode.AUTO,
        explain_method=xai.Method.RECIPROCAM,
        # target_layer="last_conv_node_name",  # target_layer - node after which XAI branch will be inserted
        target_layer="MobilenetV3/Conv_1/Conv2D",
        embed_scaling=True,  # True by default.  If set to True, saliency map scale (0 ~ 255) operation is embedded in the model
    )


.. parsed-literal::

    INFO:openvino_xai:Target insertion layer MobilenetV3/Conv_1/Conv2D is provided.
    INFO:openvino_xai:Using ReciproCAM method (for CNNs).
    INFO:openvino_xai:Explaining the model in white-box mode.
    

``Explainer`` in ``BLACKBOX`` mode
----------------------------------



``Explainer`` in ``BLACKBOX`` mode treats the model as a black box
without altering its internal structure. Therefore, this method will
work on any model that can be inferred and return class scores as
output.

While it is convenient to treat every model as a black box for
explanation purposes, black-box method may require a significant number
of inferences (AISE requires 120-500 model inferences).

Given that the quality of the saliency maps usually correlates with the
number of available inferences, we propose the following presets for the
black-box methods: ``Preset.SPEED``, ``Preset.BALANCE``,
``Preset.QUALITY`` (``Preset.BALANCE`` is used by default).

AISE (Adaptive Input Sampling for Explanation of Black-box Models)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AISE is used as a default black-box method. AISE formulates saliency map
generation as a kernel density estimation (KDE) problem, and adaptively
sample input masks using a derivative-free optimizer to maximize mask
saliency score.

.. code:: ipython3

    # Create explainer object
    explainer = xai.Explainer(
        model=model,
        task=xai.Task.CLASSIFICATION,
        preprocess_fn=preprocess_fn,
        postprocess_fn=postprocess_fn,
        explain_mode=ExplainMode.BLACKBOX,  # defaults to AUTO
    )
    
    # Generate explanation
    explanation = explainer(
        image,
        targets=retriever_class_index,
        overlay=True,
    )


.. parsed-literal::

    INFO:openvino_xai:Explaining the model in black-box mode.
    

.. code:: ipython3

    # Plot saliency map
    explanation.plot()
    
    # Save saliency map
    explanation.save(base_artifacts_dir, "blackbox_aise_")



.. image:: explainable-ai-2-deep-dive-with-output_files/explainable-ai-2-deep-dive-with-output_49_0.png


RISE (Randomized Input Sampling for Explanation of Black-box Models)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`RISE <https://arxiv.org/pdf/1806.07421>`__ probes a model by
sub-sampling the input image via random masks and records its response
to each of them. RISE creates random masks from down-scaled space
(e.g. 7×7 grid) and adds random translation shifts for the pixel-level
explanation with further up-sampling. Weighted sum of all sampled masks
used to generate the fine-grained saliency map.

.. code:: ipython3

    # Create explainer object
    explainer = xai.Explainer(
        model=model,
        task=xai.Task.CLASSIFICATION,
        preprocess_fn=preprocess_fn,
        postprocess_fn=postprocess_fn,
        explain_mode=ExplainMode.BLACKBOX,  # defaults to AUTO
        explain_method=xai.Method.RISE,  # xai.Method.AISE is used by default
    )
    
    # Generate explanation
    explanation = explainer(
        image,
        targets=retriever_class_index,
        overlay=True,
    )

.. code:: ipython3

    # Plot saliency map
    explanation.plot()
    
    # Save saliency map
    explanation.save(base_artifacts_dir, "blackbox_rise_")



.. image:: explainable-ai-2-deep-dive-with-output_files/explainable-ai-2-deep-dive-with-output_52_0.png


Advanced
--------



Import ImageNet label names and add them to saliency maps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



If ``label_names`` are not provided to the explainer call, the saved
saliency map will have the predicted class index, not the label name.
For example, ``206.jpg`` instead of ``retriever.jpg``.

To conveniently view label names in saliency maps, we provide ImageNet
label names information to the explanation call.

.. code:: ipython3

    imagenet_filename = download_file(
        "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/datasets/imagenet/imagenet_2012.txt",
        directory="data",
    )
    
    imagenet_classes = imagenet_filename.read_text().splitlines()


.. parsed-literal::

    'data/imagenet_2012.txt' already exists.
    

.. code:: ipython3

    imagenet_labels = []
    for label in imagenet_classes:
        class_label = " ".join(label.split(" ")[1:])
        first_class_label = class_label.split(",")[0].replace(" ", "_")
        imagenet_labels.append(first_class_label)
    
    print(" ".join(imagenet_labels[:10]))


.. parsed-literal::

    tench goldfish great_white_shark tiger_shark hammerhead electric_ray stingray cock hen ostrich
    

.. code:: ipython3

    # The model description states that for this model, class 0 is a background.
    # Therefore, a background must be added at the beginning of imagenet_classes.
    imagenet_labels = ["background"] + imagenet_labels

.. code:: ipython3

    # Create explainer object
    explainer = xai.Explainer(
        model=model,
        task=xai.Task.CLASSIFICATION,
        preprocess_fn=preprocess_fn,
        explain_mode=ExplainMode.WHITEBOX,
    )
    
    # Adding ImageNet label names.
    explanation = explainer(
        image,
        # Return saliency maps for 2 named labels, possible if label_names is provided
        targets=["flat-coated_retriever", "microwave"],  # slso label indices [206, 652] are possible as target
        label_names=imagenet_labels,
    )


.. parsed-literal::

    INFO:openvino_xai:Target insertion layer is not provided - trying to find it in auto mode.
    INFO:openvino_xai:Using ReciproCAM method (for CNNs).
    INFO:openvino_xai:Explaining the model in white-box mode.
    

.. code:: ipython3

    # Save saliency map
    explanation.save(base_artifacts_dir, "label_names_")

Below in ``base_artifacts_dir / "label_names"`` you can see saved
saliency maps with label name on it:

.. code:: ipython3

    # See saliency mas saved in `output` with predicted label in image name
    for file_name in base_artifacts_dir.glob("label_names_*"):
        print(file_name)


.. parsed-literal::

    artifacts/label_names_microwave.jpg
    artifacts/label_names_flat-coated_retriever.jpg
    

Activation map XAI method
~~~~~~~~~~~~~~~~~~~~~~~~~



The Activation Map method shows a general attention map without respect
to specific classes. It can be useful for understanding which areas the
model identifies as important.

If the explanation method is set to ``Method.ACTIVATIONMAP``, instead of
saliency maps for each class, the activation map is returned as
``explanation.saliency_map["per_image_map"]``.

.. code:: ipython3

    # Create explainer object
    explainer = xai.Explainer(
        model=model,
        task=xai.Task.CLASSIFICATION,
        preprocess_fn=preprocess_fn,
        explain_mode=ExplainMode.WHITEBOX,
        explain_method=xai.Method.ACTIVATIONMAP,
    )
    
    explanation = explainer(image, overlay=True)
    explanation.plot()


.. parsed-literal::

    INFO:openvino_xai:Target insertion layer is not provided - trying to find it in auto mode.
    INFO:openvino_xai:Using ActivationMap method (for CNNs).
    INFO:openvino_xai:Explaining the model in white-box mode.
    


.. image:: explainable-ai-2-deep-dive-with-output_files/explainable-ai-2-deep-dive-with-output_63_1.png

