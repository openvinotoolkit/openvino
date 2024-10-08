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

-  `Basic usage: Auto mode
   explainer <#basic-usage-auto-mode-explainer>`__

   -  `Create Explainer <#create-explainer>`__
   -  `Do explanation <#do-explanation>`__
   -  `Visualize saliency maps <#visualize-saliency-maps>`__
   -  `Save saliency maps <#save-saliency-maps>`__
   -  `Return saliency maps for all
      classes <#return-saliency-maps-for-all-classes>`__

-  `Pre- and post-process
   functions <#pre--and-post-process-functions>`__
-  `Visualization Parameters <#visualization-parameters>`__
-  `White-box explainer <#white-box-explainer>`__

   -  `ReciproCAM explain method <#reciprocam-explain-method>`__
   -  `Insert XAI branch <#insert-xai-branch>`__
   -  `Insertion-related parameters <#insertion-related-parameters>`__

-  `Black-box explainer <#black-box-explainer>`__
-  `Advanced <#advanced>`__

   -  `Import ImageNet label names and add them to saliency
      maps <#import-imagenet-label-names-and-add-them-to-saliency-maps>`__
   -  `Activation map explain method <#activation-map-explain-method>`__

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
    %pip install -q "openvino>=2024.2.0" opencv-python tqdm
    
    %pip install -q --no-deps "openvino-xai>=1.0.0"
    
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


.. parsed-literal::

    v3-small_224_1.0_float already downloaded to artifacts
    

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
    plt.imshow(image);


.. parsed-literal::

    'data/coco.jpg' already exists.
    


.. image:: explainable-ai-2-deep-dive-with-output_files/explainable-ai-2-deep-dive-with-output_10_1.png


Preprocess image for MobileNet
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    # Resize to MobileNetV3 input image shape.
    preprocessed_image = cv2.resize(src=image, dsize=(224, 224))
    # Add batch dimension
    preprocessed_image = np.expand_dims(preprocessed_image, 0)

Basic usage: Auto mode explainer
--------------------------------



The easiest way to run the explainer is to do it in Auto mode. Under the
hood of Auto mode, it will first try to run the ``White-Box`` mode. If
this fails, it will then run the ``Black-Box`` mode. See more details
about `White-Box <#white-box-explainer>`__ and
`Black-Box <#black-box-explainer>`__ modes below.

| Generating saliency maps involves model inference. The explainer will
  perform model inference, but to do so, it requires ``preprocess_fn``
  and ``postprocess_fn``.
| At this stage, we can avoid passing ``preprocess_fn`` by preprocessing
  the data beforehand (e.g., resizing and adding a batch dimension as
  shown above). We also don’t pass ``postprocess_fn`` here for
  simplicity, since the White-Box mode doesn’t fail on the example
  model.

To learn more about pre- and post-process functions, refer to the `Pre-
and post-process functions <#pre--and-post-process-functions>`__
section.

Create Explainer
~~~~~~~~~~~~~~~~



.. code:: ipython3

    # Create ov.Model
    model = ov.Core().read_model(model_xml_path)
    
    # Create explainer object
    explainer = xai.Explainer(
        model=model,
        task=xai.Task.CLASSIFICATION,
    )


.. parsed-literal::

    INFO:openvino_xai:Assigning preprocess_fn to identity function assumes that input images were already preprocessed by user before passing it to the model. Please define preprocessing function OR preprocess images beforehand.
    INFO:openvino_xai:Target insertion layer is not provided - trying to find it in auto mode.
    INFO:openvino_xai:Using ReciproCAM method (for CNNs).
    INFO:openvino_xai:Explaining the model in white-box mode.
    

Do explanation
~~~~~~~~~~~~~~



The predicted label for this image is ``flat-coated_retriever`` with
label index ``206``. So here and further we will check saliency maps for
this index.

.. code:: ipython3

    # You can choose classes to generate saliency maps for.
    # In this notebook we will check maps for predicted class 206 - flat-coated retriever
    retriever_class_index = 206

.. code:: ipython3

    explanation = explainer(
        preprocessed_image,
        targets=retriever_class_index,
        overlay=True,  # False by default
    )

Visualize saliency maps
~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    explanation: Explanation
    # Dict[int: np.ndarray] where key - class id, value - processed saliency map e.g. 354x500x3
    explanation.saliency_map
    
    # Check saved saliency maps
    print(f"Saliency maps were generated for the following classes: {explanation.targets}")
    print(f"Saliency map size: {explanation.shape}")
    
    # Show saliency maps for retriever class
    retriever_sal_map = explanation.saliency_map[retriever_class_index]
    plt.imshow(retriever_sal_map);


.. parsed-literal::

    Saliency maps were generated for the following classes: [206]
    Saliency map size: (224, 224, 3)
    


.. image:: explainable-ai-2-deep-dive-with-output_files/explainable-ai-2-deep-dive-with-output_21_1.png


Save saliency maps
~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    # Save saliency map
    output = base_artifacts_dir / "explain_auto"
    explanation.save(output)

.. code:: ipython3

    # See saved saliency maps
    image_sal_map = cv2.imread(f"{output}/target_{retriever_class_index}.jpg")
    image_sal_map = cv2.cvtColor(image_sal_map, cv2.COLOR_BGR2RGB)
    plt.imshow(image_sal_map);



.. image:: explainable-ai-2-deep-dive-with-output_files/explainable-ai-2-deep-dive-with-output_24_0.png


Return saliency maps for all classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    explanation = explainer(preprocessed_image, targets=-1)
    
    # Check saved saliency maps
    print(f"Saliency maps were generated for the following classes: {explanation.targets}")
    print(f"Saliency map size: {explanation.shape}")


.. parsed-literal::

    Saliency maps were generated for the following classes: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829, 830, 831, 832, 833, 834, 835, 836, 837, 838, 839, 840, 841, 842, 843, 844, 845, 846, 847, 848, 849, 850, 851, 852, 853, 854, 855, 856, 857, 858, 859, 860, 861, 862, 863, 864, 865, 866, 867, 868, 869, 870, 871, 872, 873, 874, 875, 876, 877, 878, 879, 880, 881, 882, 883, 884, 885, 886, 887, 888, 889, 890, 891, 892, 893, 894, 895, 896, 897, 898, 899, 900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911, 912, 913, 914, 915, 916, 917, 918, 919, 920, 921, 922, 923, 924, 925, 926, 927, 928, 929, 930, 931, 932, 933, 934, 935, 936, 937, 938, 939, 940, 941, 942, 943, 944, 945, 946, 947, 948, 949, 950, 951, 952, 953, 954, 955, 956, 957, 958, 959, 960, 961, 962, 963, 964, 965, 966, 967, 968, 969, 970, 971, 972, 973, 974, 975, 976, 977, 978, 979, 980, 981, 982, 983, 984, 985, 986, 987, 988, 989, 990, 991, 992, 993, 994, 995, 996, 997, 998, 999, 1000]
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

In Auto mode, the explainer tries to run the White-Box mode first. If it
fails, the corresponding exception will be raised, and the Black-Box
mode will be enabled as a fallback.

The Black-Box mode requires access to the output ``logits`` (activated
or not). Therefore, in such cases, ``postprocess_fn`` is required, which
accepts the raw IR model output and returns logits (see below for a
reference).

.. code:: ipython3

    def preprocess_fn(x: np.ndarray) -> np.ndarray:
        # Implementing own pre-process function based on model's implementation
        x = cv2.resize(src=x, dsize=(224, 224))
    
        # Add batch dimension
        x = np.expand_dims(x, 0)
        return x
    
    
    def postprocess_fn(x: OVDict):
        # Implementing own post-process function based on model's implementation
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
    
    # Return overlayed image
    explanation = explainer(
        preprocessed_image,
        targets=[retriever_class_index],  # target can be a single label index, label name or a list of indices/names
        overlay=True,  # False by default
        original_input_image=image,  # to apply overlay on the original image instead of preprocessed one that was used for the explainer
    )
    
    retriever_sal_map = explanation.saliency_map[retriever_class_index]
    plt.imshow(retriever_sal_map)
    
    # Save saliency map
    output = base_artifacts_dir / "overlay"
    explanation.save(output)


.. parsed-literal::

    INFO:openvino_xai:Assigning preprocess_fn to identity function assumes that input images were already preprocessed by user before passing it to the model. Please define preprocessing function OR preprocess images beforehand.
    INFO:openvino_xai:Target insertion layer is not provided - trying to find it in auto mode.
    INFO:openvino_xai:Using ReciproCAM method (for CNNs).
    INFO:openvino_xai:Explaining the model in white-box mode.
    


.. image:: explainable-ai-2-deep-dive-with-output_files/explainable-ai-2-deep-dive-with-output_32_1.png


.. code:: ipython3

    # Return low-resolution saliency map
    explanation = explainer(
        preprocessed_image,
        targets=[retriever_class_index],  # target can be a single label index, label name or a list of indices/names
        overlay=False,  # False by default
    )
    
    retriever_sal_map = explanation.saliency_map[retriever_class_index]
    plt.imshow(retriever_sal_map)
    
    # Save saliency map
    output = base_artifacts_dir / "colormap"
    explanation.save(output)



.. image:: explainable-ai-2-deep-dive-with-output_files/explainable-ai-2-deep-dive-with-output_33_0.png


.. code:: ipython3

    # Return low-resolution gray-scale saliency map
    explanation = explainer(
        preprocessed_image,
        targets=[retriever_class_index],  # target can be a single label index, label name or a list of indices/names
        resize=False,  # True by default
        colormap=False,  # True by default
    )
    
    retriever_sal_map = explanation.saliency_map[retriever_class_index]
    plt.imshow(retriever_sal_map, cmap="gray")
    
    # Save saliency map
    output = base_artifacts_dir / "grayscale"
    explanation.save(output)



.. image:: explainable-ai-2-deep-dive-with-output_files/explainable-ai-2-deep-dive-with-output_34_0.png


White-Box explainer
-------------------



ReciproCAM explain method
~~~~~~~~~~~~~~~~~~~~~~~~~



The White-Box explainer treats the model as a white box and needs to
make inner modifications. It adds extra XAI nodes after the backbone to
estimate which activations are important for model prediction.

If a method is not specified, the XAI branch will be generated using the
`ReciproCAM <https://arxiv.org/abs/2209.14074>`__ method.

By default, the insertion of the XAI branch will be done automatically
by searching for the correct node.

It works quickly and precisely, requiring only one model inference.

.. code:: ipython3

    # Create explainer object
    explainer = xai.Explainer(
        model=model,
        task=xai.Task.CLASSIFICATION,
        preprocess_fn=preprocess_fn,
        # defaults to ExplainMode.AUTO
        explain_mode=ExplainMode.WHITEBOX,
        # ReciproCAM is the default XAI method for CNNs
        explain_method=xai.Method.RECIPROCAM,
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
        target_layer="MobilenetV3/Conv_1/Conv2D",  # MobileNet V3
        embed_scaling=True,
    )


.. parsed-literal::

    INFO:openvino_xai:Target insertion layer MobilenetV3/Conv_1/Conv2D is provided.
    INFO:openvino_xai:Using ReciproCAM method (for CNNs).
    INFO:openvino_xai:Insertion of the XAI branch into the model was successful.
    

Insertion-related parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~



If automatic search for correct node fails, you can set up a correct
node manually with ``target_layer`` argument. For classification it’s
the last backbone node with shape [1, num_channels, feature_map_height,
feature_map_width]. For example, for MobileNetV3 it will be
``MobilenetV3/Conv_1/Conv2D`` layer with [1, 576, 7, 7] output shape.

To find the right ``target_layer`` for your model, check the name of the
last convolutional layer in the backbone using ``.XML`` model.

``embed_scaling`` **default True** (for speed purposes), this parameter
adds normalization to the XAI branch, which results in being able to
visualize saliency maps right away without further postprocessing.

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
    

Black-Box explainer
-------------------



The Black-Box method treats the model as a black box without altering
its structure. Therefore, this method will work on any model that can be
inferred and return class probabilities as output.

The `RISE <https://arxiv.org/pdf/1806.07421.pdf>`__ algorithm used in
Black-Box mode applies random masks to hide parts of the image,
retrieves the resulting class probabilities, and uses this information
to calculate the “importance” of each part of the image for the final
results. After performing thousands of inferences, a summarized saliency
map is generated.

While it is convenient to treat every model as a black box for
explanation purposes, this algorithm may require a large number of
inferences (defaulting to 5000) to generate a high-quality saliency map.

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
        # targets=-1,  # Explain all classes
        overlay=True,
        num_masks=1000,  # kwargs of the RISE algo
    )


.. parsed-literal::

    INFO:openvino_xai:Explaining the model in black-box mode.
    Explaining in synchronous mode: 100%|██████████| 1000/1000 [00:03<00:00, 259.73it/s]
    

.. code:: ipython3

    # Save saliency map
    output = base_artifacts_dir / "blackbox_explain"
    explanation.save(output)
    
    # See saved saliency maps
    image_sal_map = cv2.imread(f"{output}/target_{retriever_class_index}.jpg")
    image_sal_map = cv2.cvtColor(image_sal_map, cv2.COLOR_BGR2RGB)
    plt.imshow(image_sal_map);



.. image:: explainable-ai-2-deep-dive-with-output_files/explainable-ai-2-deep-dive-with-output_45_0.png


For the ``Black-Box explainer``, the number of masks and cells is
crucial for achieving good results. In the example above, it’s evident
that the number of masks was insufficient to create a high-quality map.

Varying the ``num_cells`` and ``num_masks`` parameters can achieve
different goals: - To speed up the explanation, you can reduce the
number of ``num_masks``. However, this will decrease the quality of the
resulting saliency maps, making it suitable for large and focused
objects. - Increasing ``num_cells`` provides a more fine-grained result,
but it requires a larger ``num_masks`` to converge. This approach is
more effective for classes with complex shapes.

Advanced
--------



Import ImageNet label names and add them to saliency maps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



If ``label_names`` are not provided to the explainer call, the saved
saliency map will have the predicted class index, not the name. For
example, ``image_name_target_206.jpg`` instead of
``image_name_target_retriever.jpg``.

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
        # Return saliency maps for 2 named labels
        targets=["flat-coated_retriever", "microwave"],  # Also label indices [206, 652] are possible as target
        label_names=imagenet_labels,
    )


.. parsed-literal::

    INFO:openvino_xai:Target insertion layer is not provided - trying to find it in auto mode.
    INFO:openvino_xai:Using ReciproCAM method (for CNNs).
    INFO:openvino_xai:Explaining the model in white-box mode.
    

.. code:: ipython3

    # Save saliency map
    output = base_artifacts_dir / "label_names"
    explanation.save(output)

Below in ``base_artifacts_dir / "label_names"`` you can see saved
saliency maps with label name on it:

.. code:: ipython3

    # See saliency mas saved in `output` with predicted label in image name
    for file_name in output.glob("*"):
        print(file_name)


.. parsed-literal::

    artifacts/label_names/target_microwave.jpg
    artifacts/label_names/target_flat-coated_retriever.jpg
    

Activation map explain method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



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
    
    explanation = explainer(image, targets=-1, overlay=True)
    activation_map = explanation.saliency_map["per_image_map"]
    
    plt.imshow(activation_map)
    plt.show()


.. parsed-literal::

    INFO:openvino_xai:Target insertion layer is not provided - trying to find it in auto mode.
    INFO:openvino_xai:Using ActivationMap method (for CNNs).
    INFO:openvino_xai:Explaining the model in white-box mode.
    


.. image:: explainable-ai-2-deep-dive-with-output_files/explainable-ai-2-deep-dive-with-output_57_1.png

