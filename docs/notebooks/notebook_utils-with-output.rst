Notebook Utils
==============

This notebook contains helper functions and classes for use with
OpenVINO™ Notebooks. The code is synchronized with the
``notebook_utils.py`` file in the same directory as this notebook.

There are five categories:

-  `Files <#files>`__
-  `Images <#images>`__
-  `Videos <#videos>`__
-  `Visualization <#visualization>`__
-  `OpenVINO Tools <#openvino-tools>`__
-  `Checks and Alerts <#checks-and-alerts>`__

Each category contains a test cell that also shows how to use the
functions in the section.

.. code:: ipython3

    # Install requirements
    !pip install -q "openvino>=2023.0.0" opencv-python
    !pip install -q pillow tqdm requests matplotlib


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.0 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.0 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    

Files
-----

Load an image, download a file, download an OpenVINO IR model, and
create a progress bar to show download progress.

.. code:: ipython3

    import os
    import shutil
    from PIL import Image
    from notebook_utils import load_image, download_file, download_ir_model

.. code:: ipython3

    ??load_image

.. code:: ipython3

    ??download_file

.. code:: ipython3

    ??download_ir_model

Test File Functions
~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    model_url = "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/002-example-models/segmentation.xml"
    download_ir_model(model_url, "model")
    
    assert os.path.exists("model/segmentation.xml")
    assert os.path.exists("model/segmentation.bin")



.. parsed-literal::

    model/segmentation.bin:   0%|          | 0.00/1.09M [00:00<?, ?B/s]


.. code:: ipython3

    url = "https://github.com/intel-iot-devkit/safety-gear-detector-python/raw/master/resources/Safety_Full_Hat_and_Vest.mp4"
    if os.path.exists(os.path.basename(url)):
        os.remove(os.path.basename(url))
    video_file = download_file(url)
    print(video_file)
    assert os.path.exists(video_file)



.. parsed-literal::

    Safety_Full_Hat_and_Vest.mp4:   0%|          | 0.00/26.3M [00:00<?, ?B/s]


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/notebooks/utils/Safety_Full_Hat_and_Vest.mp4


.. code:: ipython3

    url = "https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/main/README.md"
    filename = "openvino_notebooks_readme.md"
    if os.path.exists(filename):
        os.remove(filename)
    readme_file = download_file(url, filename=filename)
    print(readme_file)
    assert os.path.exists(readme_file)



.. parsed-literal::

    openvino_notebooks_readme.md:   0%|          | 0.00/13.8k [00:00<?, ?B/s]


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/notebooks/utils/openvino_notebooks_readme.md


.. code:: ipython3

    url = "https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/main/README.md"
    filename = "openvino_notebooks_readme.md"
    directory = "temp"
    video_file = download_file(
        url, filename=filename, directory=directory, show_progress=False, silent=True
    )
    print(readme_file)
    assert os.path.exists(readme_file)
    shutil.rmtree("temp")


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/notebooks/utils/openvino_notebooks_readme.md


.. code:: ipython3

    url = "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco.jpg"
    image = load_image(url)
    Image.fromarray(image[:, :, ::-1])




.. image:: notebook_utils-with-output_files/notebook_utils-with-output_12_0.png



Images
------

Convert Pixel Data
~~~~~~~~~~~~~~~~~~

Normalize image pixel values between 0 and 1, and convert images to
``RGB`` and ``BGR``.

.. code:: ipython3

    import numpy as np
    from notebook_utils import normalize_minmax, to_rgb, to_bgr

.. code:: ipython3

    ??normalize_minmax

.. code:: ipython3

    ??to_bgr

.. code:: ipython3

    ??to_rgb

Test Data Conversion Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    test_array = np.random.randint(0, 255, (100, 100, 3))
    normalized_array = normalize_minmax(test_array)
    
    assert normalized_array.min() == 0
    assert normalized_array.max() == 1

.. code:: ipython3

    bgr_array = np.ones((100, 100, 3), dtype=np.uint8)
    bgr_array[:, :, 0] = 0
    bgr_array[:, :, 1] = 1
    bgr_array[:, :, 2] = 2
    rgb_array = to_rgb(bgr_array)
    
    assert np.all(bgr_array[:, :, 0] == rgb_array[:, :, 2])
    
    bgr_array_converted = to_bgr(rgb_array)
    assert np.all(bgr_array_converted == bgr_array)

Videos
------

Video Player
~~~~~~~~~~~~

A custom video player to fulfill FPS requirements. You can set target
FPS and output size, flip the video horizontally or skip first N frames.

.. code:: ipython3

    import cv2
    from IPython.display import Image, clear_output, display
    from notebook_utils import VideoPlayer
    
    ??VideoPlayer

Test Video Player
~~~~~~~~~~~~~~~~~

.. code:: ipython3

    video = "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/video/Coco%20Walking%20in%20Berkeley.mp4"
    
    player = VideoPlayer(video, fps=15, skip_first_frames=10)
    player.start()
    for i in range(50):
        frame = player.next()
        _, encoded_img = cv2.imencode(".jpg", frame, params=[cv2.IMWRITE_JPEG_QUALITY, 90])
        img = Image(data=encoded_img)
        clear_output(wait=True)
        display(img)
    
    player.stop()
    print("Finished")



.. image:: notebook_utils-with-output_files/notebook_utils-with-output_26_0.png


.. parsed-literal::

    Finished


Visualization
-------------

Segmentation
~~~~~~~~~~~~

Define a ``SegmentationMap NamedTuple`` that keeps the labels and
colormap for a segmentation project/dataset. Create
``CityScapesSegmentation`` and ``BinarySegmentation SegmentationMaps``.
Create a function to convert a segmentation map to an ``RGB`` image with
a ``colormap``, and to show the segmentation result as an overlay over
the original image.

.. code:: ipython3

    from notebook_utils import CityScapesSegmentation, BinarySegmentation, segmentation_map_to_image, segmentation_map_to_overlay

.. code:: ipython3

    ??Label


.. parsed-literal::

    Object `Label` not found.


.. code:: ipython3

    ??SegmentationMap


.. parsed-literal::

    Object `SegmentationMap` not found.


.. code:: ipython3

    ??CityScapesSegmentation

.. code:: ipython3

    print(f"cityscapes segmentation lables: \n{CityScapesSegmentation.get_labels()}")
    print(f"cityscales segmentation colors: \n{CityScapesSegmentation.get_colormap()}")


.. parsed-literal::

    cityscapes segmentation lables: 
    ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'background']
    cityscales segmentation colors: 
    [[128  64 128]
     [244  35 232]
     [ 70  70  70]
     [102 102 156]
     [190 153 153]
     [153 153 153]
     [250 170  30]
     [220 220   0]
     [107 142  35]
     [152 251 152]
     [ 70 130 180]
     [220  20  60]
     [255   0   0]
     [  0   0 142]
     [  0   0  70]
     [  0  60 100]
     [  0  80 100]
     [  0   0 230]
     [119  11  32]
     [255 255 255]]


.. code:: ipython3

    ??BinarySegmentation

.. code:: ipython3

    print(f"binary segmentation lables: \n{BinarySegmentation.get_labels()}")
    print(f"binary segmentation colors: \n{BinarySegmentation.get_colormap()}")


.. parsed-literal::

    binary segmentation lables: 
    ['background', 'foreground']
    binary segmentation colors: 
    [[255 255 255]
     [  0   0   0]]


.. code:: ipython3

    ??segmentation_map_to_image

.. code:: ipython3

    ??segmentation_map_to_overlay

Network Results
~~~~~~~~~~~~~~~

Show network result image, optionally together with the source image and
a legend with labels.

.. code:: ipython3

    from notebook_utils import viz_result_image
    
    ??viz_result_image

Test Visualization Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    testimage = np.zeros((100, 100, 3), dtype=np.uint8)
    testimage[30:80, 30:80, :] = [0, 255, 0]
    testimage[0:10, 0:10, :] = 100
    testimage[40:60, 40:60, :] = 128
    testimage[testimage == 0] = 128
    
    
    testmask1 = np.zeros((testimage.shape[:2]))
    testmask1[30:80, 30:80] = 1
    testmask1[40:50, 40:50] = 0
    testmask1[0:15, 0:10] = 2
    
    result_image_overlay = segmentation_map_to_overlay(
        image=testimage,
        result=testmask1,
        alpha=0.6,
        colormap=np.array([[0, 0, 0], [255, 0, 0], [255, 255, 0]]),
    )
    result_image = segmentation_map_to_image(testmask1, CityScapesSegmentation.get_colormap())
    result_image_no_holes = segmentation_map_to_image(
        testmask1, CityScapesSegmentation.get_colormap(), remove_holes=True
    )
    resized_result_image = cv2.resize(result_image, (50, 50))
    overlay_result_image = segmentation_map_to_overlay(
        testimage, testmask1, 0.6, CityScapesSegmentation.get_colormap(), remove_holes=False
    )
    
    fig1 = viz_result_image(result_image, testimage)
    fig2 = viz_result_image(result_image_no_holes, testimage, labels=CityScapesSegmentation)
    fig3 = viz_result_image(
        resized_result_image,
        testimage,
        source_title="Source Image",
        result_title="Resized Result Image",
        resize=True,
    )
    fig4 = viz_result_image(
        overlay_result_image,
        labels=CityScapesSegmentation,
        result_title="Image with Result Overlay",
    )
    
    display(fig1, fig2, fig3, fig4)



.. image:: notebook_utils-with-output_files/notebook_utils-with-output_41_0.png



.. image:: notebook_utils-with-output_files/notebook_utils-with-output_41_1.png



.. image:: notebook_utils-with-output_files/notebook_utils-with-output_41_2.png



.. image:: notebook_utils-with-output_files/notebook_utils-with-output_41_3.png


Checks and Alerts
-----------------

Create an alert class to show stylized info/error/warning messages and a
``check_device`` function that checks whether a given device is
available.

.. code:: ipython3

    from notebook_utils import NotebookAlert, DeviceNotFoundAlert, check_device, check_openvino_version

.. code:: ipython3

    ??NotebookAlert

.. code:: ipython3

    ??DeviceNotFoundAlert

.. code:: ipython3

    ??check_device

.. code:: ipython3

    ??check_openvino_version

Test Alerts
~~~~~~~~~~~

.. code:: ipython3

    NotebookAlert(message="Hello, world!", alert_class="info")
    DeviceNotFoundAlert("GPU");



.. raw:: html

    <div class="alert alert-info">Hello, world!



.. raw:: html

    <div class="alert alert-warning">Running this cell requires a GPU device, which is not available on this system. The following device is available: CPU


.. code:: ipython3

    assert check_device("CPU")

.. code:: ipython3

    if check_device("HELLOWORLD"):
        print("Hello World device found.")



.. raw:: html

    <div class="alert alert-warning">Running this cell requires a HELLOWORLD device, which is not available on this system. The following device is available: CPU


.. code:: ipython3

    check_openvino_version("2022.1");



.. raw:: html

    <div class="alert alert-danger">This notebook requires OpenVINO 2022.1. The version on your system is: <i>2023.1.0-12185-9e6b00e51cd-releases/2023/1</i>.<br>Please run <span style='font-family:monospace'>pip install --upgrade -r requirements.txt</span> in the openvino_env environment to install this version. See the <a href='https://github.com/openvinotoolkit/openvino_notebooks'>OpenVINO Notebooks README</a> for detailed instructions

