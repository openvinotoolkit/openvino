Working with Open Model Zoo Models
==================================

This tutorial shows how to download a model from `Open Model
Zoo <https://github.com/openvinotoolkit/open_model_zoo>`__, convert it
to OpenVINO™ IR format, show information about the model, and benchmark
the model.

Table of contents:
^^^^^^^^^^^^^^^^^^

-  `OpenVINO and Open Model Zoo
   Tools <#openvino-and-open-model-zoo-tools>`__
-  `Preparation <#preparation>`__

   -  `Model Name <#model-name>`__
   -  `Imports <#imports>`__
   -  `Settings and Configuration <#settings-and-configuration>`__

-  `Download a Model from Open Model
   Zoo <#download-a-model-from-open-model-zoo>`__
-  `Convert a Model to OpenVINO IR
   format <#convert-a-model-to-openvino-ir-format>`__
-  `Get Model Information <#get-model-information>`__
-  `Run Benchmark Tool <#run-benchmark-tool>`__

   -  `Benchmark with Different
      Settings <#benchmark-with-different-settings>`__

OpenVINO and Open Model Zoo Tools
---------------------------------



OpenVINO and Open Model Zoo tools are listed in the table below.

+------------+--------------+-----------------------------------------+
| Tool       | Command      | Description                             |
+============+==============+=========================================+
| Model      | ``omz_downlo | Download models from Open Model Zoo.    |
| Downloader | ader``       |                                         |
+------------+--------------+-----------------------------------------+
| Model      | ``omz_conver | Convert Open Model Zoo models to        |
| Converter  | ter``        | OpenVINO’s IR format.                   |
+------------+--------------+-----------------------------------------+
| Info       | ``omz_info_d | Print information about Open Model Zoo  |
| Dumper     | umper``      | models.                                 |
+------------+--------------+-----------------------------------------+
| Benchmark  | ``benchmark_ | Benchmark model performance by          |
| Tool       | app``        | computing inference time.               |
+------------+--------------+-----------------------------------------+

.. code:: ipython3

    # Install openvino package
    %pip install -q "openvino-dev>=2024.0.0" torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


Preparation
-----------



Model Name
~~~~~~~~~~



Set ``model_name`` to the name of the Open Model Zoo model to use in
this notebook. Refer to the list of
`public <https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/index.md>`__
and
`Intel <https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/intel/index.md>`__
pre-trained models for a full list of models that can be used. Set
``model_name`` to the model you want to use.

.. code:: ipython3

    # model_name = "resnet-50-pytorch"
    model_name = "mobilenet-v2-pytorch"

Imports
~~~~~~~



.. code:: ipython3

    import json
    from pathlib import Path

    import openvino as ov
    from IPython.display import Markdown, display

    # Fetch `notebook_utils` module
    import requests

    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
    )

    open("notebook_utils.py", "w").write(r.text)
    from notebook_utils import DeviceNotFoundAlert, NotebookAlert

Settings and Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~



Set the file and directory paths. By default, this notebook downloads
models from Open Model Zoo to the ``open_model_zoo_models`` directory in
your ``$HOME`` directory. On Windows, the $HOME directory is usually
``c:\users\username``, on Linux ``/home/username``. To change the
folder, change ``base_model_dir`` in the cell below.

The following settings can be changed:

-  ``base_model_dir``: Models will be downloaded into the ``intel`` and
   ``public`` folders in this directory.
-  ``omz_cache_dir``: Cache folder for Open Model Zoo. Specifying a
   cache directory is not required for Model Downloader and Model
   Converter, but it speeds up subsequent downloads.
-  ``precision``: If specified, only models with this precision will be
   downloaded and converted.

.. code:: ipython3

    base_model_dir = Path("model")
    omz_cache_dir = Path("cache")
    precision = "FP16"

    # Check if an GPU is available on this system to use with Benchmark App.
    core = ov.Core()
    gpu_available = "GPU" in core.available_devices

    print(f"base_model_dir: {base_model_dir}, omz_cache_dir: {omz_cache_dir}, gpu_availble: {gpu_available}")


.. parsed-literal::

    base_model_dir: model, omz_cache_dir: cache, gpu_availble: False


Download a Model from Open Model Zoo
------------------------------------



Specify, display and run the Model Downloader command to download the
model.

.. code:: ipython3

    ## Uncomment the next line to show help in omz_downloader which explains the command-line options.

    # !omz_downloader --help

.. code:: ipython3

    download_command = f"omz_downloader --name {model_name} --output_dir {base_model_dir} --cache_dir {omz_cache_dir}"
    display(Markdown(f"Download command: `{download_command}`"))
    display(Markdown(f"Downloading {model_name}..."))
    ! $download_command



Download command:
``omz_downloader --name mobilenet-v2-pytorch --output_dir model --cache_dir cache``



Downloading mobilenet-v2-pytorch…


.. parsed-literal::

    ################|| Downloading mobilenet-v2-pytorch ||################

    ========== Downloading model/public/mobilenet-v2-pytorch/mobilenet_v2-b0353104.pth


.. parsed-literal::

    ... 0%, 32 KB, 954 KB/s, 0 seconds passed

.. parsed-literal::

    ... 0%, 64 KB, 986 KB/s, 0 seconds passed
... 0%, 96 KB, 1460 KB/s, 0 seconds passed
... 0%, 128 KB, 1315 KB/s, 0 seconds passed
... 1%, 160 KB, 1628 KB/s, 0 seconds passed
... 1%, 192 KB, 1916 KB/s, 0 seconds passed
... 1%, 224 KB, 2215 KB/s, 0 seconds passed
... 1%, 256 KB, 2510 KB/s, 0 seconds passed

.. parsed-literal::

    ... 2%, 288 KB, 2214 KB/s, 0 seconds passed
... 2%, 320 KB, 2450 KB/s, 0 seconds passed
... 2%, 352 KB, 2672 KB/s, 0 seconds passed
... 2%, 384 KB, 2894 KB/s, 0 seconds passed
... 2%, 416 KB, 3115 KB/s, 0 seconds passed
... 3%, 448 KB, 3332 KB/s, 0 seconds passed
... 3%, 480 KB, 3548 KB/s, 0 seconds passed
... 3%, 512 KB, 3763 KB/s, 0 seconds passed
... 3%, 544 KB, 3959 KB/s, 0 seconds passed
... 4%, 576 KB, 4178 KB/s, 0 seconds passed

.. parsed-literal::

    ... 4%, 608 KB, 3723 KB/s, 0 seconds passed
... 4%, 640 KB, 3907 KB/s, 0 seconds passed
... 4%, 672 KB, 4092 KB/s, 0 seconds passed
... 5%, 704 KB, 4274 KB/s, 0 seconds passed
... 5%, 736 KB, 4456 KB/s, 0 seconds passed
... 5%, 768 KB, 4638 KB/s, 0 seconds passed
... 5%, 800 KB, 4820 KB/s, 0 seconds passed
... 5%, 832 KB, 4998 KB/s, 0 seconds passed
... 6%, 864 KB, 5177 KB/s, 0 seconds passed
... 6%, 896 KB, 5355 KB/s, 0 seconds passed
... 6%, 928 KB, 5535 KB/s, 0 seconds passed
... 6%, 960 KB, 5712 KB/s, 0 seconds passed
... 7%, 992 KB, 5889 KB/s, 0 seconds passed
... 7%, 1024 KB, 6067 KB/s, 0 seconds passed
... 7%, 1056 KB, 6244 KB/s, 0 seconds passed
... 7%, 1088 KB, 6422 KB/s, 0 seconds passed
... 8%, 1120 KB, 6599 KB/s, 0 seconds passed
... 8%, 1152 KB, 6753 KB/s, 0 seconds passed
... 8%, 1184 KB, 6924 KB/s, 0 seconds passed
... 8%, 1216 KB, 7097 KB/s, 0 seconds passed
... 8%, 1248 KB, 6362 KB/s, 0 seconds passed
... 9%, 1280 KB, 6509 KB/s, 0 seconds passed
... 9%, 1312 KB, 6657 KB/s, 0 seconds passed
... 9%, 1344 KB, 6806 KB/s, 0 seconds passed
... 9%, 1376 KB, 6952 KB/s, 0 seconds passed
... 10%, 1408 KB, 7096 KB/s, 0 seconds passed
... 10%, 1440 KB, 7243 KB/s, 0 seconds passed
... 10%, 1472 KB, 7392 KB/s, 0 seconds passed
... 10%, 1504 KB, 7542 KB/s, 0 seconds passed
... 11%, 1536 KB, 7691 KB/s, 0 seconds passed
... 11%, 1568 KB, 7686 KB/s, 0 seconds passed
... 11%, 1600 KB, 7827 KB/s, 0 seconds passed
... 11%, 1632 KB, 7969 KB/s, 0 seconds passed
... 11%, 1664 KB, 8109 KB/s, 0 seconds passed
... 12%, 1696 KB, 8249 KB/s, 0 seconds passed
... 12%, 1728 KB, 8389 KB/s, 0 seconds passed
... 12%, 1760 KB, 8528 KB/s, 0 seconds passed
... 12%, 1792 KB, 8667 KB/s, 0 seconds passed
... 13%, 1824 KB, 8805 KB/s, 0 seconds passed
... 13%, 1856 KB, 8944 KB/s, 0 seconds passed
... 13%, 1888 KB, 9081 KB/s, 0 seconds passed
... 13%, 1920 KB, 9218 KB/s, 0 seconds passed

.. parsed-literal::

    ... 14%, 1952 KB, 9354 KB/s, 0 seconds passed
... 14%, 1984 KB, 9491 KB/s, 0 seconds passed
... 14%, 2016 KB, 9627 KB/s, 0 seconds passed
... 14%, 2048 KB, 9762 KB/s, 0 seconds passed
... 14%, 2080 KB, 9897 KB/s, 0 seconds passed
... 15%, 2112 KB, 10030 KB/s, 0 seconds passed
... 15%, 2144 KB, 10162 KB/s, 0 seconds passed
... 15%, 2176 KB, 10296 KB/s, 0 seconds passed
... 15%, 2208 KB, 10429 KB/s, 0 seconds passed
... 16%, 2240 KB, 10565 KB/s, 0 seconds passed
... 16%, 2272 KB, 10701 KB/s, 0 seconds passed
... 16%, 2304 KB, 10837 KB/s, 0 seconds passed
... 16%, 2336 KB, 10973 KB/s, 0 seconds passed
... 17%, 2368 KB, 11108 KB/s, 0 seconds passed
... 17%, 2400 KB, 11243 KB/s, 0 seconds passed
... 17%, 2432 KB, 11377 KB/s, 0 seconds passed
... 17%, 2464 KB, 11512 KB/s, 0 seconds passed
... 17%, 2496 KB, 10926 KB/s, 0 seconds passed
... 18%, 2528 KB, 11043 KB/s, 0 seconds passed
... 18%, 2560 KB, 11164 KB/s, 0 seconds passed
... 18%, 2592 KB, 11288 KB/s, 0 seconds passed
... 18%, 2624 KB, 11012 KB/s, 0 seconds passed
... 19%, 2656 KB, 11120 KB/s, 0 seconds passed
... 19%, 2688 KB, 11233 KB/s, 0 seconds passed
... 19%, 2720 KB, 11348 KB/s, 0 seconds passed
... 19%, 2752 KB, 11463 KB/s, 0 seconds passed
... 20%, 2784 KB, 11578 KB/s, 0 seconds passed
... 20%, 2816 KB, 11693 KB/s, 0 seconds passed
... 20%, 2848 KB, 11807 KB/s, 0 seconds passed
... 20%, 2880 KB, 11920 KB/s, 0 seconds passed
... 20%, 2912 KB, 12033 KB/s, 0 seconds passed
... 21%, 2944 KB, 12146 KB/s, 0 seconds passed
... 21%, 2976 KB, 12258 KB/s, 0 seconds passed
... 21%, 3008 KB, 12370 KB/s, 0 seconds passed
... 21%, 3040 KB, 12483 KB/s, 0 seconds passed
... 22%, 3072 KB, 12595 KB/s, 0 seconds passed
... 22%, 3104 KB, 12705 KB/s, 0 seconds passed
... 22%, 3136 KB, 12816 KB/s, 0 seconds passed
... 22%, 3168 KB, 12926 KB/s, 0 seconds passed
... 23%, 3200 KB, 13037 KB/s, 0 seconds passed
... 23%, 3232 KB, 13146 KB/s, 0 seconds passed
... 23%, 3264 KB, 13256 KB/s, 0 seconds passed
... 23%, 3296 KB, 13364 KB/s, 0 seconds passed
... 23%, 3328 KB, 13474 KB/s, 0 seconds passed
... 24%, 3360 KB, 13584 KB/s, 0 seconds passed
... 24%, 3392 KB, 13694 KB/s, 0 seconds passed
... 24%, 3424 KB, 13802 KB/s, 0 seconds passed
... 24%, 3456 KB, 13912 KB/s, 0 seconds passed
... 25%, 3488 KB, 14021 KB/s, 0 seconds passed
... 25%, 3520 KB, 14130 KB/s, 0 seconds passed
... 25%, 3552 KB, 14239 KB/s, 0 seconds passed
... 25%, 3584 KB, 14346 KB/s, 0 seconds passed
... 26%, 3616 KB, 14454 KB/s, 0 seconds passed
... 26%, 3648 KB, 14559 KB/s, 0 seconds passed
... 26%, 3680 KB, 14664 KB/s, 0 seconds passed
... 26%, 3712 KB, 14771 KB/s, 0 seconds passed
... 26%, 3744 KB, 14879 KB/s, 0 seconds passed
... 27%, 3776 KB, 14990 KB/s, 0 seconds passed
... 27%, 3808 KB, 15101 KB/s, 0 seconds passed
... 27%, 3840 KB, 15211 KB/s, 0 seconds passed
... 27%, 3872 KB, 15322 KB/s, 0 seconds passed
... 28%, 3904 KB, 15430 KB/s, 0 seconds passed
... 28%, 3936 KB, 15540 KB/s, 0 seconds passed
... 28%, 3968 KB, 15650 KB/s, 0 seconds passed
... 28%, 4000 KB, 15759 KB/s, 0 seconds passed
... 29%, 4032 KB, 15869 KB/s, 0 seconds passed
... 29%, 4064 KB, 15977 KB/s, 0 seconds passed
... 29%, 4096 KB, 16086 KB/s, 0 seconds passed
... 29%, 4128 KB, 16195 KB/s, 0 seconds passed

.. parsed-literal::

    ... 29%, 4160 KB, 15477 KB/s, 0 seconds passed
... 30%, 4192 KB, 15566 KB/s, 0 seconds passed
... 30%, 4224 KB, 15661 KB/s, 0 seconds passed
... 30%, 4256 KB, 15757 KB/s, 0 seconds passed
... 30%, 4288 KB, 15850 KB/s, 0 seconds passed
... 31%, 4320 KB, 15946 KB/s, 0 seconds passed
... 31%, 4352 KB, 16042 KB/s, 0 seconds passed
... 31%, 4384 KB, 16138 KB/s, 0 seconds passed
... 31%, 4416 KB, 16232 KB/s, 0 seconds passed
... 32%, 4448 KB, 16327 KB/s, 0 seconds passed
... 32%, 4480 KB, 16420 KB/s, 0 seconds passed
... 32%, 4512 KB, 16515 KB/s, 0 seconds passed
... 32%, 4544 KB, 16609 KB/s, 0 seconds passed
... 32%, 4576 KB, 16703 KB/s, 0 seconds passed
... 33%, 4608 KB, 16797 KB/s, 0 seconds passed
... 33%, 4640 KB, 16888 KB/s, 0 seconds passed
... 33%, 4672 KB, 16980 KB/s, 0 seconds passed
... 33%, 4704 KB, 17073 KB/s, 0 seconds passed
... 34%, 4736 KB, 17166 KB/s, 0 seconds passed
... 34%, 4768 KB, 17257 KB/s, 0 seconds passed
... 34%, 4800 KB, 17350 KB/s, 0 seconds passed
... 34%, 4832 KB, 17443 KB/s, 0 seconds passed
... 35%, 4864 KB, 17537 KB/s, 0 seconds passed
... 35%, 4896 KB, 17631 KB/s, 0 seconds passed
... 35%, 4928 KB, 17722 KB/s, 0 seconds passed
... 35%, 4960 KB, 17813 KB/s, 0 seconds passed
... 35%, 4992 KB, 17905 KB/s, 0 seconds passed
... 36%, 5024 KB, 17997 KB/s, 0 seconds passed
... 36%, 5056 KB, 18089 KB/s, 0 seconds passed
... 36%, 5088 KB, 18181 KB/s, 0 seconds passed
... 36%, 5120 KB, 18273 KB/s, 0 seconds passed
... 37%, 5152 KB, 18365 KB/s, 0 seconds passed
... 37%, 5184 KB, 18455 KB/s, 0 seconds passed
... 37%, 5216 KB, 18546 KB/s, 0 seconds passed
... 37%, 5248 KB, 18636 KB/s, 0 seconds passed
... 38%, 5280 KB, 18726 KB/s, 0 seconds passed
... 38%, 5312 KB, 18817 KB/s, 0 seconds passed
... 38%, 5344 KB, 18907 KB/s, 0 seconds passed
... 38%, 5376 KB, 18996 KB/s, 0 seconds passed
... 38%, 5408 KB, 19086 KB/s, 0 seconds passed
... 39%, 5440 KB, 19175 KB/s, 0 seconds passed
... 39%, 5472 KB, 19265 KB/s, 0 seconds passed
... 39%, 5504 KB, 19355 KB/s, 0 seconds passed
... 39%, 5536 KB, 19444 KB/s, 0 seconds passed
... 40%, 5568 KB, 19533 KB/s, 0 seconds passed
... 40%, 5600 KB, 19623 KB/s, 0 seconds passed
... 40%, 5632 KB, 19712 KB/s, 0 seconds passed
... 40%, 5664 KB, 19800 KB/s, 0 seconds passed
... 41%, 5696 KB, 19887 KB/s, 0 seconds passed
... 41%, 5728 KB, 19982 KB/s, 0 seconds passed
... 41%, 5760 KB, 20077 KB/s, 0 seconds passed
... 41%, 5792 KB, 20173 KB/s, 0 seconds passed
... 41%, 5824 KB, 20268 KB/s, 0 seconds passed
... 42%, 5856 KB, 20363 KB/s, 0 seconds passed
... 42%, 5888 KB, 20458 KB/s, 0 seconds passed
... 42%, 5920 KB, 20552 KB/s, 0 seconds passed
... 42%, 5952 KB, 20646 KB/s, 0 seconds passed
... 43%, 5984 KB, 20740 KB/s, 0 seconds passed
... 43%, 6016 KB, 20834 KB/s, 0 seconds passed
... 43%, 6048 KB, 20928 KB/s, 0 seconds passed
... 43%, 6080 KB, 21023 KB/s, 0 seconds passed
... 44%, 6112 KB, 21116 KB/s, 0 seconds passed
... 44%, 6144 KB, 21210 KB/s, 0 seconds passed
... 44%, 6176 KB, 21303 KB/s, 0 seconds passed
... 44%, 6208 KB, 21397 KB/s, 0 seconds passed
... 44%, 6240 KB, 21488 KB/s, 0 seconds passed
... 45%, 6272 KB, 21581 KB/s, 0 seconds passed
... 45%, 6304 KB, 21674 KB/s, 0 seconds passed
... 45%, 6336 KB, 21768 KB/s, 0 seconds passed
... 45%, 6368 KB, 21860 KB/s, 0 seconds passed
... 46%, 6400 KB, 21953 KB/s, 0 seconds passed
... 46%, 6432 KB, 22045 KB/s, 0 seconds passed
... 46%, 6464 KB, 22137 KB/s, 0 seconds passed
... 46%, 6496 KB, 22229 KB/s, 0 seconds passed
... 47%, 6528 KB, 22321 KB/s, 0 seconds passed
... 47%, 6560 KB, 22413 KB/s, 0 seconds passed
... 47%, 6592 KB, 22505 KB/s, 0 seconds passed
... 47%, 6624 KB, 22597 KB/s, 0 seconds passed
... 47%, 6656 KB, 22689 KB/s, 0 seconds passed
... 48%, 6688 KB, 22779 KB/s, 0 seconds passed
... 48%, 6720 KB, 22871 KB/s, 0 seconds passed
... 48%, 6752 KB, 22962 KB/s, 0 seconds passed
... 48%, 6784 KB, 23053 KB/s, 0 seconds passed
... 49%, 6816 KB, 23143 KB/s, 0 seconds passed
... 49%, 6848 KB, 23232 KB/s, 0 seconds passed
... 49%, 6880 KB, 23322 KB/s, 0 seconds passed
... 49%, 6912 KB, 23413 KB/s, 0 seconds passed
... 50%, 6944 KB, 23501 KB/s, 0 seconds passed
... 50%, 6976 KB, 23591 KB/s, 0 seconds passed
... 50%, 7008 KB, 23680 KB/s, 0 seconds passed
... 50%, 7040 KB, 23769 KB/s, 0 seconds passed
... 50%, 7072 KB, 23858 KB/s, 0 seconds passed
... 51%, 7104 KB, 23948 KB/s, 0 seconds passed
... 51%, 7136 KB, 24038 KB/s, 0 seconds passed
... 51%, 7168 KB, 24134 KB/s, 0 seconds passed
... 51%, 7200 KB, 24229 KB/s, 0 seconds passed
... 52%, 7232 KB, 24324 KB/s, 0 seconds passed
... 52%, 7264 KB, 24420 KB/s, 0 seconds passed
... 52%, 7296 KB, 24515 KB/s, 0 seconds passed
... 52%, 7328 KB, 24609 KB/s, 0 seconds passed
... 53%, 7360 KB, 24703 KB/s, 0 seconds passed
... 53%, 7392 KB, 24798 KB/s, 0 seconds passed
... 53%, 7424 KB, 24893 KB/s, 0 seconds passed
... 53%, 7456 KB, 24986 KB/s, 0 seconds passed
... 53%, 7488 KB, 25080 KB/s, 0 seconds passed
... 54%, 7520 KB, 25174 KB/s, 0 seconds passed
... 54%, 7552 KB, 25268 KB/s, 0 seconds passed
... 54%, 7584 KB, 25362 KB/s, 0 seconds passed
... 54%, 7616 KB, 25455 KB/s, 0 seconds passed
... 55%, 7648 KB, 25549 KB/s, 0 seconds passed
... 55%, 7680 KB, 25643 KB/s, 0 seconds passed
... 55%, 7712 KB, 25735 KB/s, 0 seconds passed
... 55%, 7744 KB, 25819 KB/s, 0 seconds passed
... 56%, 7776 KB, 25904 KB/s, 0 seconds passed
... 56%, 7808 KB, 25983 KB/s, 0 seconds passed
... 56%, 7840 KB, 26066 KB/s, 0 seconds passed
... 56%, 7872 KB, 26150 KB/s, 0 seconds passed
... 56%, 7904 KB, 26229 KB/s, 0 seconds passed
... 57%, 7936 KB, 26317 KB/s, 0 seconds passed
... 57%, 7968 KB, 26395 KB/s, 0 seconds passed
... 57%, 8000 KB, 26478 KB/s, 0 seconds passed
... 57%, 8032 KB, 26560 KB/s, 0 seconds passed
... 58%, 8064 KB, 26639 KB/s, 0 seconds passed
... 58%, 8096 KB, 26721 KB/s, 0 seconds passed
... 58%, 8128 KB, 26803 KB/s, 0 seconds passed
... 58%, 8160 KB, 26881 KB/s, 0 seconds passed
... 59%, 8192 KB, 26958 KB/s, 0 seconds passed
... 59%, 8224 KB, 27040 KB/s, 0 seconds passed
... 59%, 8256 KB, 27122 KB/s, 0 seconds passed
... 59%, 8288 KB, 27199 KB/s, 0 seconds passed
... 59%, 8320 KB, 27280 KB/s, 0 seconds passed
... 60%, 8352 KB, 27362 KB/s, 0 seconds passed
... 60%, 8384 KB, 27443 KB/s, 0 seconds passed
... 60%, 8416 KB, 27524 KB/s, 0 seconds passed
... 60%, 8448 KB, 27605 KB/s, 0 seconds passed
... 61%, 8480 KB, 27686 KB/s, 0 seconds passed
... 61%, 8512 KB, 27765 KB/s, 0 seconds passed
... 61%, 8544 KB, 27841 KB/s, 0 seconds passed
... 61%, 8576 KB, 27922 KB/s, 0 seconds passed
... 62%, 8608 KB, 28002 KB/s, 0 seconds passed
... 62%, 8640 KB, 28077 KB/s, 0 seconds passed
... 62%, 8672 KB, 28152 KB/s, 0 seconds passed
... 62%, 8704 KB, 28228 KB/s, 0 seconds passed
... 62%, 8736 KB, 28307 KB/s, 0 seconds passed
... 63%, 8768 KB, 28387 KB/s, 0 seconds passed
... 63%, 8800 KB, 28462 KB/s, 0 seconds passed
... 63%, 8832 KB, 28541 KB/s, 0 seconds passed
... 63%, 8864 KB, 28615 KB/s, 0 seconds passed
... 64%, 8896 KB, 28694 KB/s, 0 seconds passed
... 64%, 8928 KB, 28778 KB/s, 0 seconds passed
... 64%, 8960 KB, 28852 KB/s, 0 seconds passed
... 64%, 8992 KB, 28931 KB/s, 0 seconds passed
... 65%, 9024 KB, 29009 KB/s, 0 seconds passed

.. parsed-literal::

    ... 65%, 9056 KB, 29087 KB/s, 0 seconds passed
... 65%, 9088 KB, 29161 KB/s, 0 seconds passed
... 65%, 9120 KB, 29239 KB/s, 0 seconds passed
... 65%, 9152 KB, 29317 KB/s, 0 seconds passed
... 66%, 9184 KB, 29278 KB/s, 0 seconds passed
... 66%, 9216 KB, 29355 KB/s, 0 seconds passed
... 66%, 9248 KB, 29433 KB/s, 0 seconds passed
... 66%, 9280 KB, 29505 KB/s, 0 seconds passed
... 67%, 9312 KB, 29582 KB/s, 0 seconds passed
... 67%, 9344 KB, 29659 KB/s, 0 seconds passed
... 67%, 9376 KB, 29731 KB/s, 0 seconds passed
... 67%, 9408 KB, 29807 KB/s, 0 seconds passed
... 68%, 9440 KB, 29874 KB/s, 0 seconds passed
... 68%, 9472 KB, 29950 KB/s, 0 seconds passed
... 68%, 9504 KB, 30022 KB/s, 0 seconds passed
... 68%, 9536 KB, 30102 KB/s, 0 seconds passed
... 68%, 9568 KB, 30174 KB/s, 0 seconds passed
... 69%, 9600 KB, 30250 KB/s, 0 seconds passed
... 69%, 9632 KB, 30325 KB/s, 0 seconds passed
... 69%, 9664 KB, 30400 KB/s, 0 seconds passed
... 69%, 9696 KB, 30476 KB/s, 0 seconds passed
... 70%, 9728 KB, 30551 KB/s, 0 seconds passed
... 70%, 9760 KB, 30621 KB/s, 0 seconds passed
... 70%, 9792 KB, 30696 KB/s, 0 seconds passed
... 70%, 9824 KB, 30772 KB/s, 0 seconds passed
... 71%, 9856 KB, 30831 KB/s, 0 seconds passed
... 71%, 9888 KB, 30906 KB/s, 0 seconds passed
... 71%, 9920 KB, 30980 KB/s, 0 seconds passed
... 71%, 9952 KB, 31055 KB/s, 0 seconds passed
... 71%, 9984 KB, 31124 KB/s, 0 seconds passed
... 72%, 10016 KB, 31198 KB/s, 0 seconds passed
... 72%, 10048 KB, 31273 KB/s, 0 seconds passed
... 72%, 10080 KB, 31346 KB/s, 0 seconds passed
... 72%, 10112 KB, 29928 KB/s, 0 seconds passed
... 73%, 10144 KB, 29976 KB/s, 0 seconds passed
... 73%, 10176 KB, 30032 KB/s, 0 seconds passed
... 73%, 10208 KB, 30090 KB/s, 0 seconds passed
... 73%, 10240 KB, 30151 KB/s, 0 seconds passed
... 74%, 10272 KB, 30213 KB/s, 0 seconds passed
... 74%, 10304 KB, 30275 KB/s, 0 seconds passed
... 74%, 10336 KB, 30337 KB/s, 0 seconds passed
... 74%, 10368 KB, 30399 KB/s, 0 seconds passed
... 74%, 10400 KB, 30460 KB/s, 0 seconds passed
... 75%, 10432 KB, 30521 KB/s, 0 seconds passed
... 75%, 10464 KB, 30581 KB/s, 0 seconds passed
... 75%, 10496 KB, 30638 KB/s, 0 seconds passed
... 75%, 10528 KB, 30698 KB/s, 0 seconds passed
... 76%, 10560 KB, 30759 KB/s, 0 seconds passed
... 76%, 10592 KB, 30820 KB/s, 0 seconds passed
... 76%, 10624 KB, 30880 KB/s, 0 seconds passed
... 76%, 10656 KB, 30941 KB/s, 0 seconds passed
... 77%, 10688 KB, 31001 KB/s, 0 seconds passed
... 77%, 10720 KB, 31059 KB/s, 0 seconds passed
... 77%, 10752 KB, 31117 KB/s, 0 seconds passed
... 77%, 10784 KB, 31177 KB/s, 0 seconds passed
... 77%, 10816 KB, 31236 KB/s, 0 seconds passed
... 78%, 10848 KB, 31292 KB/s, 0 seconds passed
... 78%, 10880 KB, 31351 KB/s, 0 seconds passed
... 78%, 10912 KB, 31410 KB/s, 0 seconds passed
... 78%, 10944 KB, 31473 KB/s, 0 seconds passed
... 79%, 10976 KB, 31537 KB/s, 0 seconds passed
... 79%, 11008 KB, 31603 KB/s, 0 seconds passed
... 79%, 11040 KB, 31668 KB/s, 0 seconds passed
... 79%, 11072 KB, 31733 KB/s, 0 seconds passed
... 80%, 11104 KB, 31798 KB/s, 0 seconds passed
... 80%, 11136 KB, 31863 KB/s, 0 seconds passed
... 80%, 11168 KB, 31926 KB/s, 0 seconds passed
... 80%, 11200 KB, 31991 KB/s, 0 seconds passed
... 80%, 11232 KB, 32051 KB/s, 0 seconds passed
... 81%, 11264 KB, 32113 KB/s, 0 seconds passed
... 81%, 11296 KB, 32178 KB/s, 0 seconds passed
... 81%, 11328 KB, 32242 KB/s, 0 seconds passed
... 81%, 11360 KB, 32306 KB/s, 0 seconds passed
... 82%, 11392 KB, 32370 KB/s, 0 seconds passed
... 82%, 11424 KB, 32434 KB/s, 0 seconds passed
... 82%, 11456 KB, 32498 KB/s, 0 seconds passed
... 82%, 11488 KB, 32560 KB/s, 0 seconds passed
... 82%, 11520 KB, 32624 KB/s, 0 seconds passed
... 83%, 11552 KB, 32688 KB/s, 0 seconds passed
... 83%, 11584 KB, 32751 KB/s, 0 seconds passed
... 83%, 11616 KB, 32815 KB/s, 0 seconds passed
... 83%, 11648 KB, 32878 KB/s, 0 seconds passed
... 84%, 11680 KB, 32939 KB/s, 0 seconds passed
... 84%, 11712 KB, 33003 KB/s, 0 seconds passed
... 84%, 11744 KB, 33064 KB/s, 0 seconds passed
... 84%, 11776 KB, 33127 KB/s, 0 seconds passed
... 85%, 11808 KB, 33187 KB/s, 0 seconds passed
... 85%, 11840 KB, 33249 KB/s, 0 seconds passed
... 85%, 11872 KB, 33312 KB/s, 0 seconds passed
... 85%, 11904 KB, 33373 KB/s, 0 seconds passed
... 85%, 11936 KB, 33437 KB/s, 0 seconds passed
... 86%, 11968 KB, 33497 KB/s, 0 seconds passed
... 86%, 12000 KB, 33560 KB/s, 0 seconds passed
... 86%, 12032 KB, 33630 KB/s, 0 seconds passed
... 86%, 12064 KB, 33700 KB/s, 0 seconds passed
... 87%, 12096 KB, 33771 KB/s, 0 seconds passed
... 87%, 12128 KB, 33841 KB/s, 0 seconds passed
... 87%, 12160 KB, 33911 KB/s, 0 seconds passed
... 87%, 12192 KB, 33982 KB/s, 0 seconds passed
... 88%, 12224 KB, 34050 KB/s, 0 seconds passed
... 88%, 12256 KB, 34120 KB/s, 0 seconds passed
... 88%, 12288 KB, 34190 KB/s, 0 seconds passed
... 88%, 12320 KB, 34261 KB/s, 0 seconds passed
... 88%, 12352 KB, 34331 KB/s, 0 seconds passed
... 89%, 12384 KB, 34400 KB/s, 0 seconds passed
... 89%, 12416 KB, 34468 KB/s, 0 seconds passed
... 89%, 12448 KB, 34537 KB/s, 0 seconds passed
... 89%, 12480 KB, 34603 KB/s, 0 seconds passed
... 90%, 12512 KB, 34673 KB/s, 0 seconds passed
... 90%, 12544 KB, 34742 KB/s, 0 seconds passed
... 90%, 12576 KB, 34811 KB/s, 0 seconds passed
... 90%, 12608 KB, 34880 KB/s, 0 seconds passed
... 91%, 12640 KB, 34949 KB/s, 0 seconds passed
... 91%, 12672 KB, 35019 KB/s, 0 seconds passed
... 91%, 12704 KB, 35088 KB/s, 0 seconds passed
... 91%, 12736 KB, 35156 KB/s, 0 seconds passed

.. parsed-literal::

    ... 91%, 12768 KB, 35223 KB/s, 0 seconds passed
... 92%, 12800 KB, 35291 KB/s, 0 seconds passed
... 92%, 12832 KB, 35360 KB/s, 0 seconds passed
... 92%, 12864 KB, 35428 KB/s, 0 seconds passed
... 92%, 12896 KB, 35497 KB/s, 0 seconds passed
... 93%, 12928 KB, 35567 KB/s, 0 seconds passed
... 93%, 12960 KB, 35635 KB/s, 0 seconds passed
... 93%, 12992 KB, 35702 KB/s, 0 seconds passed
... 93%, 13024 KB, 35771 KB/s, 0 seconds passed
... 94%, 13056 KB, 35840 KB/s, 0 seconds passed
... 94%, 13088 KB, 35909 KB/s, 0 seconds passed
... 94%, 13120 KB, 35977 KB/s, 0 seconds passed
... 94%, 13152 KB, 36045 KB/s, 0 seconds passed
... 94%, 13184 KB, 36114 KB/s, 0 seconds passed
... 95%, 13216 KB, 36181 KB/s, 0 seconds passed
... 95%, 13248 KB, 36248 KB/s, 0 seconds passed
... 95%, 13280 KB, 36316 KB/s, 0 seconds passed
... 95%, 13312 KB, 36385 KB/s, 0 seconds passed
... 96%, 13344 KB, 36452 KB/s, 0 seconds passed
... 96%, 13376 KB, 36514 KB/s, 0 seconds passed
... 96%, 13408 KB, 36574 KB/s, 0 seconds passed
... 96%, 13440 KB, 36635 KB/s, 0 seconds passed
... 97%, 13472 KB, 36696 KB/s, 0 seconds passed
... 97%, 13504 KB, 36752 KB/s, 0 seconds passed
... 97%, 13536 KB, 36814 KB/s, 0 seconds passed
... 97%, 13568 KB, 36875 KB/s, 0 seconds passed
... 97%, 13600 KB, 36930 KB/s, 0 seconds passed
... 98%, 13632 KB, 36991 KB/s, 0 seconds passed
... 98%, 13664 KB, 37051 KB/s, 0 seconds passed
... 98%, 13696 KB, 37112 KB/s, 0 seconds passed
... 98%, 13728 KB, 37166 KB/s, 0 seconds passed
... 99%, 13760 KB, 37226 KB/s, 0 seconds passed
... 99%, 13792 KB, 37282 KB/s, 0 seconds passed
... 99%, 13824 KB, 37341 KB/s, 0 seconds passed
... 99%, 13856 KB, 37401 KB/s, 0 seconds passed
... 100%, 13879 KB, 37444 KB/s, 0 seconds passed




Convert a Model to OpenVINO IR format
-------------------------------------



Specify, display and run the Model Converter command to convert the
model to OpenVINO IR format. Model conversion may take a while. The
output of the Model Converter command will be displayed. When the
conversion is successful, the last lines of the output will include:
``[ SUCCESS ] Generated IR version 11 model.`` For downloaded models
that are already in OpenVINO IR format, conversion will be skipped.

.. code:: ipython3

    ## Uncomment the next line to show Help in omz_converter which explains the command-line options.

    # !omz_converter --help

.. code:: ipython3

    convert_command = f"omz_converter --name {model_name} --precisions {precision} --download_dir {base_model_dir} --output_dir {base_model_dir}"
    display(Markdown(f"Convert command: `{convert_command}`"))
    display(Markdown(f"Converting {model_name}..."))

    ! $convert_command



Convert command:
``omz_converter --name mobilenet-v2-pytorch --precisions FP16 --download_dir model --output_dir model``



Converting mobilenet-v2-pytorch…


.. parsed-literal::

    ========== Converting mobilenet-v2-pytorch to ONNX
    Conversion to ONNX command: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-661/.workspace/scm/ov-notebook/.venv/bin/python -- /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-661/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/omz_tools/internal_scripts/pytorch_to_onnx.py --model-name=mobilenet_v2 --weights=model/public/mobilenet-v2-pytorch/mobilenet_v2-b0353104.pth --import-module=torchvision.models --input-shape=1,3,224,224 --output-file=model/public/mobilenet-v2-pytorch/mobilenet-v2.onnx --input-names=data --output-names=prob



.. parsed-literal::

    ONNX check passed successfully.


.. parsed-literal::


    ========== Converting mobilenet-v2-pytorch to IR (FP16)
    Conversion command: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-661/.workspace/scm/ov-notebook/.venv/bin/python -- /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-661/.workspace/scm/ov-notebook/.venv/bin/mo --framework=onnx --output_dir=model/public/mobilenet-v2-pytorch/FP16 --model_name=mobilenet-v2-pytorch --input=data '--mean_values=data[123.675,116.28,103.53]' '--scale_values=data[58.624,57.12,57.375]' --reverse_input_channels --output=prob --input_model=model/public/mobilenet-v2-pytorch/mobilenet-v2.onnx '--layout=data(NCHW)' '--input_shape=[1, 3, 224, 224]' --compress_to_fp16=True



.. parsed-literal::

    [ INFO ] Generated IR will be compressed to FP16. If you get lower accuracy, please consider disabling compression explicitly by adding argument --compress_to_fp16=False.
    Find more information about compression to FP16 at https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_FP16_Compression.html
    [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release. Please use OpenVINO Model Converter (OVC). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
    Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
    [ SUCCESS ] Generated IR version 11 model.
    [ SUCCESS ] XML file: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-661/.workspace/scm/ov-notebook/notebooks/model-tools/model/public/mobilenet-v2-pytorch/FP16/mobilenet-v2-pytorch.xml
    [ SUCCESS ] BIN file: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-661/.workspace/scm/ov-notebook/notebooks/model-tools/model/public/mobilenet-v2-pytorch/FP16/mobilenet-v2-pytorch.bin





Get Model Information
---------------------



The Info Dumper prints the following information for Open Model Zoo
models:

-  Model name
-  Description
-  Framework that was used to train the model
-  License URL
-  Precisions supported by the model
-  Subdirectory: the location of the downloaded model
-  Task type

This information can be shown by running
``omz_info_dumper --name model_name`` in a terminal. The information can
also be parsed and used in scripts.

In the next cell, run Info Dumper and use ``json`` to load the
information in a dictionary.

.. code:: ipython3

    model_info_output = %sx omz_info_dumper --name $model_name
    model_info = json.loads(model_info_output.get_nlstr())

    if len(model_info) > 1:
        NotebookAlert(
            f"There are multiple IR files for the {model_name} model. The first model in the "
            "omz_info_dumper output will be used for benchmarking. Change "
            "`selected_model_info` in the cell below to select a different model from the list.",
            "warning",
        )

    model_info




.. parsed-literal::

    [{'name': 'mobilenet-v2-pytorch',
      'composite_model_name': None,
      'description': 'MobileNet V2 is image classification model pre-trained on ImageNet dataset. This is a PyTorch* implementation of MobileNetV2 architecture as described in the paper "Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification, Detection and Segmentation" <https://arxiv.org/abs/1801.04381>.\nThe model input is a blob that consists of a single image of "1, 3, 224, 224" in "RGB" order.\nThe model output is typical object classifier for the 1000 different classifications matching with those in the ImageNet database.',
      'framework': 'pytorch',
      'license_url': 'https://raw.githubusercontent.com/pytorch/vision/master/LICENSE',
      'accuracy_config': '/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-661/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/omz_tools/models/public/mobilenet-v2-pytorch/accuracy-check.yml',
      'model_config': '/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-661/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/omz_tools/models/public/mobilenet-v2-pytorch/model.yml',
      'precisions': ['FP16', 'FP32'],
      'subdirectory': 'public/mobilenet-v2-pytorch',
      'task_type': 'classification',
      'input_info': [{'name': 'data',
        'shape': [1, 3, 224, 224],
        'layout': 'NCHW'}],
      'model_stages': []}]



Having information of the model in a JSON file enables extraction of the
path to the model directory, and building the path to the OpenVINO IR
file.

.. code:: ipython3

    selected_model_info = model_info[0]
    model_path = base_model_dir / Path(selected_model_info["subdirectory"]) / Path(f"{precision}/{selected_model_info['name']}.xml")
    print(model_path, "exists:", model_path.exists())


.. parsed-literal::

    model/public/mobilenet-v2-pytorch/FP16/mobilenet-v2-pytorch.xml exists: True


Run Benchmark Tool
------------------



By default, Benchmark Tool runs inference for 60 seconds in asynchronous
mode on CPU. It returns inference speed as latency (milliseconds per
image) and throughput values (frames per second).

.. code:: ipython3

    ## Uncomment the next line to show Help in benchmark_app which explains the command-line options.
    # !benchmark_app --help

.. code:: ipython3

    benchmark_command = f"benchmark_app -m {model_path} -t 15"
    display(Markdown(f"Benchmark command: `{benchmark_command}`"))
    display(Markdown(f"Benchmarking {model_name} on CPU with async inference for 15 seconds..."))

    ! $benchmark_command



Benchmark command:
``benchmark_app -m model/public/mobilenet-v2-pytorch/FP16/mobilenet-v2-pytorch.xml -t 15``



Benchmarking mobilenet-v2-pytorch on CPU with async inference for 15
seconds…


.. parsed-literal::

    [Step 1/11] Parsing and validating input arguments
    [ INFO ] Parsing input parameters
    [Step 2/11] Loading OpenVINO Runtime


.. parsed-literal::

    [ INFO ] OpenVINO:
    [ INFO ] Build ................................. 2024.0.0-14509-34caeefd078-releases/2024/0
    [ INFO ]
    [ INFO ] Device info:
    [ INFO ] CPU
    [ INFO ] Build ................................. 2024.0.0-14509-34caeefd078-releases/2024/0
    [ INFO ]
    [ INFO ]
    [Step 3/11] Setting device configuration
    [ WARNING ] Performance hint was not explicitly specified in command line. Device(CPU) performance hint will be set to PerformanceMode.THROUGHPUT.
    [Step 4/11] Reading model files
    [ INFO ] Loading model files


.. parsed-literal::

    [ INFO ] Read model took 28.84 ms
    [ INFO ] Original model I/O parameters:
    [ INFO ] Model inputs:
    [ INFO ]     data (node: data) : f32 / [N,C,H,W] / [1,3,224,224]
    [ INFO ] Model outputs:
    [ INFO ]     prob (node: prob) : f32 / [...] / [1,1000]
    [Step 5/11] Resizing model to match image sizes and given batch
    [ INFO ] Model batch size: 1
    [Step 6/11] Configuring input of the model
    [ INFO ] Model inputs:
    [ INFO ]     data (node: data) : u8 / [N,C,H,W] / [1,3,224,224]
    [ INFO ] Model outputs:
    [ INFO ]     prob (node: prob) : f32 / [...] / [1,1000]
    [Step 7/11] Loading the model to the device


.. parsed-literal::

    [ INFO ] Compile model took 154.21 ms
    [Step 8/11] Querying optimal runtime parameters
    [ INFO ] Model:
    [ INFO ]   NETWORK_NAME: main_graph
    [ INFO ]   OPTIMAL_NUMBER_OF_INFER_REQUESTS: 6
    [ INFO ]   NUM_STREAMS: 6
    [ INFO ]   AFFINITY: Affinity.CORE
    [ INFO ]   INFERENCE_NUM_THREADS: 24
    [ INFO ]   PERF_COUNT: NO
    [ INFO ]   INFERENCE_PRECISION_HINT: <Type: 'float32'>
    [ INFO ]   PERFORMANCE_HINT: THROUGHPUT
    [ INFO ]   EXECUTION_MODE_HINT: ExecutionMode.PERFORMANCE
    [ INFO ]   PERFORMANCE_HINT_NUM_REQUESTS: 0
    [ INFO ]   ENABLE_CPU_PINNING: True
    [ INFO ]   SCHEDULING_CORE_TYPE: SchedulingCoreType.ANY_CORE
    [ INFO ]   ENABLE_HYPER_THREADING: True
    [ INFO ]   EXECUTION_DEVICES: ['CPU']
    [ INFO ]   CPU_DENORMALS_OPTIMIZATION: False
    [ INFO ]   LOG_LEVEL: Level.NO
    [ INFO ]   CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE: 1.0
    [ INFO ]   DYNAMIC_QUANTIZATION_GROUP_SIZE: 0
    [ INFO ]   KV_CACHE_PRECISION: <Type: 'float16'>
    [Step 9/11] Creating infer requests and preparing input tensors
    [ WARNING ] No input files were given for input 'data'!. This input will be filled with random values!
    [ INFO ] Fill input 'data' with random values
    [Step 10/11] Measuring performance (Start inference asynchronously, 6 inference requests, limits: 15000 ms duration)
    [ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).
    [ INFO ] First inference took 6.50 ms


.. parsed-literal::

    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            20394 iterations
    [ INFO ] Duration:         15004.33 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        4.29 ms
    [ INFO ]    Average:       4.29 ms
    [ INFO ]    Min:           2.36 ms
    [ INFO ]    Max:           12.12 ms
    [ INFO ] Throughput:   1359.21 FPS


Benchmark with Different Settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



The ``benchmark_app`` tool displays logging information that is not
always necessary. A more compact result is achieved when the output is
parsed with ``json``.

The following cells show some examples of ``benchmark_app`` with
different parameters. Below are some useful parameters:

-  ``-d`` A device to use for inference. For example: CPU, GPU, MULTI.
   Default: CPU.
-  ``-t`` Time expressed in number of seconds to run inference. Default:
   60.
-  ``-api`` Use asynchronous (async) or synchronous (sync) inference.
   Default: async.
-  ``-b`` Batch size. Default: 1.

Run ``! benchmark_app --help`` to get an overview of all possible
command-line parameters.

In the next cell, define the ``benchmark_model()`` function that calls
``benchmark_app``. This makes it easy to try different combinations. In
the cell below that, you display available devices on the system.

   **Note**: In this notebook, ``benchmark_app`` runs for 15 seconds to
   give a quick indication of performance. For more accurate
   performance, it is recommended to run inference for at least one
   minute by setting the ``t`` parameter to 60 or higher, and run
   ``benchmark_app`` in a terminal/command prompt after closing other
   applications. Copy the **benchmark command** and paste it in a
   command prompt where you have activated the ``openvino_env``
   environment.

.. code:: ipython3

    def benchmark_model(model_xml, device="CPU", seconds=60, api="async", batch=1):
        core = ov.Core()
        model_path = Path(model_xml)
        if ("GPU" in device) and ("GPU" not in core.available_devices):
            DeviceNotFoundAlert("GPU")
        else:
            benchmark_command = f"benchmark_app -m {model_path} -d {device} -t {seconds} -api {api} -b {batch}"
            display(Markdown(f"**Benchmark {model_path.name} with {device} for {seconds} seconds with {api} inference**"))
            display(Markdown(f"Benchmark command: `{benchmark_command}`"))

            benchmark_output = %sx $benchmark_command
            print("command ended")
            benchmark_result = [line for line in benchmark_output if not (line.startswith(r"[") or line.startswith("      ") or line == "")]
            print("\n".join(benchmark_result))

.. code:: ipython3

    core = ov.Core()

    # Show devices available for OpenVINO Runtime
    for device in core.available_devices:
        device_name = core.get_property(device, "FULL_DEVICE_NAME")
        print(f"{device}: {device_name}")


.. parsed-literal::

    CPU: Intel(R) Core(TM) i9-10920X CPU @ 3.50GHz


You can select inference device using device widget

.. code:: ipython3

    import ipywidgets as widgets

    device = widgets.Dropdown(
        options=core.available_devices + ["AUTO"],
        value="CPU",
        description="Device:",
        disabled=False,
    )

    device




.. parsed-literal::

    Dropdown(description='Device:', options=('CPU', 'AUTO'), value='CPU')



.. code:: ipython3

    benchmark_model(model_path, device=device.value, seconds=15, api="async")



**Benchmark mobilenet-v2-pytorch.xml with CPU for 15 seconds with async
inference**



Benchmark command:
``benchmark_app -m model/public/mobilenet-v2-pytorch/FP16/mobilenet-v2-pytorch.xml -d CPU -t 15 -api async -b 1``


.. parsed-literal::

    command ended


