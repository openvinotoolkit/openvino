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
    %pip install -q "openvino-dev>=2024.0.0"


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
    import urllib.request
    urllib.request.urlretrieve(
        url='https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/main/notebooks/utils/notebook_utils.py',
        filename='notebook_utils.py'
    )
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

    print(
        f"base_model_dir: {base_model_dir}, omz_cache_dir: {omz_cache_dir}, gpu_availble: {gpu_available}"
    )


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

    download_command = (
        f"omz_downloader --name {model_name} --output_dir {base_model_dir} --cache_dir {omz_cache_dir}"
    )
    display(Markdown(f"Download command: `{download_command}`"))
    display(Markdown(f"Downloading {model_name}..."))
    ! $download_command



Download command:
``omz_downloader --name mobilenet-v2-pytorch --output_dir model --cache_dir cache``



Downloading mobilenet-v2-pytorch…


.. parsed-literal::

    ################|| Downloading mobilenet-v2-pytorch ||################



.. parsed-literal::

    ========== Downloading model/public/mobilenet-v2-pytorch/mobilenet_v2-b0353104.pth


.. parsed-literal::

    ... 0%, 32 KB, 967 KB/s, 0 seconds passed
... 0%, 64 KB, 961 KB/s, 0 seconds passed
... 0%, 96 KB, 1403 KB/s, 0 seconds passed

.. parsed-literal::

    ... 0%, 128 KB, 1260 KB/s, 0 seconds passed
... 1%, 160 KB, 1556 KB/s, 0 seconds passed
... 1%, 192 KB, 1836 KB/s, 0 seconds passed
... 1%, 224 KB, 2101 KB/s, 0 seconds passed
... 1%, 256 KB, 2357 KB/s, 0 seconds passed

.. parsed-literal::

    ... 2%, 288 KB, 2128 KB/s, 0 seconds passed
... 2%, 320 KB, 2358 KB/s, 0 seconds passed
... 2%, 352 KB, 2586 KB/s, 0 seconds passed
... 2%, 384 KB, 2807 KB/s, 0 seconds passed
... 2%, 416 KB, 3017 KB/s, 0 seconds passed
... 3%, 448 KB, 3232 KB/s, 0 seconds passed
... 3%, 480 KB, 3455 KB/s, 0 seconds passed
... 3%, 512 KB, 3567 KB/s, 0 seconds passed
... 3%, 544 KB, 3779 KB/s, 0 seconds passed
... 4%, 576 KB, 3992 KB/s, 0 seconds passed
... 4%, 608 KB, 3572 KB/s, 0 seconds passed
... 4%, 640 KB, 3750 KB/s, 0 seconds passed
... 4%, 672 KB, 3929 KB/s, 0 seconds passed
... 5%, 704 KB, 4108 KB/s, 0 seconds passed

.. parsed-literal::

    ... 5%, 736 KB, 4238 KB/s, 0 seconds passed
... 5%, 768 KB, 4413 KB/s, 0 seconds passed
... 5%, 800 KB, 4589 KB/s, 0 seconds passed
... 5%, 832 KB, 4764 KB/s, 0 seconds passed
... 6%, 864 KB, 4939 KB/s, 0 seconds passed
... 6%, 896 KB, 5112 KB/s, 0 seconds passed
... 6%, 928 KB, 5286 KB/s, 0 seconds passed
... 6%, 960 KB, 5459 KB/s, 0 seconds passed
... 7%, 992 KB, 5598 KB/s, 0 seconds passed
... 7%, 1024 KB, 5765 KB/s, 0 seconds passed
... 7%, 1056 KB, 5935 KB/s, 0 seconds passed
... 7%, 1088 KB, 6103 KB/s, 0 seconds passed
... 8%, 1120 KB, 6272 KB/s, 0 seconds passed
... 8%, 1152 KB, 6441 KB/s, 0 seconds passed
... 8%, 1184 KB, 5729 KB/s, 0 seconds passed
... 8%, 1216 KB, 5869 KB/s, 0 seconds passed
... 8%, 1248 KB, 6013 KB/s, 0 seconds passed
... 9%, 1280 KB, 6159 KB/s, 0 seconds passed
... 9%, 1312 KB, 6304 KB/s, 0 seconds passed
... 9%, 1344 KB, 6448 KB/s, 0 seconds passed
... 9%, 1376 KB, 6591 KB/s, 0 seconds passed
... 10%, 1408 KB, 6735 KB/s, 0 seconds passed
... 10%, 1440 KB, 6878 KB/s, 0 seconds passed
... 10%, 1472 KB, 7021 KB/s, 0 seconds passed
... 10%, 1504 KB, 7163 KB/s, 0 seconds passed
... 11%, 1536 KB, 7305 KB/s, 0 seconds passed
... 11%, 1568 KB, 7447 KB/s, 0 seconds passed
... 11%, 1600 KB, 7588 KB/s, 0 seconds passed
... 11%, 1632 KB, 7728 KB/s, 0 seconds passed
... 11%, 1664 KB, 7869 KB/s, 0 seconds passed
... 12%, 1696 KB, 8008 KB/s, 0 seconds passed
... 12%, 1728 KB, 8148 KB/s, 0 seconds passed
... 12%, 1760 KB, 8288 KB/s, 0 seconds passed
... 12%, 1792 KB, 8426 KB/s, 0 seconds passed
... 13%, 1824 KB, 8565 KB/s, 0 seconds passed
... 13%, 1856 KB, 8703 KB/s, 0 seconds passed
... 13%, 1888 KB, 8841 KB/s, 0 seconds passed
... 13%, 1920 KB, 8978 KB/s, 0 seconds passed
... 14%, 1952 KB, 9115 KB/s, 0 seconds passed
... 14%, 1984 KB, 9251 KB/s, 0 seconds passed
... 14%, 2016 KB, 9388 KB/s, 0 seconds passed
... 14%, 2048 KB, 9524 KB/s, 0 seconds passed
... 14%, 2080 KB, 9658 KB/s, 0 seconds passed
... 15%, 2112 KB, 9793 KB/s, 0 seconds passed
... 15%, 2144 KB, 9927 KB/s, 0 seconds passed
... 15%, 2176 KB, 10061 KB/s, 0 seconds passed
... 15%, 2208 KB, 10195 KB/s, 0 seconds passed
... 16%, 2240 KB, 10329 KB/s, 0 seconds passed
... 16%, 2272 KB, 10462 KB/s, 0 seconds passed
... 16%, 2304 KB, 10596 KB/s, 0 seconds passed
... 16%, 2336 KB, 10729 KB/s, 0 seconds passed
... 17%, 2368 KB, 10862 KB/s, 0 seconds passed

.. parsed-literal::

    ... 17%, 2400 KB, 10090 KB/s, 0 seconds passed
... 17%, 2432 KB, 10125 KB/s, 0 seconds passed
... 17%, 2464 KB, 10240 KB/s, 0 seconds passed
... 17%, 2496 KB, 10358 KB/s, 0 seconds passed
... 18%, 2528 KB, 10477 KB/s, 0 seconds passed
... 18%, 2560 KB, 10596 KB/s, 0 seconds passed
... 18%, 2592 KB, 10715 KB/s, 0 seconds passed
... 18%, 2624 KB, 10833 KB/s, 0 seconds passed
... 19%, 2656 KB, 10951 KB/s, 0 seconds passed
... 19%, 2688 KB, 11070 KB/s, 0 seconds passed
... 19%, 2720 KB, 11187 KB/s, 0 seconds passed
... 19%, 2752 KB, 11303 KB/s, 0 seconds passed
... 20%, 2784 KB, 11421 KB/s, 0 seconds passed
... 20%, 2816 KB, 11537 KB/s, 0 seconds passed
... 20%, 2848 KB, 11655 KB/s, 0 seconds passed
... 20%, 2880 KB, 11771 KB/s, 0 seconds passed
... 20%, 2912 KB, 11888 KB/s, 0 seconds passed
... 21%, 2944 KB, 12004 KB/s, 0 seconds passed
... 21%, 2976 KB, 12120 KB/s, 0 seconds passed
... 21%, 3008 KB, 12235 KB/s, 0 seconds passed
... 21%, 3040 KB, 12350 KB/s, 0 seconds passed
... 22%, 3072 KB, 12465 KB/s, 0 seconds passed
... 22%, 3104 KB, 12579 KB/s, 0 seconds passed
... 22%, 3136 KB, 12693 KB/s, 0 seconds passed
... 22%, 3168 KB, 12806 KB/s, 0 seconds passed
... 23%, 3200 KB, 12920 KB/s, 0 seconds passed
... 23%, 3232 KB, 13034 KB/s, 0 seconds passed
... 23%, 3264 KB, 13147 KB/s, 0 seconds passed
... 23%, 3296 KB, 13260 KB/s, 0 seconds passed
... 23%, 3328 KB, 13373 KB/s, 0 seconds passed
... 24%, 3360 KB, 13486 KB/s, 0 seconds passed
... 24%, 3392 KB, 13598 KB/s, 0 seconds passed
... 24%, 3424 KB, 13710 KB/s, 0 seconds passed
... 24%, 3456 KB, 13821 KB/s, 0 seconds passed
... 25%, 3488 KB, 13931 KB/s, 0 seconds passed
... 25%, 3520 KB, 14043 KB/s, 0 seconds passed
... 25%, 3552 KB, 14154 KB/s, 0 seconds passed
... 25%, 3584 KB, 14263 KB/s, 0 seconds passed
... 26%, 3616 KB, 14373 KB/s, 0 seconds passed
... 26%, 3648 KB, 14481 KB/s, 0 seconds passed
... 26%, 3680 KB, 14591 KB/s, 0 seconds passed
... 26%, 3712 KB, 14700 KB/s, 0 seconds passed
... 26%, 3744 KB, 14815 KB/s, 0 seconds passed
... 27%, 3776 KB, 14929 KB/s, 0 seconds passed
... 27%, 3808 KB, 15044 KB/s, 0 seconds passed
... 27%, 3840 KB, 15159 KB/s, 0 seconds passed
... 27%, 3872 KB, 15273 KB/s, 0 seconds passed
... 28%, 3904 KB, 15386 KB/s, 0 seconds passed
... 28%, 3936 KB, 15499 KB/s, 0 seconds passed
... 28%, 3968 KB, 15603 KB/s, 0 seconds passed
... 28%, 4000 KB, 15712 KB/s, 0 seconds passed
... 29%, 4032 KB, 15821 KB/s, 0 seconds passed
... 29%, 4064 KB, 15926 KB/s, 0 seconds passed
... 29%, 4096 KB, 16023 KB/s, 0 seconds passed
... 29%, 4128 KB, 16109 KB/s, 0 seconds passed
... 29%, 4160 KB, 16206 KB/s, 0 seconds passed
... 30%, 4192 KB, 16303 KB/s, 0 seconds passed
... 30%, 4224 KB, 16402 KB/s, 0 seconds passed
... 30%, 4256 KB, 16502 KB/s, 0 seconds passed
... 30%, 4288 KB, 16607 KB/s, 0 seconds passed
... 31%, 4320 KB, 16713 KB/s, 0 seconds passed
... 31%, 4352 KB, 16818 KB/s, 0 seconds passed
... 31%, 4384 KB, 16578 KB/s, 0 seconds passed
... 31%, 4416 KB, 16670 KB/s, 0 seconds passed
... 32%, 4448 KB, 16767 KB/s, 0 seconds passed
... 32%, 4480 KB, 16862 KB/s, 0 seconds passed
... 32%, 4512 KB, 16958 KB/s, 0 seconds passed
... 32%, 4544 KB, 17053 KB/s, 0 seconds passed
... 32%, 4576 KB, 17153 KB/s, 0 seconds passed
... 33%, 4608 KB, 17254 KB/s, 0 seconds passed
... 33%, 4640 KB, 17106 KB/s, 0 seconds passed
... 33%, 4672 KB, 17187 KB/s, 0 seconds passed
... 33%, 4704 KB, 17278 KB/s, 0 seconds passed
... 34%, 4736 KB, 17372 KB/s, 0 seconds passed
... 34%, 4768 KB, 17465 KB/s, 0 seconds passed
... 34%, 4800 KB, 17561 KB/s, 0 seconds passed
... 34%, 4832 KB, 17659 KB/s, 0 seconds passed
... 35%, 4864 KB, 17749 KB/s, 0 seconds passed
... 35%, 4896 KB, 17841 KB/s, 0 seconds passed
... 35%, 4928 KB, 17935 KB/s, 0 seconds passed

.. parsed-literal::

    ... 35%, 4960 KB, 17758 KB/s, 0 seconds passed
... 35%, 4992 KB, 17845 KB/s, 0 seconds passed
... 36%, 5024 KB, 17939 KB/s, 0 seconds passed
... 36%, 5056 KB, 18033 KB/s, 0 seconds passed
... 36%, 5088 KB, 18127 KB/s, 0 seconds passed
... 36%, 5120 KB, 18222 KB/s, 0 seconds passed
... 37%, 5152 KB, 18316 KB/s, 0 seconds passed
... 37%, 5184 KB, 18410 KB/s, 0 seconds passed
... 37%, 5216 KB, 18504 KB/s, 0 seconds passed
... 37%, 5248 KB, 18598 KB/s, 0 seconds passed
... 38%, 5280 KB, 18679 KB/s, 0 seconds passed
... 38%, 5312 KB, 18744 KB/s, 0 seconds passed
... 38%, 5344 KB, 18835 KB/s, 0 seconds passed
... 38%, 5376 KB, 18929 KB/s, 0 seconds passed
... 38%, 5408 KB, 19021 KB/s, 0 seconds passed
... 39%, 5440 KB, 19110 KB/s, 0 seconds passed
... 39%, 5472 KB, 19179 KB/s, 0 seconds passed
... 39%, 5504 KB, 19267 KB/s, 0 seconds passed
... 39%, 5536 KB, 19358 KB/s, 0 seconds passed
... 40%, 5568 KB, 19430 KB/s, 0 seconds passed
... 40%, 5600 KB, 19521 KB/s, 0 seconds passed
... 40%, 5632 KB, 19608 KB/s, 0 seconds passed
... 40%, 5664 KB, 19695 KB/s, 0 seconds passed
... 41%, 5696 KB, 19768 KB/s, 0 seconds passed
... 41%, 5728 KB, 19835 KB/s, 0 seconds passed
... 41%, 5760 KB, 19915 KB/s, 0 seconds passed
... 41%, 5792 KB, 20005 KB/s, 0 seconds passed
... 41%, 5824 KB, 20091 KB/s, 0 seconds passed
... 42%, 5856 KB, 20158 KB/s, 0 seconds passed
... 42%, 5888 KB, 20247 KB/s, 0 seconds passed
... 42%, 5920 KB, 20314 KB/s, 0 seconds passed
... 42%, 5952 KB, 20396 KB/s, 0 seconds passed
... 43%, 5984 KB, 20484 KB/s, 0 seconds passed
... 43%, 6016 KB, 20559 KB/s, 0 seconds passed
... 43%, 6048 KB, 20639 KB/s, 0 seconds passed
... 43%, 6080 KB, 20726 KB/s, 0 seconds passed
... 44%, 6112 KB, 20811 KB/s, 0 seconds passed
... 44%, 6144 KB, 20898 KB/s, 0 seconds passed
... 44%, 6176 KB, 20983 KB/s, 0 seconds passed
... 44%, 6208 KB, 21065 KB/s, 0 seconds passed
... 44%, 6240 KB, 21117 KB/s, 0 seconds passed
... 45%, 6272 KB, 21201 KB/s, 0 seconds passed
... 45%, 6304 KB, 20427 KB/s, 0 seconds passed
... 45%, 6336 KB, 20501 KB/s, 0 seconds passed
... 45%, 6368 KB, 20582 KB/s, 0 seconds passed
... 46%, 6400 KB, 20664 KB/s, 0 seconds passed
... 46%, 6432 KB, 20746 KB/s, 0 seconds passed
... 46%, 6464 KB, 20829 KB/s, 0 seconds passed
... 46%, 6496 KB, 20911 KB/s, 0 seconds passed
... 47%, 6528 KB, 20947 KB/s, 0 seconds passed
... 47%, 6560 KB, 21027 KB/s, 0 seconds passed
... 47%, 6592 KB, 21107 KB/s, 0 seconds passed
... 47%, 6624 KB, 21188 KB/s, 0 seconds passed
... 47%, 6656 KB, 21267 KB/s, 0 seconds passed
... 48%, 6688 KB, 21347 KB/s, 0 seconds passed
... 48%, 6720 KB, 21429 KB/s, 0 seconds passed
... 48%, 6752 KB, 21511 KB/s, 0 seconds passed
... 48%, 6784 KB, 21592 KB/s, 0 seconds passed
... 49%, 6816 KB, 21673 KB/s, 0 seconds passed
... 49%, 6848 KB, 21754 KB/s, 0 seconds passed
... 49%, 6880 KB, 21835 KB/s, 0 seconds passed
... 49%, 6912 KB, 21915 KB/s, 0 seconds passed
... 50%, 6944 KB, 21996 KB/s, 0 seconds passed
... 50%, 6976 KB, 22076 KB/s, 0 seconds passed
... 50%, 7008 KB, 22156 KB/s, 0 seconds passed
... 50%, 7040 KB, 22236 KB/s, 0 seconds passed
... 50%, 7072 KB, 22317 KB/s, 0 seconds passed
... 51%, 7104 KB, 22274 KB/s, 0 seconds passed
... 51%, 7136 KB, 22315 KB/s, 0 seconds passed
... 51%, 7168 KB, 22392 KB/s, 0 seconds passed
... 51%, 7200 KB, 22471 KB/s, 0 seconds passed
... 52%, 7232 KB, 22550 KB/s, 0 seconds passed
... 52%, 7264 KB, 22629 KB/s, 0 seconds passed
... 52%, 7296 KB, 22708 KB/s, 0 seconds passed
... 52%, 7328 KB, 22786 KB/s, 0 seconds passed
... 53%, 7360 KB, 22865 KB/s, 0 seconds passed
... 53%, 7392 KB, 22943 KB/s, 0 seconds passed
... 53%, 7424 KB, 23021 KB/s, 0 seconds passed
... 53%, 7456 KB, 23097 KB/s, 0 seconds passed
... 53%, 7488 KB, 23176 KB/s, 0 seconds passed
... 54%, 7520 KB, 23255 KB/s, 0 seconds passed
... 54%, 7552 KB, 23335 KB/s, 0 seconds passed
... 54%, 7584 KB, 23415 KB/s, 0 seconds passed
... 54%, 7616 KB, 23494 KB/s, 0 seconds passed
... 55%, 7648 KB, 23573 KB/s, 0 seconds passed
... 55%, 7680 KB, 23653 KB/s, 0 seconds passed
... 55%, 7712 KB, 23733 KB/s, 0 seconds passed
... 55%, 7744 KB, 23812 KB/s, 0 seconds passed
... 56%, 7776 KB, 23891 KB/s, 0 seconds passed
... 56%, 7808 KB, 23971 KB/s, 0 seconds passed
... 56%, 7840 KB, 24049 KB/s, 0 seconds passed
... 56%, 7872 KB, 24128 KB/s, 0 seconds passed
... 56%, 7904 KB, 24207 KB/s, 0 seconds passed
... 57%, 7936 KB, 24286 KB/s, 0 seconds passed
... 57%, 7968 KB, 24364 KB/s, 0 seconds passed

.. parsed-literal::

    ... 57%, 8000 KB, 24441 KB/s, 0 seconds passed
... 57%, 8032 KB, 24519 KB/s, 0 seconds passed
... 58%, 8064 KB, 24597 KB/s, 0 seconds passed
... 58%, 8096 KB, 24674 KB/s, 0 seconds passed
... 58%, 8128 KB, 24752 KB/s, 0 seconds passed
... 58%, 8160 KB, 24830 KB/s, 0 seconds passed
... 59%, 8192 KB, 24907 KB/s, 0 seconds passed
... 59%, 8224 KB, 24984 KB/s, 0 seconds passed
... 59%, 8256 KB, 25051 KB/s, 0 seconds passed
... 59%, 8288 KB, 25127 KB/s, 0 seconds passed
... 59%, 8320 KB, 25204 KB/s, 0 seconds passed
... 60%, 8352 KB, 25276 KB/s, 0 seconds passed
... 60%, 8384 KB, 25353 KB/s, 0 seconds passed
... 60%, 8416 KB, 25429 KB/s, 0 seconds passed
... 60%, 8448 KB, 25504 KB/s, 0 seconds passed
... 61%, 8480 KB, 25580 KB/s, 0 seconds passed
... 61%, 8512 KB, 25656 KB/s, 0 seconds passed
... 61%, 8544 KB, 25732 KB/s, 0 seconds passed
... 61%, 8576 KB, 25808 KB/s, 0 seconds passed
... 62%, 8608 KB, 25883 KB/s, 0 seconds passed
... 62%, 8640 KB, 25958 KB/s, 0 seconds passed
... 62%, 8672 KB, 26032 KB/s, 0 seconds passed
... 62%, 8704 KB, 26107 KB/s, 0 seconds passed
... 62%, 8736 KB, 26180 KB/s, 0 seconds passed
... 63%, 8768 KB, 26256 KB/s, 0 seconds passed
... 63%, 8800 KB, 26331 KB/s, 0 seconds passed
... 63%, 8832 KB, 26406 KB/s, 0 seconds passed
... 63%, 8864 KB, 26480 KB/s, 0 seconds passed
... 64%, 8896 KB, 26550 KB/s, 0 seconds passed
... 64%, 8928 KB, 26624 KB/s, 0 seconds passed
... 64%, 8960 KB, 26698 KB/s, 0 seconds passed
... 64%, 8992 KB, 26773 KB/s, 0 seconds passed
... 65%, 9024 KB, 26838 KB/s, 0 seconds passed
... 65%, 9056 KB, 26907 KB/s, 0 seconds passed
... 65%, 9088 KB, 26982 KB/s, 0 seconds passed
... 65%, 9120 KB, 27055 KB/s, 0 seconds passed
... 65%, 9152 KB, 27128 KB/s, 0 seconds passed
... 66%, 9184 KB, 27201 KB/s, 0 seconds passed
... 66%, 9216 KB, 27271 KB/s, 0 seconds passed
... 66%, 9248 KB, 27348 KB/s, 0 seconds passed
... 66%, 9280 KB, 27421 KB/s, 0 seconds passed
... 67%, 9312 KB, 27494 KB/s, 0 seconds passed
... 67%, 9344 KB, 27563 KB/s, 0 seconds passed
... 67%, 9376 KB, 27634 KB/s, 0 seconds passed
... 67%, 9408 KB, 27708 KB/s, 0 seconds passed
... 68%, 9440 KB, 27781 KB/s, 0 seconds passed
... 68%, 9472 KB, 27848 KB/s, 0 seconds passed
... 68%, 9504 KB, 27924 KB/s, 0 seconds passed
... 68%, 9536 KB, 27997 KB/s, 0 seconds passed
... 68%, 9568 KB, 28069 KB/s, 0 seconds passed
... 69%, 9600 KB, 28136 KB/s, 0 seconds passed
... 69%, 9632 KB, 28208 KB/s, 0 seconds passed
... 69%, 9664 KB, 28280 KB/s, 0 seconds passed
... 69%, 9696 KB, 28351 KB/s, 0 seconds passed
... 70%, 9728 KB, 28423 KB/s, 0 seconds passed
... 70%, 9760 KB, 28490 KB/s, 0 seconds passed
... 70%, 9792 KB, 28561 KB/s, 0 seconds passed
... 70%, 9824 KB, 28622 KB/s, 0 seconds passed
... 71%, 9856 KB, 28695 KB/s, 0 seconds passed
... 71%, 9888 KB, 28765 KB/s, 0 seconds passed
... 71%, 9920 KB, 28836 KB/s, 0 seconds passed
... 71%, 9952 KB, 28906 KB/s, 0 seconds passed
... 71%, 9984 KB, 28977 KB/s, 0 seconds passed
... 72%, 10016 KB, 29042 KB/s, 0 seconds passed
... 72%, 10048 KB, 29113 KB/s, 0 seconds passed
... 72%, 10080 KB, 29183 KB/s, 0 seconds passed
... 72%, 10112 KB, 29254 KB/s, 0 seconds passed
... 73%, 10144 KB, 29319 KB/s, 0 seconds passed
... 73%, 10176 KB, 29388 KB/s, 0 seconds passed
... 73%, 10208 KB, 29453 KB/s, 0 seconds passed
... 73%, 10240 KB, 29524 KB/s, 0 seconds passed
... 74%, 10272 KB, 29593 KB/s, 0 seconds passed
... 74%, 10304 KB, 29662 KB/s, 0 seconds passed
... 74%, 10336 KB, 29728 KB/s, 0 seconds passed
... 74%, 10368 KB, 29801 KB/s, 0 seconds passed
... 74%, 10400 KB, 29865 KB/s, 0 seconds passed
... 75%, 10432 KB, 29935 KB/s, 0 seconds passed
... 75%, 10464 KB, 30004 KB/s, 0 seconds passed
... 75%, 10496 KB, 30073 KB/s, 0 seconds passed
... 75%, 10528 KB, 30137 KB/s, 0 seconds passed
... 76%, 10560 KB, 30205 KB/s, 0 seconds passed
... 76%, 10592 KB, 30273 KB/s, 0 seconds passed
... 76%, 10624 KB, 30342 KB/s, 0 seconds passed
... 76%, 10656 KB, 30411 KB/s, 0 seconds passed
... 77%, 10688 KB, 30474 KB/s, 0 seconds passed
... 77%, 10720 KB, 30543 KB/s, 0 seconds passed
... 77%, 10752 KB, 30610 KB/s, 0 seconds passed
... 77%, 10784 KB, 30678 KB/s, 0 seconds passed
... 77%, 10816 KB, 30741 KB/s, 0 seconds passed
... 78%, 10848 KB, 30809 KB/s, 0 seconds passed
... 78%, 10880 KB, 30882 KB/s, 0 seconds passed
... 78%, 10912 KB, 30948 KB/s, 0 seconds passed
... 78%, 10944 KB, 31012 KB/s, 0 seconds passed
... 79%, 10976 KB, 31079 KB/s, 0 seconds passed
... 79%, 11008 KB, 31145 KB/s, 0 seconds passed
... 79%, 11040 KB, 31213 KB/s, 0 seconds passed
... 79%, 11072 KB, 31280 KB/s, 0 seconds passed
... 80%, 11104 KB, 31341 KB/s, 0 seconds passed
... 80%, 11136 KB, 31409 KB/s, 0 seconds passed
... 80%, 11168 KB, 31476 KB/s, 0 seconds passed
... 80%, 11200 KB, 31541 KB/s, 0 seconds passed
... 80%, 11232 KB, 31604 KB/s, 0 seconds passed
... 81%, 11264 KB, 31671 KB/s, 0 seconds passed
... 81%, 11296 KB, 31723 KB/s, 0 seconds passed
... 81%, 11328 KB, 31785 KB/s, 0 seconds passed
... 81%, 11360 KB, 31845 KB/s, 0 seconds passed
... 82%, 11392 KB, 31916 KB/s, 0 seconds passed
... 82%, 11424 KB, 31987 KB/s, 0 seconds passed
... 82%, 11456 KB, 32048 KB/s, 0 seconds passed
... 82%, 11488 KB, 32113 KB/s, 0 seconds passed
... 82%, 11520 KB, 32179 KB/s, 0 seconds passed
... 83%, 11552 KB, 32245 KB/s, 0 seconds passed
... 83%, 11584 KB, 32310 KB/s, 0 seconds passed
... 83%, 11616 KB, 32375 KB/s, 0 seconds passed
... 83%, 11648 KB, 32440 KB/s, 0 seconds passed
... 84%, 11680 KB, 32501 KB/s, 0 seconds passed
... 84%, 11712 KB, 32565 KB/s, 0 seconds passed
... 84%, 11744 KB, 32631 KB/s, 0 seconds passed
... 84%, 11776 KB, 32696 KB/s, 0 seconds passed
... 85%, 11808 KB, 32760 KB/s, 0 seconds passed
... 85%, 11840 KB, 32825 KB/s, 0 seconds passed
... 85%, 11872 KB, 32885 KB/s, 0 seconds passed
... 85%, 11904 KB, 32949 KB/s, 0 seconds passed
... 85%, 11936 KB, 33013 KB/s, 0 seconds passed
... 86%, 11968 KB, 33078 KB/s, 0 seconds passed
... 86%, 12000 KB, 33142 KB/s, 0 seconds passed
... 86%, 12032 KB, 33206 KB/s, 0 seconds passed
... 86%, 12064 KB, 33265 KB/s, 0 seconds passed
... 87%, 12096 KB, 33319 KB/s, 0 seconds passed
... 87%, 12128 KB, 33383 KB/s, 0 seconds passed
... 87%, 12160 KB, 33447 KB/s, 0 seconds passed
... 87%, 12192 KB, 33510 KB/s, 0 seconds passed
... 88%, 12224 KB, 33569 KB/s, 0 seconds passed
... 88%, 12256 KB, 33633 KB/s, 0 seconds passed
... 88%, 12288 KB, 33695 KB/s, 0 seconds passed
... 88%, 12320 KB, 33759 KB/s, 0 seconds passed
... 88%, 12352 KB, 33817 KB/s, 0 seconds passed
... 89%, 12384 KB, 33880 KB/s, 0 seconds passed
... 89%, 12416 KB, 33943 KB/s, 0 seconds passed
... 89%, 12448 KB, 34006 KB/s, 0 seconds passed
... 89%, 12480 KB, 34068 KB/s, 0 seconds passed
... 90%, 12512 KB, 34132 KB/s, 0 seconds passed
... 90%, 12544 KB, 34194 KB/s, 0 seconds passed
... 90%, 12576 KB, 34256 KB/s, 0 seconds passed
... 90%, 12608 KB, 34319 KB/s, 0 seconds passed
... 91%, 12640 KB, 34377 KB/s, 0 seconds passed
... 91%, 12672 KB, 34426 KB/s, 0 seconds passed
... 91%, 12704 KB, 34482 KB/s, 0 seconds passed
... 91%, 12736 KB, 34540 KB/s, 0 seconds passed
... 91%, 12768 KB, 34610 KB/s, 0 seconds passed
... 92%, 12800 KB, 34671 KB/s, 0 seconds passed
... 92%, 12832 KB, 34733 KB/s, 0 seconds passed
... 92%, 12864 KB, 34790 KB/s, 0 seconds passed
... 92%, 12896 KB, 34857 KB/s, 0 seconds passed
... 93%, 12928 KB, 34918 KB/s, 0 seconds passed
... 93%, 12960 KB, 34979 KB/s, 0 seconds passed
... 93%, 12992 KB, 35036 KB/s, 0 seconds passed
... 93%, 13024 KB, 35098 KB/s, 0 seconds passed
... 94%, 13056 KB, 35158 KB/s, 0 seconds passed
... 94%, 13088 KB, 35220 KB/s, 0 seconds passed
... 94%, 13120 KB, 35281 KB/s, 0 seconds passed
... 94%, 13152 KB, 35342 KB/s, 0 seconds passed
... 94%, 13184 KB, 35397 KB/s, 0 seconds passed

.. parsed-literal::

    ... 95%, 13216 KB, 33905 KB/s, 0 seconds passed
... 95%, 13248 KB, 33962 KB/s, 0 seconds passed
... 95%, 13280 KB, 34026 KB/s, 0 seconds passed
... 95%, 13312 KB, 34086 KB/s, 0 seconds passed
... 96%, 13344 KB, 34130 KB/s, 0 seconds passed
... 96%, 13376 KB, 34189 KB/s, 0 seconds passed
... 96%, 13408 KB, 34247 KB/s, 0 seconds passed
... 96%, 13440 KB, 34302 KB/s, 0 seconds passed
... 97%, 13472 KB, 34360 KB/s, 0 seconds passed
... 97%, 13504 KB, 34418 KB/s, 0 seconds passed
... 97%, 13536 KB, 34477 KB/s, 0 seconds passed
... 97%, 13568 KB, 34535 KB/s, 0 seconds passed
... 97%, 13600 KB, 34592 KB/s, 0 seconds passed
... 98%, 13632 KB, 34651 KB/s, 0 seconds passed
... 98%, 13664 KB, 34709 KB/s, 0 seconds passed
... 98%, 13696 KB, 34760 KB/s, 0 seconds passed
... 98%, 13728 KB, 34819 KB/s, 0 seconds passed
... 99%, 13760 KB, 34876 KB/s, 0 seconds passed
... 99%, 13792 KB, 34934 KB/s, 0 seconds passed
... 99%, 13824 KB, 34991 KB/s, 0 seconds passed
... 99%, 13856 KB, 35044 KB/s, 0 seconds passed
... 100%, 13879 KB, 35077 KB/s, 0 seconds passed




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
    Conversion to ONNX command: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/bin/python -- /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/omz_tools/internal_scripts/pytorch_to_onnx.py --model-name=mobilenet_v2 --weights=model/public/mobilenet-v2-pytorch/mobilenet_v2-b0353104.pth --import-module=torchvision.models --input-shape=1,3,224,224 --output-file=model/public/mobilenet-v2-pytorch/mobilenet-v2.onnx --input-names=data --output-names=prob



.. parsed-literal::

    ONNX check passed successfully.


.. parsed-literal::


    ========== Converting mobilenet-v2-pytorch to IR (FP16)
    Conversion command: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/bin/python -- /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/bin/mo --framework=onnx --output_dir=model/public/mobilenet-v2-pytorch/FP16 --model_name=mobilenet-v2-pytorch --input=data '--mean_values=data[123.675,116.28,103.53]' '--scale_values=data[58.624,57.12,57.375]' --reverse_input_channels --output=prob --input_model=model/public/mobilenet-v2-pytorch/mobilenet-v2.onnx '--layout=data(NCHW)' '--input_shape=[1, 3, 224, 224]' --compress_to_fp16=True



.. parsed-literal::

    [ INFO ] Generated IR will be compressed to FP16. If you get lower accuracy, please consider disabling compression explicitly by adding argument --compress_to_fp16=False.
    Find more information about compression to FP16 at https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_FP16_Compression.html
    [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release. Please use OpenVINO Model Converter (OVC). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
    Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
    [ SUCCESS ] Generated IR version 11 model.
    [ SUCCESS ] XML file: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/notebooks/104-model-tools/model/public/mobilenet-v2-pytorch/FP16/mobilenet-v2-pytorch.xml
    [ SUCCESS ] BIN file: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/notebooks/104-model-tools/model/public/mobilenet-v2-pytorch/FP16/mobilenet-v2-pytorch.bin



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
      'accuracy_config': '/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/omz_tools/models/public/mobilenet-v2-pytorch/accuracy-check.yml',
      'model_config': '/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/omz_tools/models/public/mobilenet-v2-pytorch/model.yml',
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
    model_path = (
        base_model_dir
        / Path(selected_model_info["subdirectory"])
        / Path(f"{precision}/{selected_model_info['name']}.xml")
    )
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
    [ INFO ] OpenVINO:
    [ INFO ] Build ................................. 2024.0.0-14509-34caeefd078-releases/2024/0
    [ INFO ]
    [ INFO ] Device info:


.. parsed-literal::

    [ INFO ] CPU
    [ INFO ] Build ................................. 2024.0.0-14509-34caeefd078-releases/2024/0
    [ INFO ]
    [ INFO ]
    [Step 3/11] Setting device configuration
    [ WARNING ] Performance hint was not explicitly specified in command line. Device(CPU) performance hint will be set to PerformanceMode.THROUGHPUT.
    [Step 4/11] Reading model files
    [ INFO ] Loading model files


.. parsed-literal::

    [ INFO ] Read model took 30.94 ms
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

    [ INFO ] Compile model took 136.02 ms
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
    [ INFO ] First inference took 6.30 ms


.. parsed-literal::

    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            20280 iterations
    [ INFO ] Duration:         15006.67 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        4.31 ms
    [ INFO ]    Average:       4.31 ms
    [ INFO ]    Min:           2.75 ms
    [ INFO ]    Max:           13.73 ms
    [ INFO ] Throughput:   1351.40 FPS


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
            benchmark_result = [line for line in benchmark_output
                                if not (line.startswith(r"[") or line.startswith("      ") or line == "")]
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
        value='CPU',
        description='Device:',
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


