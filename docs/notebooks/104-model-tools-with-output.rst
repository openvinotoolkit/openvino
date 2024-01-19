Working with Open Model Zoo Models
==================================

This tutorial shows how to download a model from `Open Model
Zoo <https://github.com/openvinotoolkit/open_model_zoo>`__, convert it
to OpenVINO™ IR format, show information about the model, and benchmark
the model.

Table of contents:
^^^^^^^^^^^^^^^^^^

-  `OpenVINO and Open Model Zoo
   Tools <#OpenVINO-and-Open-Model-Zoo-Tools>`__
-  `Preparation <#Preparation>`__

   -  `Model Name <#Model-Name>`__
   -  `Imports <#Imports>`__
   -  `Settings and Configuration <#Settings-and-Configuration>`__

-  `Download a Model from Open Model
   Zoo <#Download-a-Model-from-Open-Model-Zoo>`__
-  `Convert a Model to OpenVINO IR
   format <#Convert-a-Model-to-OpenVINO-IR-format>`__
-  `Get Model Information <#Get-Model-Information>`__
-  `Run Benchmark Tool <#Run-Benchmark-Tool>`__

   -  `Benchmark with Different
      Settings <#Benchmark-with-Different-Settings>`__

OpenVINO and Open Model Zoo Tools
---------------------------------

`back to top ⬆️ <#Table-of-contents:>`__

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
    %pip install -q "openvino-dev>=2023.1.0"


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


Preparation
-----------

`back to top ⬆️ <#Table-of-contents:>`__

Model Name
~~~~~~~~~~

`back to top ⬆️ <#Table-of-contents:>`__

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

`back to top ⬆️ <#Table-of-contents:>`__

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

`back to top ⬆️ <#Table-of-contents:>`__

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
    
    # Check if an iGPU is available on this system to use with Benchmark App.
    core = ov.Core()
    gpu_available = "GPU" in core.available_devices
    
    print(
        f"base_model_dir: {base_model_dir}, omz_cache_dir: {omz_cache_dir}, gpu_availble: {gpu_available}"
    )


.. parsed-literal::

    base_model_dir: model, omz_cache_dir: cache, gpu_availble: False


Download a Model from Open Model Zoo
------------------------------------

`back to top ⬆️ <#Table-of-contents:>`__

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
    
    ========== Downloading model/public/mobilenet-v2-pytorch/mobilenet_v2-b0353104.pth


.. parsed-literal::

    ... 0%, 32 KB, 1358 KB/s, 0 seconds passed

.. parsed-literal::

    ... 0%, 64 KB, 1162 KB/s, 0 seconds passed... 0%, 96 KB, 1715 KB/s, 0 seconds passed... 0%, 128 KB, 1454 KB/s, 0 seconds passed... 1%, 160 KB, 1683 KB/s, 0 seconds passed

.. parsed-literal::

    ... 1%, 192 KB, 1604 KB/s, 0 seconds passed... 1%, 224 KB, 1859 KB/s, 0 seconds passed... 1%, 256 KB, 2119 KB/s, 0 seconds passed... 2%, 288 KB, 2263 KB/s, 0 seconds passed

.. parsed-literal::

    ... 2%, 320 KB, 2104 KB/s, 0 seconds passed... 2%, 352 KB, 2304 KB/s, 0 seconds passed... 2%, 384 KB, 2507 KB/s, 0 seconds passed... 2%, 416 KB, 2607 KB/s, 0 seconds passed... 3%, 448 KB, 2428 KB/s, 0 seconds passed... 3%, 480 KB, 2591 KB/s, 0 seconds passed... 3%, 512 KB, 2757 KB/s, 0 seconds passed... 3%, 544 KB, 2833 KB/s, 0 seconds passed

.. parsed-literal::

    ... 4%, 576 KB, 2656 KB/s, 0 seconds passed... 4%, 608 KB, 2792 KB/s, 0 seconds passed... 4%, 640 KB, 2932 KB/s, 0 seconds passed... 4%, 672 KB, 2994 KB/s, 0 seconds passed

.. parsed-literal::

    ... 5%, 704 KB, 2822 KB/s, 0 seconds passed... 5%, 736 KB, 2941 KB/s, 0 seconds passed... 5%, 768 KB, 3062 KB/s, 0 seconds passed... 5%, 800 KB, 3113 KB/s, 0 seconds passed... 5%, 832 KB, 2953 KB/s, 0 seconds passed... 6%, 864 KB, 3057 KB/s, 0 seconds passed... 6%, 896 KB, 3163 KB/s, 0 seconds passed... 6%, 928 KB, 3207 KB/s, 0 seconds passed

.. parsed-literal::

    ... 6%, 960 KB, 3055 KB/s, 0 seconds passed... 7%, 992 KB, 3149 KB/s, 0 seconds passed... 7%, 1024 KB, 3243 KB/s, 0 seconds passed... 7%, 1056 KB, 3283 KB/s, 0 seconds passed... 7%, 1088 KB, 3140 KB/s, 0 seconds passed... 8%, 1120 KB, 3224 KB/s, 0 seconds passed... 8%, 1152 KB, 3309 KB/s, 0 seconds passed

.. parsed-literal::

    ... 8%, 1184 KB, 3342 KB/s, 0 seconds passed... 8%, 1216 KB, 3208 KB/s, 0 seconds passed... 8%, 1248 KB, 3284 KB/s, 0 seconds passed... 9%, 1280 KB, 3362 KB/s, 0 seconds passed... 9%, 1312 KB, 3393 KB/s, 0 seconds passed

.. parsed-literal::

    ... 9%, 1344 KB, 3266 KB/s, 0 seconds passed... 9%, 1376 KB, 3336 KB/s, 0 seconds passed... 10%, 1408 KB, 3407 KB/s, 0 seconds passed... 10%, 1440 KB, 3304 KB/s, 0 seconds passed... 10%, 1472 KB, 3315 KB/s, 0 seconds passed... 10%, 1504 KB, 3380 KB/s, 0 seconds passed... 11%, 1536 KB, 3446 KB/s, 0 seconds passed

.. parsed-literal::

    ... 11%, 1568 KB, 3349 KB/s, 0 seconds passed... 11%, 1600 KB, 3358 KB/s, 0 seconds passed... 11%, 1632 KB, 3418 KB/s, 0 seconds passed... 11%, 1664 KB, 3478 KB/s, 0 seconds passed... 12%, 1696 KB, 3506 KB/s, 0 seconds passed

.. parsed-literal::

    ... 12%, 1728 KB, 3398 KB/s, 0 seconds passed... 12%, 1760 KB, 3454 KB/s, 0 seconds passed... 12%, 1792 KB, 3509 KB/s, 0 seconds passed... 13%, 1824 KB, 3529 KB/s, 0 seconds passed... 13%, 1856 KB, 3431 KB/s, 0 seconds passed... 13%, 1888 KB, 3484 KB/s, 0 seconds passed... 13%, 1920 KB, 3534 KB/s, 0 seconds passed... 14%, 1952 KB, 3555 KB/s, 0 seconds passed

.. parsed-literal::

    ... 14%, 1984 KB, 3458 KB/s, 0 seconds passed... 14%, 2016 KB, 3510 KB/s, 0 seconds passed... 14%, 2048 KB, 3556 KB/s, 0 seconds passed... 14%, 2080 KB, 3572 KB/s, 0 seconds passed... 15%, 2112 KB, 3485 KB/s, 0 seconds passed... 15%, 2144 KB, 3533 KB/s, 0 seconds passed

.. parsed-literal::

    ... 15%, 2176 KB, 3572 KB/s, 0 seconds passed... 15%, 2208 KB, 3592 KB/s, 0 seconds passed... 16%, 2240 KB, 3509 KB/s, 0 seconds passed... 16%, 2272 KB, 3552 KB/s, 0 seconds passed... 16%, 2304 KB, 3592 KB/s, 0 seconds passed... 16%, 2336 KB, 3610 KB/s, 0 seconds passed

.. parsed-literal::

    ... 17%, 2368 KB, 3526 KB/s, 0 seconds passed... 17%, 2400 KB, 3569 KB/s, 0 seconds passed... 17%, 2432 KB, 3606 KB/s, 0 seconds passed... 17%, 2464 KB, 3542 KB/s, 0 seconds passed... 17%, 2496 KB, 3545 KB/s, 0 seconds passed... 18%, 2528 KB, 3586 KB/s, 0 seconds passed... 18%, 2560 KB, 3622 KB/s, 0 seconds passed

.. parsed-literal::

    ... 18%, 2592 KB, 3561 KB/s, 0 seconds passed... 18%, 2624 KB, 3563 KB/s, 0 seconds passed... 19%, 2656 KB, 3602 KB/s, 0 seconds passed... 19%, 2688 KB, 3637 KB/s, 0 seconds passed... 19%, 2720 KB, 3578 KB/s, 0 seconds passed

.. parsed-literal::

    ... 19%, 2752 KB, 3579 KB/s, 0 seconds passed... 20%, 2784 KB, 3616 KB/s, 0 seconds passed... 20%, 2816 KB, 3650 KB/s, 0 seconds passed... 20%, 2848 KB, 3590 KB/s, 0 seconds passed... 20%, 2880 KB, 3592 KB/s, 0 seconds passed... 20%, 2912 KB, 3629 KB/s, 0 seconds passed... 21%, 2944 KB, 3661 KB/s, 0 seconds passed

.. parsed-literal::

    ... 21%, 2976 KB, 3603 KB/s, 0 seconds passed... 21%, 3008 KB, 3608 KB/s, 0 seconds passed... 21%, 3040 KB, 3641 KB/s, 0 seconds passed... 22%, 3072 KB, 3673 KB/s, 0 seconds passed... 22%, 3104 KB, 3621 KB/s, 0 seconds passed

.. parsed-literal::

    ... 22%, 3136 KB, 3621 KB/s, 0 seconds passed... 22%, 3168 KB, 3652 KB/s, 0 seconds passed... 23%, 3200 KB, 3684 KB/s, 0 seconds passed... 23%, 3232 KB, 3634 KB/s, 0 seconds passed... 23%, 3264 KB, 3632 KB/s, 0 seconds passed... 23%, 3296 KB, 3662 KB/s, 0 seconds passed... 23%, 3328 KB, 3693 KB/s, 0 seconds passed

.. parsed-literal::

    ... 24%, 3360 KB, 3643 KB/s, 0 seconds passed... 24%, 3392 KB, 3641 KB/s, 0 seconds passed... 24%, 3424 KB, 3671 KB/s, 0 seconds passed... 24%, 3456 KB, 3703 KB/s, 0 seconds passed... 25%, 3488 KB, 3655 KB/s, 0 seconds passed... 25%, 3520 KB, 3654 KB/s, 0 seconds passed... 25%, 3552 KB, 3681 KB/s, 0 seconds passed... 25%, 3584 KB, 3711 KB/s, 0 seconds passed

.. parsed-literal::

    ... 26%, 3616 KB, 3665 KB/s, 0 seconds passed... 26%, 3648 KB, 3664 KB/s, 0 seconds passed... 26%, 3680 KB, 3689 KB/s, 0 seconds passed... 26%, 3712 KB, 3705 KB/s, 1 seconds passed

.. parsed-literal::

    ... 26%, 3744 KB, 3647 KB/s, 1 seconds passed... 27%, 3776 KB, 3667 KB/s, 1 seconds passed... 27%, 3808 KB, 3695 KB/s, 1 seconds passed... 27%, 3840 KB, 3713 KB/s, 1 seconds passed... 27%, 3872 KB, 3656 KB/s, 1 seconds passed... 28%, 3904 KB, 3676 KB/s, 1 seconds passed... 28%, 3936 KB, 3703 KB/s, 1 seconds passed... 28%, 3968 KB, 3731 KB/s, 1 seconds passed

.. parsed-literal::

    ... 28%, 4000 KB, 3684 KB/s, 1 seconds passed... 29%, 4032 KB, 3684 KB/s, 1 seconds passed... 29%, 4064 KB, 3710 KB/s, 1 seconds passed... 29%, 4096 KB, 3737 KB/s, 1 seconds passed... 29%, 4128 KB, 3690 KB/s, 1 seconds passed

.. parsed-literal::

    ... 29%, 4160 KB, 3691 KB/s, 1 seconds passed... 30%, 4192 KB, 3718 KB/s, 1 seconds passed... 30%, 4224 KB, 3743 KB/s, 1 seconds passed... 30%, 4256 KB, 3703 KB/s, 1 seconds passed... 30%, 4288 KB, 3699 KB/s, 1 seconds passed... 31%, 4320 KB, 3724 KB/s, 1 seconds passed... 31%, 4352 KB, 3748 KB/s, 1 seconds passed

.. parsed-literal::

    ... 31%, 4384 KB, 3687 KB/s, 1 seconds passed... 31%, 4416 KB, 3687 KB/s, 1 seconds passed... 32%, 4448 KB, 3712 KB/s, 1 seconds passed... 32%, 4480 KB, 3738 KB/s, 1 seconds passed... 32%, 4512 KB, 3694 KB/s, 1 seconds passed

.. parsed-literal::

    ... 32%, 4544 KB, 3693 KB/s, 1 seconds passed... 32%, 4576 KB, 3718 KB/s, 1 seconds passed... 33%, 4608 KB, 3743 KB/s, 1 seconds passed... 33%, 4640 KB, 3702 KB/s, 1 seconds passed... 33%, 4672 KB, 3700 KB/s, 1 seconds passed... 33%, 4704 KB, 3723 KB/s, 1 seconds passed... 34%, 4736 KB, 3747 KB/s, 1 seconds passed

.. parsed-literal::

    ... 34%, 4768 KB, 3723 KB/s, 1 seconds passed... 34%, 4800 KB, 3704 KB/s, 1 seconds passed... 34%, 4832 KB, 3727 KB/s, 1 seconds passed... 35%, 4864 KB, 3751 KB/s, 1 seconds passed... 35%, 4896 KB, 3711 KB/s, 1 seconds passed

.. parsed-literal::

    ... 35%, 4928 KB, 3710 KB/s, 1 seconds passed... 35%, 4960 KB, 3732 KB/s, 1 seconds passed... 35%, 4992 KB, 3755 KB/s, 1 seconds passed... 36%, 5024 KB, 3716 KB/s, 1 seconds passed... 36%, 5056 KB, 3716 KB/s, 1 seconds passed... 36%, 5088 KB, 3738 KB/s, 1 seconds passed... 36%, 5120 KB, 3760 KB/s, 1 seconds passed

.. parsed-literal::

    ... 37%, 5152 KB, 3721 KB/s, 1 seconds passed... 37%, 5184 KB, 3722 KB/s, 1 seconds passed... 37%, 5216 KB, 3742 KB/s, 1 seconds passed... 37%, 5248 KB, 3764 KB/s, 1 seconds passed... 38%, 5280 KB, 3726 KB/s, 1 seconds passed... 38%, 5312 KB, 3726 KB/s, 1 seconds passed... 38%, 5344 KB, 3746 KB/s, 1 seconds passed

.. parsed-literal::

    ... 38%, 5376 KB, 3727 KB/s, 1 seconds passed... 38%, 5408 KB, 3729 KB/s, 1 seconds passed... 39%, 5440 KB, 3731 KB/s, 1 seconds passed... 39%, 5472 KB, 3751 KB/s, 1 seconds passed... 39%, 5504 KB, 3731 KB/s, 1 seconds passed

.. parsed-literal::

    ... 39%, 5536 KB, 3734 KB/s, 1 seconds passed... 40%, 5568 KB, 3735 KB/s, 1 seconds passed... 40%, 5600 KB, 3754 KB/s, 1 seconds passed... 40%, 5632 KB, 3735 KB/s, 1 seconds passed... 40%, 5664 KB, 3738 KB/s, 1 seconds passed... 41%, 5696 KB, 3740 KB/s, 1 seconds passed... 41%, 5728 KB, 3758 KB/s, 1 seconds passed

.. parsed-literal::

    ... 41%, 5760 KB, 3739 KB/s, 1 seconds passed... 41%, 5792 KB, 3745 KB/s, 1 seconds passed... 41%, 5824 KB, 3745 KB/s, 1 seconds passed... 42%, 5856 KB, 3763 KB/s, 1 seconds passed... 42%, 5888 KB, 3781 KB/s, 1 seconds passed... 42%, 5920 KB, 3749 KB/s, 1 seconds passed

.. parsed-literal::

    ... 42%, 5952 KB, 3748 KB/s, 1 seconds passed... 43%, 5984 KB, 3767 KB/s, 1 seconds passed... 43%, 6016 KB, 3784 KB/s, 1 seconds passed... 43%, 6048 KB, 3754 KB/s, 1 seconds passed... 43%, 6080 KB, 3754 KB/s, 1 seconds passed... 44%, 6112 KB, 3770 KB/s, 1 seconds passed... 44%, 6144 KB, 3787 KB/s, 1 seconds passed

.. parsed-literal::

    ... 44%, 6176 KB, 3757 KB/s, 1 seconds passed... 44%, 6208 KB, 3755 KB/s, 1 seconds passed... 44%, 6240 KB, 3772 KB/s, 1 seconds passed... 45%, 6272 KB, 3756 KB/s, 1 seconds passed... 45%, 6304 KB, 3758 KB/s, 1 seconds passed... 45%, 6336 KB, 3758 KB/s, 1 seconds passed... 45%, 6368 KB, 3776 KB/s, 1 seconds passed

.. parsed-literal::

    ... 46%, 6400 KB, 3759 KB/s, 1 seconds passed... 46%, 6432 KB, 3761 KB/s, 1 seconds passed... 46%, 6464 KB, 3762 KB/s, 1 seconds passed... 46%, 6496 KB, 3779 KB/s, 1 seconds passed... 47%, 6528 KB, 3762 KB/s, 1 seconds passed

.. parsed-literal::

    ... 47%, 6560 KB, 3764 KB/s, 1 seconds passed... 47%, 6592 KB, 3764 KB/s, 1 seconds passed... 47%, 6624 KB, 3781 KB/s, 1 seconds passed... 47%, 6656 KB, 3766 KB/s, 1 seconds passed... 48%, 6688 KB, 3766 KB/s, 1 seconds passed... 48%, 6720 KB, 3768 KB/s, 1 seconds passed... 48%, 6752 KB, 3784 KB/s, 1 seconds passed

.. parsed-literal::

    ... 48%, 6784 KB, 3767 KB/s, 1 seconds passed... 49%, 6816 KB, 3769 KB/s, 1 seconds passed... 49%, 6848 KB, 3771 KB/s, 1 seconds passed... 49%, 6880 KB, 3787 KB/s, 1 seconds passed... 49%, 6912 KB, 3771 KB/s, 1 seconds passed

.. parsed-literal::

    ... 50%, 6944 KB, 3771 KB/s, 1 seconds passed... 50%, 6976 KB, 3773 KB/s, 1 seconds passed... 50%, 7008 KB, 3789 KB/s, 1 seconds passed... 50%, 7040 KB, 3773 KB/s, 1 seconds passed... 50%, 7072 KB, 3774 KB/s, 1 seconds passed... 51%, 7104 KB, 3779 KB/s, 1 seconds passed... 51%, 7136 KB, 3792 KB/s, 1 seconds passed

.. parsed-literal::

    ... 51%, 7168 KB, 3779 KB/s, 1 seconds passed... 51%, 7200 KB, 3781 KB/s, 1 seconds passed... 52%, 7232 KB, 3781 KB/s, 1 seconds passed... 52%, 7264 KB, 3794 KB/s, 1 seconds passed... 52%, 7296 KB, 3782 KB/s, 1 seconds passed... 52%, 7328 KB, 3785 KB/s, 1 seconds passed

.. parsed-literal::

    ... 53%, 7360 KB, 3784 KB/s, 1 seconds passed... 53%, 7392 KB, 3797 KB/s, 1 seconds passed... 53%, 7424 KB, 3785 KB/s, 1 seconds passed... 53%, 7456 KB, 3787 KB/s, 1 seconds passed... 53%, 7488 KB, 3784 KB/s, 1 seconds passed... 54%, 7520 KB, 3799 KB/s, 1 seconds passed

.. parsed-literal::

    ... 54%, 7552 KB, 3784 KB/s, 1 seconds passed... 54%, 7584 KB, 3785 KB/s, 2 seconds passed... 54%, 7616 KB, 3786 KB/s, 2 seconds passed... 55%, 7648 KB, 3801 KB/s, 2 seconds passed... 55%, 7680 KB, 3786 KB/s, 2 seconds passed... 55%, 7712 KB, 3787 KB/s, 2 seconds passed... 55%, 7744 KB, 3791 KB/s, 2 seconds passed... 56%, 7776 KB, 3803 KB/s, 2 seconds passed

.. parsed-literal::

    ... 56%, 7808 KB, 3792 KB/s, 2 seconds passed... 56%, 7840 KB, 3790 KB/s, 2 seconds passed... 56%, 7872 KB, 3792 KB/s, 2 seconds passed... 56%, 7904 KB, 3805 KB/s, 2 seconds passed... 57%, 7936 KB, 3795 KB/s, 2 seconds passed

.. parsed-literal::

    ... 57%, 7968 KB, 3792 KB/s, 2 seconds passed... 57%, 8000 KB, 3796 KB/s, 2 seconds passed... 57%, 8032 KB, 3807 KB/s, 2 seconds passed... 58%, 8064 KB, 3797 KB/s, 2 seconds passed... 58%, 8096 KB, 3799 KB/s, 2 seconds passed... 58%, 8128 KB, 3798 KB/s, 2 seconds passed... 58%, 8160 KB, 3809 KB/s, 2 seconds passed

.. parsed-literal::

    ... 59%, 8192 KB, 3799 KB/s, 2 seconds passed... 59%, 8224 KB, 3800 KB/s, 2 seconds passed... 59%, 8256 KB, 3798 KB/s, 2 seconds passed... 59%, 8288 KB, 3811 KB/s, 2 seconds passed... 59%, 8320 KB, 3793 KB/s, 2 seconds passed

.. parsed-literal::

    ... 60%, 8352 KB, 3793 KB/s, 2 seconds passed... 60%, 8384 KB, 3799 KB/s, 2 seconds passed... 60%, 8416 KB, 3812 KB/s, 2 seconds passed... 60%, 8448 KB, 3795 KB/s, 2 seconds passed... 61%, 8480 KB, 3800 KB/s, 2 seconds passed... 61%, 8512 KB, 3803 KB/s, 2 seconds passed... 61%, 8544 KB, 3815 KB/s, 2 seconds passed

.. parsed-literal::

    ... 61%, 8576 KB, 3803 KB/s, 2 seconds passed... 62%, 8608 KB, 3802 KB/s, 2 seconds passed... 62%, 8640 KB, 3804 KB/s, 2 seconds passed... 62%, 8672 KB, 3816 KB/s, 2 seconds passed... 62%, 8704 KB, 3805 KB/s, 2 seconds passed... 62%, 8736 KB, 3805 KB/s, 2 seconds passed

.. parsed-literal::

    ... 63%, 8768 KB, 3806 KB/s, 2 seconds passed... 63%, 8800 KB, 3818 KB/s, 2 seconds passed... 63%, 8832 KB, 3806 KB/s, 2 seconds passed... 63%, 8864 KB, 3802 KB/s, 2 seconds passed... 64%, 8896 KB, 3807 KB/s, 2 seconds passed... 64%, 8928 KB, 3820 KB/s, 2 seconds passed... 64%, 8960 KB, 3808 KB/s, 2 seconds passed

.. parsed-literal::

    ... 64%, 8992 KB, 3804 KB/s, 2 seconds passed... 65%, 9024 KB, 3808 KB/s, 2 seconds passed... 65%, 9056 KB, 3821 KB/s, 2 seconds passed... 65%, 9088 KB, 3809 KB/s, 2 seconds passed... 65%, 9120 KB, 3806 KB/s, 2 seconds passed... 65%, 9152 KB, 3810 KB/s, 2 seconds passed... 66%, 9184 KB, 3822 KB/s, 2 seconds passed

.. parsed-literal::

    ... 66%, 9216 KB, 3810 KB/s, 2 seconds passed... 66%, 9248 KB, 3808 KB/s, 2 seconds passed... 66%, 9280 KB, 3811 KB/s, 2 seconds passed... 67%, 9312 KB, 3823 KB/s, 2 seconds passed... 67%, 9344 KB, 3812 KB/s, 2 seconds passed

.. parsed-literal::

    ... 67%, 9376 KB, 3809 KB/s, 2 seconds passed... 67%, 9408 KB, 3812 KB/s, 2 seconds passed... 68%, 9440 KB, 3824 KB/s, 2 seconds passed... 68%, 9472 KB, 3810 KB/s, 2 seconds passed... 68%, 9504 KB, 3816 KB/s, 2 seconds passed... 68%, 9536 KB, 3815 KB/s, 2 seconds passed... 68%, 9568 KB, 3826 KB/s, 2 seconds passed

.. parsed-literal::

    ... 69%, 9600 KB, 3816 KB/s, 2 seconds passed... 69%, 9632 KB, 3813 KB/s, 2 seconds passed... 69%, 9664 KB, 3817 KB/s, 2 seconds passed... 69%, 9696 KB, 3827 KB/s, 2 seconds passed... 70%, 9728 KB, 3817 KB/s, 2 seconds passed

.. parsed-literal::

    ... 70%, 9760 KB, 3812 KB/s, 2 seconds passed... 70%, 9792 KB, 3817 KB/s, 2 seconds passed... 70%, 9824 KB, 3828 KB/s, 2 seconds passed... 71%, 9856 KB, 3815 KB/s, 2 seconds passed... 71%, 9888 KB, 3815 KB/s, 2 seconds passed... 71%, 9920 KB, 3819 KB/s, 2 seconds passed... 71%, 9952 KB, 3830 KB/s, 2 seconds passed

.. parsed-literal::

    ... 71%, 9984 KB, 3819 KB/s, 2 seconds passed... 72%, 10016 KB, 3818 KB/s, 2 seconds passed... 72%, 10048 KB, 3820 KB/s, 2 seconds passed... 72%, 10080 KB, 3831 KB/s, 2 seconds passed... 72%, 10112 KB, 3817 KB/s, 2 seconds passed... 73%, 10144 KB, 3816 KB/s, 2 seconds passed

.. parsed-literal::

    ... 73%, 10176 KB, 3821 KB/s, 2 seconds passed... 73%, 10208 KB, 3832 KB/s, 2 seconds passed... 73%, 10240 KB, 3819 KB/s, 2 seconds passed... 74%, 10272 KB, 3819 KB/s, 2 seconds passed... 74%, 10304 KB, 3823 KB/s, 2 seconds passed... 74%, 10336 KB, 3833 KB/s, 2 seconds passed

.. parsed-literal::

    ... 74%, 10368 KB, 3821 KB/s, 2 seconds passed... 74%, 10400 KB, 3821 KB/s, 2 seconds passed... 75%, 10432 KB, 3824 KB/s, 2 seconds passed... 75%, 10464 KB, 3834 KB/s, 2 seconds passed... 75%, 10496 KB, 3822 KB/s, 2 seconds passed... 75%, 10528 KB, 3822 KB/s, 2 seconds passed... 76%, 10560 KB, 3825 KB/s, 2 seconds passed... 76%, 10592 KB, 3836 KB/s, 2 seconds passed

.. parsed-literal::

    ... 76%, 10624 KB, 3824 KB/s, 2 seconds passed... 76%, 10656 KB, 3824 KB/s, 2 seconds passed... 77%, 10688 KB, 3828 KB/s, 2 seconds passed... 77%, 10720 KB, 3837 KB/s, 2 seconds passed... 77%, 10752 KB, 3825 KB/s, 2 seconds passed

.. parsed-literal::

    ... 77%, 10784 KB, 3825 KB/s, 2 seconds passed... 77%, 10816 KB, 3829 KB/s, 2 seconds passed... 78%, 10848 KB, 3838 KB/s, 2 seconds passed... 78%, 10880 KB, 3826 KB/s, 2 seconds passed... 78%, 10912 KB, 3826 KB/s, 2 seconds passed... 78%, 10944 KB, 3830 KB/s, 2 seconds passed... 79%, 10976 KB, 3839 KB/s, 2 seconds passed

.. parsed-literal::

    ... 79%, 11008 KB, 3828 KB/s, 2 seconds passed... 79%, 11040 KB, 3828 KB/s, 2 seconds passed... 79%, 11072 KB, 3831 KB/s, 2 seconds passed... 80%, 11104 KB, 3840 KB/s, 2 seconds passed... 80%, 11136 KB, 3828 KB/s, 2 seconds passed... 80%, 11168 KB, 3829 KB/s, 2 seconds passed

.. parsed-literal::

    ... 80%, 11200 KB, 3832 KB/s, 2 seconds passed... 80%, 11232 KB, 3833 KB/s, 2 seconds passed... 81%, 11264 KB, 3830 KB/s, 2 seconds passed... 81%, 11296 KB, 3829 KB/s, 2 seconds passed... 81%, 11328 KB, 3833 KB/s, 2 seconds passed... 81%, 11360 KB, 3834 KB/s, 2 seconds passed

.. parsed-literal::

    ... 82%, 11392 KB, 3831 KB/s, 2 seconds passed... 82%, 11424 KB, 3831 KB/s, 2 seconds passed... 82%, 11456 KB, 3833 KB/s, 2 seconds passed... 82%, 11488 KB, 3834 KB/s, 2 seconds passed... 82%, 11520 KB, 3832 KB/s, 3 seconds passed... 83%, 11552 KB, 3832 KB/s, 3 seconds passed

.. parsed-literal::

    ... 83%, 11584 KB, 3834 KB/s, 3 seconds passed... 83%, 11616 KB, 3794 KB/s, 3 seconds passed... 83%, 11648 KB, 3804 KB/s, 3 seconds passed... 84%, 11680 KB, 3814 KB/s, 3 seconds passed... 84%, 11712 KB, 3824 KB/s, 3 seconds passed

.. parsed-literal::

    ... 84%, 11744 KB, 3796 KB/s, 3 seconds passed... 84%, 11776 KB, 3806 KB/s, 3 seconds passed... 85%, 11808 KB, 3815 KB/s, 3 seconds passed... 85%, 11840 KB, 3825 KB/s, 3 seconds passed

.. parsed-literal::

    ... 85%, 11872 KB, 3798 KB/s, 3 seconds passed... 85%, 11904 KB, 3807 KB/s, 3 seconds passed... 85%, 11936 KB, 3817 KB/s, 3 seconds passed... 86%, 11968 KB, 3826 KB/s, 3 seconds passed... 86%, 12000 KB, 3798 KB/s, 3 seconds passed... 86%, 12032 KB, 3808 KB/s, 3 seconds passed... 86%, 12064 KB, 3818 KB/s, 3 seconds passed... 87%, 12096 KB, 3827 KB/s, 3 seconds passed

.. parsed-literal::

    ... 87%, 12128 KB, 3800 KB/s, 3 seconds passed... 87%, 12160 KB, 3809 KB/s, 3 seconds passed... 87%, 12192 KB, 3819 KB/s, 3 seconds passed... 88%, 12224 KB, 3827 KB/s, 3 seconds passed... 88%, 12256 KB, 3801 KB/s, 3 seconds passed... 88%, 12288 KB, 3810 KB/s, 3 seconds passed... 88%, 12320 KB, 3820 KB/s, 3 seconds passed... 88%, 12352 KB, 3829 KB/s, 3 seconds passed

.. parsed-literal::

    ... 89%, 12384 KB, 3802 KB/s, 3 seconds passed... 89%, 12416 KB, 3812 KB/s, 3 seconds passed... 89%, 12448 KB, 3821 KB/s, 3 seconds passed... 89%, 12480 KB, 3830 KB/s, 3 seconds passed

.. parsed-literal::

    ... 90%, 12512 KB, 3804 KB/s, 3 seconds passed... 90%, 12544 KB, 3813 KB/s, 3 seconds passed... 90%, 12576 KB, 3822 KB/s, 3 seconds passed... 90%, 12608 KB, 3831 KB/s, 3 seconds passed... 91%, 12640 KB, 3805 KB/s, 3 seconds passed... 91%, 12672 KB, 3814 KB/s, 3 seconds passed... 91%, 12704 KB, 3823 KB/s, 3 seconds passed... 91%, 12736 KB, 3832 KB/s, 3 seconds passed

.. parsed-literal::

    ... 91%, 12768 KB, 3807 KB/s, 3 seconds passed... 92%, 12800 KB, 3815 KB/s, 3 seconds passed... 92%, 12832 KB, 3824 KB/s, 3 seconds passed... 92%, 12864 KB, 3833 KB/s, 3 seconds passed

.. parsed-literal::

    ... 92%, 12896 KB, 3808 KB/s, 3 seconds passed... 93%, 12928 KB, 3816 KB/s, 3 seconds passed... 93%, 12960 KB, 3825 KB/s, 3 seconds passed... 93%, 12992 KB, 3834 KB/s, 3 seconds passed... 93%, 13024 KB, 3809 KB/s, 3 seconds passed... 94%, 13056 KB, 3817 KB/s, 3 seconds passed... 94%, 13088 KB, 3826 KB/s, 3 seconds passed... 94%, 13120 KB, 3835 KB/s, 3 seconds passed

.. parsed-literal::

    ... 94%, 13152 KB, 3810 KB/s, 3 seconds passed... 94%, 13184 KB, 3818 KB/s, 3 seconds passed... 95%, 13216 KB, 3827 KB/s, 3 seconds passed... 95%, 13248 KB, 3836 KB/s, 3 seconds passed

.. parsed-literal::

    ... 95%, 13280 KB, 3811 KB/s, 3 seconds passed... 95%, 13312 KB, 3819 KB/s, 3 seconds passed... 96%, 13344 KB, 3827 KB/s, 3 seconds passed... 96%, 13376 KB, 3836 KB/s, 3 seconds passed... 96%, 13408 KB, 3811 KB/s, 3 seconds passed... 96%, 13440 KB, 3819 KB/s, 3 seconds passed... 97%, 13472 KB, 3828 KB/s, 3 seconds passed... 97%, 13504 KB, 3837 KB/s, 3 seconds passed

.. parsed-literal::

    ... 97%, 13536 KB, 3812 KB/s, 3 seconds passed... 97%, 13568 KB, 3821 KB/s, 3 seconds passed... 97%, 13600 KB, 3829 KB/s, 3 seconds passed... 98%, 13632 KB, 3837 KB/s, 3 seconds passed... 98%, 13664 KB, 3813 KB/s, 3 seconds passed... 98%, 13696 KB, 3822 KB/s, 3 seconds passed... 98%, 13728 KB, 3830 KB/s, 3 seconds passed... 99%, 13760 KB, 3838 KB/s, 3 seconds passed

.. parsed-literal::

    ... 99%, 13792 KB, 3815 KB/s, 3 seconds passed... 99%, 13824 KB, 3822 KB/s, 3 seconds passed... 99%, 13856 KB, 3831 KB/s, 3 seconds passed... 100%, 13879 KB, 3837 KB/s, 3 seconds passed
    


Convert a Model to OpenVINO IR format
-------------------------------------

`back to top ⬆️ <#Table-of-contents:>`__

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
    Conversion to ONNX command: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-598/.workspace/scm/ov-notebook/.venv/bin/python -- /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-598/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/openvino/model_zoo/internal_scripts/pytorch_to_onnx.py --model-name=mobilenet_v2 --weights=model/public/mobilenet-v2-pytorch/mobilenet_v2-b0353104.pth --import-module=torchvision.models --input-shape=1,3,224,224 --output-file=model/public/mobilenet-v2-pytorch/mobilenet-v2.onnx --input-names=data --output-names=prob
    


.. parsed-literal::

    ONNX check passed successfully.


.. parsed-literal::

    
    ========== Converting mobilenet-v2-pytorch to IR (FP16)
    Conversion command: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-598/.workspace/scm/ov-notebook/.venv/bin/python -- /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-598/.workspace/scm/ov-notebook/.venv/bin/mo --framework=onnx --output_dir=model/public/mobilenet-v2-pytorch/FP16 --model_name=mobilenet-v2-pytorch --input=data '--mean_values=data[123.675,116.28,103.53]' '--scale_values=data[58.624,57.12,57.375]' --reverse_input_channels --output=prob --input_model=model/public/mobilenet-v2-pytorch/mobilenet-v2.onnx '--layout=data(NCHW)' '--input_shape=[1, 3, 224, 224]' --compress_to_fp16=True
    


.. parsed-literal::

    [ INFO ] Generated IR will be compressed to FP16. If you get lower accuracy, please consider disabling compression explicitly by adding argument --compress_to_fp16=False.
    Find more information about compression to FP16 at https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_FP16_Compression.html
    [ INFO ] The model was converted to IR v11, the latest model format that corresponds to the source DL framework input/output format. While IR v11 is backwards compatible with OpenVINO Inference Engine API v1.0, please use API v2.0 (as of 2022.1) to take advantage of the latest improvements in IR v11.
    Find more information about API v2.0 and IR v11 at https://docs.openvino.ai/2023.0/openvino_2_0_transition_guide.html
    [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release. Please use OpenVINO Model Converter (OVC). OVC represents a lightweight alternative of MO and provides simplified model conversion API. 
    Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
    [ SUCCESS ] Generated IR version 11 model.
    [ SUCCESS ] XML file: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-598/.workspace/scm/ov-notebook/notebooks/104-model-tools/model/public/mobilenet-v2-pytorch/FP16/mobilenet-v2-pytorch.xml
    [ SUCCESS ] BIN file: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-598/.workspace/scm/ov-notebook/notebooks/104-model-tools/model/public/mobilenet-v2-pytorch/FP16/mobilenet-v2-pytorch.bin


.. parsed-literal::

    


Get Model Information
---------------------

`back to top ⬆️ <#Table-of-contents:>`__

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
      'accuracy_config': '/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-598/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/openvino/model_zoo/models/public/mobilenet-v2-pytorch/accuracy-check.yml',
      'model_config': '/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-598/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/openvino/model_zoo/models/public/mobilenet-v2-pytorch/model.yml',
      'precisions': ['FP16', 'FP32'],
      'quantization_output_precisions': ['FP16-INT8', 'FP32-INT8'],
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

`back to top ⬆️ <#Table-of-contents:>`__

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
    [ INFO ] Build ................................. 2023.3.0-13775-ceeafaf64f3-releases/2023/3
    [ INFO ] 
    [ INFO ] Device info:


.. parsed-literal::

    [ INFO ] CPU
    [ INFO ] Build ................................. 2023.3.0-13775-ceeafaf64f3-releases/2023/3
    [ INFO ] 
    [ INFO ] 
    [Step 3/11] Setting device configuration
    [ WARNING ] Performance hint was not explicitly specified in command line. Device(CPU) performance hint will be set to PerformanceMode.THROUGHPUT.
    [Step 4/11] Reading model files
    [ INFO ] Loading model files
    [ INFO ] Read model took 30.88 ms
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

    [ INFO ] Compile model took 154.97 ms
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
    [ INFO ]   CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE: 1.0
    [Step 9/11] Creating infer requests and preparing input tensors
    [ WARNING ] No input files were given for input 'data'!. This input will be filled with random values!
    [ INFO ] Fill input 'data' with random values 
    [Step 10/11] Measuring performance (Start inference asynchronously, 6 inference requests, limits: 15000 ms duration)
    [ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).


.. parsed-literal::

    [ INFO ] First inference took 6.26 ms


.. parsed-literal::

    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            20028 iterations
    [ INFO ] Duration:         15005.22 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        4.36 ms
    [ INFO ]    Average:       4.37 ms
    [ INFO ]    Min:           2.74 ms
    [ INFO ]    Max:           12.78 ms
    [ INFO ] Throughput:   1334.74 FPS


Benchmark with Different Settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#Table-of-contents:>`__

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


.. code:: ipython3

    benchmark_model(model_path, device="CPU", seconds=15, api="async")



**Benchmark mobilenet-v2-pytorch.xml with CPU for 15 seconds with async
inference**



Benchmark command:
``benchmark_app -m model/public/mobilenet-v2-pytorch/FP16/mobilenet-v2-pytorch.xml -d CPU -t 15 -api async -b 1``


.. parsed-literal::

    command ended
    


.. code:: ipython3

    benchmark_model(model_path, device="AUTO", seconds=15, api="async")



**Benchmark mobilenet-v2-pytorch.xml with AUTO for 15 seconds with async
inference**



Benchmark command:
``benchmark_app -m model/public/mobilenet-v2-pytorch/FP16/mobilenet-v2-pytorch.xml -d AUTO -t 15 -api async -b 1``


.. parsed-literal::

    command ended
    


.. code:: ipython3

    benchmark_model(model_path, device="GPU", seconds=15, api="async")



.. raw:: html

    <div class="alert alert-warning">Running this cell requires a GPU device, which is not available on this system. The following device is available: CPU


.. code:: ipython3

    benchmark_model(model_path, device="MULTI:CPU,GPU", seconds=15, api="async")



.. raw:: html

    <div class="alert alert-warning">Running this cell requires a GPU device, which is not available on this system. The following device is available: CPU

