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
    %pip install -q "openvino-dev>=2023.1.0"


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

    ... 0%, 32 KB, 1147 KB/s, 0 seconds passed
... 0%, 64 KB, 1047 KB/s, 0 seconds passed

.. parsed-literal::

    ... 0%, 96 KB, 1108 KB/s, 0 seconds passed
... 0%, 128 KB, 1467 KB/s, 0 seconds passed
... 1%, 160 KB, 1676 KB/s, 0 seconds passed
... 1%, 192 KB, 1999 KB/s, 0 seconds passed
... 1%, 224 KB, 1847 KB/s, 0 seconds passed
... 1%, 256 KB, 2102 KB/s, 0 seconds passed

.. parsed-literal::

    ... 2%, 288 KB, 2214 KB/s, 0 seconds passed
... 2%, 320 KB, 2445 KB/s, 0 seconds passed
... 2%, 352 KB, 2259 KB/s, 0 seconds passed
... 2%, 384 KB, 2457 KB/s, 0 seconds passed
... 2%, 416 KB, 2526 KB/s, 0 seconds passed
... 3%, 448 KB, 2712 KB/s, 0 seconds passed

.. parsed-literal::

    ... 3%, 480 KB, 2525 KB/s, 0 seconds passed
... 3%, 512 KB, 2682 KB/s, 0 seconds passed
... 3%, 544 KB, 2734 KB/s, 0 seconds passed
... 4%, 576 KB, 2884 KB/s, 0 seconds passed
... 4%, 608 KB, 2702 KB/s, 0 seconds passed
... 4%, 640 KB, 2834 KB/s, 0 seconds passed

.. parsed-literal::

    ... 4%, 672 KB, 2871 KB/s, 0 seconds passed
... 5%, 704 KB, 2999 KB/s, 0 seconds passed
... 5%, 736 KB, 2837 KB/s, 0 seconds passed
... 5%, 768 KB, 2950 KB/s, 0 seconds passed
... 5%, 800 KB, 2978 KB/s, 0 seconds passed
... 5%, 832 KB, 3091 KB/s, 0 seconds passed

.. parsed-literal::

    ... 6%, 864 KB, 2935 KB/s, 0 seconds passed
... 6%, 896 KB, 3035 KB/s, 0 seconds passed
... 6%, 928 KB, 3064 KB/s, 0 seconds passed
... 6%, 960 KB, 3161 KB/s, 0 seconds passed
... 7%, 992 KB, 3022 KB/s, 0 seconds passed
... 7%, 1024 KB, 3109 KB/s, 0 seconds passed

.. parsed-literal::

    ... 7%, 1056 KB, 3127 KB/s, 0 seconds passed
... 7%, 1088 KB, 3216 KB/s, 0 seconds passed
... 8%, 1120 KB, 3088 KB/s, 0 seconds passed
... 8%, 1152 KB, 3165 KB/s, 0 seconds passed
... 8%, 1184 KB, 3177 KB/s, 0 seconds passed
... 8%, 1216 KB, 3258 KB/s, 0 seconds passed

.. parsed-literal::

    ... 8%, 1248 KB, 3135 KB/s, 0 seconds passed
... 9%, 1280 KB, 3208 KB/s, 0 seconds passed
... 9%, 1312 KB, 3219 KB/s, 0 seconds passed
... 9%, 1344 KB, 3294 KB/s, 0 seconds passed
... 9%, 1376 KB, 3183 KB/s, 0 seconds passed
... 10%, 1408 KB, 3249 KB/s, 0 seconds passed

.. parsed-literal::

    ... 10%, 1440 KB, 3262 KB/s, 0 seconds passed
... 10%, 1472 KB, 3325 KB/s, 0 seconds passed
... 10%, 1504 KB, 3218 KB/s, 0 seconds passed
... 11%, 1536 KB, 3281 KB/s, 0 seconds passed
... 11%, 1568 KB, 3288 KB/s, 0 seconds passed
... 11%, 1600 KB, 3351 KB/s, 0 seconds passed

.. parsed-literal::

    ... 11%, 1632 KB, 3249 KB/s, 0 seconds passed
... 11%, 1664 KB, 3309 KB/s, 0 seconds passed
... 12%, 1696 KB, 3314 KB/s, 0 seconds passed
... 12%, 1728 KB, 3374 KB/s, 0 seconds passed
... 12%, 1760 KB, 3288 KB/s, 0 seconds passed
... 12%, 1792 KB, 3335 KB/s, 0 seconds passed

.. parsed-literal::

    ... 13%, 1824 KB, 3350 KB/s, 0 seconds passed
... 13%, 1856 KB, 3394 KB/s, 0 seconds passed
... 13%, 1888 KB, 3312 KB/s, 0 seconds passed
... 13%, 1920 KB, 3357 KB/s, 0 seconds passed
... 14%, 1952 KB, 3372 KB/s, 0 seconds passed
... 14%, 1984 KB, 3413 KB/s, 0 seconds passed

.. parsed-literal::

    ... 14%, 2016 KB, 3335 KB/s, 0 seconds passed
... 14%, 2048 KB, 3378 KB/s, 0 seconds passed
... 14%, 2080 KB, 3388 KB/s, 0 seconds passed
... 15%, 2112 KB, 3430 KB/s, 0 seconds passed
... 15%, 2144 KB, 3356 KB/s, 0 seconds passed
... 15%, 2176 KB, 3394 KB/s, 0 seconds passed

.. parsed-literal::

    ... 15%, 2208 KB, 3404 KB/s, 0 seconds passed
... 16%, 2240 KB, 3445 KB/s, 0 seconds passed
... 16%, 2272 KB, 3373 KB/s, 0 seconds passed
... 16%, 2304 KB, 3409 KB/s, 0 seconds passed
... 16%, 2336 KB, 3417 KB/s, 0 seconds passed
... 17%, 2368 KB, 3457 KB/s, 0 seconds passed

.. parsed-literal::

    ... 17%, 2400 KB, 3390 KB/s, 0 seconds passed
... 17%, 2432 KB, 3422 KB/s, 0 seconds passed
... 17%, 2464 KB, 3429 KB/s, 0 seconds passed
... 17%, 2496 KB, 3366 KB/s, 0 seconds passed
... 18%, 2528 KB, 3399 KB/s, 0 seconds passed

.. parsed-literal::

    ... 18%, 2560 KB, 3433 KB/s, 0 seconds passed
... 18%, 2592 KB, 3440 KB/s, 0 seconds passed
... 18%, 2624 KB, 3478 KB/s, 0 seconds passed
... 19%, 2656 KB, 3415 KB/s, 0 seconds passed
... 19%, 2688 KB, 3444 KB/s, 0 seconds passed
... 19%, 2720 KB, 3450 KB/s, 0 seconds passed

.. parsed-literal::

    ... 19%, 2752 KB, 3396 KB/s, 0 seconds passed
... 20%, 2784 KB, 3424 KB/s, 0 seconds passed
... 20%, 2816 KB, 3453 KB/s, 0 seconds passed
... 20%, 2848 KB, 3461 KB/s, 0 seconds passed
... 20%, 2880 KB, 3409 KB/s, 0 seconds passed

.. parsed-literal::

    ... 20%, 2912 KB, 3434 KB/s, 0 seconds passed
... 21%, 2944 KB, 3451 KB/s, 0 seconds passed
... 21%, 2976 KB, 3468 KB/s, 0 seconds passed
... 21%, 3008 KB, 3418 KB/s, 0 seconds passed
... 21%, 3040 KB, 3445 KB/s, 0 seconds passed
... 22%, 3072 KB, 3461 KB/s, 0 seconds passed
... 22%, 3104 KB, 3476 KB/s, 0 seconds passed

.. parsed-literal::

    ... 22%, 3136 KB, 3432 KB/s, 0 seconds passed
... 22%, 3168 KB, 3457 KB/s, 0 seconds passed
... 23%, 3200 KB, 3485 KB/s, 0 seconds passed
... 23%, 3232 KB, 3489 KB/s, 0 seconds passed
... 23%, 3264 KB, 3442 KB/s, 0 seconds passed

.. parsed-literal::

    ... 23%, 3296 KB, 3465 KB/s, 0 seconds passed
... 23%, 3328 KB, 3493 KB/s, 0 seconds passed
... 24%, 3360 KB, 3497 KB/s, 0 seconds passed
... 24%, 3392 KB, 3452 KB/s, 0 seconds passed
... 24%, 3424 KB, 3475 KB/s, 0 seconds passed
... 24%, 3456 KB, 3501 KB/s, 0 seconds passed
... 25%, 3488 KB, 3503 KB/s, 0 seconds passed

.. parsed-literal::

    ... 25%, 3520 KB, 3460 KB/s, 1 seconds passed
... 25%, 3552 KB, 3479 KB/s, 1 seconds passed
... 25%, 3584 KB, 3506 KB/s, 1 seconds passed
... 26%, 3616 KB, 3508 KB/s, 1 seconds passed

.. parsed-literal::

    ... 26%, 3648 KB, 3466 KB/s, 1 seconds passed
... 26%, 3680 KB, 3485 KB/s, 1 seconds passed
... 26%, 3712 KB, 3512 KB/s, 1 seconds passed
... 26%, 3744 KB, 3515 KB/s, 1 seconds passed
... 27%, 3776 KB, 3474 KB/s, 1 seconds passed
... 27%, 3808 KB, 3495 KB/s, 1 seconds passed
... 27%, 3840 KB, 3519 KB/s, 1 seconds passed
... 27%, 3872 KB, 3522 KB/s, 1 seconds passed

.. parsed-literal::

    ... 28%, 3904 KB, 3481 KB/s, 1 seconds passed
... 28%, 3936 KB, 3502 KB/s, 1 seconds passed
... 28%, 3968 KB, 3525 KB/s, 1 seconds passed
... 28%, 4000 KB, 3527 KB/s, 1 seconds passed

.. parsed-literal::

    ... 29%, 4032 KB, 3491 KB/s, 1 seconds passed
... 29%, 4064 KB, 3508 KB/s, 1 seconds passed
... 29%, 4096 KB, 3531 KB/s, 1 seconds passed
... 29%, 4128 KB, 3532 KB/s, 1 seconds passed
... 29%, 4160 KB, 3497 KB/s, 1 seconds passed
... 30%, 4192 KB, 3515 KB/s, 1 seconds passed
... 30%, 4224 KB, 3536 KB/s, 1 seconds passed
... 30%, 4256 KB, 3537 KB/s, 1 seconds passed

.. parsed-literal::

    ... 30%, 4288 KB, 3498 KB/s, 1 seconds passed
... 31%, 4320 KB, 3515 KB/s, 1 seconds passed
... 31%, 4352 KB, 3529 KB/s, 1 seconds passed
... 31%, 4384 KB, 3539 KB/s, 1 seconds passed

.. parsed-literal::

    ... 31%, 4416 KB, 3505 KB/s, 1 seconds passed
... 32%, 4448 KB, 3522 KB/s, 1 seconds passed
... 32%, 4480 KB, 3535 KB/s, 1 seconds passed
... 32%, 4512 KB, 3544 KB/s, 1 seconds passed
... 32%, 4544 KB, 3511 KB/s, 1 seconds passed
... 32%, 4576 KB, 3527 KB/s, 1 seconds passed
... 33%, 4608 KB, 3549 KB/s, 1 seconds passed
... 33%, 4640 KB, 3548 KB/s, 1 seconds passed

.. parsed-literal::

    ... 33%, 4672 KB, 3517 KB/s, 1 seconds passed
... 33%, 4704 KB, 3533 KB/s, 1 seconds passed
... 34%, 4736 KB, 3545 KB/s, 1 seconds passed
... 34%, 4768 KB, 3553 KB/s, 1 seconds passed

.. parsed-literal::

    ... 34%, 4800 KB, 3522 KB/s, 1 seconds passed
... 34%, 4832 KB, 3537 KB/s, 1 seconds passed
... 35%, 4864 KB, 3550 KB/s, 1 seconds passed
... 35%, 4896 KB, 3557 KB/s, 1 seconds passed
... 35%, 4928 KB, 3525 KB/s, 1 seconds passed
... 35%, 4960 KB, 3539 KB/s, 1 seconds passed
... 35%, 4992 KB, 3553 KB/s, 1 seconds passed
... 36%, 5024 KB, 3561 KB/s, 1 seconds passed

.. parsed-literal::

    ... 36%, 5056 KB, 3532 KB/s, 1 seconds passed
... 36%, 5088 KB, 3546 KB/s, 1 seconds passed
... 36%, 5120 KB, 3558 KB/s, 1 seconds passed
... 37%, 5152 KB, 3564 KB/s, 1 seconds passed

.. parsed-literal::

    ... 37%, 5184 KB, 3534 KB/s, 1 seconds passed
... 37%, 5216 KB, 3546 KB/s, 1 seconds passed
... 37%, 5248 KB, 3559 KB/s, 1 seconds passed
... 38%, 5280 KB, 3566 KB/s, 1 seconds passed
... 38%, 5312 KB, 3536 KB/s, 1 seconds passed
... 38%, 5344 KB, 3549 KB/s, 1 seconds passed
... 38%, 5376 KB, 3563 KB/s, 1 seconds passed

.. parsed-literal::

    ... 38%, 5408 KB, 3570 KB/s, 1 seconds passed
... 39%, 5440 KB, 3540 KB/s, 1 seconds passed
... 39%, 5472 KB, 3554 KB/s, 1 seconds passed
... 39%, 5504 KB, 3565 KB/s, 1 seconds passed
... 39%, 5536 KB, 3573 KB/s, 1 seconds passed

.. parsed-literal::

    ... 40%, 5568 KB, 3543 KB/s, 1 seconds passed
... 40%, 5600 KB, 3557 KB/s, 1 seconds passed
... 40%, 5632 KB, 3567 KB/s, 1 seconds passed
... 40%, 5664 KB, 3575 KB/s, 1 seconds passed
... 41%, 5696 KB, 3545 KB/s, 1 seconds passed
... 41%, 5728 KB, 3560 KB/s, 1 seconds passed
... 41%, 5760 KB, 3570 KB/s, 1 seconds passed

.. parsed-literal::

    ... 41%, 5792 KB, 3577 KB/s, 1 seconds passed
... 41%, 5824 KB, 3548 KB/s, 1 seconds passed
... 42%, 5856 KB, 3561 KB/s, 1 seconds passed
... 42%, 5888 KB, 3572 KB/s, 1 seconds passed
... 42%, 5920 KB, 3579 KB/s, 1 seconds passed

.. parsed-literal::

    ... 42%, 5952 KB, 3551 KB/s, 1 seconds passed
... 43%, 5984 KB, 3564 KB/s, 1 seconds passed
... 43%, 6016 KB, 3574 KB/s, 1 seconds passed
... 43%, 6048 KB, 3582 KB/s, 1 seconds passed
... 43%, 6080 KB, 3555 KB/s, 1 seconds passed
... 44%, 6112 KB, 3568 KB/s, 1 seconds passed
... 44%, 6144 KB, 3577 KB/s, 1 seconds passed

.. parsed-literal::

    ... 44%, 6176 KB, 3585 KB/s, 1 seconds passed
... 44%, 6208 KB, 3558 KB/s, 1 seconds passed
... 44%, 6240 KB, 3570 KB/s, 1 seconds passed
... 45%, 6272 KB, 3580 KB/s, 1 seconds passed
... 45%, 6304 KB, 3587 KB/s, 1 seconds passed

.. parsed-literal::

    ... 45%, 6336 KB, 3561 KB/s, 1 seconds passed
... 45%, 6368 KB, 3574 KB/s, 1 seconds passed
... 46%, 6400 KB, 3582 KB/s, 1 seconds passed
... 46%, 6432 KB, 3589 KB/s, 1 seconds passed
... 46%, 6464 KB, 3562 KB/s, 1 seconds passed
... 46%, 6496 KB, 3575 KB/s, 1 seconds passed
... 47%, 6528 KB, 3585 KB/s, 1 seconds passed

.. parsed-literal::

    ... 47%, 6560 KB, 3593 KB/s, 1 seconds passed
... 47%, 6592 KB, 3567 KB/s, 1 seconds passed
... 47%, 6624 KB, 3579 KB/s, 1 seconds passed
... 47%, 6656 KB, 3588 KB/s, 1 seconds passed
... 48%, 6688 KB, 3593 KB/s, 1 seconds passed

.. parsed-literal::

    ... 48%, 6720 KB, 3567 KB/s, 1 seconds passed
... 48%, 6752 KB, 3581 KB/s, 1 seconds passed
... 48%, 6784 KB, 3590 KB/s, 1 seconds passed
... 49%, 6816 KB, 3597 KB/s, 1 seconds passed
... 49%, 6848 KB, 3572 KB/s, 1 seconds passed
... 49%, 6880 KB, 3583 KB/s, 1 seconds passed
... 49%, 6912 KB, 3592 KB/s, 1 seconds passed

.. parsed-literal::

    ... 50%, 6944 KB, 3599 KB/s, 1 seconds passed
... 50%, 6976 KB, 3574 KB/s, 1 seconds passed
... 50%, 7008 KB, 3585 KB/s, 1 seconds passed
... 50%, 7040 KB, 3593 KB/s, 1 seconds passed
... 50%, 7072 KB, 3600 KB/s, 1 seconds passed

.. parsed-literal::

    ... 51%, 7104 KB, 3575 KB/s, 1 seconds passed
... 51%, 7136 KB, 3586 KB/s, 1 seconds passed
... 51%, 7168 KB, 3595 KB/s, 1 seconds passed
... 51%, 7200 KB, 3602 KB/s, 1 seconds passed
... 52%, 7232 KB, 3579 KB/s, 2 seconds passed
... 52%, 7264 KB, 3590 KB/s, 2 seconds passed
... 52%, 7296 KB, 3598 KB/s, 2 seconds passed

.. parsed-literal::

    ... 52%, 7328 KB, 3604 KB/s, 2 seconds passed
... 53%, 7360 KB, 3579 KB/s, 2 seconds passed
... 53%, 7392 KB, 3591 KB/s, 2 seconds passed
... 53%, 7424 KB, 3599 KB/s, 2 seconds passed

.. parsed-literal::

    ... 53%, 7456 KB, 3573 KB/s, 2 seconds passed
... 53%, 7488 KB, 3580 KB/s, 2 seconds passed
... 54%, 7520 KB, 3591 KB/s, 2 seconds passed
... 54%, 7552 KB, 3599 KB/s, 2 seconds passed
... 54%, 7584 KB, 3574 KB/s, 2 seconds passed
... 54%, 7616 KB, 3581 KB/s, 2 seconds passed
... 55%, 7648 KB, 3593 KB/s, 2 seconds passed

.. parsed-literal::

    ... 55%, 7680 KB, 3598 KB/s, 2 seconds passed
... 55%, 7712 KB, 3575 KB/s, 2 seconds passed
... 55%, 7744 KB, 3583 KB/s, 2 seconds passed
... 56%, 7776 KB, 3593 KB/s, 2 seconds passed
... 56%, 7808 KB, 3598 KB/s, 2 seconds passed

.. parsed-literal::

    ... 56%, 7840 KB, 3577 KB/s, 2 seconds passed
... 56%, 7872 KB, 3584 KB/s, 2 seconds passed
... 56%, 7904 KB, 3595 KB/s, 2 seconds passed
... 57%, 7936 KB, 3600 KB/s, 2 seconds passed
... 57%, 7968 KB, 3578 KB/s, 2 seconds passed
... 57%, 8000 KB, 3586 KB/s, 2 seconds passed
... 57%, 8032 KB, 3597 KB/s, 2 seconds passed

.. parsed-literal::

    ... 58%, 8064 KB, 3601 KB/s, 2 seconds passed
... 58%, 8096 KB, 3579 KB/s, 2 seconds passed
... 58%, 8128 KB, 3588 KB/s, 2 seconds passed
... 58%, 8160 KB, 3598 KB/s, 2 seconds passed
... 59%, 8192 KB, 3602 KB/s, 2 seconds passed

.. parsed-literal::

    ... 59%, 8224 KB, 3581 KB/s, 2 seconds passed
... 59%, 8256 KB, 3589 KB/s, 2 seconds passed
... 59%, 8288 KB, 3599 KB/s, 2 seconds passed
... 59%, 8320 KB, 3604 KB/s, 2 seconds passed
... 60%, 8352 KB, 3582 KB/s, 2 seconds passed
... 60%, 8384 KB, 3591 KB/s, 2 seconds passed

.. parsed-literal::

    ... 60%, 8416 KB, 3602 KB/s, 2 seconds passed
... 60%, 8448 KB, 3605 KB/s, 2 seconds passed
... 61%, 8480 KB, 3584 KB/s, 2 seconds passed
... 61%, 8512 KB, 3594 KB/s, 2 seconds passed
... 61%, 8544 KB, 3604 KB/s, 2 seconds passed
... 61%, 8576 KB, 3611 KB/s, 2 seconds passed

.. parsed-literal::

    ... 62%, 8608 KB, 3588 KB/s, 2 seconds passed
... 62%, 8640 KB, 3596 KB/s, 2 seconds passed
... 62%, 8672 KB, 3606 KB/s, 2 seconds passed
... 62%, 8704 KB, 3613 KB/s, 2 seconds passed
... 62%, 8736 KB, 3589 KB/s, 2 seconds passed
... 63%, 8768 KB, 3599 KB/s, 2 seconds passed

.. parsed-literal::

    ... 63%, 8800 KB, 3607 KB/s, 2 seconds passed
... 63%, 8832 KB, 3615 KB/s, 2 seconds passed
... 63%, 8864 KB, 3594 KB/s, 2 seconds passed
... 64%, 8896 KB, 3600 KB/s, 2 seconds passed
... 64%, 8928 KB, 3608 KB/s, 2 seconds passed
... 64%, 8960 KB, 3616 KB/s, 2 seconds passed

.. parsed-literal::

    ... 64%, 8992 KB, 3596 KB/s, 2 seconds passed
... 65%, 9024 KB, 3600 KB/s, 2 seconds passed
... 65%, 9056 KB, 3609 KB/s, 2 seconds passed
... 65%, 9088 KB, 3617 KB/s, 2 seconds passed
... 65%, 9120 KB, 3594 KB/s, 2 seconds passed

.. parsed-literal::

    ... 65%, 9152 KB, 3601 KB/s, 2 seconds passed
... 66%, 9184 KB, 3610 KB/s, 2 seconds passed
... 66%, 9216 KB, 3613 KB/s, 2 seconds passed
... 66%, 9248 KB, 3593 KB/s, 2 seconds passed
... 66%, 9280 KB, 3601 KB/s, 2 seconds passed
... 67%, 9312 KB, 3611 KB/s, 2 seconds passed
... 67%, 9344 KB, 3613 KB/s, 2 seconds passed

.. parsed-literal::

    ... 67%, 9376 KB, 3595 KB/s, 2 seconds passed
... 67%, 9408 KB, 3603 KB/s, 2 seconds passed
... 68%, 9440 KB, 3612 KB/s, 2 seconds passed
... 68%, 9472 KB, 3616 KB/s, 2 seconds passed
... 68%, 9504 KB, 3596 KB/s, 2 seconds passed

.. parsed-literal::

    ... 68%, 9536 KB, 3605 KB/s, 2 seconds passed
... 68%, 9568 KB, 3613 KB/s, 2 seconds passed
... 69%, 9600 KB, 3617 KB/s, 2 seconds passed
... 69%, 9632 KB, 3597 KB/s, 2 seconds passed
... 69%, 9664 KB, 3605 KB/s, 2 seconds passed
... 69%, 9696 KB, 3614 KB/s, 2 seconds passed
... 70%, 9728 KB, 3618 KB/s, 2 seconds passed

.. parsed-literal::

    ... 70%, 9760 KB, 3598 KB/s, 2 seconds passed
... 70%, 9792 KB, 3606 KB/s, 2 seconds passed
... 70%, 9824 KB, 3614 KB/s, 2 seconds passed
... 71%, 9856 KB, 3619 KB/s, 2 seconds passed

.. parsed-literal::

    ... 71%, 9888 KB, 3599 KB/s, 2 seconds passed
... 71%, 9920 KB, 3607 KB/s, 2 seconds passed
... 71%, 9952 KB, 3615 KB/s, 2 seconds passed
... 71%, 9984 KB, 3620 KB/s, 2 seconds passed
... 72%, 10016 KB, 3601 KB/s, 2 seconds passed
... 72%, 10048 KB, 3608 KB/s, 2 seconds passed
... 72%, 10080 KB, 3617 KB/s, 2 seconds passed
... 72%, 10112 KB, 3621 KB/s, 2 seconds passed

.. parsed-literal::

    ... 73%, 10144 KB, 3602 KB/s, 2 seconds passed
... 73%, 10176 KB, 3609 KB/s, 2 seconds passed
... 73%, 10208 KB, 3617 KB/s, 2 seconds passed
... 73%, 10240 KB, 3622 KB/s, 2 seconds passed

.. parsed-literal::

    ... 74%, 10272 KB, 3602 KB/s, 2 seconds passed
... 74%, 10304 KB, 3610 KB/s, 2 seconds passed
... 74%, 10336 KB, 3617 KB/s, 2 seconds passed
... 74%, 10368 KB, 3623 KB/s, 2 seconds passed
... 74%, 10400 KB, 3603 KB/s, 2 seconds passed
... 75%, 10432 KB, 3610 KB/s, 2 seconds passed
... 75%, 10464 KB, 3619 KB/s, 2 seconds passed
... 75%, 10496 KB, 3623 KB/s, 2 seconds passed

.. parsed-literal::

    ... 75%, 10528 KB, 3605 KB/s, 2 seconds passed
... 76%, 10560 KB, 3611 KB/s, 2 seconds passed
... 76%, 10592 KB, 3620 KB/s, 2 seconds passed
... 76%, 10624 KB, 3624 KB/s, 2 seconds passed

.. parsed-literal::

    ... 76%, 10656 KB, 3605 KB/s, 2 seconds passed
... 77%, 10688 KB, 3611 KB/s, 2 seconds passed
... 77%, 10720 KB, 3620 KB/s, 2 seconds passed
... 77%, 10752 KB, 3624 KB/s, 2 seconds passed
... 77%, 10784 KB, 3606 KB/s, 2 seconds passed
... 77%, 10816 KB, 3613 KB/s, 2 seconds passed
... 78%, 10848 KB, 3621 KB/s, 2 seconds passed
... 78%, 10880 KB, 3627 KB/s, 2 seconds passed

.. parsed-literal::

    ... 78%, 10912 KB, 3609 KB/s, 3 seconds passed
... 78%, 10944 KB, 3615 KB/s, 3 seconds passed
... 79%, 10976 KB, 3623 KB/s, 3 seconds passed
... 79%, 11008 KB, 3628 KB/s, 3 seconds passed

.. parsed-literal::

    ... 79%, 11040 KB, 3609 KB/s, 3 seconds passed
... 79%, 11072 KB, 3615 KB/s, 3 seconds passed
... 80%, 11104 KB, 3624 KB/s, 3 seconds passed
... 80%, 11136 KB, 3629 KB/s, 3 seconds passed
... 80%, 11168 KB, 3611 KB/s, 3 seconds passed
... 80%, 11200 KB, 3617 KB/s, 3 seconds passed
... 80%, 11232 KB, 3625 KB/s, 3 seconds passed
... 81%, 11264 KB, 3630 KB/s, 3 seconds passed

.. parsed-literal::

    ... 81%, 11296 KB, 3612 KB/s, 3 seconds passed
... 81%, 11328 KB, 3618 KB/s, 3 seconds passed
... 81%, 11360 KB, 3626 KB/s, 3 seconds passed
... 82%, 11392 KB, 3629 KB/s, 3 seconds passed

.. parsed-literal::

    ... 82%, 11424 KB, 3611 KB/s, 3 seconds passed
... 82%, 11456 KB, 3617 KB/s, 3 seconds passed
... 82%, 11488 KB, 3626 KB/s, 3 seconds passed
... 82%, 11520 KB, 3610 KB/s, 3 seconds passed
... 83%, 11552 KB, 3612 KB/s, 3 seconds passed
... 83%, 11584 KB, 3618 KB/s, 3 seconds passed
... 83%, 11616 KB, 3626 KB/s, 3 seconds passed

.. parsed-literal::

    ... 83%, 11648 KB, 3611 KB/s, 3 seconds passed
... 84%, 11680 KB, 3613 KB/s, 3 seconds passed
... 84%, 11712 KB, 3618 KB/s, 3 seconds passed
... 84%, 11744 KB, 3627 KB/s, 3 seconds passed
... 84%, 11776 KB, 3612 KB/s, 3 seconds passed

.. parsed-literal::

    ... 85%, 11808 KB, 3613 KB/s, 3 seconds passed
... 85%, 11840 KB, 3620 KB/s, 3 seconds passed
... 85%, 11872 KB, 3628 KB/s, 3 seconds passed
... 85%, 11904 KB, 3610 KB/s, 3 seconds passed
... 85%, 11936 KB, 3614 KB/s, 3 seconds passed
... 86%, 11968 KB, 3620 KB/s, 3 seconds passed
... 86%, 12000 KB, 3629 KB/s, 3 seconds passed

.. parsed-literal::

    ... 86%, 12032 KB, 3610 KB/s, 3 seconds passed
... 86%, 12064 KB, 3615 KB/s, 3 seconds passed
... 87%, 12096 KB, 3621 KB/s, 3 seconds passed
... 87%, 12128 KB, 3630 KB/s, 3 seconds passed

.. parsed-literal::

    ... 87%, 12160 KB, 3611 KB/s, 3 seconds passed
... 87%, 12192 KB, 3612 KB/s, 3 seconds passed
... 88%, 12224 KB, 3621 KB/s, 3 seconds passed
... 88%, 12256 KB, 3630 KB/s, 3 seconds passed
... 88%, 12288 KB, 3612 KB/s, 3 seconds passed
... 88%, 12320 KB, 3614 KB/s, 3 seconds passed
... 88%, 12352 KB, 3622 KB/s, 3 seconds passed
... 89%, 12384 KB, 3631 KB/s, 3 seconds passed

.. parsed-literal::

    ... 89%, 12416 KB, 3617 KB/s, 3 seconds passed
... 89%, 12448 KB, 3618 KB/s, 3 seconds passed
... 89%, 12480 KB, 3623 KB/s, 3 seconds passed
... 90%, 12512 KB, 3632 KB/s, 3 seconds passed

.. parsed-literal::

    ... 90%, 12544 KB, 3618 KB/s, 3 seconds passed
... 90%, 12576 KB, 3615 KB/s, 3 seconds passed
... 90%, 12608 KB, 3624 KB/s, 3 seconds passed
... 91%, 12640 KB, 3632 KB/s, 3 seconds passed
... 91%, 12672 KB, 3619 KB/s, 3 seconds passed
... 91%, 12704 KB, 3619 KB/s, 3 seconds passed
... 91%, 12736 KB, 3624 KB/s, 3 seconds passed
... 91%, 12768 KB, 3633 KB/s, 3 seconds passed

.. parsed-literal::

    ... 92%, 12800 KB, 3616 KB/s, 3 seconds passed
... 92%, 12832 KB, 3620 KB/s, 3 seconds passed
... 92%, 12864 KB, 3625 KB/s, 3 seconds passed
... 92%, 12896 KB, 3633 KB/s, 3 seconds passed

.. parsed-literal::

    ... 93%, 12928 KB, 3617 KB/s, 3 seconds passed
... 93%, 12960 KB, 3621 KB/s, 3 seconds passed
... 93%, 12992 KB, 3626 KB/s, 3 seconds passed
... 93%, 13024 KB, 3634 KB/s, 3 seconds passed
... 94%, 13056 KB, 3618 KB/s, 3 seconds passed
... 94%, 13088 KB, 3622 KB/s, 3 seconds passed
... 94%, 13120 KB, 3627 KB/s, 3 seconds passed
... 94%, 13152 KB, 3634 KB/s, 3 seconds passed

.. parsed-literal::

    ... 94%, 13184 KB, 3619 KB/s, 3 seconds passed
... 95%, 13216 KB, 3622 KB/s, 3 seconds passed
... 95%, 13248 KB, 3628 KB/s, 3 seconds passed
... 95%, 13280 KB, 3635 KB/s, 3 seconds passed

.. parsed-literal::

    ... 95%, 13312 KB, 3620 KB/s, 3 seconds passed
... 96%, 13344 KB, 3623 KB/s, 3 seconds passed
... 96%, 13376 KB, 3628 KB/s, 3 seconds passed
... 96%, 13408 KB, 3636 KB/s, 3 seconds passed
... 96%, 13440 KB, 3620 KB/s, 3 seconds passed
... 97%, 13472 KB, 3624 KB/s, 3 seconds passed

.. parsed-literal::

    ... 97%, 13504 KB, 3628 KB/s, 3 seconds passed
... 97%, 13536 KB, 3636 KB/s, 3 seconds passed
... 97%, 13568 KB, 3621 KB/s, 3 seconds passed
... 97%, 13600 KB, 3625 KB/s, 3 seconds passed
... 98%, 13632 KB, 3629 KB/s, 3 seconds passed
... 98%, 13664 KB, 3637 KB/s, 3 seconds passed

.. parsed-literal::

    ... 98%, 13696 KB, 3622 KB/s, 3 seconds passed
... 98%, 13728 KB, 3625 KB/s, 3 seconds passed
... 99%, 13760 KB, 3629 KB/s, 3 seconds passed
... 99%, 13792 KB, 3637 KB/s, 3 seconds passed
... 99%, 13824 KB, 3623 KB/s, 3 seconds passed
... 99%, 13856 KB, 3623 KB/s, 3 seconds passed

.. parsed-literal::

    ... 100%, 13879 KB, 3629 KB/s, 3 seconds passed




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
    Conversion to ONNX command: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/bin/python -- /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/openvino/model_zoo/internal_scripts/pytorch_to_onnx.py --model-name=mobilenet_v2 --weights=model/public/mobilenet-v2-pytorch/mobilenet_v2-b0353104.pth --import-module=torchvision.models --input-shape=1,3,224,224 --output-file=model/public/mobilenet-v2-pytorch/mobilenet-v2.onnx --input-names=data --output-names=prob



.. parsed-literal::

    ONNX check passed successfully.


.. parsed-literal::


    ========== Converting mobilenet-v2-pytorch to IR (FP16)
    Conversion command: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/bin/python -- /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/bin/mo --framework=onnx --output_dir=model/public/mobilenet-v2-pytorch/FP16 --model_name=mobilenet-v2-pytorch --input=data '--mean_values=data[123.675,116.28,103.53]' '--scale_values=data[58.624,57.12,57.375]' --reverse_input_channels --output=prob --input_model=model/public/mobilenet-v2-pytorch/mobilenet-v2.onnx '--layout=data(NCHW)' '--input_shape=[1, 3, 224, 224]' --compress_to_fp16=True



.. parsed-literal::

    [ INFO ] Generated IR will be compressed to FP16. If you get lower accuracy, please consider disabling compression explicitly by adding argument --compress_to_fp16=False.
    Find more information about compression to FP16 at https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_FP16_Compression.html
    [ INFO ] The model was converted to IR v11, the latest model format that corresponds to the source DL framework input/output format. While IR v11 is backwards compatible with OpenVINO Inference Engine API v1.0, please use API v2.0 (as of 2022.1) to take advantage of the latest improvements in IR v11.
    Find more information about API v2.0 and IR v11 at https://docs.openvino.ai/2023.0/openvino_2_0_transition_guide.html
    [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release. Please use OpenVINO Model Converter (OVC). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
    Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
    [ SUCCESS ] Generated IR version 11 model.
    [ SUCCESS ] XML file: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/notebooks/104-model-tools/model/public/mobilenet-v2-pytorch/FP16/mobilenet-v2-pytorch.xml
    [ SUCCESS ] BIN file: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/notebooks/104-model-tools/model/public/mobilenet-v2-pytorch/FP16/mobilenet-v2-pytorch.bin




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
      'accuracy_config': '/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/openvino/model_zoo/models/public/mobilenet-v2-pytorch/accuracy-check.yml',
      'model_config': '/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/openvino/model_zoo/models/public/mobilenet-v2-pytorch/model.yml',
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
    [ INFO ] Read model took 28.83 ms
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

    [ INFO ] Compile model took 135.30 ms
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
    [ INFO ] First inference took 6.27 ms


.. parsed-literal::

    [Step 11/11] Dumping statistics report


.. parsed-literal::

    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            20346 iterations
    [ INFO ] Duration:         15007.48 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        4.29 ms
    [ INFO ]    Average:       4.30 ms
    [ INFO ]    Min:           2.40 ms
    [ INFO ]    Max:           12.55 ms
    [ INFO ] Throughput:   1355.72 FPS


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

   **NOTE**: In this notebook, ``benchmark_app`` runs for 15 seconds to
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

