Deblur Photos with DeblurGAN-v2 and OpenVINO™
=============================================

Table of contents:
^^^^^^^^^^^^^^^^^^

-  `What is deblurring? <#what-is-deblurring>`__
-  `Preparations <#preparations>`__

   -  `Imports <#imports>`__
   -  `Settings <#settings>`__
   -  `Select inference device <#select-inference-device>`__
   -  `Download DeblurGAN-v2 Model <#download-deblurgan-v-model>`__
   -  `Prepare model <#prepare-model>`__
   -  `Convert DeblurGAN-v2 Model to OpenVINO IR
      format <#convert-deblurgan-v-model-to-openvino-ir-format>`__

-  `Load the Model <#load-the-model>`__
-  `Deblur Image <#deblur-image>`__

   -  `Load, resize and reshape input
      image <#load-resize-and-reshape-input-image>`__
   -  `Do Inference on the Input
      Image <#do-inference-on-the-input-image>`__
   -  `Display results <#display-results>`__
   -  `Save the deblurred image <#save-the-deblurred-image>`__

This tutorial demonstrates Single Image Motion Deblurring with
DeblurGAN-v2 in OpenVINO, by first converting the
`VITA-Group/DeblurGANv2 <https://github.com/VITA-Group/DeblurGANv2>`__
model to OpenVINO Intermediate Representation (OpenVINO IR) format. For
more information about the model, see the
`documentation <https://docs.openvino.ai/2023.0/omz_models_model_deblurgan_v2.html>`__.

What is deblurring?
~~~~~~~~~~~~~~~~~~~



Deblurring is the task of removing motion blurs that usually occur in
photos shot with hand-held cameras when there are moving objects in the
scene. Blurs not only reduce the human perception about the quality of
the image, but also complicate computer vision analyses.

For more information, refer to the following research paper:

Kupyn, O., Martyniuk, T., Wu, J., & Wang, Z. (2019). `DeblurGAN-v2:
Deblurring (orders-of-magnitude) faster and
better. <https://openaccess.thecvf.com/content_ICCV_2019/html/Kupyn_DeblurGAN-v2_Deblurring_Orders-of-Magnitude_Faster_and_Better_ICCV_2019_paper.html>`__
In Proceedings of the IEEE/CVF International Conference on Computer
Vision (pp. 8878-8887).

Preparations
------------



.. code:: ipython3

    %pip install -q "openvino-dev>=2023.1.0"


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.0 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    

.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


Imports
~~~~~~~



.. code:: ipython3

    import sys
    from pathlib import Path
    
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    from IPython.display import Markdown, display
    import openvino as ov
    
    sys.path.append("../utils")
    from notebook_utils import load_image

Settings
~~~~~~~~



.. code:: ipython3

    # A directory where the model will be downloaded.
    model_dir = Path("model")
    model_dir.mkdir(exist_ok=True)
    
    # The name of the model from Open Model Zoo.
    model_name = "deblurgan-v2"
    model_xml_path = model_dir / f"{model_name}.xml"
    ov_model = None
    
    precision = "FP16"

Select inference device
~~~~~~~~~~~~~~~~~~~~~~~



select device from dropdown list for running inference using OpenVINO

.. code:: ipython3

    import ipywidgets as widgets
    
    core = ov.Core()
    
    device = widgets.Dropdown(
        options=core.available_devices + ["AUTO"],
        value='AUTO',
        description='Device:',
        disabled=False,
    )
    
    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



Download DeblurGAN-v2 Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~



Model defined in
`VITA-Group/DeblurGANv2 <https://github.com/VITA-Group/DeblurGANv2>`__
repository. For converting model we should clone this repo and install
its dependencies. To reduce conversion step, we will use OMZ downloader
for downloading model weights. After downloading is finished, model
related code will be saved in ``model/public/deblurgan-v2/models/``
directory and weights in ``public/deblurgan-v2/ckpt/fpn_mobilenet.h5``

.. code:: ipython3

    download_command = (
        f"omz_downloader --name {model_name} --output_dir"
        f" {model_dir} --cache_dir {model_dir}"
    )
    display(Markdown(f"Download command: `{download_command}`"))
    display(Markdown(f"Downloading {model_name}..."))
    ! $download_command



Download command:
``omz_downloader --name deblurgan-v2 --output_dir model --cache_dir model``



Downloading deblurgan-v2…


.. parsed-literal::

    ################|| Downloading deblurgan-v2 ||################
    
    ========== Downloading model/public/deblurgan-v2/models/__init__.py


.. parsed-literal::

    
    
    ========== Downloading model/public/deblurgan-v2/models/fpn_mobilenet.py


.. parsed-literal::

    ... 100%, 5 KB, 15565 KB/s, 0 seconds passed

    
    ========== Downloading model/public/deblurgan-v2/models/mobilenet_v2.py


.. parsed-literal::

    ... 100%, 4 KB, 14887 KB/s, 0 seconds passed

    
    ========== Downloading model/public/deblurgan-v2/models/networks.py


.. parsed-literal::

    ... 100%, 12 KB, 36393 KB/s, 0 seconds passed

    
    ========== Downloading model/public/deblurgan-v2/ckpt/fpn_mobilenet.h5


.. parsed-literal::

    ... 0%, 32 KB, 1335 KB/s, 0 seconds passed

.. parsed-literal::

    ... 0%, 64 KB, 1061 KB/s, 0 seconds passed
... 0%, 96 KB, 1390 KB/s, 0 seconds passed

.. parsed-literal::

    ... 0%, 128 KB, 1313 KB/s, 0 seconds passed
... 1%, 160 KB, 1359 KB/s, 0 seconds passed
... 1%, 192 KB, 1438 KB/s, 0 seconds passed

.. parsed-literal::

    ... 1%, 224 KB, 1573 KB/s, 0 seconds passed
... 1%, 256 KB, 1792 KB/s, 0 seconds passed
... 2%, 288 KB, 2011 KB/s, 0 seconds passed
... 2%, 320 KB, 1882 KB/s, 0 seconds passed
... 2%, 352 KB, 1966 KB/s, 0 seconds passed
... 2%, 384 KB, 2139 KB/s, 0 seconds passed

.. parsed-literal::

    ... 3%, 416 KB, 2178 KB/s, 0 seconds passed
... 3%, 448 KB, 2164 KB/s, 0 seconds passed
... 3%, 480 KB, 2227 KB/s, 0 seconds passed
... 3%, 512 KB, 2368 KB/s, 0 seconds passed
... 4%, 544 KB, 2511 KB/s, 0 seconds passed

.. parsed-literal::

    ... 4%, 576 KB, 2346 KB/s, 0 seconds passed
... 4%, 608 KB, 2410 KB/s, 0 seconds passed
... 4%, 640 KB, 2530 KB/s, 0 seconds passed
... 5%, 672 KB, 2544 KB/s, 0 seconds passed
... 5%, 704 KB, 2493 KB/s, 0 seconds passed

.. parsed-literal::

    ... 5%, 736 KB, 2543 KB/s, 0 seconds passed
... 5%, 768 KB, 2649 KB/s, 0 seconds passed
... 6%, 800 KB, 2657 KB/s, 0 seconds passed
... 6%, 832 KB, 2604 KB/s, 0 seconds passed
... 6%, 864 KB, 2653 KB/s, 0 seconds passed
... 6%, 896 KB, 2744 KB/s, 0 seconds passed
... 7%, 928 KB, 2751 KB/s, 0 seconds passed

.. parsed-literal::

    ... 7%, 960 KB, 2692 KB/s, 0 seconds passed
... 7%, 992 KB, 2736 KB/s, 0 seconds passed
... 7%, 1024 KB, 2816 KB/s, 0 seconds passed
... 8%, 1056 KB, 2821 KB/s, 0 seconds passed

.. parsed-literal::

    ... 8%, 1088 KB, 2768 KB/s, 0 seconds passed
... 8%, 1120 KB, 2807 KB/s, 0 seconds passed
... 8%, 1152 KB, 2878 KB/s, 0 seconds passed
... 8%, 1184 KB, 2884 KB/s, 0 seconds passed
... 9%, 1216 KB, 2829 KB/s, 0 seconds passed
... 9%, 1248 KB, 2862 KB/s, 0 seconds passed
... 9%, 1280 KB, 2928 KB/s, 0 seconds passed

.. parsed-literal::

    ... 9%, 1312 KB, 2932 KB/s, 0 seconds passed
... 10%, 1344 KB, 2880 KB/s, 0 seconds passed
... 10%, 1376 KB, 2909 KB/s, 0 seconds passed
... 10%, 1408 KB, 2972 KB/s, 0 seconds passed
... 10%, 1440 KB, 2975 KB/s, 0 seconds passed

.. parsed-literal::

    ... 11%, 1472 KB, 2924 KB/s, 0 seconds passed
... 11%, 1504 KB, 2953 KB/s, 0 seconds passed
... 11%, 1536 KB, 3009 KB/s, 0 seconds passed
... 11%, 1568 KB, 3012 KB/s, 0 seconds passed
... 12%, 1600 KB, 2962 KB/s, 0 seconds passed

.. parsed-literal::

    ... 12%, 1632 KB, 2987 KB/s, 0 seconds passed
... 12%, 1664 KB, 3041 KB/s, 0 seconds passed
... 12%, 1696 KB, 3037 KB/s, 0 seconds passed
... 13%, 1728 KB, 2997 KB/s, 0 seconds passed
... 13%, 1760 KB, 3017 KB/s, 0 seconds passed
... 13%, 1792 KB, 3070 KB/s, 0 seconds passed
... 13%, 1824 KB, 3066 KB/s, 0 seconds passed

.. parsed-literal::

    ... 14%, 1856 KB, 3024 KB/s, 0 seconds passed
... 14%, 1888 KB, 3046 KB/s, 0 seconds passed
... 14%, 1920 KB, 3094 KB/s, 0 seconds passed

.. parsed-literal::

    ... 14%, 1952 KB, 3007 KB/s, 0 seconds passed
... 15%, 1984 KB, 3051 KB/s, 0 seconds passed
... 15%, 2016 KB, 3069 KB/s, 0 seconds passed
... 15%, 2048 KB, 3115 KB/s, 0 seconds passed
... 15%, 2080 KB, 3029 KB/s, 0 seconds passed
... 16%, 2112 KB, 3073 KB/s, 0 seconds passed
... 16%, 2144 KB, 3090 KB/s, 0 seconds passed
... 16%, 2176 KB, 3133 KB/s, 0 seconds passed

.. parsed-literal::

    ... 16%, 2208 KB, 3132 KB/s, 0 seconds passed
... 16%, 2240 KB, 3094 KB/s, 0 seconds passed
... 17%, 2272 KB, 3111 KB/s, 0 seconds passed
... 17%, 2304 KB, 3151 KB/s, 0 seconds passed
... 17%, 2336 KB, 3151 KB/s, 0 seconds passed

.. parsed-literal::

    ... 17%, 2368 KB, 3113 KB/s, 0 seconds passed
... 18%, 2400 KB, 3133 KB/s, 0 seconds passed
... 18%, 2432 KB, 3167 KB/s, 0 seconds passed
... 18%, 2464 KB, 3168 KB/s, 0 seconds passed
... 18%, 2496 KB, 3126 KB/s, 0 seconds passed

.. parsed-literal::

    ... 19%, 2528 KB, 3148 KB/s, 0 seconds passed
... 19%, 2560 KB, 3182 KB/s, 0 seconds passed
... 19%, 2592 KB, 3178 KB/s, 0 seconds passed
... 19%, 2624 KB, 3142 KB/s, 0 seconds passed
... 20%, 2656 KB, 3161 KB/s, 0 seconds passed
... 20%, 2688 KB, 3195 KB/s, 0 seconds passed
... 20%, 2720 KB, 3193 KB/s, 0 seconds passed

.. parsed-literal::

    ... 20%, 2752 KB, 3155 KB/s, 0 seconds passed
... 21%, 2784 KB, 3174 KB/s, 0 seconds passed
... 21%, 2816 KB, 3207 KB/s, 0 seconds passed
... 21%, 2848 KB, 3202 KB/s, 0 seconds passed

.. parsed-literal::

    ... 21%, 2880 KB, 3169 KB/s, 0 seconds passed
... 22%, 2912 KB, 3186 KB/s, 0 seconds passed
... 22%, 2944 KB, 3218 KB/s, 0 seconds passed
... 22%, 2976 KB, 3213 KB/s, 0 seconds passed
... 22%, 3008 KB, 3179 KB/s, 0 seconds passed
... 23%, 3040 KB, 3197 KB/s, 0 seconds passed
... 23%, 3072 KB, 3229 KB/s, 0 seconds passed

.. parsed-literal::

    ... 23%, 3104 KB, 3163 KB/s, 0 seconds passed
... 23%, 3136 KB, 3191 KB/s, 0 seconds passed
... 24%, 3168 KB, 3208 KB/s, 0 seconds passed
... 24%, 3200 KB, 3238 KB/s, 0 seconds passed

.. parsed-literal::

    ... 24%, 3232 KB, 3173 KB/s, 1 seconds passed
... 24%, 3264 KB, 3200 KB/s, 1 seconds passed
... 24%, 3296 KB, 3217 KB/s, 1 seconds passed
... 25%, 3328 KB, 3246 KB/s, 1 seconds passed
... 25%, 3360 KB, 3183 KB/s, 1 seconds passed
... 25%, 3392 KB, 3209 KB/s, 1 seconds passed

.. parsed-literal::

    ... 25%, 3424 KB, 3226 KB/s, 1 seconds passed
... 26%, 3456 KB, 3254 KB/s, 1 seconds passed
... 26%, 3488 KB, 3253 KB/s, 1 seconds passed
... 26%, 3520 KB, 3220 KB/s, 1 seconds passed
... 26%, 3552 KB, 3236 KB/s, 1 seconds passed
... 27%, 3584 KB, 3262 KB/s, 1 seconds passed

.. parsed-literal::

    ... 27%, 3616 KB, 3202 KB/s, 1 seconds passed
... 27%, 3648 KB, 3229 KB/s, 1 seconds passed
... 27%, 3680 KB, 3243 KB/s, 1 seconds passed
... 28%, 3712 KB, 3270 KB/s, 1 seconds passed

.. parsed-literal::

    ... 28%, 3744 KB, 3212 KB/s, 1 seconds passed
... 28%, 3776 KB, 3235 KB/s, 1 seconds passed
... 28%, 3808 KB, 3249 KB/s, 1 seconds passed
... 29%, 3840 KB, 3275 KB/s, 1 seconds passed
... 29%, 3872 KB, 3220 KB/s, 1 seconds passed
... 29%, 3904 KB, 3244 KB/s, 1 seconds passed
... 29%, 3936 KB, 3257 KB/s, 1 seconds passed
... 30%, 3968 KB, 3282 KB/s, 1 seconds passed

.. parsed-literal::

    ... 30%, 4000 KB, 3228 KB/s, 1 seconds passed
... 30%, 4032 KB, 3251 KB/s, 1 seconds passed
... 30%, 4064 KB, 3265 KB/s, 1 seconds passed
... 31%, 4096 KB, 3287 KB/s, 1 seconds passed

.. parsed-literal::

    ... 31%, 4128 KB, 3235 KB/s, 1 seconds passed
... 31%, 4160 KB, 3257 KB/s, 1 seconds passed
... 31%, 4192 KB, 3271 KB/s, 1 seconds passed
... 32%, 4224 KB, 3273 KB/s, 1 seconds passed
... 32%, 4256 KB, 3241 KB/s, 1 seconds passed

.. parsed-literal::

    ... 32%, 4288 KB, 3258 KB/s, 1 seconds passed
... 32%, 4320 KB, 3275 KB/s, 1 seconds passed
... 32%, 4352 KB, 3280 KB/s, 1 seconds passed
... 33%, 4384 KB, 3249 KB/s, 1 seconds passed
... 33%, 4416 KB, 3270 KB/s, 1 seconds passed
... 33%, 4448 KB, 3283 KB/s, 1 seconds passed
... 33%, 4480 KB, 3285 KB/s, 1 seconds passed

.. parsed-literal::

    ... 34%, 4512 KB, 3254 KB/s, 1 seconds passed
... 34%, 4544 KB, 3275 KB/s, 1 seconds passed
... 34%, 4576 KB, 3288 KB/s, 1 seconds passed
... 34%, 4608 KB, 3291 KB/s, 1 seconds passed

.. parsed-literal::

    ... 35%, 4640 KB, 3260 KB/s, 1 seconds passed
... 35%, 4672 KB, 3281 KB/s, 1 seconds passed
... 35%, 4704 KB, 3293 KB/s, 1 seconds passed
... 35%, 4736 KB, 3296 KB/s, 1 seconds passed
... 36%, 4768 KB, 3265 KB/s, 1 seconds passed
... 36%, 4800 KB, 3285 KB/s, 1 seconds passed
... 36%, 4832 KB, 3297 KB/s, 1 seconds passed

.. parsed-literal::

    ... 36%, 4864 KB, 3300 KB/s, 1 seconds passed
... 37%, 4896 KB, 3270 KB/s, 1 seconds passed
... 37%, 4928 KB, 3290 KB/s, 1 seconds passed
... 37%, 4960 KB, 3302 KB/s, 1 seconds passed
... 37%, 4992 KB, 3304 KB/s, 1 seconds passed

.. parsed-literal::

    ... 38%, 5024 KB, 3275 KB/s, 1 seconds passed
... 38%, 5056 KB, 3295 KB/s, 1 seconds passed
... 38%, 5088 KB, 3306 KB/s, 1 seconds passed
... 38%, 5120 KB, 3307 KB/s, 1 seconds passed
... 39%, 5152 KB, 3280 KB/s, 1 seconds passed
... 39%, 5184 KB, 3299 KB/s, 1 seconds passed

.. parsed-literal::

    ... 39%, 5216 KB, 3309 KB/s, 1 seconds passed
... 39%, 5248 KB, 3310 KB/s, 1 seconds passed
... 40%, 5280 KB, 3284 KB/s, 1 seconds passed
... 40%, 5312 KB, 3302 KB/s, 1 seconds passed
... 40%, 5344 KB, 3314 KB/s, 1 seconds passed
... 40%, 5376 KB, 3316 KB/s, 1 seconds passed

.. parsed-literal::

    ... 41%, 5408 KB, 3290 KB/s, 1 seconds passed
... 41%, 5440 KB, 3307 KB/s, 1 seconds passed
... 41%, 5472 KB, 3318 KB/s, 1 seconds passed
... 41%, 5504 KB, 3320 KB/s, 1 seconds passed

.. parsed-literal::

    ... 41%, 5536 KB, 3295 KB/s, 1 seconds passed
... 42%, 5568 KB, 3310 KB/s, 1 seconds passed
... 42%, 5600 KB, 3322 KB/s, 1 seconds passed
... 42%, 5632 KB, 3323 KB/s, 1 seconds passed
... 42%, 5664 KB, 3298 KB/s, 1 seconds passed
... 43%, 5696 KB, 3314 KB/s, 1 seconds passed
... 43%, 5728 KB, 3325 KB/s, 1 seconds passed

.. parsed-literal::

    ... 43%, 5760 KB, 3326 KB/s, 1 seconds passed
... 43%, 5792 KB, 3302 KB/s, 1 seconds passed
... 44%, 5824 KB, 3318 KB/s, 1 seconds passed
... 44%, 5856 KB, 3328 KB/s, 1 seconds passed
... 44%, 5888 KB, 3329 KB/s, 1 seconds passed

.. parsed-literal::

    ... 44%, 5920 KB, 3304 KB/s, 1 seconds passed
... 45%, 5952 KB, 3319 KB/s, 1 seconds passed
... 45%, 5984 KB, 3330 KB/s, 1 seconds passed
... 45%, 6016 KB, 3332 KB/s, 1 seconds passed
... 45%, 6048 KB, 3307 KB/s, 1 seconds passed

.. parsed-literal::

    ... 46%, 6080 KB, 3323 KB/s, 1 seconds passed
... 46%, 6112 KB, 3333 KB/s, 1 seconds passed
... 46%, 6144 KB, 3334 KB/s, 1 seconds passed
... 46%, 6176 KB, 3312 KB/s, 1 seconds passed
... 47%, 6208 KB, 3326 KB/s, 1 seconds passed
... 47%, 6240 KB, 3336 KB/s, 1 seconds passed
... 47%, 6272 KB, 3337 KB/s, 1 seconds passed

.. parsed-literal::

    ... 47%, 6304 KB, 3315 KB/s, 1 seconds passed
... 48%, 6336 KB, 3329 KB/s, 1 seconds passed
... 48%, 6368 KB, 3339 KB/s, 1 seconds passed
... 48%, 6400 KB, 3341 KB/s, 1 seconds passed

.. parsed-literal::

    ... 48%, 6432 KB, 3318 KB/s, 1 seconds passed
... 49%, 6464 KB, 3333 KB/s, 1 seconds passed
... 49%, 6496 KB, 3342 KB/s, 1 seconds passed
... 49%, 6528 KB, 3310 KB/s, 1 seconds passed
... 49%, 6560 KB, 3320 KB/s, 1 seconds passed
... 49%, 6592 KB, 3334 KB/s, 1 seconds passed
... 50%, 6624 KB, 3344 KB/s, 1 seconds passed

.. parsed-literal::

    ... 50%, 6656 KB, 3313 KB/s, 2 seconds passed
... 50%, 6688 KB, 3322 KB/s, 2 seconds passed
... 50%, 6720 KB, 3336 KB/s, 2 seconds passed
... 51%, 6752 KB, 3345 KB/s, 2 seconds passed
... 51%, 6784 KB, 3348 KB/s, 2 seconds passed

.. parsed-literal::

    ... 51%, 6816 KB, 3325 KB/s, 2 seconds passed
... 51%, 6848 KB, 3339 KB/s, 2 seconds passed
... 52%, 6880 KB, 3348 KB/s, 2 seconds passed
... 52%, 6912 KB, 3351 KB/s, 2 seconds passed

.. parsed-literal::

    ... 52%, 6944 KB, 3328 KB/s, 2 seconds passed
... 52%, 6976 KB, 3342 KB/s, 2 seconds passed
... 53%, 7008 KB, 3351 KB/s, 2 seconds passed
... 53%, 7040 KB, 3354 KB/s, 2 seconds passed
... 53%, 7072 KB, 3331 KB/s, 2 seconds passed
... 53%, 7104 KB, 3344 KB/s, 2 seconds passed
... 54%, 7136 KB, 3353 KB/s, 2 seconds passed

.. parsed-literal::

    ... 54%, 7168 KB, 3325 KB/s, 2 seconds passed
... 54%, 7200 KB, 3333 KB/s, 2 seconds passed
... 54%, 7232 KB, 3346 KB/s, 2 seconds passed
... 55%, 7264 KB, 3349 KB/s, 2 seconds passed

.. parsed-literal::

    ... 55%, 7296 KB, 3328 KB/s, 2 seconds passed
... 55%, 7328 KB, 3335 KB/s, 2 seconds passed
... 55%, 7360 KB, 3348 KB/s, 2 seconds passed
... 56%, 7392 KB, 3358 KB/s, 2 seconds passed
... 56%, 7424 KB, 3331 KB/s, 2 seconds passed
... 56%, 7456 KB, 3338 KB/s, 2 seconds passed
... 56%, 7488 KB, 3351 KB/s, 2 seconds passed
... 57%, 7520 KB, 3360 KB/s, 2 seconds passed

.. parsed-literal::

    ... 57%, 7552 KB, 3332 KB/s, 2 seconds passed
... 57%, 7584 KB, 3340 KB/s, 2 seconds passed
... 57%, 7616 KB, 3353 KB/s, 2 seconds passed
... 57%, 7648 KB, 3363 KB/s, 2 seconds passed
... 58%, 7680 KB, 3364 KB/s, 2 seconds passed

.. parsed-literal::

    ... 58%, 7712 KB, 3342 KB/s, 2 seconds passed
... 58%, 7744 KB, 3355 KB/s, 2 seconds passed
... 58%, 7776 KB, 3365 KB/s, 2 seconds passed
... 59%, 7808 KB, 3366 KB/s, 2 seconds passed

.. parsed-literal::

    ... 59%, 7840 KB, 3343 KB/s, 2 seconds passed
... 59%, 7872 KB, 3356 KB/s, 2 seconds passed
... 59%, 7904 KB, 3366 KB/s, 2 seconds passed
... 60%, 7936 KB, 3368 KB/s, 2 seconds passed
... 60%, 7968 KB, 3345 KB/s, 2 seconds passed
... 60%, 8000 KB, 3357 KB/s, 2 seconds passed
... 60%, 8032 KB, 3368 KB/s, 2 seconds passed
... 61%, 8064 KB, 3370 KB/s, 2 seconds passed

.. parsed-literal::

    ... 61%, 8096 KB, 3346 KB/s, 2 seconds passed
... 61%, 8128 KB, 3359 KB/s, 2 seconds passed
... 61%, 8160 KB, 3369 KB/s, 2 seconds passed

.. parsed-literal::

    ... 62%, 8192 KB, 3344 KB/s, 2 seconds passed
... 62%, 8224 KB, 3348 KB/s, 2 seconds passed
... 62%, 8256 KB, 3360 KB/s, 2 seconds passed
... 62%, 8288 KB, 3371 KB/s, 2 seconds passed
... 63%, 8320 KB, 3374 KB/s, 2 seconds passed
... 63%, 8352 KB, 3350 KB/s, 2 seconds passed
... 63%, 8384 KB, 3362 KB/s, 2 seconds passed
... 63%, 8416 KB, 3373 KB/s, 2 seconds passed

.. parsed-literal::

    ... 64%, 8448 KB, 3348 KB/s, 2 seconds passed
... 64%, 8480 KB, 3352 KB/s, 2 seconds passed
... 64%, 8512 KB, 3364 KB/s, 2 seconds passed
... 64%, 8544 KB, 3375 KB/s, 2 seconds passed

.. parsed-literal::

    ... 65%, 8576 KB, 3347 KB/s, 2 seconds passed
... 65%, 8608 KB, 3353 KB/s, 2 seconds passed
... 65%, 8640 KB, 3365 KB/s, 2 seconds passed
... 65%, 8672 KB, 3370 KB/s, 2 seconds passed

.. parsed-literal::

    ... 65%, 8704 KB, 3348 KB/s, 2 seconds passed
... 66%, 8736 KB, 3355 KB/s, 2 seconds passed
... 66%, 8768 KB, 3366 KB/s, 2 seconds passed
... 66%, 8800 KB, 3371 KB/s, 2 seconds passed
... 66%, 8832 KB, 3353 KB/s, 2 seconds passed
... 67%, 8864 KB, 3357 KB/s, 2 seconds passed
... 67%, 8896 KB, 3368 KB/s, 2 seconds passed
... 67%, 8928 KB, 3379 KB/s, 2 seconds passed

.. parsed-literal::

    ... 67%, 8960 KB, 3352 KB/s, 2 seconds passed
... 68%, 8992 KB, 3359 KB/s, 2 seconds passed
... 68%, 9024 KB, 3369 KB/s, 2 seconds passed
... 68%, 9056 KB, 3380 KB/s, 2 seconds passed

.. parsed-literal::

    ... 68%, 9088 KB, 3353 KB/s, 2 seconds passed
... 69%, 9120 KB, 3360 KB/s, 2 seconds passed
... 69%, 9152 KB, 3370 KB/s, 2 seconds passed
... 69%, 9184 KB, 3373 KB/s, 2 seconds passed
... 69%, 9216 KB, 3355 KB/s, 2 seconds passed
... 70%, 9248 KB, 3361 KB/s, 2 seconds passed
... 70%, 9280 KB, 3372 KB/s, 2 seconds passed

.. parsed-literal::

    ... 70%, 9312 KB, 3376 KB/s, 2 seconds passed
... 70%, 9344 KB, 3357 KB/s, 2 seconds passed
... 71%, 9376 KB, 3364 KB/s, 2 seconds passed
... 71%, 9408 KB, 3374 KB/s, 2 seconds passed
... 71%, 9440 KB, 3378 KB/s, 2 seconds passed

.. parsed-literal::

    ... 71%, 9472 KB, 3358 KB/s, 2 seconds passed
... 72%, 9504 KB, 3365 KB/s, 2 seconds passed
... 72%, 9536 KB, 3375 KB/s, 2 seconds passed
... 72%, 9568 KB, 3382 KB/s, 2 seconds passed

.. parsed-literal::

    ... 72%, 9600 KB, 3360 KB/s, 2 seconds passed
... 73%, 9632 KB, 3368 KB/s, 2 seconds passed
... 73%, 9664 KB, 3377 KB/s, 2 seconds passed
... 73%, 9696 KB, 3383 KB/s, 2 seconds passed
... 73%, 9728 KB, 3362 KB/s, 2 seconds passed
... 74%, 9760 KB, 3369 KB/s, 2 seconds passed
... 74%, 9792 KB, 3378 KB/s, 2 seconds passed
... 74%, 9824 KB, 3381 KB/s, 2 seconds passed

.. parsed-literal::

    ... 74%, 9856 KB, 3362 KB/s, 2 seconds passed
... 74%, 9888 KB, 3370 KB/s, 2 seconds passed
... 75%, 9920 KB, 3379 KB/s, 2 seconds passed
... 75%, 9952 KB, 3383 KB/s, 2 seconds passed

.. parsed-literal::

    ... 75%, 9984 KB, 3364 KB/s, 2 seconds passed
... 75%, 10016 KB, 3371 KB/s, 2 seconds passed
... 76%, 10048 KB, 3380 KB/s, 2 seconds passed
... 76%, 10080 KB, 3384 KB/s, 2 seconds passed
... 76%, 10112 KB, 3366 KB/s, 3 seconds passed
... 76%, 10144 KB, 3372 KB/s, 3 seconds passed
... 77%, 10176 KB, 3382 KB/s, 3 seconds passed

.. parsed-literal::

    ... 77%, 10208 KB, 3386 KB/s, 3 seconds passed
... 77%, 10240 KB, 3365 KB/s, 3 seconds passed
... 77%, 10272 KB, 3371 KB/s, 3 seconds passed
... 78%, 10304 KB, 3380 KB/s, 3 seconds passed
... 78%, 10336 KB, 3388 KB/s, 3 seconds passed

.. parsed-literal::

    ... 78%, 10368 KB, 3368 KB/s, 3 seconds passed
... 78%, 10400 KB, 3374 KB/s, 3 seconds passed
... 79%, 10432 KB, 3382 KB/s, 3 seconds passed
... 79%, 10464 KB, 3389 KB/s, 3 seconds passed

.. parsed-literal::

    ... 79%, 10496 KB, 3369 KB/s, 3 seconds passed
... 79%, 10528 KB, 3375 KB/s, 3 seconds passed
... 80%, 10560 KB, 3384 KB/s, 3 seconds passed
... 80%, 10592 KB, 3387 KB/s, 3 seconds passed
... 80%, 10624 KB, 3371 KB/s, 3 seconds passed
... 80%, 10656 KB, 3376 KB/s, 3 seconds passed
... 81%, 10688 KB, 3385 KB/s, 3 seconds passed
... 81%, 10720 KB, 3389 KB/s, 3 seconds passed

.. parsed-literal::

    ... 81%, 10752 KB, 3372 KB/s, 3 seconds passed
... 81%, 10784 KB, 3377 KB/s, 3 seconds passed
... 82%, 10816 KB, 3385 KB/s, 3 seconds passed
... 82%, 10848 KB, 3389 KB/s, 3 seconds passed

.. parsed-literal::

    ... 82%, 10880 KB, 3372 KB/s, 3 seconds passed
... 82%, 10912 KB, 3378 KB/s, 3 seconds passed
... 82%, 10944 KB, 3386 KB/s, 3 seconds passed
... 83%, 10976 KB, 3390 KB/s, 3 seconds passed
... 83%, 11008 KB, 3373 KB/s, 3 seconds passed

.. parsed-literal::

    ... 83%, 11040 KB, 3379 KB/s, 3 seconds passed
... 83%, 11072 KB, 3387 KB/s, 3 seconds passed
... 84%, 11104 KB, 3391 KB/s, 3 seconds passed
... 84%, 11136 KB, 3374 KB/s, 3 seconds passed
... 84%, 11168 KB, 3380 KB/s, 3 seconds passed
... 84%, 11200 KB, 3388 KB/s, 3 seconds passed
... 85%, 11232 KB, 3392 KB/s, 3 seconds passed

.. parsed-literal::

    ... 85%, 11264 KB, 3375 KB/s, 3 seconds passed
... 85%, 11296 KB, 3381 KB/s, 3 seconds passed
... 85%, 11328 KB, 3389 KB/s, 3 seconds passed
... 86%, 11360 KB, 3393 KB/s, 3 seconds passed

.. parsed-literal::

    ... 86%, 11392 KB, 3376 KB/s, 3 seconds passed
... 86%, 11424 KB, 3382 KB/s, 3 seconds passed
... 86%, 11456 KB, 3390 KB/s, 3 seconds passed
... 87%, 11488 KB, 3393 KB/s, 3 seconds passed
... 87%, 11520 KB, 3377 KB/s, 3 seconds passed
... 87%, 11552 KB, 3383 KB/s, 3 seconds passed
... 87%, 11584 KB, 3391 KB/s, 3 seconds passed

.. parsed-literal::

    ... 88%, 11616 KB, 3394 KB/s, 3 seconds passed
... 88%, 11648 KB, 3378 KB/s, 3 seconds passed
... 88%, 11680 KB, 3384 KB/s, 3 seconds passed
... 88%, 11712 KB, 3392 KB/s, 3 seconds passed
... 89%, 11744 KB, 3395 KB/s, 3 seconds passed

.. parsed-literal::

    ... 89%, 11776 KB, 3379 KB/s, 3 seconds passed
... 89%, 11808 KB, 3385 KB/s, 3 seconds passed
... 89%, 11840 KB, 3393 KB/s, 3 seconds passed
... 90%, 11872 KB, 3396 KB/s, 3 seconds passed
... 90%, 11904 KB, 3379 KB/s, 3 seconds passed

.. parsed-literal::

    ... 90%, 11936 KB, 3386 KB/s, 3 seconds passed
... 90%, 11968 KB, 3394 KB/s, 3 seconds passed
... 90%, 12000 KB, 3397 KB/s, 3 seconds passed
... 91%, 12032 KB, 3380 KB/s, 3 seconds passed
... 91%, 12064 KB, 3388 KB/s, 3 seconds passed
... 91%, 12096 KB, 3395 KB/s, 3 seconds passed
... 91%, 12128 KB, 3399 KB/s, 3 seconds passed

.. parsed-literal::

    ... 92%, 12160 KB, 3383 KB/s, 3 seconds passed
... 92%, 12192 KB, 3389 KB/s, 3 seconds passed
... 92%, 12224 KB, 3396 KB/s, 3 seconds passed
... 92%, 12256 KB, 3399 KB/s, 3 seconds passed

.. parsed-literal::

    ... 93%, 12288 KB, 3383 KB/s, 3 seconds passed
... 93%, 12320 KB, 3389 KB/s, 3 seconds passed
... 93%, 12352 KB, 3396 KB/s, 3 seconds passed
... 93%, 12384 KB, 3382 KB/s, 3 seconds passed
... 94%, 12416 KB, 3382 KB/s, 3 seconds passed
... 94%, 12448 KB, 3390 KB/s, 3 seconds passed
... 94%, 12480 KB, 3397 KB/s, 3 seconds passed

.. parsed-literal::

    ... 94%, 12512 KB, 3383 KB/s, 3 seconds passed
... 95%, 12544 KB, 3383 KB/s, 3 seconds passed
... 95%, 12576 KB, 3391 KB/s, 3 seconds passed
... 95%, 12608 KB, 3398 KB/s, 3 seconds passed

.. parsed-literal::

    ... 95%, 12640 KB, 3385 KB/s, 3 seconds passed
... 96%, 12672 KB, 3384 KB/s, 3 seconds passed
... 96%, 12704 KB, 3391 KB/s, 3 seconds passed
... 96%, 12736 KB, 3399 KB/s, 3 seconds passed
... 96%, 12768 KB, 3386 KB/s, 3 seconds passed

.. parsed-literal::

    ... 97%, 12800 KB, 3384 KB/s, 3 seconds passed
... 97%, 12832 KB, 3392 KB/s, 3 seconds passed
... 97%, 12864 KB, 3400 KB/s, 3 seconds passed
... 97%, 12896 KB, 3403 KB/s, 3 seconds passed
... 98%, 12928 KB, 3388 KB/s, 3 seconds passed
... 98%, 12960 KB, 3393 KB/s, 3 seconds passed
... 98%, 12992 KB, 3401 KB/s, 3 seconds passed

.. parsed-literal::

    ... 98%, 13024 KB, 3387 KB/s, 3 seconds passed
... 98%, 13056 KB, 3386 KB/s, 3 seconds passed
... 99%, 13088 KB, 3393 KB/s, 3 seconds passed
... 99%, 13120 KB, 3401 KB/s, 3 seconds passed

.. parsed-literal::

    ... 99%, 13152 KB, 3385 KB/s, 3 seconds passed
... 99%, 13184 KB, 3387 KB/s, 3 seconds passed
... 100%, 13188 KB, 3388 KB/s, 3 seconds passed

    
    ========== Replacing text in model/public/deblurgan-v2/models/networks.py
    ========== Replacing text in model/public/deblurgan-v2/models/fpn_mobilenet.py
    ========== Replacing text in model/public/deblurgan-v2/models/fpn_mobilenet.py
    


Prepare model
~~~~~~~~~~~~~



DeblurGAN-v2 is PyTorch model for converting it to OpenVINO Intermediate
Representation format, we should first instantiate model class and load
checkpoint weights.

.. code:: ipython3

    sys.path.append("model/public/deblurgan-v2")
    
    import torch
    
    from models.networks import get_generator
    
    
    class DeblurV2(torch.nn.Module):
        def __init__(self, weights, model_name):
            super().__init__()
    
            parameters = {'g_name': model_name, 'norm_layer': 'instance'}
            self.impl = get_generator(parameters)
            checkpoint = torch.load(weights, map_location='cpu')['model']
            self.impl.load_state_dict(checkpoint)
            self.impl.train(True)
    
        def forward(self, image):
            out = self.impl(image)
            # convert out to [0, 1] range
            out = (out + 1) / 2
            return out

Convert DeblurGAN-v2 Model to OpenVINO IR format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



For best results with OpenVINO, it is recommended to convert the model
to OpenVINO IR format. To convert the PyTorch model, we will use model
conversion Python API. The ``ov.convert_model`` Python function returns
an OpenVINO model ready to load on a device and start making
predictions. We can save the model on the disk for next usage with
``ov.save_model``. For more information about model conversion Python
API, see this
`page <https://docs.openvino.ai/2023.0/openvino_docs_model_processing_introduction.html>`__.

Model conversion may take a while.

.. code:: ipython3

    deblur_gan_model = DeblurV2("model/public/deblurgan-v2/ckpt/fpn_mobilenet.h5", "fpn_mobilenet")
    
    with torch.no_grad():
        deblur_gan_model.eval()
        ov_model = ov.convert_model(deblur_gan_model, example_input=torch.ones((1,3,736,1312), dtype=torch.float32), input=[[1,3,736,1312]])
        ov.save_model(ov_model, model_xml_path, compress_to_fp16=(precision == "FP16"))

Load the Model
--------------



Load and compile the DeblurGAN-v2 model in the OpenVINO Runtime with
``core.read_model`` and compile it for the specified device with
``core.compile_model``. Get input and output keys and the expected input
shape for the model.

.. code:: ipython3

    model = core.read_model(model=model_xml_path)
    compiled_model = core.compile_model(model=model, device_name=device.value)

.. code:: ipython3

    model_input_layer = compiled_model.input(0)
    model_output_layer = compiled_model.output(0)

.. code:: ipython3

    model_input_layer




.. parsed-literal::

    <ConstOutput: names[image] shape[1,3,736,1312] type: f32>



.. code:: ipython3

    model_output_layer




.. parsed-literal::

    <ConstOutput: names[] shape[1,3,736,1312] type: f32>



Deblur Image
------------



Load, resize and reshape input image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



The input image is read by using the default ``load_image`` function
from ``notebooks.utils``. Then, resized to meet the network expected
input sizes, and reshaped to ``(N, C, H, W)``, where ``N`` is a number
of images in the batch, ``C`` is a number of channels, ``H`` is the
height, and ``W`` is the width.

.. code:: ipython3

    # Image filename (local path or URL)
    filename = "https://raw.githubusercontent.com/VITA-Group/DeblurGANv2/master/test_img/000027.png"

.. code:: ipython3

    # Load the input image.
    # Load image returns image in BGR format
    image = load_image(filename)
    
    # Convert the image to expected by model RGB format
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # N,C,H,W = batch size, number of channels, height, width.
    N, C, H, W = model_input_layer.shape
    
    # Resize the image to meet network expected input sizes.
    resized_image = cv2.resize(image, (W, H))
    
    # Convert image to float32 precision anf normalize in [-1, 1] range
    input_image = (resized_image.astype(np.float32) - 127.5) / 127.5
    
    # Add batch dimension to input image tensor
    input_image = np.expand_dims(input_image.transpose(2, 0, 1), 0) 

.. code:: ipython3

    plt.imshow(image);



.. image:: 217-vision-deblur-with-output_files/217-vision-deblur-with-output_25_0.png


Do Inference on the Input Image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



Do the inference, convert the result to an image shape and resize it to
the original image size.

.. code:: ipython3

    # Inference.
    result = compiled_model([input_image])[model_output_layer]
    
    # Convert the result to an image shape and [0, 255] range
    result_image = result[0].transpose((1, 2, 0)) * 255
    
    h, w = image.shape[:2]
    
    # Resize to the original image size and convert to original u8 precision
    resized_result_image = cv2.resize(result_image, (w, h)).astype(np.uint8)

.. code:: ipython3

    plt.imshow(resized_result_image);



.. image:: 217-vision-deblur-with-output_files/217-vision-deblur-with-output_28_0.png


Display results
~~~~~~~~~~~~~~~



.. code:: ipython3

    # Create subplot(r,c) by providing the no. of rows (r),
    # number of columns (c) and figure size.
    f, ax = plt.subplots(1, 2, figsize=(20, 20))
    
    # Use the created array and display the images horizontally.
    ax[0].set_title("Blurred")
    ax[0].imshow(image)
    
    ax[1].set_title("DeblurGAN-v2")
    ax[1].imshow(resized_result_image);



.. image:: 217-vision-deblur-with-output_files/217-vision-deblur-with-output_30_0.png


Save the deblurred image
~~~~~~~~~~~~~~~~~~~~~~~~



Save the output image of the DeblurGAN-v2 model in the current
directory.

.. code:: ipython3

    savename = "deblurred.png"
    cv2.imwrite(savename, resized_result_image);
