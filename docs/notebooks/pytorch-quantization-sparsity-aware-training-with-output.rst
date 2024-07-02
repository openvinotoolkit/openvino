Quantization-Sparsity Aware Training with NNCF, using PyTorch framework
=======================================================================

This notebook is based on `ImageNet training in
PyTorch <https://github.com/pytorch/examples/blob/master/imagenet/main.py>`__.

The goal of this notebook is to demonstrate how to use the Neural
Network Compression Framework
`NNCF <https://github.com/openvinotoolkit/nncf>`__ 8-bit quantization to
optimize a PyTorch model for inference with OpenVINO Toolkit. The
optimization process contains the following steps:

-  Transforming the original dense ``FP32`` model to sparse ``INT8``
-  Using fine-tuning to improve the accuracy.
-  Exporting optimized and original models to OpenVINO IR
-  Measuring and comparing the performance of models.

For more advanced usage, refer to these
`examples <https://github.com/openvinotoolkit/nncf/tree/develop/examples>`__.

This tutorial uses the ResNet-50 model with the ImageNet dataset. The
dataset must be downloaded separately. To see ResNet models, visit
`PyTorch hub <https://pytorch.org/hub/pytorch_vision_resnet/>`__.

Table of contents:
^^^^^^^^^^^^^^^^^^

-  `Imports and Settings <#Imports-and-Settings>`__
-  `Pre-train Floating-Point Model <#Pre-train-Floating-Point-Model>`__

   -  `Train Function <#Train-Function>`__
   -  `Validate Function <#Validate-Function>`__
   -  `Helpers <#Helpers>`__
   -  `Get a Pre-trained FP32 Model <#Get-a-Pre-trained-FP32-Model>`__

-  `Create and Initialize
   Quantization <#Create-and-Initialize-Quantization>`__
-  `Fine-tune the Compressed Model <#Fine-tune-the-Compressed-Model>`__
-  `Export INT8 Sparse Model to OpenVINO
   IR <#Export-INT8-Model-to-OpenVINO-IR>`__
-  `Benchmark Model Performance by Computing Inference
   Time <#Benchmark-Model-Performance-by-Computing-Inference-Time>`__

.. code:: ipython3

    %pip install -q --extra-index-url https://download.pytorch.org/whl/cpu  "openvino>=2024.0.0" "torch" "torchvision" "tqdm"
    %pip install -q "nncf>=2.9.0"


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.
    Note: you may need to restart the kernel to use updated packages.


Imports and Settings
--------------------

`back to top ⬆️ <#Table-of-contents:>`__

On Windows, add the required C++ directories to the system PATH.

Import NNCF and all auxiliary packages from your Python code. Set a name
for the model, and the image width and height that will be used for the
network. Also define paths where PyTorch and OpenVINO IR versions of the
models will be stored.

   **NOTE**: All NNCF logging messages below ERROR level (INFO and
   WARNING) are disabled to simplify the tutorial. For production use,
   it is recommended to enable logging by removing
   ``set_log_level(logging.ERROR)``.

.. code:: ipython3

    import time
    import warnings  # To disable warnings on export model
    from pathlib import Path
    
    import torch
    
    import torch.nn as nn
    import torch.nn.parallel
    import torch.optim
    import torch.utils.data
    import torch.utils.data.distributed
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    import torchvision.models as models
    
    import openvino as ov
    from torch.jit import TracerWarning
    
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")
    
    MODEL_DIR = Path("model")
    OUTPUT_DIR = Path("output")
    # DATA_DIR = Path("...")  # Insert path to folder containing imagenet folder
    # DATASET_DIR = DATA_DIR / "imagenet"


.. parsed-literal::

    Using cpu device


.. code:: ipython3

    # Fetch `notebook_utils` module
    import zipfile
    import requests
    
    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
    )
    open("notebook_utils.py", "w").write(r.text)
    from notebook_utils import download_file
    
    DATA_DIR = Path("data")
    
    
    def download_tiny_imagenet_200(
        data_dir: Path,
        url="http://cs231n.stanford.edu/tiny-imagenet-200.zip",
        tarname="tiny-imagenet-200.zip",
    ):
        archive_path = data_dir / tarname
        download_file(url, directory=data_dir, filename=tarname)
        zip_ref = zipfile.ZipFile(archive_path, "r")
        zip_ref.extractall(path=data_dir)
        zip_ref.close()
    
    
    def prepare_tiny_imagenet_200(dataset_dir: Path):
        # Format validation set the same way as train set is formatted.
        val_data_dir = dataset_dir / "val"
        val_annotations_file = val_data_dir / "val_annotations.txt"
        with open(val_annotations_file, "r") as f:
            val_annotation_data = map(lambda line: line.split("\t")[:2], f.readlines())
        val_images_dir = val_data_dir / "images"
        for image_filename, image_label in val_annotation_data:
            from_image_filepath = val_images_dir / image_filename
            to_image_dir = val_data_dir / image_label
            if not to_image_dir.exists():
                to_image_dir.mkdir()
            to_image_filepath = to_image_dir / image_filename
            from_image_filepath.rename(to_image_filepath)
        val_annotations_file.unlink()
        val_images_dir.rmdir()
    
    
    DATASET_DIR = DATA_DIR / "tiny-imagenet-200"
    if not DATASET_DIR.exists():
        download_tiny_imagenet_200(DATA_DIR)
        prepare_tiny_imagenet_200(DATASET_DIR)
        print(f"Successfully downloaded and prepared dataset at: {DATASET_DIR}")
    
    BASE_MODEL_NAME = "resnet18"
    image_size = 64
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    
    # Paths where PyTorch and OpenVINO IR models will be stored.
    fp32_pth_path = Path(MODEL_DIR / (BASE_MODEL_NAME + "_fp32")).with_suffix(".pth")
    fp32_ir_path = fp32_pth_path.with_suffix(".xml")
    int8_sparse_ir_path = Path(MODEL_DIR / (BASE_MODEL_NAME + "_int8_sparse")).with_suffix(".xml")



.. parsed-literal::

    data/tiny-imagenet-200.zip:   0%|          | 0.00/237M [00:00<?, ?B/s]


.. parsed-literal::

    Successfully downloaded and prepared dataset at: data/tiny-imagenet-200


Train Function
~~~~~~~~~~~~~~

`back to top ⬆️ <#Table-of-contents:>`__

.. code:: ipython3

    def train(train_loader, model, compression_ctrl, criterion, optimizer, epoch):
        batch_time = AverageMeter("Time", ":3.3f")
        losses = AverageMeter("Loss", ":2.3f")
        top1 = AverageMeter("Acc@1", ":2.2f")
        top5 = AverageMeter("Acc@5", ":2.2f")
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, losses, top1, top5],
            prefix="Epoch:[{}]".format(epoch),
        )
    
        # Switch to train mode.
        model.train()
    
        end = time.time()
        for i, (images, target) in enumerate(train_loader):
            images = images.to(device)
            target = target.to(device)
    
            # Compute output.
            output = model(images)
            loss = criterion(output, target)
    
            # Measure accuracy and record loss.
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
    
            # Compute gradient and do opt step.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            # Measure elapsed time.
            batch_time.update(time.time() - end)
            end = time.time()
    
            print_frequency = 50
            if i % print_frequency == 0:
                progress.display(i)
            compression_ctrl.scheduler.step()

Validate Function
~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#Table-of-contents:>`__

.. code:: ipython3

    def validate(val_loader, model, criterion):
        batch_time = AverageMeter("Time", ":3.3f")
        losses = AverageMeter("Loss", ":2.3f")
        top1 = AverageMeter("Acc@1", ":2.2f")
        top5 = AverageMeter("Acc@5", ":2.2f")
        progress = ProgressMeter(len(val_loader), [batch_time, losses, top1, top5], prefix="Test: ")
    
        # Switch to evaluate mode.
        model.eval()
    
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(val_loader):
                images = images.to(device)
                target = target.to(device)
    
                # Compute output.
                output = model(images)
                loss = criterion(output, target)
    
                # Measure accuracy and record loss.
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))
    
                # Measure elapsed time.
                batch_time.update(time.time() - end)
                end = time.time()
    
                print_frequency = 10
                if i % print_frequency == 0:
                    progress.display(i)
    
            print(" * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}".format(top1=top1, top5=top5))
        return top1.avg

Helpers
~~~~~~~

`back to top ⬆️ <#Table-of-contents:>`__

.. code:: ipython3

    class AverageMeter(object):
        """Computes and stores the average and current value"""
    
        def __init__(self, name, fmt=":f"):
            self.name = name
            self.fmt = fmt
            self.reset()
    
        def reset(self):
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0
    
        def update(self, val, n=1):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count
    
        def __str__(self):
            fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
            return fmtstr.format(**self.__dict__)
    
    
    class ProgressMeter(object):
        def __init__(self, num_batches, meters, prefix=""):
            self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
            self.meters = meters
            self.prefix = prefix
    
        def display(self, batch):
            entries = [self.prefix + self.batch_fmtstr.format(batch)]
            entries += [str(meter) for meter in self.meters]
            print("\t".join(entries))
    
        def _get_batch_fmtstr(self, num_batches):
            num_digits = len(str(num_batches // 1))
            fmt = "{:" + str(num_digits) + "d}"
            return "[" + fmt + "/" + fmt.format(num_batches) + "]"
    
    
    def accuracy(output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)
    
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
    
            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

Get a Pre-trained FP32 Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#Table-of-contents:>`__

А pre-trained floating-point model is a prerequisite for quantization.
It can be obtained by tuning from scratch with the code below.

.. code:: ipython3

    num_classes = 1000
    init_lr = 1e-4
    batch_size = 128
    epochs = 20
    
    # model = models.resnet50(pretrained=True)
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(in_features=512, out_features=200, bias=True)
    model.to(device)
    
    
    # Data loading code.
    train_dir = DATASET_DIR / "train"
    val_dir = DATASET_DIR / "val"
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    train_dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose(
            [
                transforms.Resize([image_size, image_size]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    val_dataset = datasets.ImageFolder(
        val_dir,
        transforms.Compose(
            [
                transforms.Resize([256, 256]),
                transforms.CenterCrop([image_size, image_size]),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        sampler=None,
    )
    
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
    
    # Define loss function (criterion) and optimizer.
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-717/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
      warnings.warn(
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-717/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=ResNet18_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet18_Weights.DEFAULT` to get the most up-to-date weights.
      warnings.warn(msg)


Export the ``FP32`` model to OpenVINO™ Intermediate Representation, to
benchmark it in comparison with the ``INT8`` model.

.. code:: ipython3

    dummy_input = torch.randn(1, 3, image_size, image_size).to(device)
    
    ov_model = ov.convert_model(model, example_input=dummy_input, input=[1, 3, image_size, image_size])
    ov.save_model(ov_model, fp32_ir_path, compress_to_fp16=False)
    print(f"FP32 model was exported to {fp32_ir_path}.")


.. parsed-literal::

    FP32 model was exported to model/resnet18_fp32.xml.


Create and Initialize Quantization and Sparsity Training
--------------------------------------------------------

`back to top ⬆️ <#Table-of-contents:>`__

NNCF enables compression-aware training by integrating into regular
training pipelines. The framework is designed so that modifications to
your original training code are minor.

.. code:: ipython3

    from nncf import NNCFConfig
    from nncf.torch import create_compressed_model, register_default_init_args
    
    # load
    nncf_config = NNCFConfig.from_json("config.json")
    nncf_config = register_default_init_args(nncf_config, train_loader)
    
    # Creating a compressed model
    compression_ctrl, compressed_model = create_compressed_model(model, nncf_config)
    compression_ctrl.scheduler.epoch_step()


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, tensorflow, onnx, openvino
    INFO:nncf:Ignored adding weight sparsifier for operation: ResNet/NNCFConv2d[conv1]/conv2d_0
    INFO:nncf:Collecting tensor statistics |█               | 8 / 79
    INFO:nncf:Collecting tensor statistics |███             | 16 / 79
    INFO:nncf:Collecting tensor statistics |████            | 24 / 79
    INFO:nncf:Collecting tensor statistics |██████          | 32 / 79
    INFO:nncf:Collecting tensor statistics |████████        | 40 / 79
    INFO:nncf:Collecting tensor statistics |█████████       | 48 / 79
    INFO:nncf:Collecting tensor statistics |███████████     | 56 / 79
    INFO:nncf:Collecting tensor statistics |████████████    | 64 / 79
    INFO:nncf:Collecting tensor statistics |██████████████  | 72 / 79
    INFO:nncf:Collecting tensor statistics |████████████████| 79 / 79
    INFO:nncf:Compiling and loading torch extension: quantized_functions_cpu...
    INFO:nncf:Finished loading torch extension: quantized_functions_cpu


.. parsed-literal::

    2024-07-02 01:41:27.244480: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-07-02 01:41:27.276044: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2024-07-02 01:41:27.882264: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


.. parsed-literal::

    INFO:nncf:BatchNorm statistics adaptation |█               | 1 / 16
    INFO:nncf:BatchNorm statistics adaptation |██              | 2 / 16
    INFO:nncf:BatchNorm statistics adaptation |███             | 3 / 16
    INFO:nncf:BatchNorm statistics adaptation |████            | 4 / 16
    INFO:nncf:BatchNorm statistics adaptation |█████           | 5 / 16
    INFO:nncf:BatchNorm statistics adaptation |██████          | 6 / 16
    INFO:nncf:BatchNorm statistics adaptation |███████         | 7 / 16
    INFO:nncf:BatchNorm statistics adaptation |████████        | 8 / 16
    INFO:nncf:BatchNorm statistics adaptation |█████████       | 9 / 16
    INFO:nncf:BatchNorm statistics adaptation |██████████      | 10 / 16
    INFO:nncf:BatchNorm statistics adaptation |███████████     | 11 / 16
    INFO:nncf:BatchNorm statistics adaptation |████████████    | 12 / 16
    INFO:nncf:BatchNorm statistics adaptation |█████████████   | 13 / 16
    INFO:nncf:BatchNorm statistics adaptation |██████████████  | 14 / 16
    INFO:nncf:BatchNorm statistics adaptation |███████████████ | 15 / 16
    INFO:nncf:BatchNorm statistics adaptation |████████████████| 16 / 16


Validate Compressed Model

Evaluate the new model on the validation set after initialization of
quantization and sparsity.

.. code:: ipython3

    acc1 = validate(val_loader, compressed_model, criterion)
    print(f"Accuracy of initialized sparse INT8 model: {acc1:.3f}")


.. parsed-literal::

    Test: [ 0/79]	Time 0.355 (0.355)	Loss 6.069 (6.069)	Acc@1 0.00 (0.00)	Acc@5 4.69 (4.69)
    Test: [10/79]	Time 0.147 (0.163)	Loss 5.368 (5.689)	Acc@1 0.78 (0.07)	Acc@5 3.91 (2.41)
    Test: [20/79]	Time 0.155 (0.157)	Loss 5.921 (5.653)	Acc@1 0.00 (0.56)	Acc@5 2.34 (3.16)
    Test: [30/79]	Time 0.150 (0.155)	Loss 5.664 (5.670)	Acc@1 0.00 (0.50)	Acc@5 0.78 (2.90)
    Test: [40/79]	Time 0.142 (0.155)	Loss 5.608 (5.632)	Acc@1 1.56 (0.59)	Acc@5 3.12 (3.09)
    Test: [50/79]	Time 0.144 (0.154)	Loss 5.170 (5.618)	Acc@1 0.00 (0.72)	Acc@5 2.34 (3.32)
    Test: [60/79]	Time 0.145 (0.154)	Loss 6.619 (5.634)	Acc@1 0.00 (0.67)	Acc@5 0.00 (3.00)
    Test: [70/79]	Time 0.158 (0.153)	Loss 5.771 (5.653)	Acc@1 0.00 (0.57)	Acc@5 1.56 (2.77)
     * Acc@1 0.570 Acc@5 2.770
    Accuracy of initialized sparse INT8 model: 0.570


Fine-tune the Compressed Model
------------------------------

`back to top ⬆️ <#Table-of-contents:>`__

At this step, a regular fine-tuning process is applied to further
improve quantized model accuracy. Normally, several epochs of tuning are
required with a small learning rate, the same that is usually used at
the end of the training of the original model. No other changes in the
training pipeline are required. Here is a simple example.

.. code:: ipython3

    compression_lr = init_lr / 10
    optimizer = torch.optim.Adam(compressed_model.parameters(), lr=compression_lr)
    nr_epochs = 10
    # Train for one epoch with NNCF.
    print("Training")
    for epoch in range(nr_epochs):
        compression_ctrl.scheduler.epoch_step()
        train(train_loader, compressed_model, compression_ctrl, criterion, optimizer, epoch=epoch)
    
    # Evaluate on validation set after Quantization-Aware Training (QAT case).
    print("Validating")
    acc1_int8_sparse = validate(val_loader, compressed_model, criterion)
    
    print(f"Accuracy of tuned INT8 sparse model: {acc1_int8_sparse:.3f}")
    print(f"Accuracy drop of tuned INT8 sparse model over pre-trained FP32 model: {acc1 - acc1_int8_sparse:.3f}")


.. parsed-literal::

    Training
    Epoch:[0][  0/782]	Time 0.555 (0.555)	Loss 5.673 (5.673)	Acc@1 0.78 (0.78)	Acc@5 3.12 (3.12)
    Epoch:[0][ 50/782]	Time 0.336 (0.342)	Loss 5.643 (5.644)	Acc@1 0.00 (0.78)	Acc@5 2.34 (3.12)
    Epoch:[0][100/782]	Time 0.335 (0.340)	Loss 5.565 (5.604)	Acc@1 0.78 (0.80)	Acc@5 2.34 (3.23)
    Epoch:[0][150/782]	Time 0.340 (0.340)	Loss 5.540 (5.559)	Acc@1 0.78 (0.90)	Acc@5 3.91 (3.53)
    Epoch:[0][200/782]	Time 0.338 (0.340)	Loss 5.273 (5.515)	Acc@1 2.34 (1.07)	Acc@5 7.81 (3.98)
    Epoch:[0][250/782]	Time 0.354 (0.340)	Loss 5.358 (5.473)	Acc@1 1.56 (1.24)	Acc@5 6.25 (4.52)
    Epoch:[0][300/782]	Time 0.340 (0.340)	Loss 5.226 (5.431)	Acc@1 1.56 (1.45)	Acc@5 7.03 (5.10)
    Epoch:[0][350/782]	Time 0.349 (0.340)	Loss 5.104 (5.388)	Acc@1 1.56 (1.67)	Acc@5 10.16 (5.81)
    Epoch:[0][400/782]	Time 0.341 (0.341)	Loss 5.052 (5.351)	Acc@1 0.78 (1.84)	Acc@5 12.50 (6.42)
    Epoch:[0][450/782]	Time 0.365 (0.341)	Loss 5.049 (5.312)	Acc@1 3.91 (2.11)	Acc@5 10.94 (7.15)
    Epoch:[0][500/782]	Time 0.334 (0.341)	Loss 4.855 (5.275)	Acc@1 5.47 (2.38)	Acc@5 13.28 (7.91)
    Epoch:[0][550/782]	Time 0.345 (0.341)	Loss 4.707 (5.237)	Acc@1 10.16 (2.74)	Acc@5 24.22 (8.75)
    Epoch:[0][600/782]	Time 0.339 (0.342)	Loss 4.622 (5.197)	Acc@1 7.81 (3.14)	Acc@5 25.00 (9.72)
    Epoch:[0][650/782]	Time 0.340 (0.342)	Loss 4.615 (5.160)	Acc@1 10.16 (3.55)	Acc@5 22.66 (10.64)
    Epoch:[0][700/782]	Time 0.340 (0.342)	Loss 4.655 (5.122)	Acc@1 7.03 (3.99)	Acc@5 22.66 (11.62)
    Epoch:[0][750/782]	Time 0.336 (0.342)	Loss 4.461 (5.084)	Acc@1 15.62 (4.51)	Acc@5 34.38 (12.66)
    Epoch:[1][  0/782]	Time 0.662 (0.662)	Loss 4.331 (4.331)	Acc@1 15.62 (15.62)	Acc@5 35.16 (35.16)
    Epoch:[1][ 50/782]	Time 0.338 (0.351)	Loss 4.327 (4.228)	Acc@1 14.06 (16.68)	Acc@5 32.03 (37.44)
    Epoch:[1][100/782]	Time 0.334 (0.346)	Loss 4.208 (4.187)	Acc@1 17.97 (18.04)	Acc@5 35.94 (38.38)
    Epoch:[1][150/782]	Time 0.344 (0.345)	Loss 4.060 (4.166)	Acc@1 17.97 (18.56)	Acc@5 42.97 (38.90)
    Epoch:[1][200/782]	Time 0.349 (0.346)	Loss 4.100 (4.142)	Acc@1 17.97 (18.94)	Acc@5 41.41 (39.69)
    Epoch:[1][250/782]	Time 0.334 (0.346)	Loss 4.081 (4.119)	Acc@1 21.88 (19.23)	Acc@5 43.75 (40.24)
    Epoch:[1][300/782]	Time 0.337 (0.345)	Loss 4.199 (4.099)	Acc@1 15.62 (19.49)	Acc@5 37.50 (40.77)
    Epoch:[1][350/782]	Time 0.347 (0.345)	Loss 3.830 (4.077)	Acc@1 25.78 (19.82)	Acc@5 45.31 (41.33)
    Epoch:[1][400/782]	Time 0.351 (0.345)	Loss 4.089 (4.054)	Acc@1 21.09 (20.27)	Acc@5 39.06 (41.95)
    Epoch:[1][450/782]	Time 0.346 (0.345)	Loss 3.782 (4.034)	Acc@1 26.56 (20.62)	Acc@5 44.53 (42.39)
    Epoch:[1][500/782]	Time 0.350 (0.345)	Loss 3.816 (4.012)	Acc@1 26.56 (21.00)	Acc@5 50.78 (43.00)
    Epoch:[1][550/782]	Time 0.341 (0.345)	Loss 3.620 (3.989)	Acc@1 26.56 (21.37)	Acc@5 52.34 (43.58)
    Epoch:[1][600/782]	Time 0.349 (0.345)	Loss 3.694 (3.971)	Acc@1 28.91 (21.63)	Acc@5 47.66 (44.06)
    Epoch:[1][650/782]	Time 0.355 (0.345)	Loss 3.738 (3.952)	Acc@1 22.66 (21.86)	Acc@5 45.31 (44.52)
    Epoch:[1][700/782]	Time 0.341 (0.346)	Loss 3.735 (3.936)	Acc@1 25.00 (22.09)	Acc@5 44.53 (44.90)
    Epoch:[1][750/782]	Time 0.353 (0.346)	Loss 3.630 (3.918)	Acc@1 29.69 (22.32)	Acc@5 53.12 (45.32)
    Epoch:[2][  0/782]	Time 0.684 (0.684)	Loss 3.419 (3.419)	Acc@1 32.03 (32.03)	Acc@5 57.81 (57.81)
    Epoch:[2][ 50/782]	Time 0.349 (0.356)	Loss 3.397 (3.466)	Acc@1 32.03 (29.34)	Acc@5 56.25 (54.96)
    Epoch:[2][100/782]	Time 0.343 (0.353)	Loss 3.293 (3.432)	Acc@1 33.59 (30.02)	Acc@5 59.38 (56.53)
    Epoch:[2][150/782]	Time 0.345 (0.352)	Loss 3.358 (3.422)	Acc@1 33.59 (30.30)	Acc@5 59.38 (56.64)
    Epoch:[2][200/782]	Time 0.354 (0.352)	Loss 3.215 (3.410)	Acc@1 34.38 (30.50)	Acc@5 63.28 (56.97)
    Epoch:[2][250/782]	Time 0.348 (0.351)	Loss 3.369 (3.392)	Acc@1 32.81 (30.82)	Acc@5 57.81 (57.15)
    Epoch:[2][300/782]	Time 0.345 (0.351)	Loss 3.487 (3.379)	Acc@1 25.78 (30.96)	Acc@5 51.56 (57.35)
    Epoch:[2][350/782]	Time 0.347 (0.352)	Loss 3.336 (3.370)	Acc@1 34.38 (31.04)	Acc@5 60.94 (57.51)
    Epoch:[2][400/782]	Time 0.342 (0.352)	Loss 3.434 (3.359)	Acc@1 25.78 (31.16)	Acc@5 59.38 (57.66)
    Epoch:[2][450/782]	Time 0.359 (0.352)	Loss 3.440 (3.348)	Acc@1 28.12 (31.42)	Acc@5 57.81 (57.85)
    Epoch:[2][500/782]	Time 0.346 (0.352)	Loss 3.129 (3.336)	Acc@1 35.16 (31.59)	Acc@5 66.41 (58.09)
    Epoch:[2][550/782]	Time 0.339 (0.352)	Loss 3.388 (3.322)	Acc@1 26.56 (31.77)	Acc@5 52.34 (58.40)
    Epoch:[2][600/782]	Time 0.348 (0.351)	Loss 3.078 (3.311)	Acc@1 36.72 (31.89)	Acc@5 63.28 (58.57)
    Epoch:[2][650/782]	Time 0.351 (0.351)	Loss 3.172 (3.300)	Acc@1 36.72 (32.08)	Acc@5 64.84 (58.76)
    Epoch:[2][700/782]	Time 0.344 (0.351)	Loss 3.152 (3.287)	Acc@1 32.03 (32.23)	Acc@5 58.59 (58.98)
    Epoch:[2][750/782]	Time 0.355 (0.351)	Loss 3.228 (3.275)	Acc@1 36.72 (32.45)	Acc@5 56.25 (59.21)
    Epoch:[3][  0/782]	Time 0.669 (0.669)	Loss 3.060 (3.060)	Acc@1 32.03 (32.03)	Acc@5 66.41 (66.41)
    Epoch:[3][ 50/782]	Time 0.348 (0.355)	Loss 2.926 (2.958)	Acc@1 44.53 (37.94)	Acc@5 62.50 (65.10)
    Epoch:[3][100/782]	Time 0.349 (0.351)	Loss 3.022 (2.938)	Acc@1 34.38 (38.18)	Acc@5 61.72 (65.66)
    Epoch:[3][150/782]	Time 0.352 (0.350)	Loss 2.760 (2.934)	Acc@1 40.62 (38.10)	Acc@5 69.53 (65.46)
    Epoch:[3][200/782]	Time 0.351 (0.351)	Loss 3.039 (2.928)	Acc@1 34.38 (38.21)	Acc@5 60.94 (65.38)
    Epoch:[3][250/782]	Time 0.355 (0.351)	Loss 2.829 (2.924)	Acc@1 33.59 (38.16)	Acc@5 67.19 (65.41)
    Epoch:[3][300/782]	Time 0.344 (0.351)	Loss 2.895 (2.919)	Acc@1 43.75 (38.16)	Acc@5 72.66 (65.39)
    Epoch:[3][350/782]	Time 0.348 (0.351)	Loss 2.767 (2.914)	Acc@1 41.41 (38.23)	Acc@5 68.75 (65.42)
    Epoch:[3][400/782]	Time 0.345 (0.350)	Loss 3.116 (2.908)	Acc@1 30.47 (38.20)	Acc@5 60.16 (65.48)
    Epoch:[3][450/782]	Time 0.344 (0.349)	Loss 2.914 (2.903)	Acc@1 35.94 (38.30)	Acc@5 62.50 (65.54)
    Epoch:[3][500/782]	Time 0.345 (0.349)	Loss 2.719 (2.895)	Acc@1 44.53 (38.36)	Acc@5 67.97 (65.71)
    Epoch:[3][550/782]	Time 0.353 (0.349)	Loss 3.138 (2.889)	Acc@1 32.81 (38.40)	Acc@5 60.16 (65.79)
    Epoch:[3][600/782]	Time 0.354 (0.349)	Loss 3.042 (2.884)	Acc@1 32.03 (38.43)	Acc@5 58.59 (65.82)
    Epoch:[3][650/782]	Time 0.345 (0.349)	Loss 2.931 (2.877)	Acc@1 42.19 (38.54)	Acc@5 67.19 (65.96)
    Epoch:[3][700/782]	Time 0.343 (0.348)	Loss 2.968 (2.870)	Acc@1 32.81 (38.57)	Acc@5 61.72 (66.06)
    Epoch:[3][750/782]	Time 0.341 (0.348)	Loss 2.799 (2.864)	Acc@1 37.50 (38.71)	Acc@5 65.62 (66.12)
    Epoch:[4][  0/782]	Time 0.714 (0.714)	Loss 2.625 (2.625)	Acc@1 46.09 (46.09)	Acc@5 68.75 (68.75)
    Epoch:[4][ 50/782]	Time 0.340 (0.367)	Loss 2.682 (2.727)	Acc@1 46.09 (40.18)	Acc@5 67.97 (67.98)
    Epoch:[4][100/782]	Time 0.352 (0.355)	Loss 2.824 (2.699)	Acc@1 33.59 (41.11)	Acc@5 64.84 (68.60)
    Epoch:[4][150/782]	Time 0.345 (0.352)	Loss 2.703 (2.690)	Acc@1 46.09 (41.44)	Acc@5 64.84 (68.91)
    Epoch:[4][200/782]	Time 0.345 (0.351)	Loss 2.523 (2.683)	Acc@1 46.88 (41.64)	Acc@5 74.22 (69.03)
    Epoch:[4][250/782]	Time 0.351 (0.350)	Loss 2.381 (2.677)	Acc@1 49.22 (41.80)	Acc@5 74.22 (69.10)
    Epoch:[4][300/782]	Time 0.356 (0.350)	Loss 2.633 (2.674)	Acc@1 42.19 (41.82)	Acc@5 65.62 (68.98)
    Epoch:[4][350/782]	Time 0.344 (0.350)	Loss 2.621 (2.671)	Acc@1 46.09 (41.86)	Acc@5 71.88 (69.01)
    Epoch:[4][400/782]	Time 0.347 (0.349)	Loss 2.472 (2.662)	Acc@1 42.97 (42.02)	Acc@5 75.00 (69.15)
    Epoch:[4][450/782]	Time 0.355 (0.349)	Loss 2.529 (2.659)	Acc@1 42.19 (42.03)	Acc@5 75.78 (69.18)
    Epoch:[4][500/782]	Time 0.355 (0.349)	Loss 2.793 (2.654)	Acc@1 37.50 (42.12)	Acc@5 64.84 (69.27)
    Epoch:[4][550/782]	Time 0.353 (0.349)	Loss 2.474 (2.646)	Acc@1 45.31 (42.31)	Acc@5 67.97 (69.32)
    Epoch:[4][600/782]	Time 0.345 (0.349)	Loss 2.383 (2.642)	Acc@1 51.56 (42.36)	Acc@5 73.44 (69.34)
    Epoch:[4][650/782]	Time 0.335 (0.349)	Loss 2.595 (2.638)	Acc@1 43.75 (42.41)	Acc@5 71.88 (69.35)
    Epoch:[4][700/782]	Time 0.342 (0.349)	Loss 2.541 (2.634)	Acc@1 39.84 (42.44)	Acc@5 74.22 (69.37)
    Epoch:[4][750/782]	Time 0.348 (0.349)	Loss 2.408 (2.628)	Acc@1 45.31 (42.52)	Acc@5 75.00 (69.51)
    Epoch:[5][  0/782]	Time 0.690 (0.690)	Loss 2.310 (2.310)	Acc@1 48.44 (48.44)	Acc@5 75.00 (75.00)
    Epoch:[5][ 50/782]	Time 0.353 (0.358)	Loss 2.585 (2.521)	Acc@1 42.97 (43.66)	Acc@5 68.75 (71.32)
    Epoch:[5][100/782]	Time 0.352 (0.354)	Loss 2.263 (2.491)	Acc@1 48.44 (44.46)	Acc@5 74.22 (71.88)
    Epoch:[5][150/782]	Time 0.354 (0.353)	Loss 2.296 (2.480)	Acc@1 52.34 (44.62)	Acc@5 75.00 (71.90)
    Epoch:[5][200/782]	Time 0.349 (0.352)	Loss 2.430 (2.479)	Acc@1 48.44 (44.75)	Acc@5 70.31 (71.79)
    Epoch:[5][250/782]	Time 0.347 (0.351)	Loss 2.566 (2.482)	Acc@1 40.62 (44.74)	Acc@5 69.53 (71.70)
    Epoch:[5][300/782]	Time 0.343 (0.351)	Loss 2.414 (2.476)	Acc@1 40.62 (44.86)	Acc@5 78.12 (71.78)
    Epoch:[5][350/782]	Time 0.352 (0.351)	Loss 2.301 (2.477)	Acc@1 50.78 (44.74)	Acc@5 75.78 (71.62)
    Epoch:[5][400/782]	Time 0.354 (0.350)	Loss 2.414 (2.472)	Acc@1 44.53 (44.87)	Acc@5 72.66 (71.71)
    Epoch:[5][450/782]	Time 0.348 (0.351)	Loss 2.352 (2.466)	Acc@1 50.78 (44.94)	Acc@5 72.66 (71.85)
    Epoch:[5][500/782]	Time 0.346 (0.351)	Loss 2.423 (2.464)	Acc@1 47.66 (44.97)	Acc@5 74.22 (71.84)
    Epoch:[5][550/782]	Time 0.353 (0.350)	Loss 2.407 (2.459)	Acc@1 40.62 (45.03)	Acc@5 71.88 (71.88)
    Epoch:[5][600/782]	Time 0.349 (0.350)	Loss 2.326 (2.457)	Acc@1 48.44 (45.05)	Acc@5 77.34 (71.91)
    Epoch:[5][650/782]	Time 0.349 (0.350)	Loss 2.283 (2.452)	Acc@1 47.66 (45.13)	Acc@5 71.88 (72.01)
    Epoch:[5][700/782]	Time 0.348 (0.351)	Loss 2.217 (2.446)	Acc@1 46.88 (45.21)	Acc@5 72.66 (72.09)
    Epoch:[5][750/782]	Time 0.352 (0.351)	Loss 2.474 (2.442)	Acc@1 50.78 (45.29)	Acc@5 65.62 (72.12)
    Epoch:[6][  0/782]	Time 0.687 (0.687)	Loss 2.568 (2.568)	Acc@1 44.53 (44.53)	Acc@5 64.06 (64.06)
    Epoch:[6][ 50/782]	Time 0.336 (0.357)	Loss 2.411 (2.321)	Acc@1 45.31 (47.50)	Acc@5 68.75 (74.17)
    Epoch:[6][100/782]	Time 0.344 (0.352)	Loss 2.401 (2.333)	Acc@1 48.44 (47.05)	Acc@5 72.66 (73.89)
    Epoch:[6][150/782]	Time 0.345 (0.352)	Loss 2.220 (2.331)	Acc@1 46.88 (47.11)	Acc@5 75.78 (73.85)
    Epoch:[6][200/782]	Time 0.351 (0.352)	Loss 2.330 (2.329)	Acc@1 49.22 (47.21)	Acc@5 73.44 (73.77)
    Epoch:[6][250/782]	Time 0.343 (0.352)	Loss 2.581 (2.330)	Acc@1 43.75 (47.22)	Acc@5 67.97 (73.84)
    Epoch:[6][300/782]	Time 0.344 (0.351)	Loss 2.457 (2.321)	Acc@1 42.97 (47.57)	Acc@5 73.44 (74.00)
    Epoch:[6][350/782]	Time 0.336 (0.350)	Loss 2.332 (2.321)	Acc@1 50.78 (47.49)	Acc@5 73.44 (73.98)
    Epoch:[6][400/782]	Time 0.353 (0.350)	Loss 2.057 (2.317)	Acc@1 53.91 (47.56)	Acc@5 80.47 (74.01)
    Epoch:[6][450/782]	Time 0.343 (0.350)	Loss 2.379 (2.316)	Acc@1 45.31 (47.41)	Acc@5 71.09 (74.02)
    Epoch:[6][500/782]	Time 0.348 (0.350)	Loss 2.337 (2.313)	Acc@1 48.44 (47.44)	Acc@5 71.09 (74.10)
    Epoch:[6][550/782]	Time 0.345 (0.349)	Loss 2.207 (2.309)	Acc@1 46.88 (47.54)	Acc@5 74.22 (74.18)
    Epoch:[6][600/782]	Time 0.353 (0.349)	Loss 2.191 (2.305)	Acc@1 57.03 (47.63)	Acc@5 77.34 (74.22)
    Epoch:[6][650/782]	Time 0.348 (0.349)	Loss 2.120 (2.303)	Acc@1 53.12 (47.62)	Acc@5 77.34 (74.23)
    Epoch:[6][700/782]	Time 0.339 (0.349)	Loss 2.312 (2.298)	Acc@1 39.84 (47.71)	Acc@5 71.88 (74.30)
    Epoch:[6][750/782]	Time 0.350 (0.349)	Loss 2.080 (2.295)	Acc@1 53.12 (47.77)	Acc@5 79.69 (74.34)
    Epoch:[7][  0/782]	Time 0.693 (0.693)	Loss 2.192 (2.192)	Acc@1 44.53 (44.53)	Acc@5 78.12 (78.12)
    Epoch:[7][ 50/782]	Time 0.341 (0.360)	Loss 2.139 (2.214)	Acc@1 50.78 (48.56)	Acc@5 76.56 (75.32)
    Epoch:[7][100/782]	Time 0.337 (0.354)	Loss 2.266 (2.213)	Acc@1 57.03 (49.16)	Acc@5 71.88 (75.45)
    Epoch:[7][150/782]	Time 0.352 (0.353)	Loss 1.987 (2.209)	Acc@1 54.69 (49.10)	Acc@5 82.03 (75.53)
    Epoch:[7][200/782]	Time 0.352 (0.352)	Loss 2.232 (2.203)	Acc@1 43.75 (49.37)	Acc@5 75.00 (75.62)
    Epoch:[7][250/782]	Time 0.341 (0.354)	Loss 2.216 (2.203)	Acc@1 48.44 (49.27)	Acc@5 78.91 (75.66)
    Epoch:[7][300/782]	Time 0.346 (0.353)	Loss 2.393 (2.202)	Acc@1 49.22 (49.30)	Acc@5 71.09 (75.70)
    Epoch:[7][350/782]	Time 0.347 (0.353)	Loss 2.084 (2.196)	Acc@1 44.53 (49.47)	Acc@5 80.47 (75.84)
    Epoch:[7][400/782]	Time 0.363 (0.353)	Loss 1.682 (2.194)	Acc@1 65.62 (49.55)	Acc@5 83.59 (75.82)
    Epoch:[7][450/782]	Time 0.348 (0.352)	Loss 2.193 (2.194)	Acc@1 47.66 (49.62)	Acc@5 75.78 (75.82)
    Epoch:[7][500/782]	Time 0.350 (0.351)	Loss 2.166 (2.192)	Acc@1 45.31 (49.59)	Acc@5 78.12 (75.81)
    Epoch:[7][550/782]	Time 0.352 (0.351)	Loss 2.126 (2.187)	Acc@1 47.66 (49.70)	Acc@5 78.91 (75.84)
    Epoch:[7][600/782]	Time 0.352 (0.350)	Loss 2.222 (2.184)	Acc@1 49.22 (49.73)	Acc@5 73.44 (75.87)
    Epoch:[7][650/782]	Time 0.348 (0.350)	Loss 2.075 (2.181)	Acc@1 50.00 (49.79)	Acc@5 78.12 (75.89)
    Epoch:[7][700/782]	Time 0.335 (0.350)	Loss 2.181 (2.179)	Acc@1 47.66 (49.81)	Acc@5 75.78 (75.89)
    Epoch:[7][750/782]	Time 0.346 (0.349)	Loss 2.071 (2.177)	Acc@1 53.12 (49.82)	Acc@5 75.78 (75.89)
    Epoch:[8][  0/782]	Time 0.696 (0.696)	Loss 1.829 (1.829)	Acc@1 58.59 (58.59)	Acc@5 82.03 (82.03)
    Epoch:[8][ 50/782]	Time 0.348 (0.357)	Loss 2.171 (2.096)	Acc@1 50.78 (51.04)	Acc@5 78.91 (77.51)
    Epoch:[8][100/782]	Time 0.357 (0.353)	Loss 2.207 (2.089)	Acc@1 52.34 (51.26)	Acc@5 74.22 (77.56)
    Epoch:[8][150/782]	Time 0.353 (0.352)	Loss 2.289 (2.100)	Acc@1 49.22 (51.13)	Acc@5 73.44 (77.32)
    Epoch:[8][200/782]	Time 0.357 (0.352)	Loss 2.175 (2.101)	Acc@1 46.88 (51.00)	Acc@5 77.34 (77.29)
    Epoch:[8][250/782]	Time 0.350 (0.351)	Loss 2.239 (2.092)	Acc@1 47.66 (51.30)	Acc@5 71.88 (77.35)
    Epoch:[8][300/782]	Time 0.345 (0.350)	Loss 2.070 (2.087)	Acc@1 49.22 (51.40)	Acc@5 75.78 (77.41)
    Epoch:[8][350/782]	Time 0.353 (0.350)	Loss 1.868 (2.083)	Acc@1 52.34 (51.38)	Acc@5 82.81 (77.39)
    Epoch:[8][400/782]	Time 0.347 (0.350)	Loss 2.345 (2.084)	Acc@1 40.62 (51.47)	Acc@5 71.88 (77.34)
    Epoch:[8][450/782]	Time 0.371 (0.350)	Loss 1.731 (2.085)	Acc@1 63.28 (51.43)	Acc@5 82.81 (77.32)
    Epoch:[8][500/782]	Time 0.344 (0.349)	Loss 2.142 (2.082)	Acc@1 46.09 (51.40)	Acc@5 77.34 (77.35)
    Epoch:[8][550/782]	Time 0.346 (0.349)	Loss 2.173 (2.080)	Acc@1 53.91 (51.45)	Acc@5 73.44 (77.40)
    Epoch:[8][600/782]	Time 0.348 (0.349)	Loss 2.184 (2.077)	Acc@1 54.69 (51.55)	Acc@5 73.44 (77.43)
    Epoch:[8][650/782]	Time 0.353 (0.349)	Loss 2.118 (2.075)	Acc@1 49.22 (51.60)	Acc@5 76.56 (77.43)
    Epoch:[8][700/782]	Time 0.352 (0.349)	Loss 2.254 (2.074)	Acc@1 51.56 (51.61)	Acc@5 72.66 (77.37)
    Epoch:[8][750/782]	Time 0.347 (0.349)	Loss 2.056 (2.071)	Acc@1 53.91 (51.67)	Acc@5 75.78 (77.41)
    Epoch:[9][  0/782]	Time 0.683 (0.683)	Loss 1.824 (1.824)	Acc@1 59.38 (59.38)	Acc@5 85.16 (85.16)
    Epoch:[9][ 50/782]	Time 0.346 (0.353)	Loss 2.063 (1.996)	Acc@1 50.78 (53.09)	Acc@5 80.47 (78.65)
    Epoch:[9][100/782]	Time 0.350 (0.351)	Loss 1.874 (1.999)	Acc@1 58.59 (53.12)	Acc@5 82.03 (78.38)
    Epoch:[9][150/782]	Time 0.347 (0.350)	Loss 2.026 (1.994)	Acc@1 50.78 (53.17)	Acc@5 78.91 (78.80)
    Epoch:[9][200/782]	Time 0.350 (0.349)	Loss 1.877 (1.994)	Acc@1 59.38 (53.10)	Acc@5 82.81 (78.68)
    Epoch:[9][250/782]	Time 0.341 (0.349)	Loss 2.166 (1.996)	Acc@1 46.09 (53.00)	Acc@5 73.44 (78.60)
    Epoch:[9][300/782]	Time 0.345 (0.349)	Loss 2.125 (1.997)	Acc@1 51.56 (53.01)	Acc@5 76.56 (78.49)
    Epoch:[9][350/782]	Time 0.345 (0.349)	Loss 2.210 (1.995)	Acc@1 46.88 (52.89)	Acc@5 75.00 (78.60)
    Epoch:[9][400/782]	Time 0.350 (0.349)	Loss 1.897 (1.994)	Acc@1 57.81 (52.86)	Acc@5 79.69 (78.56)
    Epoch:[9][450/782]	Time 0.345 (0.349)	Loss 2.045 (1.989)	Acc@1 50.78 (53.00)	Acc@5 76.56 (78.62)
    Epoch:[9][500/782]	Time 0.336 (0.349)	Loss 2.300 (1.990)	Acc@1 46.88 (52.97)	Acc@5 72.66 (78.62)
    Epoch:[9][550/782]	Time 0.351 (0.349)	Loss 1.604 (1.990)	Acc@1 64.06 (53.02)	Acc@5 82.81 (78.61)
    Epoch:[9][600/782]	Time 0.352 (0.349)	Loss 1.763 (1.987)	Acc@1 54.69 (53.07)	Acc@5 85.16 (78.65)
    Epoch:[9][650/782]	Time 0.344 (0.349)	Loss 1.664 (1.984)	Acc@1 63.28 (53.11)	Acc@5 82.81 (78.71)
    Epoch:[9][700/782]	Time 0.347 (0.349)	Loss 2.284 (1.982)	Acc@1 42.97 (53.12)	Acc@5 78.12 (78.76)
    Epoch:[9][750/782]	Time 0.353 (0.349)	Loss 1.698 (1.983)	Acc@1 59.38 (53.11)	Acc@5 82.03 (78.72)
    Validating
    Test: [ 0/79]	Time 0.409 (0.409)	Loss 4.175 (4.175)	Acc@1 7.81 (7.81)	Acc@5 29.69 (29.69)
    Test: [10/79]	Time 0.148 (0.163)	Loss 5.955 (4.803)	Acc@1 3.12 (7.81)	Acc@5 7.03 (21.02)
    Test: [20/79]	Time 0.136 (0.151)	Loss 6.302 (5.109)	Acc@1 0.00 (5.21)	Acc@5 3.12 (17.22)
    Test: [30/79]	Time 0.173 (0.145)	Loss 5.520 (5.327)	Acc@1 1.56 (4.26)	Acc@5 16.41 (14.36)
    Test: [40/79]	Time 0.116 (0.142)	Loss 5.560 (5.399)	Acc@1 6.25 (4.12)	Acc@5 7.81 (13.34)
    Test: [50/79]	Time 0.150 (0.142)	Loss 4.887 (5.498)	Acc@1 7.81 (3.92)	Acc@5 21.88 (12.68)
    Test: [60/79]	Time 0.139 (0.142)	Loss 5.905 (5.512)	Acc@1 0.00 (3.98)	Acc@5 7.03 (12.58)
    Test: [70/79]	Time 0.139 (0.142)	Loss 4.785 (5.526)	Acc@1 2.34 (3.75)	Acc@5 11.72 (11.99)
     * Acc@1 5.320 Acc@5 15.300
    Accuracy of tuned INT8 sparse model: 5.320
    Accuracy drop of tuned INT8 sparse model over pre-trained FP32 model: -4.750


Export INT8 Sparse Model to OpenVINO IR
---------------------------------------

`back to top ⬆️ <#Table-of-contents:>`__

.. code:: ipython3

    warnings.filterwarnings("ignore", category=TracerWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    # Export INT8 model to OpenVINO™ IR
    ov_model = ov.convert_model(compressed_model, example_input=dummy_input, input=[1, 3, image_size, image_size])
    ov.save_model(ov_model, int8_sparse_ir_path)
    print(f"INT8 sparse model exported to {int8_sparse_ir_path}.")


.. parsed-literal::

    WARNING:tensorflow:Please fix your imports. Module tensorflow.python.training.tracking.base has been moved to tensorflow.python.trackable.base. The old module will be deleted in version 2.11.
    INT8 sparse model exported to model/resnet18_int8_sparse.xml.


Benchmark Model Performance by Computing Inference Time
-------------------------------------------------------

`back to top ⬆️ <#Table-of-contents:>`__

Finally, measure the inference performance of the ``FP32`` and ``INT8``
models, using `Benchmark
Tool <https://docs.openvino.ai/2024/learn-openvino/openvino-samples/benchmark-tool.html>`__
- inference performance measurement tool in OpenVINO. By default,
Benchmark Tool runs inference for 60 seconds in asynchronous mode on
CPU. It returns inference speed as latency (milliseconds per image) and
throughput (frames per second) values.

   **NOTE**: This notebook runs ``benchmark_app`` for 15 seconds to give
   a quick indication of performance. For more accurate performance, it
   is recommended to run ``benchmark_app`` in a terminal/command prompt
   after closing other applications. Run
   ``benchmark_app -m model.xml -d CPU`` to benchmark async inference on
   CPU for one minute. Change CPU to GPU to benchmark on GPU. Run
   ``benchmark_app --help`` to see an overview of all command-line
   options.

.. code:: ipython3

    import ipywidgets as widgets
    
    # Initialize OpenVINO runtime
    core = ov.Core()
    device = widgets.Dropdown(
        options=core.available_devices,
        value="CPU",
        description="Device:",
        disabled=False,
    )
    
    device




.. parsed-literal::

    Dropdown(description='Device:', options=('CPU',), value='CPU')



.. code:: ipython3

    def parse_benchmark_output(benchmark_output):
        parsed_output = [line for line in benchmark_output if "FPS" in line]
        print(*parsed_output, sep="\n")
    
    
    print("Benchmark FP32 model (IR)")
    benchmark_output = ! benchmark_app -m $fp32_ir_path -d $device.value -api async -t 15
    parse_benchmark_output(benchmark_output)
    
    print("Benchmark INT8 sparse model (IR)")
    benchmark_output = ! benchmark_app -m $int8_ir_path -d $device.value -api async -t 15
    parse_benchmark_output(benchmark_output)


.. parsed-literal::

    Benchmark FP32 model (IR)
    [ INFO ] Throughput:   2947.28 FPS
    Benchmark INT8 sparse model (IR)
    


Show Device Information for reference.

.. code:: ipython3

    core.get_property(device.value, "FULL_DEVICE_NAME")




.. parsed-literal::

    'Intel(R) Core(TM) i9-10920X CPU @ 3.50GHz'


