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


**Table of contents:**


-  `Imports and Settings <#imports-and-settings>`__
-  `Pre-train Floating-Point Model <#pre-train-floating-point-model>`__

   -  `Train Function <#train-function>`__
   -  `Validate Function <#validate-function>`__
   -  `Helpers <#helpers>`__
   -  `Get a Pre-trained FP32 Model <#get-a-pre-trained-fp32-model>`__

-  `Create and Initialize
   Quantization <#create-and-initialize-quantization>`__
-  `Fine-tune the Compressed Model <#fine-tune-the-compressed-model>`__
-  `Export INT8 Sparse Model to OpenVINO
   IR <#export-int8-model-to-openvino-ir>`__
-  `Benchmark Model Performance by Computing Inference
   Time <#benchmark-model-performance-by-computing-inference-time>`__

Installation Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~

This is a self-contained example that relies solely on its own code.

We recommend running the notebook in a virtual environment. You only
need a Jupyter server to start. For details, please refer to
`Installation
Guide <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide>`__.

.. code:: ipython3

    %pip install -q --extra-index-url https://download.pytorch.org/whl/cpu  "openvino>=2024.0.0" "torch" "torchvision" "tqdm"
    %pip install -q "nncf>=2.9.0"


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.
    Note: you may need to restart the kernel to use updated packages.


Imports and Settings
--------------------



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
    from notebook_utils import download_file, device_widget
    
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

    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
      warnings.warn(
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=ResNet18_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet18_Weights.DEFAULT` to get the most up-to-date weights.
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
    WARNING:nncf:NNCF provides best results with torch==2.4.*, while current torch version is 2.2.2+cpu. If you encounter issues, consider switching to torch==2.4.*
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

    2024-10-23 03:39:01.390395: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-10-23 03:39:01.422771: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2024-10-23 03:39:02.047453: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


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

    Test: [ 0/79]	Time 0.425 (0.425)	Loss 6.069 (6.069)	Acc@1 0.00 (0.00)	Acc@5 4.69 (4.69)
    Test: [10/79]	Time 0.120 (0.164)	Loss 5.368 (5.689)	Acc@1 0.78 (0.07)	Acc@5 3.91 (2.41)
    Test: [20/79]	Time 0.142 (0.157)	Loss 5.921 (5.653)	Acc@1 0.00 (0.56)	Acc@5 2.34 (3.16)
    Test: [30/79]	Time 0.141 (0.150)	Loss 5.664 (5.670)	Acc@1 0.00 (0.50)	Acc@5 0.78 (2.90)
    Test: [40/79]	Time 0.155 (0.147)	Loss 5.608 (5.632)	Acc@1 1.56 (0.59)	Acc@5 3.12 (3.09)
    Test: [50/79]	Time 0.119 (0.144)	Loss 5.170 (5.618)	Acc@1 0.00 (0.72)	Acc@5 2.34 (3.32)
    Test: [60/79]	Time 0.141 (0.144)	Loss 6.619 (5.634)	Acc@1 0.00 (0.67)	Acc@5 0.00 (3.00)
    Test: [70/79]	Time 0.187 (0.144)	Loss 5.771 (5.653)	Acc@1 0.00 (0.57)	Acc@5 1.56 (2.77)
     * Acc@1 0.570 Acc@5 2.770
    Accuracy of initialized sparse INT8 model: 0.570


Fine-tune the Compressed Model
------------------------------



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
    Epoch:[0][  0/782]	Time 0.715 (0.715)	Loss 5.673 (5.673)	Acc@1 0.78 (0.78)	Acc@5 3.12 (3.12)
    Epoch:[0][ 50/782]	Time 0.335 (0.346)	Loss 5.634 (5.644)	Acc@1 0.00 (0.74)	Acc@5 3.12 (3.12)
    Epoch:[0][100/782]	Time 0.337 (0.342)	Loss 5.572 (5.606)	Acc@1 0.78 (0.77)	Acc@5 2.34 (3.29)
    Epoch:[0][150/782]	Time 0.363 (0.351)	Loss 5.531 (5.559)	Acc@1 1.56 (0.90)	Acc@5 3.12 (3.52)
    Epoch:[0][200/782]	Time 0.340 (0.348)	Loss 5.276 (5.515)	Acc@1 1.56 (1.07)	Acc@5 7.03 (3.96)
    Epoch:[0][250/782]	Time 0.340 (0.346)	Loss 5.361 (5.473)	Acc@1 0.00 (1.21)	Acc@5 6.25 (4.52)
    Epoch:[0][300/782]	Time 0.458 (0.348)	Loss 5.242 (5.431)	Acc@1 1.56 (1.41)	Acc@5 7.81 (5.04)
    Epoch:[0][350/782]	Time 0.337 (0.349)	Loss 5.092 (5.389)	Acc@1 3.12 (1.65)	Acc@5 11.72 (5.78)
    Epoch:[0][400/782]	Time 0.336 (0.348)	Loss 5.052 (5.351)	Acc@1 0.78 (1.85)	Acc@5 12.50 (6.39)
    Epoch:[0][450/782]	Time 0.338 (0.347)	Loss 5.033 (5.312)	Acc@1 3.12 (2.14)	Acc@5 12.50 (7.11)
    Epoch:[0][500/782]	Time 0.337 (0.349)	Loss 4.859 (5.275)	Acc@1 5.47 (2.41)	Acc@5 13.28 (7.85)
    Epoch:[0][550/782]	Time 0.337 (0.348)	Loss 4.697 (5.237)	Acc@1 10.94 (2.75)	Acc@5 23.44 (8.72)
    Epoch:[0][600/782]	Time 0.338 (0.347)	Loss 4.616 (5.197)	Acc@1 8.59 (3.15)	Acc@5 25.78 (9.74)
    Epoch:[0][650/782]	Time 0.420 (0.348)	Loss 4.610 (5.160)	Acc@1 9.38 (3.55)	Acc@5 23.44 (10.65)
    Epoch:[0][700/782]	Time 0.338 (0.348)	Loss 4.633 (5.122)	Acc@1 7.81 (4.00)	Acc@5 23.44 (11.65)
    Epoch:[0][750/782]	Time 0.355 (0.347)	Loss 4.462 (5.083)	Acc@1 17.19 (4.53)	Acc@5 34.38 (12.69)
    Epoch:[1][  0/782]	Time 0.761 (0.761)	Loss 4.324 (4.324)	Acc@1 17.19 (17.19)	Acc@5 33.59 (33.59)
    Epoch:[1][ 50/782]	Time 0.340 (0.376)	Loss 4.327 (4.226)	Acc@1 12.50 (16.76)	Acc@5 33.59 (37.47)
    Epoch:[1][100/782]	Time 0.341 (0.359)	Loss 4.195 (4.187)	Acc@1 17.19 (17.98)	Acc@5 37.50 (38.35)
    Epoch:[1][150/782]	Time 0.338 (0.352)	Loss 4.044 (4.166)	Acc@1 21.09 (18.47)	Acc@5 39.84 (38.98)
    Epoch:[1][200/782]	Time 0.463 (0.352)	Loss 4.096 (4.142)	Acc@1 18.75 (18.82)	Acc@5 39.06 (39.77)
    Epoch:[1][250/782]	Time 0.340 (0.354)	Loss 4.091 (4.119)	Acc@1 19.53 (19.14)	Acc@5 42.19 (40.31)
    Epoch:[1][300/782]	Time 0.341 (0.352)	Loss 4.201 (4.098)	Acc@1 14.84 (19.40)	Acc@5 34.38 (40.81)
    Epoch:[1][350/782]	Time 0.340 (0.350)	Loss 3.818 (4.076)	Acc@1 26.56 (19.74)	Acc@5 45.31 (41.34)
    Epoch:[1][400/782]	Time 0.338 (0.352)	Loss 4.093 (4.053)	Acc@1 18.75 (20.20)	Acc@5 36.72 (41.94)
    Epoch:[1][450/782]	Time 0.341 (0.351)	Loss 3.788 (4.033)	Acc@1 25.78 (20.55)	Acc@5 44.53 (42.41)
    Epoch:[1][500/782]	Time 0.338 (0.350)	Loss 3.821 (4.011)	Acc@1 25.78 (20.93)	Acc@5 50.78 (43.01)
    Epoch:[1][550/782]	Time 0.481 (0.351)	Loss 3.625 (3.988)	Acc@1 28.91 (21.30)	Acc@5 50.00 (43.57)
    Epoch:[1][600/782]	Time 0.388 (0.351)	Loss 3.691 (3.969)	Acc@1 28.12 (21.55)	Acc@5 46.09 (44.08)
    Epoch:[1][650/782]	Time 0.345 (0.350)	Loss 3.736 (3.951)	Acc@1 22.66 (21.75)	Acc@5 47.66 (44.59)
    Epoch:[1][700/782]	Time 0.345 (0.350)	Loss 3.740 (3.935)	Acc@1 25.00 (21.97)	Acc@5 44.53 (44.94)
    Epoch:[1][750/782]	Time 0.345 (0.352)	Loss 3.619 (3.917)	Acc@1 32.03 (22.25)	Acc@5 52.34 (45.38)
    Epoch:[2][  0/782]	Time 0.761 (0.761)	Loss 3.427 (3.427)	Acc@1 28.91 (28.91)	Acc@5 58.59 (58.59)
    Epoch:[2][ 50/782]	Time 0.345 (0.355)	Loss 3.394 (3.465)	Acc@1 34.38 (29.43)	Acc@5 58.59 (55.38)
    Epoch:[2][100/782]	Time 0.464 (0.354)	Loss 3.294 (3.432)	Acc@1 34.38 (30.09)	Acc@5 60.16 (56.66)
    Epoch:[2][150/782]	Time 0.343 (0.356)	Loss 3.359 (3.422)	Acc@1 32.81 (30.34)	Acc@5 61.72 (56.80)
    Epoch:[2][200/782]	Time 0.342 (0.353)	Loss 3.217 (3.409)	Acc@1 35.16 (30.47)	Acc@5 64.06 (57.05)
    Epoch:[2][250/782]	Time 0.352 (0.351)	Loss 3.366 (3.391)	Acc@1 30.47 (30.75)	Acc@5 57.03 (57.27)
    Epoch:[2][300/782]	Time 0.344 (0.354)	Loss 3.484 (3.378)	Acc@1 25.00 (30.86)	Acc@5 51.56 (57.40)
    Epoch:[2][350/782]	Time 0.345 (0.353)	Loss 3.327 (3.369)	Acc@1 33.59 (30.97)	Acc@5 61.72 (57.54)
    Epoch:[2][400/782]	Time 0.342 (0.352)	Loss 3.425 (3.358)	Acc@1 25.78 (31.11)	Acc@5 58.59 (57.65)
    Epoch:[2][450/782]	Time 0.341 (0.354)	Loss 3.440 (3.346)	Acc@1 28.12 (31.40)	Acc@5 59.38 (57.85)
    Epoch:[2][500/782]	Time 0.342 (0.353)	Loss 3.122 (3.335)	Acc@1 35.16 (31.58)	Acc@5 66.41 (58.07)
    Epoch:[2][550/782]	Time 0.341 (0.352)	Loss 3.368 (3.321)	Acc@1 28.91 (31.76)	Acc@5 55.47 (58.41)
    Epoch:[2][600/782]	Time 0.341 (0.351)	Loss 3.077 (3.309)	Acc@1 37.50 (31.89)	Acc@5 64.06 (58.55)
    Epoch:[2][650/782]	Time 0.345 (0.353)	Loss 3.177 (3.299)	Acc@1 35.94 (32.09)	Acc@5 64.84 (58.72)
    Epoch:[2][700/782]	Time 0.342 (0.352)	Loss 3.139 (3.286)	Acc@1 33.59 (32.24)	Acc@5 60.94 (58.95)
    Epoch:[2][750/782]	Time 0.373 (0.351)	Loss 3.238 (3.274)	Acc@1 35.94 (32.48)	Acc@5 57.81 (59.19)
    Epoch:[3][  0/782]	Time 0.880 (0.880)	Loss 3.069 (3.069)	Acc@1 33.59 (33.59)	Acc@5 64.84 (64.84)
    Epoch:[3][ 50/782]	Time 0.343 (0.369)	Loss 2.916 (2.955)	Acc@1 44.53 (38.13)	Acc@5 64.06 (65.06)
    Epoch:[3][100/782]	Time 0.342 (0.357)	Loss 3.027 (2.936)	Acc@1 34.38 (38.15)	Acc@5 63.28 (65.73)
    Epoch:[3][150/782]	Time 0.343 (0.353)	Loss 2.753 (2.933)	Acc@1 39.84 (38.11)	Acc@5 70.31 (65.45)
    Epoch:[3][200/782]	Time 0.343 (0.358)	Loss 3.030 (2.928)	Acc@1 35.16 (38.22)	Acc@5 59.38 (65.42)
    Epoch:[3][250/782]	Time 0.370 (0.355)	Loss 2.841 (2.923)	Acc@1 33.59 (38.19)	Acc@5 67.19 (65.40)
    Epoch:[3][300/782]	Time 0.341 (0.354)	Loss 2.888 (2.918)	Acc@1 42.97 (38.22)	Acc@5 71.88 (65.44)
    Epoch:[3][350/782]	Time 0.344 (0.356)	Loss 2.760 (2.914)	Acc@1 40.62 (38.24)	Acc@5 69.53 (65.44)
    Epoch:[3][400/782]	Time 0.344 (0.355)	Loss 3.104 (2.907)	Acc@1 30.47 (38.21)	Acc@5 59.38 (65.51)
    Epoch:[3][450/782]	Time 0.344 (0.355)	Loss 2.911 (2.901)	Acc@1 35.94 (38.32)	Acc@5 62.50 (65.57)
    Epoch:[3][500/782]	Time 0.340 (0.354)	Loss 2.736 (2.894)	Acc@1 41.41 (38.37)	Acc@5 64.84 (65.73)
    Epoch:[3][550/782]	Time 0.340 (0.355)	Loss 3.151 (2.888)	Acc@1 29.69 (38.40)	Acc@5 60.16 (65.81)
    Epoch:[3][600/782]	Time 0.341 (0.354)	Loss 3.021 (2.883)	Acc@1 30.47 (38.44)	Acc@5 59.38 (65.83)
    Epoch:[3][650/782]	Time 0.342 (0.353)	Loss 2.929 (2.876)	Acc@1 41.41 (38.55)	Acc@5 66.41 (65.97)
    Epoch:[3][700/782]	Time 0.345 (0.355)	Loss 2.975 (2.869)	Acc@1 33.59 (38.60)	Acc@5 62.50 (66.06)
    Epoch:[3][750/782]	Time 0.344 (0.354)	Loss 2.790 (2.863)	Acc@1 39.06 (38.73)	Acc@5 64.84 (66.12)
    Epoch:[4][  0/782]	Time 0.759 (0.759)	Loss 2.629 (2.629)	Acc@1 46.88 (46.88)	Acc@5 67.97 (67.97)
    Epoch:[4][ 50/782]	Time 0.343 (0.353)	Loss 2.676 (2.725)	Acc@1 45.31 (40.36)	Acc@5 67.19 (68.01)
    Epoch:[4][100/782]	Time 0.341 (0.363)	Loss 2.824 (2.698)	Acc@1 32.81 (41.15)	Acc@5 66.41 (68.69)
    Epoch:[4][150/782]	Time 0.341 (0.356)	Loss 2.700 (2.689)	Acc@1 46.88 (41.41)	Acc@5 62.50 (69.01)
    Epoch:[4][200/782]	Time 0.345 (0.353)	Loss 2.516 (2.682)	Acc@1 46.88 (41.59)	Acc@5 75.00 (69.14)
    Epoch:[4][250/782]	Time 0.345 (0.357)	Loss 2.395 (2.676)	Acc@1 49.22 (41.80)	Acc@5 73.44 (69.18)
    Epoch:[4][300/782]	Time 0.346 (0.354)	Loss 2.625 (2.673)	Acc@1 42.19 (41.85)	Acc@5 65.62 (69.08)
    Epoch:[4][350/782]	Time 0.341 (0.353)	Loss 2.616 (2.670)	Acc@1 46.88 (41.88)	Acc@5 71.88 (69.12)
    Epoch:[4][400/782]	Time 0.473 (0.353)	Loss 2.459 (2.661)	Acc@1 42.97 (42.00)	Acc@5 72.66 (69.26)
    Epoch:[4][450/782]	Time 0.344 (0.354)	Loss 2.520 (2.657)	Acc@1 45.31 (42.02)	Acc@5 75.00 (69.26)
    Epoch:[4][500/782]	Time 0.343 (0.353)	Loss 2.788 (2.653)	Acc@1 37.50 (42.08)	Acc@5 64.84 (69.31)
    Epoch:[4][550/782]	Time 0.346 (0.352)	Loss 2.466 (2.645)	Acc@1 43.75 (42.25)	Acc@5 68.75 (69.41)
    Epoch:[4][600/782]	Time 0.341 (0.354)	Loss 2.392 (2.640)	Acc@1 51.56 (42.30)	Acc@5 73.44 (69.44)
    Epoch:[4][650/782]	Time 0.342 (0.353)	Loss 2.593 (2.636)	Acc@1 41.41 (42.33)	Acc@5 71.09 (69.45)
    Epoch:[4][700/782]	Time 0.344 (0.352)	Loss 2.537 (2.633)	Acc@1 38.28 (42.34)	Acc@5 73.44 (69.46)
    Epoch:[4][750/782]	Time 0.469 (0.354)	Loss 2.407 (2.626)	Acc@1 42.19 (42.42)	Acc@5 76.56 (69.60)
    Epoch:[5][  0/782]	Time 0.753 (0.753)	Loss 2.314 (2.314)	Acc@1 49.22 (49.22)	Acc@5 73.44 (73.44)
    Epoch:[5][ 50/782]	Time 0.343 (0.353)	Loss 2.585 (2.519)	Acc@1 43.75 (43.64)	Acc@5 69.53 (71.03)
    Epoch:[5][100/782]	Time 0.343 (0.348)	Loss 2.277 (2.489)	Acc@1 46.88 (44.35)	Acc@5 76.56 (71.71)
    Epoch:[5][150/782]	Time 0.355 (0.357)	Loss 2.283 (2.479)	Acc@1 52.34 (44.65)	Acc@5 75.78 (71.80)
    Epoch:[5][200/782]	Time 0.344 (0.354)	Loss 2.444 (2.478)	Acc@1 46.88 (44.71)	Acc@5 69.53 (71.70)
    Epoch:[5][250/782]	Time 0.344 (0.352)	Loss 2.566 (2.481)	Acc@1 42.97 (44.73)	Acc@5 69.53 (71.67)
    Epoch:[5][300/782]	Time 0.463 (0.354)	Loss 2.404 (2.474)	Acc@1 42.19 (44.81)	Acc@5 77.34 (71.83)
    Epoch:[5][350/782]	Time 0.344 (0.354)	Loss 2.306 (2.476)	Acc@1 50.78 (44.73)	Acc@5 77.34 (71.68)
    Epoch:[5][400/782]	Time 0.341 (0.352)	Loss 2.418 (2.471)	Acc@1 43.75 (44.84)	Acc@5 72.66 (71.75)
    Epoch:[5][450/782]	Time 0.343 (0.351)	Loss 2.359 (2.465)	Acc@1 51.56 (44.92)	Acc@5 74.22 (71.87)
    Epoch:[5][500/782]	Time 0.345 (0.353)	Loss 2.418 (2.463)	Acc@1 47.66 (44.95)	Acc@5 75.00 (71.86)
    Epoch:[5][550/782]	Time 0.343 (0.353)	Loss 2.405 (2.459)	Acc@1 42.19 (45.00)	Acc@5 71.09 (71.89)
    Epoch:[5][600/782]	Time 0.345 (0.352)	Loss 2.330 (2.457)	Acc@1 50.00 (45.04)	Acc@5 76.56 (71.92)
    Epoch:[5][650/782]	Time 0.345 (0.353)	Loss 2.273 (2.451)	Acc@1 48.44 (45.10)	Acc@5 72.66 (72.01)
    Epoch:[5][700/782]	Time 0.341 (0.353)	Loss 2.231 (2.446)	Acc@1 46.09 (45.19)	Acc@5 72.66 (72.09)
    Epoch:[5][750/782]	Time 0.341 (0.352)	Loss 2.482 (2.442)	Acc@1 50.78 (45.26)	Acc@5 67.19 (72.14)
    Epoch:[6][  0/782]	Time 0.760 (0.760)	Loss 2.563 (2.563)	Acc@1 43.75 (43.75)	Acc@5 64.06 (64.06)
    Epoch:[6][ 50/782]	Time 0.342 (0.381)	Loss 2.414 (2.318)	Acc@1 46.09 (47.76)	Acc@5 70.31 (74.16)
    Epoch:[6][100/782]	Time 0.342 (0.363)	Loss 2.413 (2.332)	Acc@1 46.88 (47.07)	Acc@5 71.88 (73.90)
    Epoch:[6][150/782]	Time 0.344 (0.357)	Loss 2.217 (2.330)	Acc@1 48.44 (47.10)	Acc@5 75.78 (73.78)
    Epoch:[6][200/782]	Time 0.342 (0.360)	Loss 2.341 (2.328)	Acc@1 48.44 (47.20)	Acc@5 73.44 (73.74)
    Epoch:[6][250/782]	Time 0.343 (0.357)	Loss 2.578 (2.330)	Acc@1 43.75 (47.19)	Acc@5 67.19 (73.85)
    Epoch:[6][300/782]	Time 0.344 (0.355)	Loss 2.454 (2.321)	Acc@1 43.75 (47.48)	Acc@5 71.88 (74.04)
    Epoch:[6][350/782]	Time 0.342 (0.353)	Loss 2.336 (2.320)	Acc@1 49.22 (47.46)	Acc@5 75.00 (74.05)
    Epoch:[6][400/782]	Time 0.342 (0.356)	Loss 2.060 (2.316)	Acc@1 50.78 (47.57)	Acc@5 81.25 (74.07)
    Epoch:[6][450/782]	Time 0.343 (0.354)	Loss 2.363 (2.316)	Acc@1 46.09 (47.42)	Acc@5 71.88 (74.07)
    Epoch:[6][500/782]	Time 0.342 (0.354)	Loss 2.333 (2.312)	Acc@1 49.22 (47.43)	Acc@5 70.31 (74.11)
    Epoch:[6][550/782]	Time 0.342 (0.355)	Loss 2.198 (2.308)	Acc@1 46.88 (47.51)	Acc@5 75.00 (74.18)
    Epoch:[6][600/782]	Time 0.342 (0.354)	Loss 2.199 (2.304)	Acc@1 58.59 (47.62)	Acc@5 77.34 (74.20)
    Epoch:[6][650/782]	Time 0.346 (0.353)	Loss 2.126 (2.303)	Acc@1 51.56 (47.62)	Acc@5 80.47 (74.24)
    Epoch:[6][700/782]	Time 0.467 (0.353)	Loss 2.313 (2.298)	Acc@1 39.84 (47.71)	Acc@5 71.88 (74.32)
    Epoch:[6][750/782]	Time 0.343 (0.354)	Loss 2.078 (2.294)	Acc@1 55.47 (47.77)	Acc@5 78.12 (74.35)
    Epoch:[7][  0/782]	Time 0.761 (0.761)	Loss 2.202 (2.202)	Acc@1 43.75 (43.75)	Acc@5 75.78 (75.78)
    Epoch:[7][ 50/782]	Time 0.343 (0.352)	Loss 2.119 (2.211)	Acc@1 53.12 (48.94)	Acc@5 76.56 (75.41)
    Epoch:[7][100/782]	Time 0.340 (0.363)	Loss 2.285 (2.211)	Acc@1 55.47 (49.30)	Acc@5 71.09 (75.46)
    Epoch:[7][150/782]	Time 0.343 (0.356)	Loss 1.987 (2.207)	Acc@1 56.25 (49.26)	Acc@5 81.25 (75.51)
    Epoch:[7][200/782]	Time 0.348 (0.353)	Loss 2.240 (2.202)	Acc@1 47.66 (49.49)	Acc@5 75.00 (75.61)
    Epoch:[7][250/782]	Time 0.343 (0.352)	Loss 2.206 (2.202)	Acc@1 48.44 (49.41)	Acc@5 77.34 (75.70)
    Epoch:[7][300/782]	Time 0.346 (0.355)	Loss 2.387 (2.201)	Acc@1 51.56 (49.46)	Acc@5 69.53 (75.77)
    Epoch:[7][350/782]	Time 0.343 (0.353)	Loss 2.073 (2.195)	Acc@1 42.19 (49.53)	Acc@5 81.25 (75.92)
    Epoch:[7][400/782]	Time 0.342 (0.352)	Loss 1.702 (2.193)	Acc@1 63.28 (49.61)	Acc@5 84.38 (75.91)
    Epoch:[7][450/782]	Time 0.343 (0.354)	Loss 2.209 (2.193)	Acc@1 48.44 (49.64)	Acc@5 76.56 (75.92)
    Epoch:[7][500/782]	Time 0.343 (0.353)	Loss 2.164 (2.191)	Acc@1 48.44 (49.61)	Acc@5 76.56 (75.86)
    Epoch:[7][550/782]	Time 0.344 (0.352)	Loss 2.102 (2.186)	Acc@1 46.88 (49.73)	Acc@5 78.91 (75.89)
    Epoch:[7][600/782]	Time 0.477 (0.353)	Loss 2.209 (2.183)	Acc@1 47.66 (49.76)	Acc@5 71.88 (75.90)
    Epoch:[7][650/782]	Time 0.342 (0.353)	Loss 2.071 (2.180)	Acc@1 49.22 (49.84)	Acc@5 75.78 (75.91)
    Epoch:[7][700/782]	Time 0.344 (0.353)	Loss 2.158 (2.178)	Acc@1 47.66 (49.87)	Acc@5 75.00 (75.93)
    Epoch:[7][750/782]	Time 0.341 (0.352)	Loss 2.076 (2.177)	Acc@1 52.34 (49.89)	Acc@5 76.56 (75.91)
    Epoch:[8][  0/782]	Time 0.772 (0.772)	Loss 1.827 (1.827)	Acc@1 57.81 (57.81)	Acc@5 82.81 (82.81)
    Epoch:[8][ 50/782]	Time 0.343 (0.353)	Loss 2.204 (2.097)	Acc@1 50.00 (51.07)	Acc@5 77.34 (77.28)
    Epoch:[8][100/782]	Time 0.344 (0.348)	Loss 2.199 (2.089)	Acc@1 53.12 (51.28)	Acc@5 73.44 (77.47)
    Epoch:[8][150/782]	Time 0.461 (0.349)	Loss 2.295 (2.101)	Acc@1 47.66 (51.02)	Acc@5 74.22 (77.14)
    Epoch:[8][200/782]	Time 0.343 (0.353)	Loss 2.163 (2.101)	Acc@1 46.09 (50.88)	Acc@5 77.34 (77.20)
    Epoch:[8][250/782]	Time 0.342 (0.352)	Loss 2.244 (2.092)	Acc@1 46.09 (51.15)	Acc@5 71.88 (77.31)
    Epoch:[8][300/782]	Time 0.344 (0.350)	Loss 2.068 (2.087)	Acc@1 51.56 (51.28)	Acc@5 76.56 (77.41)
    Epoch:[8][350/782]	Time 0.345 (0.353)	Loss 1.878 (2.083)	Acc@1 53.12 (51.31)	Acc@5 81.25 (77.41)
    Epoch:[8][400/782]	Time 0.343 (0.352)	Loss 2.356 (2.084)	Acc@1 39.84 (51.37)	Acc@5 72.66 (77.38)
    Epoch:[8][450/782]	Time 0.355 (0.352)	Loss 1.727 (2.084)	Acc@1 61.72 (51.35)	Acc@5 82.81 (77.35)
    Epoch:[8][500/782]	Time 0.471 (0.353)	Loss 2.142 (2.082)	Acc@1 46.09 (51.33)	Acc@5 78.12 (77.37)
    Epoch:[8][550/782]	Time 0.342 (0.353)	Loss 2.170 (2.079)	Acc@1 52.34 (51.39)	Acc@5 74.22 (77.42)
    Epoch:[8][600/782]	Time 0.342 (0.352)	Loss 2.189 (2.076)	Acc@1 54.69 (51.51)	Acc@5 74.22 (77.46)
    Epoch:[8][650/782]	Time 0.351 (0.351)	Loss 2.114 (2.074)	Acc@1 50.00 (51.54)	Acc@5 79.69 (77.47)
    Epoch:[8][700/782]	Time 0.342 (0.353)	Loss 2.255 (2.074)	Acc@1 53.12 (51.53)	Acc@5 73.44 (77.44)
    Epoch:[8][750/782]	Time 0.343 (0.352)	Loss 2.060 (2.071)	Acc@1 54.69 (51.57)	Acc@5 76.56 (77.46)
    Epoch:[9][  0/782]	Time 0.778 (0.778)	Loss 1.831 (1.831)	Acc@1 56.25 (56.25)	Acc@5 84.38 (84.38)
    Epoch:[9][ 50/782]	Time 0.385 (0.369)	Loss 2.054 (1.996)	Acc@1 48.44 (53.31)	Acc@5 81.25 (78.81)
    Epoch:[9][100/782]	Time 0.342 (0.364)	Loss 1.864 (1.998)	Acc@1 59.38 (53.34)	Acc@5 82.03 (78.36)
    Epoch:[9][150/782]	Time 0.344 (0.358)	Loss 2.027 (1.993)	Acc@1 51.56 (53.24)	Acc@5 80.47 (78.79)
    Epoch:[9][200/782]	Time 0.342 (0.354)	Loss 1.873 (1.994)	Acc@1 57.81 (53.24)	Acc@5 81.25 (78.72)
    Epoch:[9][250/782]	Time 0.343 (0.358)	Loss 2.171 (1.996)	Acc@1 47.66 (53.16)	Acc@5 75.00 (78.60)
    Epoch:[9][300/782]	Time 0.342 (0.356)	Loss 2.138 (1.997)	Acc@1 52.34 (53.12)	Acc@5 75.78 (78.49)
    Epoch:[9][350/782]	Time 0.342 (0.354)	Loss 2.202 (1.995)	Acc@1 44.53 (53.02)	Acc@5 75.00 (78.57)
    Epoch:[9][400/782]	Time 0.343 (0.356)	Loss 1.884 (1.994)	Acc@1 59.38 (52.95)	Acc@5 81.25 (78.53)
    Epoch:[9][450/782]	Time 0.345 (0.355)	Loss 2.046 (1.988)	Acc@1 51.56 (53.07)	Acc@5 75.78 (78.60)
    Epoch:[9][500/782]	Time 0.345 (0.354)	Loss 2.284 (1.990)	Acc@1 46.88 (53.00)	Acc@5 72.66 (78.62)
    Epoch:[9][550/782]	Time 0.342 (0.354)	Loss 1.614 (1.990)	Acc@1 65.62 (53.05)	Acc@5 82.81 (78.60)
    Epoch:[9][600/782]	Time 0.345 (0.355)	Loss 1.783 (1.986)	Acc@1 53.12 (53.10)	Acc@5 85.16 (78.65)
    Epoch:[9][650/782]	Time 0.343 (0.355)	Loss 1.669 (1.983)	Acc@1 60.94 (53.14)	Acc@5 82.81 (78.71)
    Epoch:[9][700/782]	Time 0.355 (0.354)	Loss 2.272 (1.982)	Acc@1 41.41 (53.14)	Acc@5 75.78 (78.75)
    Epoch:[9][750/782]	Time 0.349 (0.355)	Loss 1.714 (1.982)	Acc@1 59.38 (53.12)	Acc@5 80.47 (78.71)
    Validating
    Test: [ 0/79]	Time 0.444 (0.444)	Loss 4.184 (4.184)	Acc@1 8.59 (8.59)	Acc@5 31.25 (31.25)
    Test: [10/79]	Time 0.139 (0.173)	Loss 5.948 (4.814)	Acc@1 3.12 (7.67)	Acc@5 6.25 (21.24)
    Test: [20/79]	Time 0.119 (0.158)	Loss 6.329 (5.114)	Acc@1 0.00 (4.95)	Acc@5 3.91 (17.26)
    Test: [30/79]	Time 0.176 (0.153)	Loss 5.530 (5.322)	Acc@1 0.78 (4.11)	Acc@5 17.19 (14.42)
    Test: [40/79]	Time 0.139 (0.150)	Loss 5.589 (5.396)	Acc@1 6.25 (4.04)	Acc@5 8.59 (13.38)
    Test: [50/79]	Time 0.119 (0.147)	Loss 4.862 (5.493)	Acc@1 7.03 (3.80)	Acc@5 23.44 (12.65)
    Test: [60/79]	Time 0.119 (0.146)	Loss 5.924 (5.506)	Acc@1 0.00 (3.92)	Acc@5 6.25 (12.59)
    Test: [70/79]	Time 0.150 (0.146)	Loss 4.818 (5.519)	Acc@1 3.12 (3.71)	Acc@5 10.16 (11.95)
     * Acc@1 5.190 Acc@5 15.180
    Accuracy of tuned INT8 sparse model: 5.190
    Accuracy drop of tuned INT8 sparse model over pre-trained FP32 model: -4.620


Export INT8 Sparse Model to OpenVINO IR
---------------------------------------



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

    # Initialize OpenVINO runtime
    core = ov.Core()
    device = device_widget()
    
    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



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
    [ INFO ] Throughput:   2946.50 FPS
    Benchmark INT8 sparse model (IR)
    


Show Device Information for reference.

.. code:: ipython3

    import openvino.properties as props
    
    
    core.get_property(device.value, props.device.full_name)




.. parsed-literal::

    'AUTO'


