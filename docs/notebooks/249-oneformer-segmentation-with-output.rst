Universal Segmentation with OneFormer and OpenVINO
==================================================

This tutorial demonstrates how to use the
`OneFormer <https://arxiv.org/abs/2211.06220>`__ model from HuggingFace
with OpenVINO. It describes how to download weights and create PyTorch
model using Hugging Face transformers library, then convert model to
OpenVINO Intermediate Representation format (IR) using OpenVINO Model
Optimizer API and run model inference

|image0|

OneFormer is a follow-up work of
`Mask2Former <https://arxiv.org/abs/2112.01527>`__. The latter still
requires training on instance/semantic/panoptic datasets separately to
get state-of-the-art results.

OneFormer incorporates a text module in the Mask2Former framework, to
condition the model on the respective subtask (instance, semantic or
panoptic). This gives even more accurate results, but comes with a cost
of increased latency, however.

.. |image0| image:: https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/oneformer_architecture.png

Install required libraries
--------------------------

.. code:: ipython3

    !pip install -q "transformers>=4.26.0" "openvino==2023.1.0.dev20230728" gradio torch scipy ipywidgets Pillow matplotlib

Prepare the environment
-----------------------

Import all required packages and set paths for models and constant
variables.

.. code:: ipython3

    import warnings
    from collections import defaultdict
    from pathlib import Path
    import sys
    
    from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
    from transformers.models.oneformer.modeling_oneformer import OneFormerForUniversalSegmentationOutput
    import torch
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from PIL import Image
    from PIL import ImageOps
    
    import openvino
    
    sys.path.append("../utils")
    from notebook_utils import download_file


.. parsed-literal::

    2023-08-13 20:13:13.033722: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2023-08-13 20:13:13.205781: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2023-08-13 20:13:14.052205: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


.. code:: ipython3

    IR_PATH = Path("oneformer.xml")
    OUTPUT_NAMES = ['class_queries_logits', 'masks_queries_logits']

Load OneFormer fine-tuned on COCO for universal segmentation
------------------------------------------------------------

Here we use the ``from_pretrained`` method of
``OneFormerForUniversalSegmentation`` to load the `HuggingFace OneFormer
model <https://huggingface.co/docs/transformers/model_doc/oneformer>`__
based on Swin-L backbone and trained on
`COCO <https://cocodataset.org/>`__ dataset.

Also, we use HuggingFace processor to prepare the model inputs from
images and post-process model outputs for visualization.

.. code:: ipython3

    processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_coco_swin_large")
    model = OneFormerForUniversalSegmentation.from_pretrained(
        "shi-labs/oneformer_coco_swin_large",
    )
    id2label = model.config.id2label

.. code:: ipython3

    task_seq_length = processor.task_seq_length
    shape = (800, 800)
    dummy_input = {
        "pixel_values": torch.randn(1, 3, *shape),
        "task_inputs": torch.randn(1, task_seq_length),
        "pixel_mask": torch.randn(1, *shape),
    }

Convert the model to OpenVINO IR format
---------------------------------------

Convert the PyTorch model to IR format to take advantage of OpenVINO
optimization tools and features. The ``openvino.convert_model`` python
function in OpenVINO Converter can convert the model. The function
returns instance of OpenVINO Model class, which is ready to use in
Python interface. However, it can also be serialized to OpenVINO IR
format for future execution using ``save_model`` function. PyTorch to
OpenVINO conversion is based on TorchScript tracing. HuggingFace models
have specific configuration parameter ``torchscript``, which can be used
for making the model more suitable for tracing. For preparing model. we
should provide PyTorch model instance and example input to
``openvino.convert_model``.

.. code:: ipython3

    model.config.torchscript = True
    
    if not IR_PATH.exists():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = openvino.convert_model(model, example_input=dummy_input)
        openvino.save_model(model, IR_PATH, compress_to_fp16=False)

Select inference device
-----------------------

Select device from dropdown list for running inference using OpenVINO

.. code:: ipython3

    import ipywidgets as widgets
    
    core = openvino.Core()
    
    device = widgets.Dropdown(
        options=core.available_devices + ["AUTO"],
        value='AUTO',
        description='Device:',
        disabled=False,
    )
    
    device




.. parsed-literal::

    Dropdown(description='Device:', index=4, options=('CPU', 'GPU.0', 'GPU.1', 'GPU.2', 'AUTO'), value='AUTO')



We can prepare the image using the HuggingFace processor. OneFormer
leverages a processor which internally consists of an image processor
(for the image modality) and a tokenizer (for the text modality).
OneFormer is actually a multimodal model, since it incorporates both
images and text to solve image segmentation.

.. code:: ipython3

    def prepare_inputs(image: Image.Image, task: str):
        """Convert image to model input"""
        image = ImageOps.pad(image, shape)
        inputs = processor(image, [task], return_tensors="pt")
        converted = {
            'pixel_values': inputs['pixel_values'],
            'task_inputs': inputs['task_inputs']
        }
        return converted

.. code:: ipython3

    def process_output(d):
        """Convert OpenVINO model output to HuggingFace representation for visualization"""
        hf_kwargs = {
            output_name: torch.tensor(d[output_name]) for output_name in OUTPUT_NAMES
        }
    
        return OneFormerForUniversalSegmentationOutput(**hf_kwargs)

.. code:: ipython3

    # Read the model from files.
    model = core.read_model(model=IR_PATH)
    # Compile the model.
    compiled_model = core.compile_model(model=model, device_name=device.value)

Model predicts ``class_queries_logits`` of shape
``(batch_size, num_queries)`` and ``masks_queries_logits`` of shape
``(batch_size, num_queries, height, width)``.

Here we define functions for visualization of network outputs to show
the inference results.

.. code:: ipython3

    class Visualizer:
        @staticmethod
        def extract_legend(handles):
            fig = plt.figure()
            fig.legend(handles=handles, ncol=len(handles) // 20 + 1, loc='center')
            fig.tight_layout()
            return fig
        
        @staticmethod
        def predicted_semantic_map_to_figure(predicted_map):
            segmentation = predicted_map[0]
            # get the used color map
            viridis = plt.get_cmap('viridis', torch.max(segmentation))
            # get all the unique numbers
            labels_ids = torch.unique(segmentation).tolist()
            fig, ax = plt.subplots()
            ax.imshow(segmentation)
            ax.set_axis_off()
            handles = []
            for label_id in labels_ids:
                label = id2label[label_id]
                color = viridis(label_id)
                handles.append(mpatches.Patch(color=color, label=label))
            fig_legend = Visualizer.extract_legend(handles=handles)
            fig.tight_layout()
            return fig, fig_legend
            
        @staticmethod
        def predicted_instance_map_to_figure(predicted_map):
            segmentation = predicted_map[0]['segmentation']
            segments_info = predicted_map[0]['segments_info']
            # get the used color map
            viridis = plt.get_cmap('viridis', torch.max(segmentation))
            fig, ax = plt.subplots()
            ax.imshow(segmentation)
            ax.set_axis_off()
            instances_counter = defaultdict(int)
            handles = []
            # for each segment, draw its legend
            for segment in segments_info:
                segment_id = segment['id']
                segment_label_id = segment['label_id']
                segment_label = id2label[segment_label_id]
                label = f"{segment_label}-{instances_counter[segment_label_id]}"
                instances_counter[segment_label_id] += 1
                color = viridis(segment_id)
                handles.append(mpatches.Patch(color=color, label=label))
                
            fig_legend = Visualizer.extract_legend(handles)
            fig.tight_layout()
            return fig, fig_legend
    
        @staticmethod
        def predicted_panoptic_map_to_figure(predicted_map):
            segmentation = predicted_map[0]['segmentation']
            segments_info = predicted_map[0]['segments_info']
            # get the used color map
            viridis = plt.get_cmap('viridis', torch.max(segmentation))
            fig, ax = plt.subplots()
            ax.imshow(segmentation)
            ax.set_axis_off()
            instances_counter = defaultdict(int)
            handles = []
            # for each segment, draw its legend
            for segment in segments_info:
                segment_id = segment['id']
                segment_label_id = segment['label_id']
                segment_label = id2label[segment_label_id]
                label = f"{segment_label}-{instances_counter[segment_label_id]}"
                instances_counter[segment_label_id] += 1
                color = viridis(segment_id)
                handles.append(mpatches.Patch(color=color, label=label))
                
            fig_legend = Visualizer.extract_legend(handles)
            fig.tight_layout()
            return fig, fig_legend

.. code:: ipython3

    def segment(img: Image.Image, task: str):
        """
        Apply segmentation on an image.
    
        Args:
            img: Input image. It will be resized to 800x800.
            task: String describing the segmentation task. Supported values are: "semantic", "instance" and "panoptic".
        Returns:
            Tuple[Figure, Figure]: Segmentation map and legend charts.
        """
        if img is None:
            raise gr.Error("Please load the image or use one from the examples list")
        inputs = prepare_inputs(img, task)
        outputs = compiled_model(inputs)
        hf_output = process_output(outputs)
        predicted_map = getattr(processor, f"post_process_{task}_segmentation")(
            hf_output, target_sizes=[img.size[::-1]]
        )
        return getattr(Visualizer, f"predicted_{task}_map_to_figure")(predicted_map)

.. code:: ipython3

    image = download_file("http://images.cocodataset.org/val2017/000000439180.jpg", "sample.jpg")
    image = Image.open("sample.jpg")
    image



.. parsed-literal::

    sample.jpg:   0%|          | 0.00/194k [00:00<?, ?B/s]




.. image:: 249-oneformer-segmentation-with-output_files/249-oneformer-segmentation-with-output_22_1.png



Choose a segmentation task
--------------------------

.. code:: ipython3

    from ipywidgets import Dropdown
    
    task = Dropdown(options=["semantic", "instance", "panoptic"], value="semantic")
    task




.. parsed-literal::

    Dropdown(options=('semantic', 'instance', 'panoptic'), value='semantic')



Inference
---------

.. code:: ipython3

    import matplotlib
    matplotlib.use("Agg")  # disable showing figures
    
    def stack_images_horizontally(img1: Image, img2: Image):
        res = Image.new("RGB", (img1.width + img2.width, max(img1.height, img2.height)), (255, 255,255))
        res.paste(img1, (0, 0))
        res.paste(img2, (img1.width, 0))
        return res
    
    result, legend = segment(image, task.value)
    
    result.savefig("result.jpg", bbox_inches="tight")
    legend.savefig("legend.jpg", bbox_inches="tight")
    result = Image.open("result.jpg")
    legend = Image.open("legend.jpg")
    stack_images_horizontally(result, legend)




.. image:: 249-oneformer-segmentation-with-output_files/249-oneformer-segmentation-with-output_26_0.png



Interactive Demo
----------------

.. code:: ipython3

    import gradio as gr
    
    
    def compile_model(device):
        global compiled_model
        compiled_model = core.compile_model(model=model, device_name=device)
    
    
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                inp_img = gr.Image(label="Image", type="pil")
                inp_task = gr.Radio(
                    ["semantic", "instance", "panoptic"], label="Task", value="semantic"
                )
                inp_device = gr.Dropdown(
                    label="Device", choices=core.available_devices + ["AUTO"], value="AUTO"
                )
            with gr.Column():
                out_result = gr.Plot(label="Result")
                out_legend = gr.Plot(label="Legend")
        btn = gr.Button()
        gr.Examples(
            examples=[["sample.jpg", "semantic"]], inputs=[inp_img, inp_task]
        )
        btn.click(segment, [inp_img, inp_task], [out_result, out_legend])
    
        def on_device_change_begin():
            return (
                btn.update(value="Changing device...", interactive=False),
                inp_device.update(interactive=False)
            )
    
        def on_device_change_end():
            return (btn.update(value="Run", interactive=True), inp_device.update(interactive=True))
    
        inp_device.change(on_device_change_begin, outputs=[btn, inp_device]).then(
            compile_model, inp_device
        ).then(on_device_change_end, outputs=[btn, inp_device])
    
    
    try:
        demo.launch(debug=True)
    except Exception:
        demo.launch(share=True, debug=True)
    # if you are launching remotely, specify server_name and server_port
    # demo.launch(server_name='your server name', server_port='server port in int')
    # Read more in the docs: https://gradio.app/docs/
