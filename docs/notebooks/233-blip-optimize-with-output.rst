Post-Training Quantization and Weights Compression of OpenAI BLIP model with NNCF
=================================================================================

The goal of this tutorial is to demonstrate how to speed up the model by
applying 8-bit post-training quantization and data free int8 weight
compression from `NNCF <https://github.com/openvinotoolkit/nncf/>`__
(Neural Network Compression Framework) to OpenVINO IR models and infer
optimized BLIP model via OpenVINO™ Toolkit. The optimization process
contains the following steps:

1. Download and preprocess dataset for quantization.
2. Quantize the converted vision and text encoder OpenVINO models from
   `notebook <233-blip-convert.ipynb>`__ with NNCF.
3. Compress weights of the OpenVINO text decoder model from
   `notebook <233-blip-convert.ipynb>`__ with NNCF.
4. Check the model result using the same input data from the
   `notebook <233-blip-convert.ipynb>`__.
5. Compare model size of converted and optimized models.
6. Compare performance of converted and optimized models.

..

   **NOTE**: you should run
   `233-blip-convert <233-blip-convert.ipynb>`__ notebook first to
   generate OpenVINO IR models that are used for optimization.

**Table of contents:**


-  `Prerequisites <#prerequisites>`__
-  `Quantize <#quantize>`__

   -  `Prepare dataset <#prepare-dataset>`__
   -  `Quantize Vision Model <#quantize-vision-model>`__
   -  `Quantize Text Encoder <#quantize-text-encoder>`__

-  `Compress text decoder weights <#compress-weights>`__
-  `Run optimized OpenVINO
   model <#run-optimized-openvino-model>`__

   -  `Image Captioning <#image-captioning>`__
   -  `Question Answering <#question-answering>`__
   -  `Compare file sizes <#compare-file-sizes>`__
   -  `Compare inference time of the FP16 and optimized
      models <#compare-inference-time-of-the-fp-and-optimized-models>`__

Prerequisites 
-------------------------------------------------------

.. code:: ipython3

    %pip install -q datasets

.. code:: ipython3

    from pathlib import Path
    
    VISION_MODEL_OV = Path("blip_vision_model.xml")
    TEXT_ENCODER_OV = Path("blip_text_encoder.xml")
    TEXT_DECODER_OV = Path("blip_text_decoder_with_past.xml")
    
    if not (VISION_MODEL_OV.exists() and TEXT_ENCODER_OV.exists() and TEXT_DECODER_OV.exists()):
        raise RuntimeError('This notebook should be run after 233-blip-convert notebook')

.. code:: ipython3

    from transformers import BlipProcessor
    
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")

Quantize 
--------------------------------------------------

`NNCF <https://github.com/openvinotoolkit/nncf/>`__ enables
post-training quantization by adding the quantization layers into the
model graph and then using a subset of the training dataset to
initialize the parameters of these additional quantization layers. The
framework is designed so that modifications to your original training
code are minor.

The optimization process contains the following steps:

1. Create a dataset for quantization.
2. Run ``nncf.quantize`` to get a quantized model from the pre-trained
   ``FP16`` model.
3. Serialize the ``INT8`` model using ``openvino.save_model`` function.

..

   **NOTE**: Quantization is time and memory consuming operation.
   Running quantization code below may take some time.

Prepare dataset 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `VQAv2 <https://visualqa.org/>`__ is a dataset containing
open-ended questions about images. These questions require an
understanding of vision, language and commonsense knowledge to answer.

.. code:: ipython3

    import numpy as np
    from datasets import load_dataset
    from tqdm.notebook import tqdm
    
    def preprocess_batch(batch, vision_model, inputs_info):
        """
        Preprocesses a dataset batch by loading and transforming image and text data.
        VQAv2 dataset contains multiple questions to image.
        To reduce dataset preparation time we will store preprocessed images in `inputs_info`.
        """
        image_id = batch["image_id"]
        if image_id in inputs_info:
            inputs = processor(text=batch['question'], return_tensors="np")
            pixel_values = inputs_info[image_id]["pixel_values"]
            encoder_hidden_states = inputs_info[image_id]["encoder_hidden_states"]
        else:
            inputs = processor(images=batch["image"], text=batch["question"], return_tensors="np")
            pixel_values = inputs["pixel_values"]
            encoder_hidden_states = vision_model(pixel_values)[vision_model.output(0)]
            inputs_info[image_id] = {
                "pixel_values": pixel_values,
                "encoder_hidden_states": encoder_hidden_states,
                "text_encoder_inputs": []
            }
    
        text_encoder_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"]
        }
        inputs_info[image_id]["text_encoder_inputs"].append(text_encoder_inputs)
    
    
    def prepare_input_data(dataloader, vision_model, opt_init_steps):
        """
        Store calibration subset in List to reduce quantization time.
        """
        inputs_info = {}
        for batch in tqdm(dataloader, total=opt_init_steps, desc="Prepare calibration data"):
            preprocess_batch(batch, vision_model, inputs_info)
    
        calibration_subset = []
        for image_id in inputs_info:
            pixel_values = inputs_info[image_id]["pixel_values"]
            encoder_hidden_states = inputs_info[image_id]["encoder_hidden_states"]
            encoder_attention_mask = np.ones(encoder_hidden_states.shape[:-1], dtype=int)
            for text_encoder_inputs in inputs_info[image_id]["text_encoder_inputs"]:
                text_encoder_inputs["encoder_hidden_states"] = encoder_hidden_states
                text_encoder_inputs["encoder_attention_mask"] = encoder_attention_mask
                blip_inputs = {
                    "vision_model_inputs": {"pixel_values": pixel_values},
                    "text_encoder_inputs": text_encoder_inputs,
                }
                calibration_subset.append(blip_inputs)
        return calibration_subset
    
    
    def prepare_dataset(vision_model, opt_init_steps=300, streaming=True):
        """
        Prepares a vision-text dataset for quantization.
        """
        dataset = load_dataset("HuggingFaceM4/VQAv2", split="train", streaming=streaming)
        train_dataset = dataset.shuffle(seed=42).take(opt_init_steps)
        calibration_subset = prepare_input_data(train_dataset, vision_model, opt_init_steps)
        return calibration_subset

Loading and processing the dataset in streaming mode may take a long
time and depends on your internet connection.

.. code:: ipython3

    import nncf
    import openvino as ov
    
    comp_vision_model = ov.compile_model(VISION_MODEL_OV)
    calibration_data = prepare_dataset(comp_vision_model)


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, tensorflow, onnx, openvino


.. parsed-literal::

    Repo card metadata block was not found. Setting CardData to empty.



.. parsed-literal::

    Prepare calibration data:   0%|          | 0/300 [00:00<?, ?it/s]


Vision model 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    VISION_MODEL_OV_INT8 = Path(str(VISION_MODEL_OV).replace(".xml", "_int8.xml"))
    
    core = ov.Core()
    ov_vision_model = core.read_model(VISION_MODEL_OV)
    vision_dataset = nncf.Dataset(calibration_data, lambda x: x["vision_model_inputs"])
    
    quantized_model = nncf.quantize(
        model=ov_vision_model,
        calibration_dataset=vision_dataset,
        model_type=nncf.ModelType.TRANSFORMER
    )
    
    ov.save_model(quantized_model, VISION_MODEL_OV_INT8)


.. parsed-literal::

    Statistics collection: 100%|██████████| 300/300 [00:21<00:00, 14.06it/s]
    Applying Smooth Quant: 100%|██████████| 48/48 [00:01<00:00, 29.72it/s]


.. parsed-literal::

    INFO:nncf:36 ignored nodes was found by name in the NNCFGraph


.. parsed-literal::

    Statistics collection: 100%|██████████| 300/300 [01:17<00:00,  3.89it/s]
    Applying Fast Bias correction: 100%|██████████| 49/49 [00:27<00:00,  1.80it/s]


Text encoder 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    TEXT_ENCODER_OV_INT8 = Path(str(TEXT_ENCODER_OV).replace(".xml", "_int8.xml"))
    
    text_encoder_dataset = nncf.Dataset(calibration_data, lambda x: x["text_encoder_inputs"])
    ov_text_encoder = core.read_model(TEXT_ENCODER_OV)
    quantized_model = nncf.quantize(
        model=ov_text_encoder,
        calibration_dataset=text_encoder_dataset,
        model_type=nncf.ModelType.TRANSFORMER
    )
    ov.save_model(quantized_model, TEXT_ENCODER_OV_INT8)


.. parsed-literal::

    Statistics collection: 100%|██████████| 300/300 [00:10<00:00, 28.70it/s]
    Applying Smooth Quant: 100%|██████████| 73/73 [00:02<00:00, 28.87it/s]


.. parsed-literal::

    INFO:nncf:72 ignored nodes was found by name in the NNCFGraph


.. parsed-literal::

    Statistics collection: 100%|██████████| 300/300 [00:31<00:00,  9.54it/s]
    Applying Fast Bias correction: 100%|██████████| 120/120 [00:38<00:00,  3.11it/s]


Compress weights 
----------------------------------------------------------

The quantization of the text decoder leads to significant accuracy loss.
Instead of post-training quantization, we can use data free weights
compression to reduce the model footprint.

The optimization process contains the following steps:

1. Run ``nncf.compress_weights`` to get a model with compressed weights.
2. Serialize the ``OpenVINO`` model using ``openvino.save_model``
   function.

.. code:: ipython3

    TEXT_DECODER_OV_INT8 = Path(str(TEXT_DECODER_OV).replace(".xml", "_int8.xml"))
    
    text_decoder = core.read_model(TEXT_DECODER_OV)
    compressed_text_decoder = nncf.compress_weights(text_decoder)
    ov.save_model(compressed_text_decoder, str(TEXT_DECODER_OV_INT8))

Run optimized OpenVINO model 
----------------------------------------------------------------------

The steps for making predictions with the optimized OpenVINO BLIP model
are similar to the PyTorch model. Let us check the model result using
the same input data from the `first
notebook <233-blip-convert.ipynb>`__.

.. code:: ipython3

    q_ov_vision_model = ov.compile_model(VISION_MODEL_OV_INT8)
    q_ov_text_encoder = ov.compile_model(TEXT_ENCODER_OV_INT8)
    q_ov_text_decoder_with_past = ov.compile_model(TEXT_DECODER_OV_INT8)

.. code:: ipython3

    from functools import partial
    from transformers import BlipForQuestionAnswering
    from blip_model import OVBlipModel, text_decoder_forward
    
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    text_decoder = model.text_decoder
    text_decoder.eval()
    
    text_decoder.forward = partial(text_decoder_forward, ov_text_decoder_with_past=q_ov_text_decoder_with_past)
    int8_model = OVBlipModel(model.config, model.decoder_start_token_id, q_ov_vision_model, q_ov_text_encoder, text_decoder)

.. code:: ipython3

    from PIL import Image
    
    raw_image = Image.open("demo.jpg").convert('RGB')
    question = "how many dogs are in the picture?"
    # preprocess input data
    inputs = processor(raw_image, question, return_tensors="pt")

Image Captioning 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    from utils import visualize_results
    
    out = int8_model.generate_caption(inputs["pixel_values"], max_length=20)
    caption = processor.decode(out[0], skip_special_tokens=True)
    fig = visualize_results(raw_image, caption)



.. image:: 233-blip-optimize-with-output_files/233-blip-optimize-with-output_23_0.png


Question Answering 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    out = int8_model.generate_answer(**inputs, max_length=20)
    answer = processor.decode(out[0], skip_special_tokens=True)
    fig = visualize_results(raw_image, answer, question)



.. image:: 233-blip-optimize-with-output_files/233-blip-optimize-with-output_25_0.png


Compare file sizes 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    def calculate_compression_rate(ov_model_path):
        fp16_ir_model_size = Path(ov_model_path).with_suffix(".bin").stat().st_size / 1024
        int8_model_path = str(ov_model_path).replace(".xml", "_int8.xml")
        quantized_model_size = Path(int8_model_path).with_suffix(".bin").stat().st_size / 1024
        print(f'{ov_model_path.as_posix().split(".")[0]}')
        print(f"    * FP16 IR model size: {fp16_ir_model_size:.2f} KB")
        print(f"    * INT8 model size: {quantized_model_size:.2f} KB")
        print(f"    * Model compression rate: {fp16_ir_model_size / quantized_model_size:.3f}")

.. code:: ipython3

    for model_path in [VISION_MODEL_OV, TEXT_ENCODER_OV, TEXT_DECODER_OV]:
        calculate_compression_rate(model_path)


.. parsed-literal::

    blip_vision_model
        * FP16 IR model size: 168145.68 KB
        * INT8 model size: 84915.75 KB
        * Model compression rate: 1.980
    blip_text_encoder
        * FP16 IR model size: 268087.17 KB
        * INT8 model size: 134677.23 KB
        * Model compression rate: 1.991
    blip_text_decoder_with_past
        * FP16 IR model size: 269303.42 KB
        * INT8 model size: 135450.65 KB
        * Model compression rate: 1.988


Compare inference time of the FP16 and optimized models 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To measure the inference performance of the ``FP16`` and ``INT8``
models, we use median inference time on 100 samples of the calibration
dataset. So we can approximately estimate the speed up of the dynamic
quantized models.

   **NOTE**: For the most accurate performance estimation, it is
   recommended to run ``benchmark_app`` in a terminal/command prompt
   after closing other applications with static shapes.

.. code:: ipython3

    import time
    import torch
    
    def calculate_inference_time(blip_model, calibration_data, generate_caption):
        inference_time = []
        for inputs in calibration_data:
            pixel_values = torch.from_numpy(inputs["vision_model_inputs"]["pixel_values"])
            input_ids = torch.from_numpy(inputs["text_encoder_inputs"]["input_ids"])
            attention_mask = torch.from_numpy(inputs["text_encoder_inputs"]["attention_mask"])
    
            start = time.perf_counter()
            if generate_caption:
                _ = blip_model.generate_caption(pixel_values, max_length=20)
            else:
                _ = blip_model.generate_answer(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, max_length=20)
            end = time.perf_counter()
            delta = end - start
            inference_time.append(delta)
        return np.median(inference_time)

.. code:: ipython3

    fp_original_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    fp_text_decoder = fp_original_model.text_decoder
    fp_text_decoder.eval()
    
    comp_text_encoder = ov.compile_model(TEXT_ENCODER_OV)
    comp_text_decoder_with_past = ov.compile_model(TEXT_DECODER_OV)
    fp_text_decoder.forward = partial(text_decoder_forward, ov_text_decoder_with_past=comp_text_decoder_with_past)
    fp16_model = OVBlipModel(model.config, model.decoder_start_token_id, comp_vision_model, comp_text_encoder, fp_text_decoder)

.. code:: ipython3

    validation_data = calibration_data[:100]
    
    int8_caption_latency = calculate_inference_time(int8_model, validation_data, generate_caption=True)
    fp16_caption_latency = calculate_inference_time(fp16_model, validation_data, generate_caption=True)
    
    print(f"Image Captioning speed up: {fp16_caption_latency / int8_caption_latency:.3f}")

.. code:: ipython3

    int8_generate_answer_latency = calculate_inference_time(int8_model, validation_data, generate_caption=False)
    fp16_generate_answer_latency = calculate_inference_time(fp16_model, validation_data, generate_caption=False)
    print(f"Question Answering speed up: {fp16_generate_answer_latency / int8_generate_answer_latency:.3f}")
