.. {#llm_inference_native_API}

Inference with OpenVINO API
===============================

To run Generative AI models using native OpenVINO APIs you need to follow regular **Сonvert -> Optimize -> Deploy** path with a few simplifications.

To convert model from Hugging Face you can use Optimum-Intel export feature that allows to export model in OpenVINO format without invoking conversion API and tools directly, as it is shown above.
In this case, the conversion process is a bit more simplified. You can still use a regular conversion path if model comes from outside of Hugging Face ecosystem, i.e., in source framework format (PyTorch, etc.)

Model optimization can be performed within Hugging Face or directly using NNCF as described in the :doc:`weight compression guide <weight_compression>`.

Inference code that uses native API cannot benefit from Hugging Face pipelines. You need to write your custom code or take it from the available examples. Below are some examples of popular Generative AI scenarios:

* In case of LLMs for text generation, you need to handle tokenization, inference and token selection loop, and de-tokenization. If token selection involves beam search, it also needs to be written.
* For image generation models, you need to make a pipeline that includes several model inferences: inference for source (e.g., text) encoder models, inference loop for diffusion process and inference for decoding part. Scheduler code is also required.

To write such pipelines, you can follow the examples provided as part of OpenVINO:

* `llama2.openvino <https://github.com/OpenVINO-dev-contest/llama2.openvino>`__
* `LLM optimization by custom operation embedding for OpenVINO <https://github.com/luo-cheng2021/ov.cpu.llm.experimental>`__
* `C++ Implementation of Stable Diffusion <https://github.com/yangsu2022/OV_SD_CPP>`__

An inference pipeline for a text generation LLM is set up in the following stages:

1.	Read and compile the model.
2.	Tokenize text and set model inputs.
3.	Run token generation loop.
4.	De-tokenize outputs.


1. Read and Compile LLM
###################################

Models that have been converted to OpenVINO IR format can be read and compiled using
``ov.core.read_model`` and ``ov.core.compile_model``.

When reading the model, you need to get the key names for its input and output layers.
These key/value and input/output names will be used to apply inputs to the model and retrieve
its outputs. For more information, see Stateful Models.

The code below shows how to read an LLM in OpenVINO IR format, get a mapping of keys and values
for its input and output layers, and then compile it on the CPU.

.. code-block:: python

  from openvino.runtime import Core
  from pathlib import Path

  core = Core()
  ir_model_path = Path("ov_model") # your IR model folder path
  ir_model = ir_model_path / "openvino_model.xml" # your model file

  print(" --- reading model --- ")
  # Read the model and corresponding weights from file
  model = core.read_model(ir_model)

  # Get the names of the key-value inputs and outputs
  input_names = {key.get_any_name(): idx for idx, key in      enumerate(model.inputs)}
  output_names = {key.get_any_name(): idx for idx, key in      enumerate(model.outputs)}

  key_value_input_names = [key for key in input_names if "key_values" in key]
  key_value_output_names = [key for key in output_names if "present" in key]

  print(" --- compiling model --- ")
  # Compile model for the specified device
  request = core.compile_model(model=model, device_name="CPU").create_infer_request()



2. Tokenize Text and Setup Model Inputs
##########################################

Tokenization is a mandatory step in the process of generating text using LLMs. Tokenization
converts the input text into a sequence of tokens, which are essentially the format that the
model can understand and process. The input text string must be tokenized and set up in the
structure expected by the model before running inference.

The ``AutoTokenizer`` function from the Hugging Face Transformers library can be used to convert
the input text string into tokens, as shown below. The ``AutoTokenizer`` is simple to use, but
requires installing the Transformers dependencies in the environment. For a fully OpenVINO-based
option, see the Inference with OpenVINO Tokenizers section.

.. code-block:: python

  from transformers import AutoTokenizer

  model_path = "HuggingFaceH4/zephyr-7b-beta"
  tokenizer = AutoTokenizer.from_pretrained(model_path)

  input_text = "Your input text here"
  input_ids = tokenizer.encode(input_text, return_tensors="np")

To prepare the model inputs, the input prompt is defined and then tokenized with AutoTokenizer.
Once the prompt is tokenized, several other parameters need to be configured before running inference.
These parameters include attention masks, which help the model focus on relevant parts of the input,
and position IDs, which provide the model with the sequence order of tokens. Additionally, inputs
need to be formatted correctly according to the model's requirements, ensuring that the model
receives the data in a structure it can process effectively.

This code snippet shows how to tokenize the input prompt and configure the model’s inputs:

.. code-block:: python

  from transformers import AutoTokenizer

  # Load tokenizer
  print(" --- load tokenizer --- ")
  tokenizer = AutoTokenizer.from_pretrained("ov_model")
  eos_token_id = tokenizer.eos_token_id

  # Tokenize prompt and prepare initial input configuration
  inputs = tokenizer("your prompt", return_tensors="np", add_special_tokens=False)
  input_ids = inputs["input_ids"]
  attention_mask = inputs["attention_mask"]

  # Set positional encodings if the model requires them
  if "position_ids" in input_names:
          position_ids = np.arange(0, input_ids.shape[1], dtype=np.int64)
          position_ids = np.expand_dims(position_ids, axis=0)


3. Run Token Generation Loop
####################################

The core of text generation lies in the inference and token selection loop. In each iteration
of this loop, the model runs inference on the input sequence, generates and selects a new token,
and appends it to the existing sequence.

.. code-block:: python

  # Make an asynchronous request to the model and wait for the result
  request.start_async(inputs, share_inputs=True)
  request.wait()

  # Retrieve the logits and past_key_values from the model output
  logits = request.get_tensor("logits").data
  past_key_values = tuple(request.get_tensor(key).data for key in key_value_output_names)

In this phase, the model makes an asynchronous request and waits for the result. Then it retrieves
the logits and past_key_values from the model output.

**Token Selection**

The loop continues until it reaches a maximum sequence length or generates an end-of-sequence
token. Within each iteration, the model performs an inference based on the current sequence of
tokens, and a subsequent token is selected based on the model’s output. This selection can be
straightforward, like choosing the token with the highest probability, or involve more
sophisticated methods like beam search or sampling.

.. code-block:: python

  # Select the logits for the next token and process them
  next_token_logits = logits[:, cur_input_len - 1, :]
  next_token_scores = process_logits(len(input_ids[0]),
                                            next_token_logits, eos_token_id)
  next_tokens = np.argmax(next_token_scores, axis=-1)

Here, the logits for the next token are selected and processed. The model selects the next token
based on the highest score.

**Stopping criteria**

.. code-block:: python

  # Append the next token to the answer_tokens and update attention_mask
  answer_tokens = np.concatenate((answer_tokens, [next_tokens]), axis=-1)
  attention_mask = np.concatenate((attention_mask, [[1] * len(next_tokens)]), axis=-1)

  # Check if the maximum length or end-of-sequence token is reached
  if answer_length == max_sequence_length or next_tokens == eos_token_id:
      break

The stopping criteria involves appending the next token to the answer tokens and updating the
attention mask. The loop breaks if the maximum sequence length is reached or if an end-of-sequence
token is generated.

4. De-Tokenize Outputs
##########################

The final step in the process is de-tokenization, where the sequence of token IDs generated by
the model is converted back into human-readable text.
This step is essential for interpreting the model's output.

.. code-block:: python

  # Convert token IDs back to text
  output_text = tokenizer.decode(generated_sequence[0], skip_special_tokens=True)

