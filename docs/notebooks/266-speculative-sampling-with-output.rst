Text Generation via Speculative Sampling, KV Caching, and OpenVINO™
===================================================================

As model sizes grow, Generative AI implementations require significant
inference resources. This not only increases the cost per generation
from a prompt, but also increases the power consumption used to serve
such requests.

Inference optimizations for text generation are essential for reducing
costs and power consumption. When optimizing the inference process, the
amount of time and energy required to generate text can be significantly
reduced. This can lead to cost savings in terms of hardware and
software, as well as reduced power consumption. Additionally, inference
optimizations can help improve the accuracy of text generation as well
as the speed at which it can be generated. This can lead to an improved
user experience and increased efficiency in text-generation tasks. In
summary, inference optimizations for text generation are essential to
reduce costs and power consumption, while also improving the accuracy
and speed of text generation.

Another necessary condition is that the optimizations are compatible
with each other. That is, implementing a certain optimization should not
preclude other optimizations. There are several levels of optimizations
that can provide significant speedup without “bumping into each other”
in a way that will compromise overall efficiency.

For details on this method, please refer to the paper by Chen et al,
http://arxiv.org/abs/2302.01318. Additionally, there’s an interesting
proof of correctness of speculative sampling (showing that the original
distribution is preserved) by Leviathan et al,
http://arxiv.org/abs/2211.17192

Our blog article describing this implementation with OpenVino is
available at openvino.ai

**Table of contents:**

-  `Prerequisites <#prerequisites>`__

   -  `Select inference device <#select-inference-device>`__

-  `Create autoregressive and speculative forms of sampling with KV
   Cache
   support <#create-autoregressive-and-speculative-forms-of-sampling-with-kv-cache-support>`__

   -  `Setup imports <#setup-imports>`__
   -  `Prepare autoregressive
      sampling <#prepare-autoregressive-sampling>`__
   -  `Prepare speculative sampling <#prepare-speculative-sampling>`__

-  `Main generation function <#main-generation-function>`__

   -  `Download and Convert Model <#download-and-convert-model>`__

Prerequisites
-------------



First, we should install the `Hugging Face
Optimum <https://huggingface.co/docs/optimum/installation>`__ library
accelerated by OpenVINO integration. The Hugging Face Optimum Intel API
is a high-level API that enables us to convert and quantize models from
the Hugging Face Transformers library to the OpenVINO™ IR format. For
more details, refer to the `Hugging Face Optimum Intel
documentation <https://huggingface.co/docs/optimum/intel/inference>`__.

We will also need to install transformers (HuggingFace) and some other
useful modules.

.. code:: ipython3

    %pip install -q --upgrade pip
    %pip install -q --upgrade transformers torch gradio openvino accelerate onnx onnxruntime ipywidgets
    %pip install -q "git+https://github.com/huggingface/optimum-intel.git"

Select inference device
~~~~~~~~~~~~~~~~~~~~~~~



Select the device from dropdown list for running inference using
OpenVINO.

.. code:: ipython3

    import ipywidgets as widgets
    from openvino.runtime import Core
    
    core = Core()
    
    device = widgets.Dropdown(
        options=core.available_devices + ["AUTO"],
        value='CPU',
        description='Device:',
        disabled=False,
    )
    
    device

Create autoregressive and speculative forms of sampling with KV Cache support
-----------------------------------------------------------------------------



Text generation is often done in an autoregressive fashion. We will all
support a KV cache (aka Past Value Cache) in the code. Note that we are
using greedy sampling. We do not adjust other text generation parameters
(e.g. temperature) so keep this illustration of speculative sampling as
simple and understandable as possible.

Setup imports
~~~~~~~~~~~~~



.. code:: ipython3

    import time
    import numpy as np
    import torch
    import gradio as gr

Prepare autoregressive sampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    def max_fn(x):
        x_max = torch.where(x > 0, x, torch.zeros_like(x))
        return x_max / torch.sum(x_max)
    
    def autoregressive_sampling_with_pkv(x, model, N):
        n = len(x)
        T = n + N
        input = x
        past_kv = None
    
        while n < T:
            res = model(input, attention_mask=torch.ones(input.size(), dtype=torch.long), past_key_values=past_kv)
            model_out = torch.softmax(res.logits, dim=2)
            past_kv = res.past_key_values
            next_token = torch.reshape(torch.argmax(model_out[-1][-1]), (1, 1))
            x = torch.cat((x, next_token), dim=1)
            n += 1
            input = next_token
    
        return x

Prepare speculative sampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~



-  Step 1: With speculative sampling, we first generate K samples from
   the draft model (in an autoregressive manner).
-  Step 2: These are now candidates to examine using the target model
   (step 2) using a batch size of K.
-  Step 3: We now determine if the K candidates from the draft model are
   acceptable based on the logits generated from the target model in
   step 2.
-  Step 4: We can sample another token with no additional cost (assuming
   that all the candidates were accepted).

Regarding the acceptance criterion for step 3, we need to compare logits
from the target model and compare with the draft model. If the ratio is
high enough, it’s likely to be accepted (using a random number).

.. code:: ipython3

    def speculative_sampling_with_pkv(x, draft_model, target_model, N, K):
        n = x.size(1)
        T = n + N
        target_past_kv = None
        while n < T:
            # Step 1: autoregressive decode of K candidate tokens from
            # the draft model and get final p for this batch of candidates
            x_draft = None
            draft_past_kv = None
            x_draft_input = x
            p_cum = None
            for _ in range(K):
                res_draft = draft_model(x_draft_input, attention_mask=torch.ones(x_draft_input.size(), dtype=torch.long), past_key_values=draft_past_kv, use_cache=True)
                p = res_draft.logits
                p = torch.softmax(p, dim=2)
                draft_past_kv = res_draft.past_key_values
                next_token = torch.reshape(torch.argmax(p[-1][-1]), (1, 1))
                x_draft_input = next_token
                if p_cum is None:
                    p_cum = p[:, -1].unsqueeze(1)
                    x_draft = next_token
                else:
                    p_cum = torch.cat((p_cum, p), dim=1)
                    x_draft = torch.cat((x_draft, next_token), dim=1)
            # Step 2: target model forward passes on x_draft
            if target_past_kv is None:
                x_draft_target_input = torch.cat((x, x_draft), dim=1)
            else:
                x_draft_target_input = x_draft
    
            res = target_model(x_draft_target_input, attention_mask=torch.ones(x_draft_target_input.size(), dtype=torch.long), use_cache=False)
            q = res.logits
    
            target_new_past_kv = res.past_key_values
            # Step 3: append draft tokens based on acceptance-rejection criterion and resample a token on rejection
            all_accepted = True
            for k in range(K):
                j = x_draft[0][k].item()
    
                q_item = q[-1][k][j].detach().numpy()
                p_item = p_cum[-1][k][j].detach().numpy()
    
                if np.random.random() < min(1, (q_item / p_item)):  # accepted
                    x = torch.cat((x, torch.tensor(j).reshape(1,1)), dim=1)
                    n += 1
                else:                                               # rejected
                    q_p = max_fn(q[0][k] - p_cum[0][k])
                    resampled_output = torch.argmax(q_p)      
                    resampled_output = torch.reshape(resampled_output, (1,1))
                    x = torch.cat((x, resampled_output), dim=1)
                    n += 1
                    all_accepted = False
                    break
                
            target_past_kv = target_new_past_kv
            # Step 4: if all draft tokens were accepted, sample a final token
            if all_accepted:
                x = torch.cat((x, torch.reshape(torch.argmax(q[-1][-1]), (1,1))), dim=1)
                n += 1
    
        return x

Main generation function
------------------------



Download and Convert Model
~~~~~~~~~~~~~~~~~~~~~~~~~~



Optimum Intel can be used to load optimized models from the `Hugging
Face Hub <https://huggingface.co/docs/optimum/intel/hf.co/models>`__ and
create pipelines to run an inference with OpenVINO Runtime using Hugging
Face APIs. The Optimum Inference models are API compatible with Hugging
Face Transformers models. This means we just need to replace
``AutoModelForXxx`` class with the corresponding ``OVModelForXxx``
class.

Below is an example of the Dolly model

.. code:: diff

   -from transformers import AutoModelForCausalLM
   +from optimum.intel.openvino import OVModelForCausalLM
   from transformers import AutoTokenizer, pipeline

   model_id = "databricks/dolly-v2-3b"
   -model = AutoModelForCausalLM.from_pretrained(model_id)
   +model = OVModelForCausalLM.from_pretrained(model_id, from_transformers=True)

Model class initialization starts with calling ``from_pretrained``
method. When downloading and converting Transformers model, the
parameter ``from_transformers=True`` should be added. We can save the
converted model for the next usage with the ``save_pretrained`` method.
Tokenizer class and pipelines API are compatible with Optimum models.

.. code:: ipython3

    from pathlib import Path
    from transformers import AutoTokenizer
    from optimum.intel.openvino import OVModelForCausalLM
    
    #  If you are on a large system with lots of memory, you can run a larger model like DollyV2
    # draft_model_id = "databricks/dolly-v2-3b"
    # draft_model_path = Path("dolly-v2-3b")
    # target_model_id = "databricks/dolly-v2-12b"
    # target_model_path = Path("dolly-v2-12b")
    #  If you are on a system with limited memory, you can try the smaller GPT2 models
    draft_model_id = "gpt2"
    draft_model_path = Path("gpt2-local")
    target_model_id = "gpt2-xl"
    target_model_path = Path("gpt2-xl-local")
    
    target_tokenizer = AutoTokenizer.from_pretrained(target_model_id)
    
    current_device = device.value
    
    # Save local copies for subsequent runs
    if draft_model_path.exists():
        draft_ov_model = OVModelForCausalLM.from_pretrained(draft_model_path, device=current_device)
    else:
        draft_ov_model = OVModelForCausalLM.from_pretrained(draft_model_id, device=current_device, from_transformers=True)
        draft_ov_model.save_pretrained(draft_model_path)
    if target_model_path.exists():
        target_ov_model = OVModelForCausalLM.from_pretrained(target_model_path, device=current_device)
    else:
        target_ov_model = OVModelForCausalLM.from_pretrained(target_model_id, device=current_device, from_transformers=True)
        target_ov_model.save_pretrained(target_model_path)


.. code:: ipython3

    def main(
        prompt: str = "Explain the difference between fission and fusion",
        n_tokens_to_generate: int = 100,
        K: int = 5,
        seed: int = 5555,
    ):
        # seed numpy rng
        np.random.seed(seed)
        draft_model = draft_ov_model
        target_model = target_ov_model
        
    
        input_ids = target_tokenizer(prompt, return_tensors="pt")['input_ids']
    
        def run_autoregressive_sampling_fn(decode_fn, input_ids, **kwargs):
            start = time.perf_counter()
            output_ids = decode_fn(x=input_ids, **kwargs)
            text = target_tokenizer.decode(output_ids[0], skip_special_tokens=True)
            elapsed_time = time.perf_counter() - start
            return text, elapsed_time
    
        def run_speculative_sampling_fn(decode_fn, input_ids, **kwargs):
            start = time.perf_counter()
            output_ids = decode_fn(x=input_ids, **kwargs)
            text = target_tokenizer.decode(output_ids[0], skip_special_tokens=True)
            elapsed_time = time.perf_counter() - start
            return text, elapsed_time
    
        autoregressive_text, autoregressive_time = run_autoregressive_sampling_fn(
            autoregressive_sampling_with_pkv,
            input_ids,
            model=target_model,
            N=n_tokens_to_generate,
        )
    
        speculative_text, speculative_time = run_speculative_sampling_fn(
            speculative_sampling_with_pkv,
            input_ids,
            target_model=target_model,
            draft_model=draft_model,
            N=n_tokens_to_generate,
            K=K,
        )
    
    #   Format results for output in gradio
        out = "\n" + "Autoregressive Decode" + "\n" + "---------------------" + "\n"
        out = out + f"Time = {autoregressive_time:.2f}s" + "\n" + f"Text = {autoregressive_text}" + "\n"
        out = out + "\n" + "Speculative Decode" + "\n" + "------------------" + "\n"
        out = out + f"Time = {speculative_time:.2f}s" + "\n" + f"Text = {speculative_text}"
        return out
    
    if __name__ == "__main__":
        with gr.Blocks() as demo:
            gr.Markdown(
                """
                # Speculative Sampling Demo
                ## The output will show a comparison of Autoregressive Sampling vs Speculative Sampling
                - Target Model: Dolly V2 12B
                - Draft Model: Dolly V2 3B
                - K = 5
                > Some improvements can be made to acceptance criterion and adjusting temperature to improve text quality.
                """)
            with gr.Row():
                inp = gr.Textbox(placeholder="THIS CANNOT BE EMPTY", label="Input Prompt")
                out = gr.Textbox(label="Output")
            btn = gr.Button("Run")
            btn.click(fn=main, inputs=inp, outputs=out)
        demo.launch()
