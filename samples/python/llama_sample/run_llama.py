import argparse
import json
import os
import time
import numpy as np
from openvino.runtime import Core, Tensor
import openvino.runtime as ov
from transformers import AutoTokenizer

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    summation = e_x.sum(axis=-1, keepdims=True)
    return e_x / summation

def process_logits(cur_length, scores, eos_token_id, min_length=0):
    """
    reduce probability for padded indicies

    Parameters:
      cur_length - current length of input sequence
      scores - model output logits
      eos_token_id - index of end of string token in model vocab
      min_length - minimum length for appling postprocessing
    """
    if cur_length < min_length:
        scores[:, eos_token_id] = -float("inf")
    return scores


def get_top_k_logits(scores, top_k):
    """
    perform top-k sampling

    Parameters:
      scores - model output logits
      top_k - number of elements with highest probability to select
    """
    filter_value = -float("inf")
    top_k = min(max(top_k, 1), scores.shape[-1])
    top_k_scores = -np.sort(-scores)[:, :top_k]
    indices_to_remove = scores < np.min(top_k_scores)
    filtred_scores = np.ma.array(scores, mask=indices_to_remove,
                                 fill_value=filter_value).filled()
    return filtred_scores

def prepare_next_input(next_tokens, is_merged_model, outputs, outputs_names):
    next_input = {"input_ids": np.array([next_tokens]), "attention_mask": np.array([[1]])}
    if is_merged_model == True:
        next_input["use_cache_branch"] = np.array([True])
    for i, output_name in enumerate(outputs_names):
        if output_name == "logits":
            pass
        else:
            past_output_name = "past_key_values.{}".format(output_name[8:])
            next_input[past_output_name] = outputs[output_name]

    return next_input

def generate_sequence(input_ids, attention_mask, eos_token_id, n_head, max_sequence_length=128,
                      dynamic_shapes=True, args=None):
    """
    text prediction cycle.

    Parameters:
      input_ids: tokenized input ids for model
      attention_mask: attention mask for model
      max_sequence_length: maximum sequence length for stop iteration
      eos_token_ids: end of sequence index from vocab
      dynamic_shapes: use dynamic shapes for inference or pad model input to max_sequece_length
    Returns:
      predicted token ids sequence
    """
    first_iteration = True
    next_input = {}
    is_merged_model = False
    output_names = []

    inputs_ports = compiled_model.inputs
    input_names = [port.get_any_name() for port in inputs_ports]
    np.random.seed(args.seed)
    while True:
        cur_input_len = len(input_ids[0])
        if not dynamic_shapes:
            pad_len = max_sequence_length - cur_input_len
            model_input_ids = np.concatenate((input_ids, [[eos_token_id] * pad_len]), axis=-1)
            model_input_attention_mask = np.concatenate((attention_mask, [[0] * pad_len]), axis=-1)
        else:
            model_input_ids = input_ids
            model_input_attention_mask = attention_mask

        if first_iteration:
            first_input = {"input_ids": model_input_ids,
                           "attention_mask": model_input_attention_mask}
            if "use_cache_branch" in input_names:
                is_merged_model = True
                first_input["use_cache_branch"] = np.array([False])
            outputs_ports = compiled_model.outputs
            output_names = [port.get_any_name() for port in outputs_ports]
            for i, input_name in enumerate(input_names) :
                if "past_key_values" in input_name:
                    past_key_values_array = np.zeros([
                        1, n_head, 0, 128
                    ]).astype(np.float32)
                    first_input[input_name] = Tensor(past_key_values_array[:, :, :, :])

            outputs = compiled_model(first_input)
            logits = outputs['logits']
            next_token_logits = logits[:, cur_input_len - 1, :]
            first_iteration=False
        else:
            outputs = compiled_model(next_input)
            logits = outputs['logits']
            next_token_logits = logits[0]
        # pre-process distribution
        next_token_scores = process_logits(cur_input_len,
                                            next_token_logits, eos_token_id)
        top_k = 50
        next_token_scores = get_top_k_logits(next_token_scores, top_k)
        # get next token id
        probs = softmax(next_token_scores)
        next_tokens = np.ones(1, dtype=int) * np.argmax(probs[0], axis=0)
        # break the loop if max length or end of text token is reached
        if cur_input_len == max_sequence_length or next_tokens == eos_token_id:
            break
        else:
            input_ids = np.concatenate((input_ids, [next_tokens]), axis=-1)
            next_input = prepare_next_input(next_tokens, is_merged_model, outputs, output_names)

    return input_ids

def question_answer(text, args):
    print("Input text: ", text)
    print("Start generate sequence ...")
    model_path = os.path.dirname(args.ir)
    print(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    hf_config_path = os.path.join(model_path, "config.json")
    hf_config = json.load(open(hf_config_path))
    n_head  = hf_config["num_attention_heads"]
    t1 = time.time()
    inputs = tokenizer(text, return_tensors="np")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    print(f"Input tokenizer time: {time.time()-t1}")
    t1 = time.time()
    output_ids = generate_sequence(input_ids, attention_mask, eos_token_id=tokenizer.eos_token_id, n_head = n_head,
                                   max_sequence_length=128, dynamic_shapes=True, args=args)
    print(f"Infer time: {time.time()-t1}")
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('-p','--prompt', type=str, default="What is OpenVINO?", required=False,
                         help="Specify input prompt. Default is 'What is OpenVINO'")
    parser.add_argument('-d','--device', type=str, default="CPU", required=False,
                         help="Specify target device. Default is CPU")
    parser.add_argument('--ir', type=str, default="ov_model/llama-7b.xml", required=False,
                         help="Specify OpenVino IR")
    parser.add_argument('-c', '--cache_dir', type=str, required=False, help="OV Cache DIR")
    parser.add_argument('-s', '--seed', type=int, default=42, required=False, 
                        help="Specific random seed to generate fix result. Default 42, -1:means do not use seed.")
    # Parse the argument
    args = parser.parse_args()

    # initialize openvino core
    core = Core()
    print(f"Init OpenVINO model: {args.ir}")
    print("OpenVINO version:", ov.get_version())
    if args.cache_dir != None :
        core.set_property({"CACHE_DIR": args.cache_dir})
    # read the model and corresponding weights from file
    t1 = time.time()
    ov_model = core.read_model(args.ir)
    print(f"Load model time: {time.time()-t1}")
    t1 = time.time()
    ov_config = {'PERFORMANCE_HINT': 'LATENCY', 'NUM_STREAMS':1, "CACHE_DIR":""}
    compiled_model = core.compile_model(ov_model, args.device, ov_config)
    print(f"Compile model time: {time.time()-t1}")
    out_text = question_answer(args.prompt, args)
    print(out_text)