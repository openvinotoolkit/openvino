import gc
import logging
import os
import shutil
import tempfile

import pytest
from optimum.intel.openvino import (OVModelForCausalLM,
                                    OVWeightQuantizationConfig)
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import whowhatbench as wwb
from whowhatbench.wwb import genai_gen_text
from whowhatbench.model_loaders import load_model
from openvino_tokenizers import convert_tokenizer
from openvino import save_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_IDS = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    # "Qwen/Qwen2-0.5B-Instruct",
]
DEVICES = [
    # "CPU",
    "GPU.1",
]
NUMBER_OF_SAMPLES = 1
METRIC_OF_INTEREST = "similarity"
PREC_INT8 = "INT8"
PREC_INT4 = "INT4"

REFERENCES = {
    "llama": {PREC_INT8: 0.95, PREC_INT4: 0.95},
    "qwen2": {PREC_INT8: 0.86, PREC_INT4: 0.82},
}
ACCURACY_THRESHOLDS = {
    PREC_INT8: 0.05,
    PREC_INT4: 0.05,
}

# XXX: need to roll back
# tmp_dir = tempfile.mkdtemp(dir=os.getcwd())
# logger.info(f"Created temporary directory: {tmp_dir}")
tmp_dir = os.path.join(os.getcwd(), "test_models_cache")
os.makedirs(tmp_dir, exist_ok=True)
logger.info(f"Using directory: {tmp_dir}")

def get_model_path(model_name, prec):
    """
    Returns the path to the model based on its name and precision.
    """
    return os.path.join(tmp_dir, f"{model_name}_{prec}")

def get_gt_path(model_name):
    """
    Returns the path to the ground truth data based on the model name.
    """
    return os.path.join(tmp_dir, f"{model_name}_gt.json")

def init_test_scope():
    test_scope = []

    for model_id in MODEL_IDS:
        logger.info(f"Downloading and quantizing model: {model_id}")
        model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model_type = model.config.model_type
        model_path = os.path.join(tmp_dir, model_type)
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        ov_tokenizer, ov_detokenizer = convert_tokenizer(tokenizer, with_detokenizer=True)

        int8_model_path = get_model_path(model_type, PREC_INT8)
        if not os.path.exists(int8_model_path):
            print(f'Saving int8 OpenVINO model: {int8_model_path}')
            ov_model = OVModelForCausalLM.from_pretrained(model_path, load_in_8bit=True)
            ov_model.save_pretrained(int8_model_path)
            tokenizer.save_pretrained(int8_model_path)
            del ov_model
            save_model(ov_tokenizer, os.path.join(int8_model_path, "openvino_tokenizer.xml"))
            save_model(ov_detokenizer, os.path.join(int8_model_path, "openvino_detokenizer.xml"))
        gc.collect()

        int4_model_path = get_model_path(model_type, PREC_INT4)
        if not os.path.exists(int4_model_path):
            print(f'Quantizing model to INT4: {model_type}')
            quantization_config = OVWeightQuantizationConfig(bits=4, ratio=0.8)
            quantized_model = OVModelForCausalLM.from_pretrained(
                model_path, quantization_config=quantization_config
            )
            quantized_model.save_pretrained(int4_model_path)
            tokenizer.save_pretrained(int4_model_path)
            del quantized_model
            save_model(ov_tokenizer, os.path.join(int4_model_path, "openvino_tokenizer.xml"))
            save_model(ov_detokenizer, os.path.join(int4_model_path, "openvino_detokenizer.xml"))
        gc.collect()

        set_seed(42)
        gt_path = get_gt_path(model_type)
        if not os.path.exists(gt_path):
            evaluator = wwb.Evaluator(
                base_model=model, tokenizer=tokenizer, num_samples=NUMBER_OF_SAMPLES, use_chat_template=True
            )
            print(f'{gt_path} does not exist, creating ground truth data...')
            evaluator.dump_gt(gt_path)
        [
            test_scope.append((model_type, PREC_INT8, device))
            for device in DEVICES
        ]
        [
            test_scope.append((model_type, PREC_INT4, device))
            for device in DEVICES
        ]

    return test_scope


# XXX: need to roll back
# def teardown_module():
    # logger.info(f"Deleting temporary directory: {tmp_dir}")
    # shutil.rmtree(tmp_dir)


test_scope = init_test_scope()

print(test_scope)

@pytest.mark.parametrize(
    ("model_type", "precision", "device"),
    test_scope,
)
def test_accuracy_conformance(model_type, precision, device):
    os.environ["OV_GPU_DYNAMIC_QUANTIZATION_THRESHOLD"] = "1"

    task        = 'text'
    ov_config   = None
    hf          = None
    use_genai   = True
    dont_use_llamacpp = False
    model_path = get_model_path(model_type, precision)
    gt_data = get_gt_path(model_type)
    print(f"Testing model: {model_path}, type: {model_type}, precision: {precision}, device: {device}")
    target_model = load_model(task, model_path, device, ov_config, hf, use_genai, dont_use_llamacpp)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    use_chat_template = (tokenizer is not None and tokenizer.chat_template is not None)

    evaluator = wwb.Evaluator(
        base_model=None,
        gt_data=gt_data,
        tokenizer=tokenizer,
        similarity_model_id='sentence-transformers/all-mpnet-base-v2',
        num_samples=NUMBER_OF_SAMPLES,
        gen_answer_fn=genai_gen_text,
        use_chat_template=use_chat_template,
    )

    set_seed(42)
    _, all_metrics = evaluator.score(target_model, evaluator.get_generation_fn())
    metric = all_metrics[METRIC_OF_INTEREST].values[0]
    evaluator.dump_predictions(os.path.join(tmp_dir, "target.csv"))
    abs_metric_diff = abs(REFERENCES[model_type][precision] - metric)
    print(metric, REFERENCES[model_type][precision], model_type, precision)
    assert abs_metric_diff <= ACCURACY_THRESHOLDS[precision]
