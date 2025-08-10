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

# Test Configuration Catalog
TEST_CATALOG = {
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {
        "GPU": {
            "INT8": {"reference": 0.98, "threshold": 0.03},
            "INT4": {"reference": 0.90, "threshold": 0.03},
        },
        "CPU": {
            "INT8": {"reference": 0.94, "threshold": 0.03},
            "INT4": {"reference": 0.88, "threshold": 0.03},
        },
    },
    "Qwen/Qwen2-0.5B-Instruct": {
        "GPU": {
            "INT8": {"reference": 0.96, "threshold": 0.03},
            "INT4": {"reference": 0.72, "threshold": 0.03},
        },
        "CPU": {
            "INT8": {"reference": 0.91, "threshold": 0.03},
            "INT4": {"reference": 0.73, "threshold": 0.03},
        },
    },
}

# Extract configuration from catalog
MODEL_IDS = list(TEST_CATALOG.keys())
DEVICES = list(set(device for model_config in TEST_CATALOG.values() 
                  for device in model_config.keys()))

NUMBER_OF_SAMPLES = 15
METRIC_OF_INTEREST = "similarity"
PREC_INT8 = "INT8"
PREC_INT4 = "INT4"
GPU_SUFFIX = 0   # Suffix to append to device name (e.g., '.1' for GPU.1). To be replaced by option
DO_NOT_CLEANUP = False

def get_reference(model_id, device, precision):
    """Get reference value from catalog"""
    return TEST_CATALOG[model_id][device][precision]["reference"]

def get_threshold(model_id, device, precision):
    """Get threshold value from catalog"""
    return TEST_CATALOG[model_id][device][precision]["threshold"]

def get_tmp_dir():
    """Get temporary directory based on cleanup preference"""
    # Check environment variable set by pytest option
    do_not_cleanup = DO_NOT_CLEANUP
    
    if do_not_cleanup:
        # Use fixed directory by default (could use tempfile.mkdtemp for true temporary)
        tmp_dir = os.path.join(os.getcwd(), "test_models_cache")
        os.makedirs(tmp_dir, exist_ok=True)
    else:
        # Use temp directory when cleanup is disabled
        tmp_dir = tempfile.mkdtemp(dir=os.getcwd())
    logger.info(f"Using directory: {tmp_dir}")
    return tmp_dir

tmp_dir = get_tmp_dir()

def get_model_path(model_id, prec):
    """
    Returns the path to the model based on its id and precision.
    """
    return os.path.join(tmp_dir, f"{model_id.replace('/', '_')}_{prec}")

def get_gt_path(model_id):
    """
    Returns the path to the ground truth data based on the model name.
    """
    return os.path.join(tmp_dir, f"{model_id.replace('/', '_')}_gt_{NUMBER_OF_SAMPLES}.json")

def init_test_scope():
    test_scope = []

    for model_id in MODEL_IDS:
        logger.info(f"Downloading and quantizing model: {model_id}")
        model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model_path = get_model_path(model_id, "org")
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        ov_tokenizer, ov_detokenizer = convert_tokenizer(tokenizer, with_detokenizer=True)

        int8_model_path = get_model_path(model_id, PREC_INT8)
        if not os.path.exists(int8_model_path):
            logger.info(f'Saving int8 OpenVINO model: {int8_model_path}')
            ov_model = OVModelForCausalLM.from_pretrained(model_path, load_in_8bit=True)
            ov_model.save_pretrained(int8_model_path)
            tokenizer.save_pretrained(int8_model_path)
            del ov_model
            save_model(ov_tokenizer, os.path.join(int8_model_path, "openvino_tokenizer.xml"))
            save_model(ov_detokenizer, os.path.join(int8_model_path, "openvino_detokenizer.xml"))
        gc.collect()

        int4_model_path = get_model_path(model_id, PREC_INT4)
        if not os.path.exists(int4_model_path):
            logger.info(f'Quantizing model to INT4: {int4_model_path}')
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
        gt_path = get_gt_path(model_id)
        if not os.path.exists(gt_path):
            evaluator = wwb.Evaluator(
                base_model=model, tokenizer=tokenizer, num_samples=NUMBER_OF_SAMPLES, use_chat_template=True
            )
            logger.info(f'{gt_path} does not exist, creating ground truth data...')
            evaluator.dump_gt(gt_path)

        # Generate test cases for all device/precision combinations in catalog
        for device in TEST_CATALOG[model_id].keys():
            for precision in TEST_CATALOG[model_id][device].keys():
                test_scope.append((model_id, precision, device))

    return test_scope


def teardown_module():
    """Clean up temporary directory based on cleanup option"""
    do_not_cleanup = DO_NOT_CLEANUP   # to be replaced by pytest option

    if do_not_cleanup:
        logger.info(f"Cleanup disabled - preserving directory: {tmp_dir}")
    else:
        logger.info(f"Deleting temporary directory: {tmp_dir}")
        shutil.rmtree(tmp_dir)


test_scope = init_test_scope()

@pytest.mark.parametrize(
    ("model_id", "precision", "device"),
    test_scope,
)
def test_accuracy_conformance(model_id, precision, device):
    # os.environ["OV_GPU_DYNAMIC_QUANTIZATION_THRESHOLD"] = "1"

    task        = 'text'
    ov_config   = None
    hf          = None
    use_genai   = True
    dont_use_llamacpp = False
    model_path = get_model_path(model_id, precision)
    gt_data = get_gt_path(model_id)
    actual_device = f'{device}.{GPU_SUFFIX}' if GPU_SUFFIX and device == "GPU" else device
    print(f"Testing model: {model_path}, precision: {precision}, device: {actual_device}")
    target_model = load_model(task, model_path, actual_device, ov_config, hf, use_genai, dont_use_llamacpp)

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
    evaluator.dump_predictions(os.path.join(tmp_dir, f"{get_model_path(model_id, precision)}_{actual_device}target.csv"))

    # Get expected values from catalog (use original device for lookup)
    expected_reference = get_reference(model_id, device, precision)
    threshold = get_threshold(model_id, device, precision)

    abs_metric_diff = abs(expected_reference - metric)
    print(f"Metric: {metric}, Expected: {expected_reference}, Model: {model_id}, Precision: {precision}")
    assert abs_metric_diff <= threshold, f"Metric difference {abs_metric_diff} exceeds threshold {threshold}"
