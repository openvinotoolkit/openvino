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
def add_test_case(catalog, model, gpu_int8_ref, gpu_int4_ref, cpu_int8_ref, cpu_int4_ref, threshold = 0.03):
    catalog[model] = {
        "GPU": {
            "INT8": {"reference": gpu_int8_ref, "threshold": threshold},
            "INT4": {"reference": gpu_int4_ref, "threshold": threshold},
        },
        "CPU": {
            "INT8": {"reference": cpu_int8_ref, "threshold": threshold},
            "INT4": {"reference": cpu_int4_ref, "threshold": threshold},
        },
    }

TEST_CATALOG = {}
NOTEST=0.0
#                           NAME,                                   GPU_i8, GPU_i4, CPU_i8, CPU_i4
add_test_case(TEST_CATALOG, "TinyLlama/TinyLlama-1.1B-Chat-v1.0",   0.98, 0.71, 0.94,   0.77)
add_test_case(TEST_CATALOG, "Qwen/Qwen2-0.5B-Instruct",             0.86, 0.74, 0.82,   0.68)

# Extract configuration from catalog
MODEL_IDS = list(TEST_CATALOG.keys())
DOWNLOADED_MODELS = set()
DEVICES = list(set(device for model_config in TEST_CATALOG.values()
                  for device in model_config.keys()))

METRIC_OF_INTEREST = "similarity"
PREC_INT8 = "INT8"
PREC_INT4 = "INT4"
# Get configuration from environment variables (set by pytest options)
NUMBER_OF_SAMPLES = int(os.environ.get("PYTEST_SAMPLES", "15"))
GPU_SUFFIX = int(os.environ.get("PYTEST_GPU_SUFFIX", "0"))
CLEANUP_AFTER_TEST = os.environ.get("PYTEST_DO_NOT_CLEANUP", "false").lower() == "false"

# Log current configuration
logger.info(f"Configuration: SAMPLES={NUMBER_OF_SAMPLES}, GPU_SUFFIX={GPU_SUFFIX}, CLEANUP_AFTER_TEST={CLEANUP_AFTER_TEST}")

def get_reference(model_id, device, precision):
    """Get reference value from catalog"""
    return TEST_CATALOG[model_id][device][precision]["reference"]

def get_threshold(model_id, device, precision):
    """Get threshold value from catalog"""
    return TEST_CATALOG[model_id][device][precision]["threshold"]

def get_tmp_dir():
    """Get temporary directory based on cleanup preference"""
    # Check environment variable set by pytest option
    cleanup_after_test = CLEANUP_AFTER_TEST

    if cleanup_after_test:
        # Use temp directory when cleanup is enabled
        tmp_dir = tempfile.mkdtemp(dir=os.getcwd())
    else:
        # Use fixed directory by default (could use tempfile.mkdtemp for true temporary)
        tmp_dir = os.path.join(os.getcwd(), "test_models_cache")
        os.makedirs(tmp_dir, exist_ok=True)
    logger.info(f"Using directory: {tmp_dir}")
    return tmp_dir

def get_model_path(model_id, prec):
    """
    Returns the path to the model based on its id and precision.
    """
    return os.path.join(tmp_dir, f"{model_id.replace('/', '_')}_{prec}")

def get_gt_path(model_id, use_chat_template):
    """
    Returns the path to the ground truth data based on the model name.
    """
    return os.path.join(tmp_dir, f"{model_id.replace('/', '_')}_gt_{NUMBER_OF_SAMPLES}_{'chat_template' if use_chat_template else 'no-chat-template'}.json")

def setup_model(model_id):
    """
    Download and prepare models (original, INT8, INT4) and ground truth data.
    Only downloads if not already present.
    """
    if model_id in DOWNLOADED_MODELS:
        logger.info(f"Model {model_id} already prepared, skipping setup")
        return

    logger.info(f"Setting up model: {model_id}")

    # Download original model
    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Save original model
    model_path = get_model_path(model_id, "org")
    if not os.path.exists(model_path):
        logger.info(f"Saving original model: {model_path}")
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)

    # Convert tokenizer for OpenVINO
    ov_tokenizer, ov_detokenizer = convert_tokenizer(tokenizer, with_detokenizer=True)

    # Prepare INT8 model
    int8_model_path = get_model_path(model_id, PREC_INT8)
    if not os.path.exists(int8_model_path):
        logger.info(f'Creating INT8 OpenVINO model: {int8_model_path}')
        ov_model = OVModelForCausalLM.from_pretrained(model_path, load_in_8bit=True)
        ov_model.save_pretrained(int8_model_path)
        tokenizer.save_pretrained(int8_model_path)
        del ov_model
        save_model(ov_tokenizer, os.path.join(int8_model_path, "openvino_tokenizer.xml"))
        save_model(ov_detokenizer, os.path.join(int8_model_path, "openvino_detokenizer.xml"))
    gc.collect()

    # Prepare INT4 model
    int4_model_path = get_model_path(model_id, PREC_INT4)
    if not os.path.exists(int4_model_path):
        logger.info(f'Creating INT4 OpenVINO model: {int4_model_path}')
        quantization_config = OVWeightQuantizationConfig(bits=4, ratio=0.8)
        quantized_model = OVModelForCausalLM.from_pretrained(model_path, quantization_config=quantization_config)
        quantized_model.save_pretrained(int4_model_path)
        tokenizer.save_pretrained(int4_model_path)
        del quantized_model
        save_model(ov_tokenizer, os.path.join(int4_model_path, "openvino_tokenizer.xml"))
        save_model(ov_detokenizer, os.path.join(int4_model_path, "openvino_detokenizer.xml"))
    gc.collect()

    # Prepare ground truth data
    set_seed(42)
    use_chat_template = False # (tokenizer is not None and tokenizer.chat_template is not None)
    gt_path = get_gt_path(model_id, use_chat_template)
    if not os.path.exists(gt_path):
        logger.info(f'Creating ground truth data: {gt_path}')
        evaluator = wwb.Evaluator(
            base_model=model,
            tokenizer=tokenizer,
            num_samples=NUMBER_OF_SAMPLES,
            use_chat_template=use_chat_template
        )
        evaluator.dump_gt(gt_path)

    # Mark model as downloaded
    DOWNLOADED_MODELS.add(model_id)
    logger.info(f"Model setup completed: {model_id}")

def init_test_scope():
    """
    Initialize test scope with all model/device/precision combinations.
    Model setup is handled separately by setup_model().
    """
    test_scope = []

    for model_id in MODEL_IDS:
        # Generate test cases for all device/precision combinations in catalog
        for device in TEST_CATALOG[model_id].keys():
            for precision in TEST_CATALOG[model_id][device].keys():
                test_scope.append((model_id, precision, device))

    return test_scope

def teardown_module():
    """Clean up temporary directory based on cleanup option"""
    cleanup_after_test = CLEANUP_AFTER_TEST

    if cleanup_after_test:
        logger.info(f"Deleting temporary directory: {tmp_dir}")
        if os.path.exists(tmp_dir):
            # removing directory may fail because of mmap of the model
            shutil.rmtree(tmp_dir, ignore_errors=True)
            logger.info(f"Deleted(may have failed): {tmp_dir}")
        else:
            logger.info(f"Directory already removed: {tmp_dir}")
    else:
        logger.info(f"Cleanup disabled - preserving directory: {tmp_dir}")

tmp_dir = get_tmp_dir()

test_scope = init_test_scope()

@pytest.mark.parametrize(
    ("model_id", "precision", "device"),
    test_scope,
)
def test_accuracy_conformance(model_id, precision, device):
    # Get expected values from catalog (use original device for lookup)
    expected_reference = get_reference(model_id, device, precision)
    if expected_reference == NOTEST:
        pytest.xfail(f'Test is skipped for {model_id}, {precision}, {device}. Ticket 172236')
        return

    # Ensure model is set up
    if model_id not in DOWNLOADED_MODELS:
        logger.info(f"Model {model_id} not found in downloaded models, setting up now...")
        setup_model(model_id)

    # os.environ["OV_GPU_DYNAMIC_QUANTIZATION_THRESHOLD"] = "1"

    task        = 'text'
    ov_config   = None
    hf          = None
    use_genai   = True
    dont_use_llamacpp = False
    model_path = get_model_path(model_id, precision)
    actual_device = f'{device}.{GPU_SUFFIX}' if GPU_SUFFIX and device == "GPU" else device
    print(f"Testing model: {model_path}, precision: {precision}, device: {actual_device}")
    target_model = load_model(task, model_path, actual_device, ov_config, hf, use_genai, dont_use_llamacpp)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    use_chat_template = False # (tokenizer is not None and tokenizer.chat_template is not None)

    gt_data = get_gt_path(model_id, use_chat_template)
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
    evaluator.dump_predictions(os.path.join(tmp_dir, f"{get_model_path(model_id, precision)}_{actual_device}_target.csv"))

    threshold = get_threshold(model_id, device, precision)

    abs_metric_diff = abs(expected_reference - metric)
    print(f"Metric: {metric}, Expected: {expected_reference}, Model: {model_id}, Precision: {precision}")
    assert abs_metric_diff <= threshold, f"Metric difference {abs_metric_diff} exceeds threshold {threshold}"
