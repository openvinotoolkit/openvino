import gc
import logging
import os
import shutil
import tempfile

import pytest
from optimum.intel.openvino import (OVModelForCausalLM,
                                    OVWeightQuantizationConfig)
import whowhatbench as wwb
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_IDS = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Qwen/Qwen2-0.5B-Instruct",
]
DEVICES = [
    "CPU",
    "GPU",
]
NUMBER_OF_SAMPLES = 15
METRIC_OF_INTEREST = "similarity"

REFERENCES = {
    "llama": {"INT8": 0.95, "INT4": 0.95},
    "qwen2": {"INT8": 0.86, "INT4": 0.82},
}
ACCURACY_THRESHOLDS = {
    "INT8": 0.05,
    "INT4": 0.05,
}

tmp_dir = tempfile.mkdtemp(dir=os.getcwd())
logger.info(f"Created temporary directory: {tmp_dir}")

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

        ov_model = OVModelForCausalLM.from_pretrained(model_path, load_in_8bit=True)
        ov_model_path = os.path.join(tmp_dir, model_type + "_ov")
        ov_model.save_pretrained(ov_model_path)
        tokenizer.save_pretrained(ov_model_path)
        del ov_model
        gc.collect()

        quantization_config = OVWeightQuantizationConfig(bits=4, ratio=0.5)
        quantized_model = OVModelForCausalLM.from_pretrained(
            model_path, quantization_config=quantization_config
        )
        quantized_model_path = os.path.join(tmp_dir, model_type + "_ov_int4")
        quantized_model.save_pretrained(quantized_model_path)
        tokenizer.save_pretrained(quantized_model_path)
        del quantized_model
        gc.collect()

        set_seed(42)
        evaluator = wwb.Evaluator(
            base_model=model, tokenizer=tokenizer, num_samples=NUMBER_OF_SAMPLES
        )
        gt_path = os.path.join(tmp_dir, model_type + "_gt.json")
        evaluator.dump_gt(gt_path)
        [
            test_scope.append((ov_model_path, model_type, "INT8", gt_path, device))
            for device in DEVICES
        ]
        [
            test_scope.append((ov_model_path, model_type, "INT4", gt_path, device))
            for device in DEVICES
        ]

    return test_scope


def teardown_module():
    logger.info(f"Deleting temporary directory: {tmp_dir}")
    shutil.rmtree(tmp_dir)


test_scope = init_test_scope()


@pytest.mark.parametrize(
    ("model_path", "model_type", "precision", "gt_data", "device"),
    test_scope,
)
def test_accuracy_conformance(model_path, model_type, precision, gt_data, device):
    target_model = OVModelForCausalLM.from_pretrained(model_path, device=device, ov_config={"KV_CACHE_PRECISION": "f16"})
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    evaluator = wwb.Evaluator(
        base_model=None,
        tokenizer=tokenizer,
        gt_data=gt_data,
        num_samples=NUMBER_OF_SAMPLES,
    )

    set_seed(42)
    _, all_metrics = evaluator.score(target_model)
    metric = all_metrics[METRIC_OF_INTEREST].values[0]
    abs_metric_diff = abs(REFERENCES[model_type][precision] - metric)
    print(metric, REFERENCES[model_type][precision], model_type, precision)
    assert abs_metric_diff <= ACCURACY_THRESHOLDS[precision]
