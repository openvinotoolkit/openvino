import glob
import os
from e2e_oss.plugins.first_inference_tests.conftest import redefine_mo_out_path, get_non_infer_config
from e2e_oss._utils.test_utils import check_mo_precision
from utils.e2e.common.pipeline import Pipeline


def get_executable_cmd(args: dict):
    """Generate common part of cmd from arguments to execute"""
    reshape_shapes = args.get("reshape_shapes")
    return [
        str(args["executable"].resolve(strict=True)),
        f"-m={args.get('model')}",
        f"-d={args.get('device')}",
        f'-reshape_shapes="{reshape_shapes}"' if reshape_shapes else ''
    ]


def get_reformat_shapes(unformat_shapes: dict):
    """Convert shapes into string for timetest_infer/memtest_infer usage"""
    format_shapes = []
    for key, shapes in unformat_shapes.items():
        new_shapes = []
        for item_shape in shapes:
            new_shapes.append(f"{item_shape[0]}..{item_shape[1]}" if isinstance(item_shape, list) else f"{item_shape}")
        shape = ", ".join(new_shapes)
        format_shapes.append(f"{key}*{shape}")
    return "&".join(format_shapes)


def get_redef_ir(instance, test_name, log):
    instance_ie_pipeline = instance.ie_pipeline
    check_mo_precision(instance_ie_pipeline)

    try:
        redef_ir_path = redefine_mo_out_path(instance, instance_ie_pipeline)
        ie_pipeline = Pipeline(get_non_infer_config(instance_ie_pipeline))
        log.info('Executing MO pipeline for {}'.format(test_name))
        ie_pipeline.run()
    except Exception as err:
        raise Exception("MO pipeline failed") from err

    # First inference part
    ir_path = glob.glob(os.path.join(redef_ir_path, '*.xml'))[0]

    return ir_path


def deviation(reference: float, current: float) -> float:
    """
    Returns absolute deviation between two values
    """
    assert current != 0, "Reference must not be 0"
    return round(abs(reference/current), 3)


def get_trend(ratio: float, bottom: float, upp: float, memory_test=True) -> str:
    """
    Gets trend of ratio depend on thresholds boundaries [bottom, upp] and test type.
    In case first memory inference dynamism test, the flag memory_test is set to True
    if first time inference dynamism test the flag set to False
    """
    deg_branch = ratio <= bottom if memory_test else ratio >= upp
    if bottom <= ratio <= upp:
        ret = "Stable"
    elif deg_branch:
        ret = "Degradation"
    else:
        ret = "Improvement"
    return ret