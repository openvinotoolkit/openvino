import openvino as ov
import openvino as properties
from utils import get_model

def main():
    model = get_model()
    core = ov.Core()

    #! [set_manual_affinities]
    for op in model.get_ops():
        rt_info = op.get_rt_info()
        rt_info["affinity"] = "CPU"
    #! [set_manual_affinities]

    if "GPU" not in core.available_devices:
        return 0

    #! [fix_automatic_affinities]
    # This example demonstrates how to perform default affinity initialization and then
    # correct affinity manually for some layers
    device = "HETERO:GPU,CPU"

    # query_model result contains mapping of supported operations to devices
    supported_ops = core.query_model(model, device)

    # update default affinities manually for specific operations
    supported_ops["operation_name"] = "CPU"

    # set affinities to a model
    for node in model.get_ops():
        affinity = supported_ops[node.get_friendly_name()]
        node.get_rt_info()["affinity"] = "CPU"

    # load model with manually set affinities
    compiled_model = core.compile_model(model, device)
    #! [fix_automatic_affinities]

    #! [compile_model]
    import openvino.device as device

    compiled_model = core.compile_model(model, device_name="HETERO:GPU,CPU")
    # device priorities via configuration property
    compiled_model = core.compile_model(
        model, device_name="HETERO", config={device.priorities: "GPU,CPU"}
    )
    #! [compile_model]

    #! [configure_fallback_devices]
    import openvino.hint as hints

    core.set_property("HETERO", {device.priorities: "GPU,CPU"})
    core.set_property("GPU", {properties.enable_profiling: True})
    core.set_property("CPU", {hints.inference_precision: ov.Type.f32})
    compiled_model = core.compile_model(model=model, device_name="HETERO")
    #! [configure_fallback_devices]

    #! [set_pipeline_parallelism]
    import openvino.properties.hint as hints

    compiled_model = core.compile_model(
        model,
        device_name="HETERO:GPU.1,GPU.2",
        config={
            hints.model_distribution_policy:
            "PIPELINE_PARALLEL"
        })
    #! [set_pipeline_parallelism]
