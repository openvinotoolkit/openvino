#include <openvino/runtime/core.hpp>
#include <openvino/runtime/intel_gpu/properties.hpp>


int main() {
{
    //! [compile_model_default_gpu]
    ov::Core core;
    auto model = core.read_model("model.xml");
    auto compiled_model = core.compile_model(model, "GPU");
    //! [compile_model_default_gpu]
}

{
    //! [compile_model_gpu_with_id]
    ov::Core core;
    auto model = core.read_model("model.xml");
    auto compiled_model = core.compile_model(model, "GPU.1");
    //! [compile_model_gpu_with_id]
}

{
    //! [compile_model_gpu_with_id_and_tile]
    ov::Core core;
    auto model = core.read_model("model.xml");
    auto compiled_model = core.compile_model(model, "GPU.1.0");
    //! [compile_model_gpu_with_id_and_tile]
}

{
    //! [compile_model_multi]
    ov::Core core;
    auto model = core.read_model("model.xml");
    auto compiled_model = core.compile_model(model, "MULTI:GPU.1,GPU.0");
    //! [compile_model_multi]
}

{
    //! [compile_model_auto]
    ov::Core core;
    auto model = core.read_model("model.xml");
    auto compiled_model = core.compile_model(model, "AUTO:GPU.1,CPU.0", ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT));
    //! [compile_model_auto]
}

{
    //! [compile_model_batch_plugin]
    ov::Core core;
    auto model = core.read_model("model.xml");
    auto compiled_model = core.compile_model(model, "BATCH:GPU");
    //! [compile_model_batch_plugin]
}

{
    //! [compile_model_auto_batch]
    ov::Core core;
    auto model = core.read_model("model.xml");
    auto compiled_model = core.compile_model(model, "GPU", ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));
    //! [compile_model_auto_batch]
}
    return 0;
}
