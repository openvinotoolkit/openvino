#include <openvino/runtime/core.hpp>


int main() {
    {
        //! [compile_model_default]
        ov::Core core;
        auto model = core.read_model("model.xml");
        auto compiled_model = core.compile_model(model, "CPU");
        //! [compile_model_default]
    }

    {
        //! [compile_model_multi]
        ov::Core core;
        auto model = core.read_model("model.xml");
        auto compiled_model = core.compile_model(model, "MULTI:CPU,GPU.0");
        //! [compile_model_multi]
    }

    {
        //! [compile_model_auto]
        ov::Core core;
        auto model = core.read_model("model.xml");
        auto compiled_model = core.compile_model(model, "AUTO:CPU,GPU.0", ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT));
        //! [compile_model_auto]
    }
}
