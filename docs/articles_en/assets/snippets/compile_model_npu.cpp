#include <openvino/runtime/core.hpp>

int main() {
{
    //! [compile_model_default_npu]
    ov::Core core;
    auto model = core.read_model("model.xml");
    auto compiled_model = core.compile_model(model, "NPU");
    //! [compile_model_default_npu]
}
    return 0;
}
