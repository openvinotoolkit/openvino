#include <openvino/runtime/core.hpp>

int main() {
//! [part2]
ov::Core core;
auto model = core.read_model("sample.xml");
{
    auto compiled_model = core.compile_model(model, "HETERO:GPU,CPU");
}
{
    auto compiled_model = core.compile_model(model, "HETERO", ov::device::priorities("GPU", "CPU"));
}
{
    auto compiled_model = core.compile_model(model, "HETERO", ov::device::priorities("GPU,CPU"));
}
//! [part2]
return 0;
}
