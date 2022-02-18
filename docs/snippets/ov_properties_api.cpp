#include <openvino/runtime/core.hpp>

int main() {
//! [part0]
ov::Core core;
auto available_devices = core.get_available_devices();
//! [part0]

//! [part1]
auto device_priorites = core.get_property("HETERO", ov::device::priorities);
//! [part1]

//! [part2]
auto cpu_device_name = core.get_property("GPU", ov::device::full_name);
//! [part2]

//! [part3]
auto model = core.read_model("sample.xml");
{
    auto compiled_model = core.compile_model(model, "CPU",
        ov::hint::performance(oc::hint::Performance::THROUGHPUT),
        ov::hint::inference_precision(ov::element::f32));
}
//! [part3]

//! [part4]
{
    auto compiled_model = core.compile_model(model, "CPU");
    auto nireq = compiled_model.get_property(ov::optimal_number_of_infer_requests);
}
//! [part4]

//! [part5]
{
    auto compiled_model = core.compile_model(model, "MYRIAD");
    auto temperature = compiled_model.get_property(ov::device::thermal);
}
//! [part5]

//! [part6]
{
    auto compiled_model = core.compile_model(model, "CPU");
    auto nthreads = compiled_model.get_property(ov::inference_num_threads);
}
//! [part6]
return 0;
}
