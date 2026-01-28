#include <gtest/gtest.h>
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include <iostream>

namespace LayerTestsDefinitions {

// THE SCIENTIFIC TRAP (Permanently Added)
TEST(PrecisionTrapTest, GPU_HighPrecision_Floor_Check) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f64, ov::Shape{1});
    auto floor_op = std::make_shared<ov::op::v0::Floor>(param);
    auto result = std::make_shared<ov::op::v0::Result>(floor_op);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{result}, ov::ParameterVector{param});

    ov::Core core;
    bool gpu_found = false;
    for(auto&& d : core.get_available_devices()) { if(d.find("GPU") != std::string::npos) gpu_found = true; }
    if (!gpu_found) {
        std::cout << "[SKIP] No GPU found." << std::endl;
        return;
    }

    ov::AnyMap config;
    config[ov::hint::model_priority.name()] = ov::hint::Priority::HIGH; 

    auto compiled_model = core.compile_model(model, "GPU", config);
    auto request = compiled_model.create_infer_request();

    // 1.0 - 1e-15 should be 0.999... -> Floor -> 0.0
    // If precision is lost, it becomes 1.0 -> Floor -> 1.0 (Fail)
    double trap_value = 1.0 - 1.0e-15; 

    request.get_input_tensor().data<double>()[0] = trap_value;
    request.infer();

    double output_val = request.get_output_tensor().data<double>()[0];
    
    ASSERT_EQ(output_val, 0.0) << "FAILURE: Kernel used FLOAT precision! (Output was 1.0)";
}
} // namespace
