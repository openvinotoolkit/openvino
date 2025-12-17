// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>

#include "openvino/openvino.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/result.hpp"
#include "common_test_utils/test_constants.hpp"
#include "common_test_utils/subgraph_builders/weights_decompression_builders.hpp"


using namespace ov;
using namespace ov::test;

namespace {

class U2_IR_Serialization : public ::testing::Test {
protected:
    void SetUp() override {
        test_dir = std::filesystem::temp_directory_path() / "ov_u2_ir_test";
        std::filesystem::create_directories(test_dir);
        
        xml_path = test_dir / "test_u2_model.xml";
        bin_path = test_dir / "test_u2_model.bin";
    }

    void TearDown() override {
        // Cleanup
        std::filesystem::remove_all(test_dir);
    }

    std::filesystem::path test_dir;
    std::filesystem::path xml_path;
    std::filesystem::path bin_path;
};

TEST_F(U2_IR_Serialization, RoundTrip) {
    const size_t M = 256;
    const size_t N = 128;
    const size_t num_weights = M * N;
    auto input = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 1, M});
    const auto weights_subgraph = ov::test::utils::initMatMulDecompressionSubgraph(
        Shape{M, N},                         
        -1,                                   
        element::f32,                         
        element::u2,                          
        element::f32,                         
        element::dynamic,                     
        false,                              
        ov::test::utils::DecompressionType::full,   
        ov::test::utils::DecompressionType::empty,   
        false      
    );
    auto matmul = std::make_shared<op::v0::MatMul>(input, weights_subgraph, false, false);
    auto result = std::make_shared<op::v0::Result>(matmul);
    auto model = std::make_shared<Model>(ResultVector{result}, ParameterVector{input}, "u2_ir_serialization_test");

    // === BASELINE INFERENCE ===
    Core core;
    auto compiled_baseline = core.compile_model(model, utils::DEVICE_GPU);
    auto infer_req_baseline = compiled_baseline.create_infer_request();
    Tensor input_tensor(element::f32, Shape{1, 1, M});
    auto* input_data = input_tensor.data<float>();
    for (size_t i = 0; i < M; i++) {
        input_data[i] = static_cast<float>(i % 10) / 10.0f;
    }

    infer_req_baseline.set_input_tensor(input_tensor);
    infer_req_baseline.infer();
    auto baseline_output = infer_req_baseline.get_output_tensor();
    ASSERT_NO_THROW(ov::serialize(model, xml_path.string()));
    ASSERT_TRUE(std::filesystem::exists(xml_path)) << "XML file not created";
    ASSERT_TRUE(std::filesystem::exists(bin_path)) << "BIN file not created";

    // === BINARY SIZE VALIDATION  ===
    size_t theoretical_payload = (num_weights * 2 + 7) / 8;  
    size_t scale_size = N * sizeof(float);
    size_t expected_compressed_payload = theoretical_payload + scale_size;
    size_t actual_size = std::filesystem::file_size(bin_path);
    size_t failure_threshold = theoretical_payload * 3;

    std::cout << "[IR Validation] Theoretical 2-bit payload: " << theoretical_payload << " bytes\n";
    std::cout << "[IR Validation] Expected total (weights + scales): " << expected_compressed_payload << " bytes\n";
    std::cout << "[IR Validation] Actual binary size: " << actual_size << " bytes\n";
    std::cout << "[IR Validation] Failure threshold (3x weights): " << failure_threshold << " bytes\n";
    ASSERT_GE(actual_size, theoretical_payload) 
        << "Binary smaller than theoretical minimum";
    ASSERT_LE(actual_size, expected_compressed_payload + 512) 
        << "Binary has excessive overhead (>" << expected_compressed_payload + 512 << " bytes)";
    ASSERT_LT(actual_size, failure_threshold) 
        << "Binary size suggests u8 expansion instead of u2 compression";

    // === DESERIALIZATION ===
    std::shared_ptr<Model> loaded_model;
    ASSERT_NO_THROW(loaded_model = core.read_model(xml_path.string()));
    ASSERT_NE(loaded_model, nullptr) << "Loaded model is null";

    // === RUNTIME INSPECTION  ===
    try {
        auto compiled_loaded = core.compile_model(loaded_model, utils::DEVICE_GPU);
        auto runtime_model = compiled_loaded.get_runtime_model();
        
        std::cout << "[Runtime Diagnostic] Runtime model operations count: " 
                  << runtime_model->get_ops().size() << "\n";
        for (const auto& op : runtime_model->get_ops()) {
            std::cout << "[Runtime Diagnostic] Op: " << op->get_friendly_name() 
                      << " Type: " << op->get_type_name() << "\n";
        }
    } catch (const std::exception& e) {
        std::cout << "[Runtime Diagnostic] Runtime inspection unavailable: " << e.what() << "\n";
    }

    auto compiled_loaded = core.compile_model(loaded_model, utils::DEVICE_GPU);
    auto infer_req_loaded = compiled_loaded.create_infer_request();
    infer_req_loaded.set_input_tensor(input_tensor);
    infer_req_loaded.infer();
    auto loaded_output = infer_req_loaded.get_output_tensor();
    ASSERT_EQ(baseline_output.get_shape(), loaded_output.get_shape()) 
        << "Output shapes differ";
    ASSERT_EQ(baseline_output.get_element_type(), loaded_output.get_element_type()) 
        << "Output types differ";

    auto* baseline_data = baseline_output.data<float>();
    auto* loaded_data = loaded_output.data<float>();
    size_t output_size = baseline_output.get_size();
    float max_diff = 0.0f;
    for (size_t i = 0; i < output_size; i++) {
        float diff = std::abs(baseline_data[i] - loaded_data[i]);
        max_diff = std::max(max_diff, diff);
        ASSERT_LT(diff, 1e-5f) << "Output mismatch at index " << i 
                               << ": baseline=" << baseline_data[i] 
                               << " loaded=" << loaded_data[i];
    }
    std::cout << "[IR Validation] Maximum output difference: " << max_diff << "\n";
    std::cout << "[IR Validation] ✓ All checks PASSED\n";
}

}  // namespace
