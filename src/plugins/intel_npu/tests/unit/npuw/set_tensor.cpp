// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <iostream>

#include "openvino/openvino.hpp"
#include "model_generator/model_generator.hpp"

// FIXME: parametrize all the tests below

TEST(SetTensorNPUW, RemoteTensorOutputJust) {
    // Only run this test on NPU device
    ov::Core ov_core;
    auto core_devices = ov_core.get_available_devices();
    if (std::find(core_devices.begin(), core_devices.end(), "NPU") == core_devices.end()) {
        GTEST_SKIP() << "No available devices.";
    }

    // Device
    const std::string device = "NPU";

    // Create model
    ModelGenerator mg;
    auto model = mg.get_model_with_one_op();

    ov::element::Type element_type = ov::element::i64;
    auto output_tensor_shape = model->outputs()[0].get_shape();
    // Calculate total number of elements
    size_t total_elements = ov::shape_size(output_tensor_shape);

    // Create output data
    std::vector<int64_t> data = std::vector<int64_t>(total_elements, 0);
    std::iota(data.begin(), data.end(), 1);

    // Create the remote tensor output
    auto npu_context = ov_core.get_default_context(device);
    auto output = npu_context.create_host_tensor(element_type, output_tensor_shape);

    // Initialize remote input with non-zero data
    ov::Tensor values(element_type, output_tensor_shape, data.data());
    values.copy_to(output);

    // NPUW config
    ov::AnyMap config = {
        {"NPU_USE_NPUW", "YES"},
        {"NPUW_FUNCALL_FOR_ALL", "YES"},
        {"NPUW_DEVICES", "NPU"},
        {"NPUW_FOLD" , "YES"}
    };

    // Compile NPUW
    auto compiled = ov_core.compile_model(model, device, config);

    // Create infer request
    auto request = compiled.create_infer_request();

    // Set remote io
    request.set_tensor(compiled.outputs()[0], output);

    // Check output tensor is not zero
    auto output_tensor = request.get_tensor(compiled.outputs()[0]);

    auto check_non_zero = [](const ov::Tensor& t, size_t size) {
        int64_t* tdata = t.data<int64_t>();
        for (size_t i = 0; i < size; ++i) {
            if (tdata[i] == 0) {
                return false;
            }
        }
        return true;
    };
    
    EXPECT_EQ(output_tensor.data(), output.data());
    EXPECT_TRUE(check_non_zero(output_tensor, total_elements));
}

using RemoteTensorInputParams = std::tuple<ov::AnyMap>; // additional config to pick infer request class

class RemoteTensorInputTestsBase {
protected:
    ov::AnyMap extra_config;

public:
    void SetUp(const RemoteTensorInputParams& getParam) {
        std::tie(extra_config) = getParam;
    }
    std::string ToString() const {
        std::string ret = "WithConfig";
        for (const auto& el : extra_config) {
            ret += "_" + el.first + "_" + el.second.as<std::string>();
        }
        return ret;
    }
};

template <class T>
class RemoteTensorInputTestsTmpl : public ::testing::Test,
                           public T,
                           public ::testing::WithParamInterface<RemoteTensorInputParams> {
protected:
    void SetUp() override {
        T::SetUp(GetParam());
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<RemoteTensorInputParams>& obj) {
        T _bt;
        _bt.SetUp(obj.param);
        return _bt.ToString();
    }
};

using SetTensorNPUW_RemoteTensorInputTests = RemoteTensorInputTestsTmpl<RemoteTensorInputTestsBase>;

TEST_P(SetTensorNPUW_RemoteTensorInputTests, RemoteTensorInput) {
    // Only run this test on NPU device
    ov::Core ov_core;
    auto core_devices = ov_core.get_available_devices();
    if (std::find(core_devices.begin(), core_devices.end(), "NPU") == core_devices.end()) {
        GTEST_SKIP() << "No available devices.";
    }

    // Device
    const std::string device = "NPU";

    // Create model
    ModelGenerator mg;
    auto model = mg.get_model_with_repeated_blocks();

    ov::element::Type element_type = ov::element::i32;
    auto input_tensor_shape = model->inputs()[0].get_shape();
    // Calculate total number of elements
    size_t total_elements = ov::shape_size(input_tensor_shape);

    // Create input data
    std::vector<int> data = std::vector<int>(total_elements, 0);
    std::iota(data.begin(), data.end(), 1);

    // Create the remote tensor input
    auto npu_context = ov_core.get_default_context(device);
    auto input = npu_context.create_host_tensor(element_type, input_tensor_shape);

    // Initialize remote input with non-zero data
    ov::Tensor values(element_type, input_tensor_shape, data.data());
    values.copy_to(input);

    // NPUW config
    ov::AnyMap config = {
        {"NPU_USE_NPUW", "YES"},
        {"NPUW_FUNCALL_FOR_ALL", "YES"},
        {"NPUW_DEVICES", "NPU"},
        {"NPUW_FOLD" , "YES"}
    };

    // Apply parametrized config
    for (const auto& el : extra_config) {
        config[el.first] = el.second;
    }

    // Compile NPUW
    auto compiled = ov_core.compile_model(model, device, config);

    // Create infer request
    auto request = compiled.create_infer_request();

    // Set remote io
    request.set_tensor(compiled.inputs()[0], input);

    // Infer
    request.infer();

    // Check input tensor wasn't reallocated by NPUW
    auto input_tensor = request.get_tensor(compiled.inputs()[0]);

    EXPECT_EQ(input_tensor.data(), input.data());
}

const auto TestCases = ::testing::Combine(
        ::testing::ValuesIn({ov::AnyMap{}, ov::AnyMap{{"NPUW_UNFOLD_IREQS", "YES"}}}));

INSTANTIATE_TEST_SUITE_P(SetTensorNPUW_RemoteTensorInputTests, SetTensorNPUW_RemoteTensorInputTests,
                         TestCases, SetTensorNPUW_RemoteTensorInputTests::getTestCaseName);
