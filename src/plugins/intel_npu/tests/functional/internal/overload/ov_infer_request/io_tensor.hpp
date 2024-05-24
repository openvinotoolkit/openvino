// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <common_test_utils/ov_tensor_utils.hpp>
#include "behavior/ov_infer_request/io_tensor.hpp"
#include "common_test_utils/subgraph_builders/conv_pool_relu.hpp"
#include "overload/overload_test_utils_npu.hpp"

namespace ov {
namespace test {
namespace behavior {

struct OVInferRequestIOTensorTestNPU : public OVInferRequestIOTensorTest {
    void SetUp() override {
        // Skip test according to plugin specific disabledTestPatterns() (if any)
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        std::tie(target_device, configuration) = this->GetParam();
        // Skip test according to plugin specific disabledTestPatterns() (if any)
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        APIBaseTest::SetUp();
        function = ov::test::behavior::getDefaultNGraphFunctionForTheDeviceNPU();
        ov::AnyMap params;
        for (auto&& v : configuration) {
            params.emplace(v.first, v.second);
        }
        execNet = core->compile_model(function, target_device, params);
        try {
            req = execNet.create_infer_request();
        } catch (const std::exception& ex) {
            FAIL() << "Can't Create Infer Requiest in SetUp \nException [" << ex.what() << "]" << std::endl;
        }
        input = execNet.input();
        output = execNet.output();
    }
};

TEST_P(OVInferRequestIOTensorTestNPU, failToSetNullptrForInput) {
    ASSERT_THROW(req.set_tensor(input, {}), ov::Exception);
}

TEST_P(OVInferRequestIOTensorTestNPU, failToSetNullptrForOutput) {
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    ASSERT_THROW(req.set_tensor(output, {}), ov::Exception);
}

TEST_P(OVInferRequestIOTensorTestNPU, canSetAndGetInput) {
    auto tensor = utils::create_and_fill_tensor(input.get_element_type(), input.get_shape());
    OV_ASSERT_NO_THROW(req.set_tensor(input, tensor));
    ov::Tensor actual_tensor;
    OV_ASSERT_NO_THROW(actual_tensor = req.get_tensor(input));

    ASSERT_TRUE(actual_tensor);
    ASSERT_NE(nullptr, actual_tensor.data());
    ASSERT_EQ(tensor.data(), actual_tensor.data());
    ASSERT_EQ(input.get_element_type(), actual_tensor.get_element_type());
    ASSERT_EQ(input.get_shape(), actual_tensor.get_shape());
}

TEST_P(OVInferRequestIOTensorTestNPU, canSetAndGetOutput) {
    auto tensor = utils::create_and_fill_tensor(output.get_element_type(), output.get_shape());
    req.set_tensor(output, tensor);
    auto actual_tensor = req.get_tensor(output);

    ASSERT_TRUE(actual_tensor);
    ASSERT_FALSE(actual_tensor.data() == nullptr);
    ASSERT_EQ(actual_tensor.data(), tensor.data());
    ASSERT_EQ(output.get_element_type(), actual_tensor.get_element_type());
    ASSERT_EQ(output.get_shape(), actual_tensor.get_shape());
}

TEST_P(OVInferRequestIOTensorTestNPU, failToSetTensorWithIncorrectName) {
    auto tensor = utils::create_and_fill_tensor(input.get_element_type(), input.get_shape());
    ASSERT_THROW(req.set_tensor("incorrect_input", tensor), ov::Exception);
}

TEST_P(OVInferRequestIOTensorTestNPU, failToSetInputWithIncorrectSizes) {
    auto shape = input.get_shape();
    shape[0] *= 2;
    auto tensor = utils::create_and_fill_tensor(input.get_element_type(), shape);
    ASSERT_THROW(req.set_tensor(input, tensor), ov::Exception);
}

TEST_P(OVInferRequestIOTensorTestNPU, failToSetOutputWithIncorrectSizes) {
    auto shape = output.get_shape();
    shape[0] *= 2;
    auto tensor = utils::create_and_fill_tensor(output.get_element_type(), shape);
    ASSERT_THROW(req.set_tensor(output, tensor), ov::Exception);
}

TEST_P(OVInferRequestIOTensorTestNPU, canInferWithoutSetAndGetInOutSync) {
    OV_ASSERT_NO_THROW(req.infer());
}

TEST_P(OVInferRequestIOTensorTestNPU, canInferWithoutSetAndGetInOutAsync) {
    OV_ASSERT_NO_THROW(req.start_async());
    OV_ASSERT_NO_THROW(req.wait());
}

TEST_P(OVInferRequestIOTensorTestNPU, secondCallGetInputDoNotReAllocateData) {
    ov::Tensor tensor1, tensor2;
    OV_ASSERT_NO_THROW(tensor1 = req.get_tensor(input));
    OV_ASSERT_NO_THROW(tensor2 = req.get_tensor(input));
    ASSERT_EQ(tensor1.data(), tensor2.data());
}

TEST_P(OVInferRequestIOTensorTestNPU, secondCallGetOutputDoNotReAllocateData) {
    ov::Tensor tensor1, tensor2;
    OV_ASSERT_NO_THROW(tensor1 = req.get_tensor(output));
    OV_ASSERT_NO_THROW(tensor2 = req.get_tensor(output));
    ASSERT_EQ(tensor1.data(), tensor2.data());
}

TEST_P(OVInferRequestIOTensorTestNPU, secondCallGetInputAfterInferSync) {
    ov::Tensor tensor1, tensor2;
    OV_ASSERT_NO_THROW(req.infer());
    OV_ASSERT_NO_THROW(tensor1 = req.get_tensor(input));
    OV_ASSERT_NO_THROW(req.start_async());
    OV_ASSERT_NO_THROW(req.wait());
    OV_ASSERT_NO_THROW(tensor2 = req.get_tensor(input));
    ASSERT_EQ(tensor1.data(), tensor2.data());
}

TEST_P(OVInferRequestIOTensorTestNPU, secondCallGetOutputAfterInferSync) {
    ov::Tensor tensor1, tensor2;
    OV_ASSERT_NO_THROW(req.infer());
    OV_ASSERT_NO_THROW(tensor1 = req.get_tensor(output));
    OV_ASSERT_NO_THROW(req.start_async());
    OV_ASSERT_NO_THROW(req.wait());
    OV_ASSERT_NO_THROW(tensor2 = req.get_tensor(output));
    ASSERT_EQ(tensor1.data(), tensor2.data());
}

TEST_P(OVInferRequestIOTensorTestNPU, canInferWithSetInOutBlobs) {
    auto input_tensor = utils::create_and_fill_tensor(input.get_element_type(), input.get_shape());
    OV_ASSERT_NO_THROW(req.set_tensor(input, input_tensor));
    auto output_tensor = utils::create_and_fill_tensor(output.get_element_type(), output.get_shape());
    OV_ASSERT_NO_THROW(req.set_tensor(output, output_tensor));
    OV_ASSERT_NO_THROW(req.infer());

    auto actual_input_tensor = req.get_tensor(input);
    ASSERT_EQ(actual_input_tensor.data(), input_tensor.data());
    auto actual_output_tensor = req.get_tensor(output);
    ASSERT_EQ(actual_output_tensor.data(), output_tensor.data());
}

TEST_P(OVInferRequestIOTensorTestNPU, canInferWithGetIn) {
    ov::Tensor input_tensor;
    OV_ASSERT_NO_THROW(input_tensor = req.get_tensor(input));
    OV_ASSERT_NO_THROW(req.infer());
    OV_ASSERT_NO_THROW(req.start_async());
    OV_ASSERT_NO_THROW(req.wait());
    OV_ASSERT_NO_THROW(req.get_tensor(output));
}

TEST_P(OVInferRequestIOTensorTestNPU, canInferAfterIOBlobReallocation) {
    ov::Tensor input_tensor, output_tensor;
    auto in_shape = input.get_shape();
    auto out_shape = output.get_shape();

    // imitates blob reallocation
    OV_ASSERT_NO_THROW(input_tensor = req.get_tensor(input));
    OV_ASSERT_NO_THROW(input_tensor.set_shape({5, 5, 5, 5}));
    OV_ASSERT_NO_THROW(input_tensor.set_shape(in_shape));

    OV_ASSERT_NO_THROW(output_tensor = req.get_tensor(output));
    OV_ASSERT_NO_THROW(output_tensor.set_shape({20, 20, 20, 20}));
    OV_ASSERT_NO_THROW(output_tensor.set_shape(out_shape));

    OV_ASSERT_NO_THROW(req.infer());
    OV_ASSERT_NO_THROW(req.start_async());
    OV_ASSERT_NO_THROW(req.wait());
    OV_ASSERT_NO_THROW(req.get_tensor(output));
}

TEST_P(OVInferRequestIOTensorTestNPU, InferStaticNetworkSetChangedInputTensorThrow) {
    const ov::Shape shape1 = {1, 2, 32, 32};
    const ov::Shape shape2 = {1, 2, 40, 40};
    std::map<std::string, ov::PartialShape> shapes;
    shapes[function->inputs().back().get_any_name()] = shape1;
    OV_ASSERT_NO_THROW(function->reshape(shapes));
    // Load ov::Model to target plugins
    std::shared_ptr<ov::Core> ie = utils::PluginCache::get().core();
    auto execNet = ie->compile_model(function, target_device, configuration);
    // Create InferRequest
    ov::InferRequest req;
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    OV_ASSERT_NO_THROW(req.infer());
    // Get input_tensor
    ov::Tensor tensor;
    OV_ASSERT_NO_THROW(tensor = req.get_tensor(function->inputs().back().get_any_name()));
    // Set shape
    OV_ASSERT_NO_THROW(tensor.set_shape(shape2));
    ASSERT_ANY_THROW(req.infer());
}

TEST_P(OVInferRequestIOTensorTestNPU, InferStaticNetworkSetChangedOutputTensorThrow) {
    const ov::Shape shape1 = {1, 2, 32, 32};
    ov::Shape shape2;
    shape2 = ov::Shape{1, 20};

    std::map<std::string, ov::PartialShape> shapes;
    shapes[function->inputs().back().get_any_name()] = shape1;
    OV_ASSERT_NO_THROW(function->reshape(shapes));
    // Load ov::Model to target plugins
    std::shared_ptr<ov::Core> ie = utils::PluginCache::get().core();
    auto execNet = ie->compile_model(function, target_device, configuration);
    // Create InferRequest
    ov::InferRequest req;
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    OV_ASSERT_NO_THROW(req.infer());
    // Get output_tensor
    ov::Tensor tensor;
    OV_ASSERT_NO_THROW(tensor = req.get_tensor(function->outputs().back().get_any_name()););
    // Set shape
    OV_ASSERT_NO_THROW(tensor.set_shape(shape2));
    ASSERT_ANY_THROW(req.infer());
}

struct OVInferRequestIOTensorSetPrecisionTestNPU : OVInferRequestIOTensorSetPrecisionTest {
    void SetUp() override {
        std::tie(element_type, target_device, config) = this->GetParam();
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        APIBaseTest::SetUp();
        function = ov::test::utils::make_conv_pool_relu();
        execNet = core->compile_model(function, target_device, config);
        req = execNet.create_infer_request();
    }
};

TEST_P(OVInferRequestIOTensorSetPrecisionTestNPU, CanSetInBlobWithDifferentPrecision) {
    for (auto&& output : execNet.outputs()) {
        auto output_tensor = utils::create_and_fill_tensor(element_type, output.get_shape());
        if (output.get_element_type() == element_type) {
            OV_ASSERT_NO_THROW(req.set_tensor(output, output_tensor));
        } else {
            ASSERT_THROW(req.set_tensor(output, output_tensor), ov::Exception);
        }
    }
}

TEST_P(OVInferRequestIOTensorSetPrecisionTestNPU, CanSetOutBlobWithDifferentPrecision) {
    for (auto&& input : execNet.inputs()) {
        auto input_tensor = utils::create_and_fill_tensor(element_type, input.get_shape());
        if (input.get_element_type() == element_type) {
            OV_ASSERT_NO_THROW(req.set_tensor(input, input_tensor));
        } else {
            ASSERT_THROW(req.set_tensor(input, input_tensor), ov::Exception);
        }
    }
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
