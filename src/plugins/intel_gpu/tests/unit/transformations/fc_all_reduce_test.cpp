// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>
#include "openvino/openvino.hpp"
#include <openvino/core/model.hpp>
#include <openvino/core/node_vector.hpp>
#include <openvino/pass/manager.hpp>
#include "common_test_utils/ov_test_utils.hpp"
#include <transformations/utils/utils.hpp>
#include "common_test_utils/ov_tensor_utils.hpp"
#include <plugin/transformations/fc_all_reduce.hpp>
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/variadic_split.hpp"
#include "intel_gpu/op/fully_connected.hpp"
#include "intel_gpu/op/fully_connected_compressed.hpp"
#include "intel_gpu/op/placeholder.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/relu.hpp"
// #include "intel_gpu/graph/ccl_messenger.hpp"

#include <chrono>
using namespace std::chrono;

using namespace testing;
using namespace ov::intel_gpu;

TEST(TransformationTestsF1, FullyConnectedSplitInput11) {
    {
        // -------- Construct model
        unsigned long test_size = 2;
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{test_size, test_size});
        std::vector<float> weights(test_size * test_size, 2);
        auto input2 = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{test_size, test_size}, {1, 2, 3, 4});
        std::cout << "\n" << "weights: ";
        for (size_t i = 0; i < input2->get_vector<float>().size(); i++) {
            std::cout << input2->get_vector<float>()[i] << ", ";
        }
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<op::FullyConnected>(input1, input2, no_bias, ov::element::f32);
        const auto relu = std::make_shared<ov::op::v0::Relu>(fc);
        auto model = std::make_shared<ov::Model>(ov::NodeVector{relu}, ov::ParameterVector{input1});
        //auto model = std::make_shared<ov::Model>(ov::NodeVector{fc}, ov::ParameterVector{input1});

        // ov::serialize(model, "./model_fc.xml", "./model_fc.bin");

        // -------- Loading a model to the device --------
        ov::Core core;
        ov::CompiledModel compiled_model = core.compile_model(model, "GPU");

        // -------- Create an infer request --------
        ov::InferRequest infer_request = compiled_model.create_infer_request();

        // -------- Prepare input --------
        auto input_generate = ov::test::utils::InputGenerateData(0, 5);
        auto tensor = ov::test::utils::create_and_fill_tensor(infer_request.get_input_tensor().get_element_type(),
                                                              infer_request.get_input_tensor().get_shape(),
                                                              input_generate);
        std::cout << "\n" << "input_tensor: ";
        for (size_t i = 0; i < tensor.get_size(); i++) {
            std::cout << tensor.data<float>()[i] << ", ";
        }
        std::cout << std::endl;

        infer_request.set_input_tensor(tensor);

        // -------- Do inference synchronously --------
        infer_request.infer();
        for (int iter = 0; iter < 2; iter++) {
             infer_request.infer();
        }

        // -------- Process output
        auto output_tensor = infer_request.get_output_tensor();
        std::cout << "\n"
                  << "output_tensor: " << output_tensor.get_shape() << std::endl;
        for (size_t i = 0; i < output_tensor.get_size(); i++) {
            std::cout << output_tensor.data<float>()[i] << ", ";
        }
        std::cout << std::endl;
    }
}

TEST(TransformationTestsF1, FullyConnectedSplitInput16) {
    {
        // -------- Construct model
        unsigned long test_size = 16;
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{test_size, test_size});
        std::vector<float> input_data = {
            1, 2, 1, 2, 2, 3, 1, 3, 1, 1, 1, 3, 2, 3, 3, 1, 2, 2, 2, 2, 1, 2, 1, 1, 2, 3, 1, 3, 1, 2, 2, 3,
            1, 2, 2, 2, 1, 3, 1, 3, 3, 1, 3, 1, 1, 1, 2, 2, 3, 1, 1, 2, 1, 2, 3, 3, 1, 2, 2, 2, 2, 3, 3, 3,
            1, 3, 2, 1, 2, 3, 1, 1, 3, 1, 1, 1, 1, 1, 1, 3, 1, 3, 2, 2, 2, 1, 2, 2, 2, 1, 2, 3, 1, 2, 3, 1,
            3, 1, 2, 3, 3, 2, 1, 2, 2, 1, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 1, 2, 3, 3, 2, 3, 2, 1, 3, 3, 1, 3,
            1, 1, 3, 1, 3, 3, 3, 1, 1, 1, 2, 3, 1, 2, 3, 3, 3, 2, 1, 1, 1, 1, 3, 3, 1, 1, 1, 2, 3, 1, 1, 2,
            1, 3, 2, 2, 2, 1, 1, 1, 2, 2, 3, 2, 1, 1, 2, 3, 2, 2, 2, 1, 1, 1, 2, 3, 2, 2, 1, 1, 2, 3, 1, 3,
            3, 2, 2, 2, 3, 1, 1, 2, 1, 3, 3, 1, 3, 3, 3, 2, 2, 1, 1, 1, 3, 3, 2, 2, 1, 1, 3, 2, 2, 3, 1, 2,
            2, 2, 2, 2, 3, 3, 3, 1, 2, 3, 2, 2, 2, 2, 1, 3, 3, 1, 2, 2, 1, 1, 1, 3, 2, 3, 2, 3, 1, 3, 1, 2};
        std::vector<float> weights = {
            2, 2, 1, 1, 1, 1, 1, 1, 2, 1, 3, 3, 2, 2, 2, 3, 2, 2, 1, 2, 3, 2, 2, 1, 1, 2, 2, 3, 2, 3, 3, 2,
            3, 3, 1, 2, 2, 2, 3, 2, 2, 2, 3, 1, 2, 1, 2, 2, 3, 1, 3, 1, 3, 1, 3, 1, 1, 1, 1, 2, 3, 1, 1, 2,
            3, 2, 1, 1, 1, 1, 1, 2, 1, 3, 3, 3, 1, 1, 3, 1, 3, 3, 1, 2, 2, 2, 1, 1, 2, 1, 1, 2, 3, 3, 1, 3,
            2, 2, 3, 2, 3, 3, 2, 3, 1, 1, 3, 2, 2, 3, 3, 2, 1, 3, 1, 3, 1, 3, 2, 3, 2, 1, 2, 2, 3, 3, 1, 2,
            1, 1, 1, 3, 1, 2, 1, 3, 2, 1, 2, 3, 1, 1, 1, 3, 2, 1, 1, 1, 1, 2, 3, 3, 2, 1, 2, 2, 2, 2, 2, 2,
            3, 3, 2, 2, 2, 3, 1, 1, 1, 2, 1, 2, 1, 3, 3, 2, 1, 1, 1, 1, 1, 3, 2, 1, 3, 1, 3, 2, 1, 1, 3, 2,
            3, 3, 2, 1, 1, 3, 2, 2, 1, 1, 3, 1, 2, 3, 3, 2, 1, 1, 2, 1, 1, 3, 1, 1, 2, 2, 2, 2, 2, 2, 3, 1,
            3, 1, 2, 1, 3, 1, 1, 2, 3, 2, 2, 2, 1, 1, 3, 1, 1, 3, 2, 3, 3, 1, 3, 3, 1, 3, 2, 1, 2, 3, 2, 2};

        std::vector<float> result = {
            52, 66, 58, 48, 53, 59, 74, 67, 53, 56, 64, 51, 64, 54, 53, 66, 55, 63, 59, 50, 55, 60, 66, 60, 53, 52,
            64, 51, 60, 52, 54, 63, 51, 55, 61, 44, 50, 53, 69, 63, 54, 54, 56, 55, 62, 51, 53, 61, 61, 72, 71, 59,
            62, 66, 80, 71, 59, 67, 69, 58, 74, 57, 60, 75, 46, 51, 54, 44, 40, 54, 59, 55, 45, 45, 54, 47, 54, 44,
            46, 54, 54, 64, 61, 50, 55, 56, 71, 62, 51, 54, 61, 52, 62, 51, 56, 67, 69, 79, 74, 67, 64, 76, 88, 77,
            66, 67, 74, 62, 76, 62, 68, 78, 66, 76, 82, 69, 65, 78, 88, 84, 65, 71, 77, 60, 82, 62, 65, 89, 57, 70,
            63, 60, 54, 59, 79, 62, 54, 59, 66, 59, 67, 57, 58, 68, 48, 54, 59, 53, 48, 55, 61, 58, 46, 54, 51, 42,
            57, 41, 47, 59, 55, 60, 61, 48, 53, 56, 66, 58, 51, 49, 58, 51, 59, 48, 53, 64, 51, 57, 59, 51, 48, 59,
            66, 62, 49, 55, 57, 45, 61, 46, 50, 66, 62, 75, 73, 62, 65, 69, 83, 69, 54, 61, 72, 55, 75, 59, 66, 80,
            54, 65, 62, 54, 51, 60, 74, 64, 51, 57, 60, 52, 65, 51, 53, 65, 60, 73, 73, 64, 58, 69, 79, 71, 58, 62,
            70, 59, 70, 58, 61, 77, 56, 63, 61, 52, 59, 60, 70, 63, 56, 57, 62, 49, 62, 51, 58, 67};

        auto input2 = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{test_size, test_size}, weights.data());
        std::cout << "\n" << "weights: ";
        for (size_t i = 0; i < input2->get_vector<float>().size(); i++) {
            std::cout << input2->get_vector<float>()[i] << ", ";
        }
        std::cout << std::endl;
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<op::FullyConnected>(input1, input2, no_bias, ov::element::f32);
        const auto relu = std::make_shared<ov::op::v0::Relu>(fc);
        auto model = std::make_shared<ov::Model>(ov::NodeVector{relu}, ov::ParameterVector{input1});

        // ov::serialize(model, "./model_fc.xml", "./model_fc.bin");

        // -------- Loading a model to the device --------
        ov::Core core;
        ov::CompiledModel compiled_model = core.compile_model(model, "GPU");

        // -------- Create an infer request --------
        ov::InferRequest infer_request = compiled_model.create_infer_request();

        // -------- Prepare input --------
        auto tensor = ov::Tensor(infer_request.get_input_tensor().get_element_type(),
                                 infer_request.get_input_tensor().get_shape(),
                                 input_data.data());
        std::cout << "\n" << "input_tensor: ";
        for (size_t i = 0; i < tensor.get_size(); i++) {
            std::cout << tensor.data<float>()[i] << ", ";
        }
        std::cout << std::endl;

        infer_request.set_input_tensor(tensor);

        // -------- Do inference synchronously --------
        infer_request.infer();
        for (int iter = 0; iter < 2; iter++) {
             infer_request.infer();
        }

        // -------- Process output
        auto output_tensor = infer_request.get_output_tensor();
        std::cout << "\n"
                  << "output_tensor: " << output_tensor.get_shape() << std::endl;
        for (size_t i = 0; i < output_tensor.get_size(); i++) {
            if(output_tensor.data<float>()[i] != result[i]) {
                std::cout << "Result is incorrect..." << std::endl;
                return;
            }
        }
        std::cout << "Result is correct!" << std::endl;
        std::cout << std::endl;
    }
}