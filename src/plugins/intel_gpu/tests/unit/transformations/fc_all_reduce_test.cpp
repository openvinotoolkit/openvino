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