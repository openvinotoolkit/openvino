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
// #include "intel_gpu/graph/ccl_messenger.hpp"

#include <chrono>
using namespace std::chrono;

using namespace testing;
using namespace ov::intel_gpu;

TEST(TransformationTestsF1, FullyConnectedSplitInput1) {
    {
        // -------- Construct model
        // unsigned long test_size = 2;
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{8, 5120});
        std::vector<float> weights(152064 * 5120, 2);
        srand(time(0));
        std::generate(weights.begin(), weights.end(), rand);
        auto input2 = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{152064, 5120}, weights);
        // std::cout << "\n" << "weights: ";
        // for (size_t i = 0; i < input2->get_vector<float>().size(); i++) {
        //     std::cout << input2->get_vector<float>()[i] << ", ";
        // }
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<op::FullyConnected>(input1, input2, no_bias, ov::element::f32);
        auto model = std::make_shared<ov::Model>(ov::NodeVector{fc}, ov::ParameterVector{input1});

        ov::serialize(model, "./model_fc_test_qw.xml", "./model_fc_test_qw.bin");

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
        // std::cout << "\n" << "input_tensor: ";
        // for (size_t i = 0; i < tensor.get_size(); i++) {
        //     std::cout << tensor.data<float>()[i] << ", ";
        // }

        infer_request.set_input_tensor(tensor);

        // -------- Do inference synchronously --------
        infer_request.infer();
        sleep(1);
        std::vector<int64_t> tp_host_times_each_iter;
        for (int iter = 0; iter < 100; iter++) {
            auto start = std::chrono::high_resolution_clock::now();
            infer_request.infer();
            auto end = std::chrono::high_resolution_clock::now();
            // std::cout << "iter: " << iter << ", " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << std::endl;
            tp_host_times_each_iter.push_back(
                std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
        }
        double avg = static_cast<double>(std::accumulate(tp_host_times_each_iter.begin(),
                                                         tp_host_times_each_iter.end(),
                                                         (size_t)0,
                                                         std::plus<size_t>())) /
                     100;
        double max = *max_element(tp_host_times_each_iter.begin(), tp_host_times_each_iter.end());
        double min = *min_element(tp_host_times_each_iter.begin(), tp_host_times_each_iter.end());
        std::cout << "Max(ms): " << max << ", Min(ms): " << min << ", AVG(ms): " << avg << std::endl;
        // -------- Process output
        // auto output_tensor = infer_request.get_output_tensor();
        // std::cout << "\n"
        //           << "output_tensor: " << output_tensor.get_shape() << std::endl;
        // for (size_t i = 0; i < output_tensor.get_size(); i++) {
        //     std::cout << output_tensor.data<float>()[i] << ", ";
        // }
    }
}

// TEST(TransformationTestsF1, FullyConnectedSplitInput11) {
//     {
//         // -------- Construct model
//         unsigned long test_size = 2;
//         auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{test_size, test_size});
//         std::vector<float> weights(test_size * test_size, 2);
//         auto input2 = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{test_size, test_size}, {1, 2, 3, 4});
//         std::cout << "\n" << "weights: ";
//         for (size_t i = 0; i < input2->get_vector<float>().size(); i++) {
//             std::cout << input2->get_vector<float>()[i] << ", ";
//         }
//         auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
//         auto fc = std::make_shared<op::FullyConnected>(input1, input2, no_bias, ov::element::f32);
//         auto model = std::make_shared<ov::Model>(ov::NodeVector{fc}, ov::ParameterVector{input1});

//         // ov::serialize(model, "./model_fc.xml", "./model_fc.bin");

//         // -------- Loading a model to the device --------
//         ov::Core core;
//         ov::CompiledModel compiled_model = core.compile_model(model, "GPU");

//         // -------- Create an infer request --------
//         ov::InferRequest infer_request = compiled_model.create_infer_request();

//         // -------- Prepare input --------
//         auto input_generate = ov::test::utils::InputGenerateData(0, 5);
//         auto tensor = ov::test::utils::create_and_fill_tensor(infer_request.get_input_tensor().get_element_type(),
//                                                               infer_request.get_input_tensor().get_shape(),
//                                                               input_generate);
//         std::cout << "\n" << "input_tensor: ";
//         for (size_t i = 0; i < tensor.get_size(); i++) {
//             std::cout << tensor.data<float>()[i] << ", ";
//         }

//         infer_request.set_input_tensor(tensor);

//         // -------- Do inference synchronously --------
//         // infer_request.infer();
//         for (int iter = 0; iter < 100000; iter++) {
//             infer_request.infer();
//         }

//         // -------- Process output
//         auto output_tensor = infer_request.get_output_tensor();
//         std::cout << "\n"
//                   << "output_tensor: " << output_tensor.get_shape() << std::endl;
//         for (size_t i = 0; i < output_tensor.get_size(); i++) {
//             std::cout << output_tensor.data<float>()[i] << ", ";
//         }
//     }
// }


TEST(TransformationTestsF1, FullyConnectedSplitInput2) {
    // -------- Construct model
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 4, 4});

    auto start = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {0});
    auto stop = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {2});
    auto step = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
    auto axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
    auto input_slice_0 = std::make_shared<ov::op::v8::Slice>(input, start, stop, step, axis);

    auto start1 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {2});
    auto stop1 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {4});
    auto step1 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
    auto axis1 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
    auto input_slice_1 = std::make_shared<ov::op::v8::Slice>(input, start1, stop1, step1, axis1);

    auto model = std::make_shared<ov::Model>(ov::NodeVector{input_slice_0, input_slice_1}, ov::ParameterVector{input});

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
    infer_request.set_input_tensor(tensor);

    // -------- Do inference synchronously --------
    infer_request.infer();

    // -------- Process output
    // const ov::Tensor& output_tensor = infer_request.get_output_tensor();
    auto output_tensor_fc_0 = infer_request.get_output_tensor(0);
    std::cout << "\n" << "output_tensor_fc_0: ";
    for (size_t i = 0; i < output_tensor_fc_0.get_size(); i++) {
        std::cout << output_tensor_fc_0.data<float>()[i] << ", ";
    }

    auto output_tensor_fc_1 = infer_request.get_output_tensor(1);
    std::cout << "\n" << "output_tensor_fc_1: ";
    for (size_t i = 0; i < output_tensor_fc_1.get_size(); i++) {
        std::cout << output_tensor_fc_1.data<float>()[i] << ", ";
    }
}

TEST(TransformationTestsF1, FullyConnectedSplitInput3) {
    {
        // -------- Construct model
        // unsigned long test_size = 2;
        // input0 dims 0 (1, 8 16, ...1024)
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{8, 5120});
        std::vector<float> weights(152064 * 5120, 2);
        auto input2 = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{152064, 5120}, weights);
        // std::cout << "\n" << "weights: ";
        // for (size_t i = 0; i < input2->get_vector<float>().size(); i++) {
        //     std::cout << input2->get_vector<float>()[i] << ", ";
        // }
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<op::FullyConnected>(input1, input2, no_bias, ov::element::f32);
        auto model = std::make_shared<ov::Model>(ov::NodeVector{fc}, ov::ParameterVector{input1});

        ov::serialize(model, "./model_fc_test_qw_allreduce.xml", "./model_fc_test_qw_allreduce.bin");

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
        // std::cout << "\n" << "input_tensor: ";
        // for (size_t i = 0; i < tensor.get_size(); i++) {
        //     std::cout << tensor.data<float>()[i] << ", ";
        // }

        infer_request.set_input_tensor(tensor);

        // -------- Do inference synchronously --------
        infer_request.infer();
        // sleep();
        for (int iter = 0; iter < 100; iter++) {
            infer_request.infer();
        }
        // avg, max, min

        // -------- Process output
        auto output_tensor = infer_request.get_output_tensor();
        std::cout << "\n"
                  << "output_tensor: " << output_tensor.get_shape() << std::endl;
        // for (size_t i = 0; i < output_tensor.get_size(); i++) {
        //     std::cout << output_tensor.data<float>()[i] << ", ";
        // }
    }
}