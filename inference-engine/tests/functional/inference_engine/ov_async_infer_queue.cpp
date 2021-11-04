// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <map>

#include <openvino/core/function.hpp>
#include <openvino/core/shape.hpp>
#include <openvino/core/type/element_type.hpp>

#include <openvino/op/add.hpp>
#include <openvino/op/parameter.hpp>

#include <openvino/runtime/core.hpp>
#include <openvino/runtime/infer_request.hpp>
#include <openvino/runtime/executable_network.hpp>
#include <openvino/runtime/tensor.hpp>
#include <openvino/runtime/async_infer_queue.hpp>

using namespace ::testing;
using namespace std;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

TEST(AsyncInferQueueOVTests, flowTest) {
    ov::Shape input_shape{8};

    auto arg0 = make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape);
    auto arg1 = make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape);
    auto add = make_shared<ov::op::v1::Add>(arg0, arg1);
    ASSERT_EQ(add->input_value(0).get_node_shared_ptr(), arg0);
    ASSERT_EQ(add->input_value(1).get_node_shared_ptr(), arg1);

    auto model = make_shared<ov::Function>(add, ov::ParameterVector{arg0, arg1});

    ov::runtime::Core core;
    ov::runtime::ExecutableNetwork net = core.compile_model(model, "CPU");

    ov::runtime::AsyncInferQueue my_first_queue(net, 6);

    size_t counter = 0;
    size_t *counter_ptr = &counter;

    auto f = [counter_ptr](std::exception_ptr e, ov::runtime::InferRequest &request) {
        auto tensor = request.get_output_tensor();
        float* data_ptr = reinterpret_cast<float*>(tensor.data());
        vector<float> values(data_ptr, data_ptr + tensor.get_size());

        *counter_ptr += 1;

        for (auto i : values) {
            std::cout << i << ' ';
        }
        std::cout << "request: " << &request << std::endl;
    };

    my_first_queue.set_callback(f);

    size_t num_of_runs = 16;

    for (size_t i = 0; i < num_of_runs; i++) {
        std::vector<float> input_data(input_shape[0], i);
        auto t = ov::runtime::Tensor(ov::element::f32, input_shape);
        std::memcpy(t.data(), reinterpret_cast<void*>(&input_data[0]), t.get_byte_size());

        std::map<size_t, ov::runtime::Tensor> my_map;
        my_map.insert({0, t});
        my_map.insert({1, t});

        my_first_queue.start_async(my_map);
    }

    std::cout << "counter: " << counter << std::endl;

    my_first_queue.wait_all();

    std::cout << "counter: " << counter << std::endl;

    ASSERT_EQ(counter, num_of_runs);
}
