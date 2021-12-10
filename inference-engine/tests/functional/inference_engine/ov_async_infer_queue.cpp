// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock-matchers.h>

#include <map>

#include <openvino/core/model.hpp>
#include <openvino/core/shape.hpp>
#include <openvino/core/type/element_type.hpp>

#include <openvino/op/add.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/parameter.hpp>

#include <openvino/runtime/core.hpp>
#include <openvino/runtime/infer_request.hpp>
#include <openvino/runtime/executable_network.hpp>
#include <openvino/runtime/tensor.hpp>
#include <openvino/runtime/async_infer_queue.hpp>

using namespace ::testing;
using namespace std;

std::shared_ptr<ov::Model> create_add_model(ov::Shape shape) {
    auto arg0 = make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    auto arg1 = make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    auto add = make_shared<ov::op::v1::Add>(arg0, arg1);

    return make_shared<ov::Model>(add, ov::ParameterVector{arg0, arg1});
}

std::shared_ptr<ov::Model> create_mul_model(ov::Shape shape) {
    auto arg0 = make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    auto arg1 = make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    auto mul = make_shared<ov::op::v1::Multiply>(arg0, arg1);

    return make_shared<ov::Model>(mul, ov::ParameterVector{arg0, arg1});
}

TEST(AsyncInferQueueOVTests, BasicFlowTest) {
    ov::runtime::Core core;
    ov::runtime::ExecutableNetwork net =
        core.compile_model(create_add_model({8}), "CPU");

    ov::runtime::AsyncInferQueue basic_queue(net);

    size_t counter = 0;
    std::vector<ov::Any> callback_data;
    std::mutex user_mutex;

    auto f = [&counter, &callback_data, &user_mutex](std::exception_ptr e,
                                                     ov::runtime::InferRequest &request,
                                                     const ov::Any &data) {
        try {
            if (e) {
                std::rethrow_exception(e);
            }
        } catch (const std::exception& err) {
            throw ov::Exception(err.what());
        }
        std::unique_lock<std::mutex> lock(user_mutex);
        counter += 1;
        callback_data.push_back(data);
    };

    basic_queue.set_callback(f);

    size_t num_of_runs = 4;

    for (size_t i = 0; i < num_of_runs; i++) {
        basic_queue.start_async(i);
    }

    basic_queue.wait_all();

    std::vector<size_t> converted_data;
    std::transform(callback_data.begin(), callback_data.end(),
                   std::back_inserter(converted_data),
                   [](ov::Any val) -> std::size_t { return val.as<size_t>(); });

    ASSERT_EQ(counter, num_of_runs);
    ASSERT_THAT(converted_data, UnorderedElementsAreArray({0, 1, 2, 3}));
}

TEST(AsyncInferQueueOVTests, AccessRequestInsideCallbackTest) {
    size_t num_of_runs = 5;
    ov::Shape input_shape{1};
    std::vector<float> results;
    std::vector<float> expected_results;

    ov::runtime::Core core;
    ov::runtime::ExecutableNetwork net =
        core.compile_model(create_add_model(input_shape), "CPU");

    ov::runtime::AsyncInferQueue basic_queue(net, 3);

    std::mutex user_mutex;

    auto f = [&results, &user_mutex](std::exception_ptr e,
                                     ov::runtime::InferRequest &request,
                                     const ov::Any &data) {
        try {
            if (e) {
                std::rethrow_exception(e);
            }
        } catch (const std::exception& err) {
            throw ov::Exception(err.what());
        }
        float* data_ptr = request.get_output_tensor().data<float>();
        std::unique_lock<std::mutex> lock(user_mutex);
        results.push_back(data_ptr[0]);
    };

    basic_queue.set_callback(f);

    for (size_t i = 0; i < num_of_runs; i++) {
        std::vector<float> input_data(input_shape[0], i);

        expected_results.push_back(i + i);

        auto t = ov::runtime::Tensor(ov::element::f32, input_shape);
        std::copy_n(input_data.begin(), t.get_size(), t.data<float>());

        std::map<size_t, ov::runtime::Tensor> my_map;
        my_map.insert({0, t});
        my_map.insert({1, t});

        basic_queue.start_async(my_map);
    }

    basic_queue.wait_all();

    ASSERT_THAT(results, UnorderedElementsAreArray(expected_results));
}

TEST(AsyncInferQueueOVTests, AccessRequestOutsideCallbackTest) {
    size_t num_of_runs = 6;
    ov::Shape input_shape{1};
    std::vector<float> results;
    std::vector<float> expected_results;

    ov::runtime::Core core;
    ov::runtime::ExecutableNetwork net =
        core.compile_model(create_add_model(input_shape), "CPU");

    ov::runtime::AsyncInferQueue basic_queue(net, num_of_runs);

    for (size_t i = 0; i < num_of_runs; i++) {
        std::vector<float> input_data(input_shape[0], i);

        expected_results.push_back(i + i);

        auto t = ov::runtime::Tensor(ov::element::f32, input_shape);
        std::copy_n(input_data.begin(), t.get_size(), t.data<float>());

        std::map<size_t, ov::runtime::Tensor> my_map;
        my_map.insert({0, t});
        my_map.insert({1, t});

        basic_queue.start_async(my_map);
    }

    basic_queue.wait_all();

    for (size_t i = 0; i < num_of_runs; i++) {
        float* data_ptr = basic_queue[i].get_output_tensor().data<float>();
        results.push_back(data_ptr[0]);
    }

    ASSERT_THAT(results, UnorderedElementsAreArray(expected_results));
}

TEST(AsyncInferQueueOVTests, AccessInferRequestInLoopTest) {
    size_t num_of_runs = 4;
    ov::Shape input_shape{1};
    std::vector<float> results;
    std::vector<float> expected_results;

    ov::runtime::Core core;
    ov::runtime::ExecutableNetwork net =
        core.compile_model(create_mul_model(input_shape), "CPU");

    ov::runtime::AsyncInferQueue basic_queue(net);

    std::mutex user_mutex;

    auto f = [&results, &user_mutex](std::exception_ptr e,
                                     ov::runtime::InferRequest &request,
                                     const ov::Any &data) {
        try {
            if (e) {
                std::rethrow_exception(e);
            }
        } catch (const std::exception& err) {
            throw ov::Exception(err.what());
        }
        float *data_ptr = request.get_output_tensor().data<float>();
        std::unique_lock<std::mutex> lock(user_mutex);
        results.push_back(data_ptr[0]);
    };

    basic_queue.set_callback(f);

    for (size_t i = 0; i < num_of_runs; i++) {
        if (basic_queue.is_ready()) {
            std::vector<float> input_data(input_shape[0], i);

            expected_results.push_back(i * i);

            auto t = ov::runtime::Tensor(ov::element::f32, input_shape);
            std::copy_n(input_data.begin(), t.get_size(), t.data<float>());

            auto handle = basic_queue.get_idle_handle();

            basic_queue[handle].set_input_tensor(0, t);
            basic_queue[handle].set_input_tensor(1, t);

            basic_queue.start_async();
        }
    }

    basic_queue.wait_all();

    ASSERT_THAT(results, UnorderedElementsAreArray(expected_results));
}

TEST(AsyncInferQueueOVTests, ConnectAsyncInferQueuesTest) {
    size_t num_of_runs = 8;
    ov::Shape input_shape{1};
    std::vector<float> results;
    std::vector<float> expected_results;

    ov::runtime::Core core;
    ov::runtime::ExecutableNetwork add_net =
        core.compile_model(create_add_model(input_shape), "CPU");

   ov::runtime::ExecutableNetwork mul_net =
        core.compile_model(create_mul_model(input_shape), "CPU");

    ov::runtime::AsyncInferQueue add_queue(add_net, 4);
    ov::runtime::AsyncInferQueue mul_queue(mul_net, 2);

    auto f = [input_shape, &mul_queue](std::exception_ptr e,
                                       ov::runtime::InferRequest &request,
                                       const ov::Any &data) {
        try {
            if (e) {
                std::rethrow_exception(e);
            }
        } catch (const std::exception& err) {
            throw ov::Exception(err.what());
        }
        float *data_ptr = request.get_output_tensor().data<float>();
        auto t = ov::runtime::Tensor(ov::element::f32, input_shape);
        std::copy_n(&data_ptr[0], t.get_size(), t.data<float>());

        std::map<size_t, ov::runtime::Tensor> tmp;
        tmp.insert({0, t});
        tmp.insert({1, t});

        mul_queue.start_async(tmp);
    };

    add_queue.set_callback(f);

    std::mutex user_mutex;

    auto h = [&results, &user_mutex](std::exception_ptr e,
                                     ov::runtime::InferRequest &request,
                                     const ov::Any &data) {
        try {
            if (e) {
                std::rethrow_exception(e);
            }
        } catch (const std::exception& err) {
            throw ov::Exception(err.what());
        }
        float *data_ptr = request.get_output_tensor().data<float>();
        std::unique_lock<std::mutex> lock(user_mutex);
        results.push_back(data_ptr[0]);
    };

    mul_queue.set_callback(h);

    for (size_t i = 0; i < num_of_runs; i++) {
        if (add_queue.is_ready()) {
            std::vector<float> input_data(input_shape[0], i);

            expected_results.push_back((i + i) * (i + i));

            auto t = ov::runtime::Tensor(ov::element::f32, input_shape);
            std::copy_n(input_data.begin(), t.get_size(), t.data<float>());

            std::map<size_t, ov::runtime::Tensor> my_map;
            my_map.insert({0, t});
            my_map.insert({1, t});

            add_queue.start_async(my_map);
        }
    }

    add_queue.wait_all();
    mul_queue.wait_all();

    ASSERT_THAT(results, UnorderedElementsAreArray(expected_results));
}
