// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <future>

#include "base/ov_behavior_test_utils.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/node_builders/fully_connected.hpp"

namespace ov {
namespace test {
namespace behavior {
using OVInferRequestCallbackTests = OVInferRequestTests;

TEST_P(OVInferRequestCallbackTests, canCallAsyncWithCompletionCallback) {
    ov::InferRequest req;
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    bool is_called = false;
    OV_ASSERT_NO_THROW(req.set_callback([&](std::exception_ptr exception_ptr) {
        // HSD_1805940120: Wait on starting callback return HDDL_ERROR_INVAL_TASK_HANDLE
        ASSERT_EQ(exception_ptr, nullptr);
        is_called = true;
    }));
    OV_ASSERT_NO_THROW(req.start_async());
    OV_ASSERT_NO_THROW(req.wait());
    ASSERT_TRUE(is_called);
}

TEST_P(OVInferRequestCallbackTests, syncInferDoesNotCallCompletionCallback) {
    ov::InferRequest req;
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    bool is_called = false;
    req.set_callback([&](std::exception_ptr exception_ptr) {
        ASSERT_EQ(nullptr, exception_ptr);
        is_called = true;
    });
    req.infer();
    ASSERT_FALSE(is_called);
}

// test that can wait all callbacks on dtor
TEST_P(OVInferRequestCallbackTests, canStartSeveralAsyncInsideCompletionCallbackWithSafeDtor) {
    const int NUM_ITER = 10;
    struct TestUserData {
        std::atomic<int> numIter = {0};
        std::promise<bool> promise;
    };
    TestUserData data;

    ov::InferRequest req;
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    OV_ASSERT_NO_THROW(req.set_callback([&](std::exception_ptr exception_ptr) {
        if (exception_ptr) {
            data.promise.set_exception(exception_ptr);
        } else {
            if (data.numIter.fetch_add(1) != NUM_ITER) {
                req.start_async();
            } else {
                data.promise.set_value(true);
            }
        }
    }));
    auto future = data.promise.get_future();
    OV_ASSERT_NO_THROW(req.start_async());
    OV_ASSERT_NO_THROW(req.wait());
    future.wait();
    auto callbackStatus = future.get();
    ASSERT_TRUE(callbackStatus);
    auto dataNumIter = data.numIter - 1;
    ASSERT_EQ(NUM_ITER, dataNumIter);
}

TEST_P(OVInferRequestCallbackTests, returnGeneralErrorIfCallbackThrowException) {
    ov::InferRequest req;
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    OV_ASSERT_NO_THROW(req.set_callback([](std::exception_ptr) {
        OPENVINO_THROW("Throw");
    }));
    OV_ASSERT_NO_THROW(req.start_async());
    ASSERT_THROW(req.wait(), ov::Exception);
}

TEST_P(OVInferRequestCallbackTests, ReturnResultNotReadyFromWaitInAsyncModeForTooSmallTimeout) {
    auto Basic_LSTM_S_GetNetwork = [](size_t thirdDimOut, size_t hiddenSize) {
        size_t num_cells = 10;
        std::pair<float, float> weights_range = {0.f, 10.f};
        std::vector<float>* hidden_memory_init_out = nullptr;
        std::vector<float>* cell_memory_init_out = nullptr;
        auto elem_type = ov::element::f32;

        auto param = std::make_shared<ov::op::v0::Parameter>(elem_type, ov::Shape{1, num_cells * thirdDimOut});

        const size_t batch_size = 1;

        // Reshape_1 [1,thirdDimOut*num_cells] -> [1, num_cells, thirdDimOut]
        std::vector<uint64_t> outFormShapes1 = {batch_size, num_cells, thirdDimOut};
        auto pattern1 = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{3}, outFormShapes1);
        auto reshape1 = std::make_shared<ov::op::v1::Reshape>(param, pattern1, false);

        auto reshape1_shape = reshape1->output(0).get_shape();
        auto H_init = ov::test::utils::deprecated::make_constant<float>(elem_type,
                                                                        {batch_size, hiddenSize},
                                                                        {},
                                                                        true,
                                                                        weights_range.second,
                                                                        weights_range.first);
        auto C_init = ov::test::utils::deprecated::make_constant<float>(elem_type,
                                                                        {batch_size, hiddenSize},
                                                                        {},
                                                                        true,
                                                                        weights_range.second,
                                                                        weights_range.first);
        if (hidden_memory_init_out != nullptr) {
            *hidden_memory_init_out = std::static_pointer_cast<ov::op::v0::Constant>(H_init)->cast_vector<float>();
        }
        if (cell_memory_init_out != nullptr) {
            *cell_memory_init_out = std::static_pointer_cast<ov::op::v0::Constant>(C_init)->cast_vector<float>();
        }
        auto H_t = std::make_shared<ov::op::v0::Parameter>(elem_type, ov::Shape{batch_size, hiddenSize});
        auto C_t = std::make_shared<ov::op::v0::Parameter>(elem_type, ov::Shape{batch_size, hiddenSize});
        H_t->set_friendly_name("hidden_state_1");
        C_t->set_friendly_name("cell_state_1");
        // Body
        auto X = std::make_shared<ov::op::v0::Parameter>(elem_type, ov::Shape{batch_size, 1, reshape1_shape[2]});
        auto weightsNode = ov::test::utils::deprecated::make_constant<float>(elem_type,
                                                                             {4 * hiddenSize, reshape1_shape[2]},
                                                                             {},
                                                                             true,
                                                                             weights_range.second,
                                                                             weights_range.first);
        auto reccurrenceWeightsNode = ov::test::utils::deprecated::make_constant<float>(elem_type,
                                                                                        {4 * hiddenSize, hiddenSize},
                                                                                        {},
                                                                                        true,
                                                                                        weights_range.second,
                                                                                        weights_range.first);

        // lstm [1, 10], [1, 118], [1, 118] -> [1, 118], [1, 118]
        outFormShapes1 = {batch_size, reshape1_shape[2]};
        auto constantX = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{2}, outFormShapes1);
        auto lstm1 = std::make_shared<ov::op::v4::LSTMCell>(std::make_shared<ov::op::v1::Reshape>(X, constantX, false),
                                                            H_t,
                                                            C_t,
                                                            weightsNode,
                                                            reccurrenceWeightsNode,
                                                            hiddenSize);

        auto H_o = lstm1->output(0);
        auto C_o = lstm1->output(1);

        // TensorIterator [1, num_cells, thirdDimOut] [1, 118], [1, 118] -> [1, 118]
        auto body = std::make_shared<ov::Model>(ov::OutputVector{H_o, C_o}, ov::ParameterVector{X, H_t, C_t});

        auto tensor_iterator = std::make_shared<ov::op::v0::TensorIterator>();
        tensor_iterator->set_body(body);

        // input tensor shape: [1, num_cells, thirdDimOut] chunk shape: [1, 1, thirdDimOut]
        tensor_iterator->set_sliced_input(X, reshape1, 0, 1, 1, -1, 1);
        tensor_iterator->set_merged_input(H_t, H_init, H_o);
        tensor_iterator->set_merged_input(C_t, C_init, C_o);

        auto out0 = tensor_iterator->get_iter_value(H_o, -1);

        const size_t output_size = 12;
        auto fc1 = ov::test::utils::make_fully_connected(out0,
                                                         elem_type,
                                                         output_size,
                                                         true,
                                                         {hiddenSize, output_size},
                                                         {weights_range.second},
                                                         {0.f});

        return std::make_shared<ov::Model>(fc1->outputs(), ov::ParameterVector{param}, "Basic_LSTM_S");
    };

    // GetNetwork(3000, 380) make inference around 20ms on GNA SW
    // so increases chances for getting RESULT_NOT_READY
    OV_ASSERT_NO_THROW(execNet = core->compile_model(Basic_LSTM_S_GetNetwork(300, 38), target_device, configuration));
    ov::InferRequest req;
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    std::promise<std::chrono::system_clock::time_point> callbackTimeStamp;
    auto callbackTimeStampFuture = callbackTimeStamp.get_future();
    // add a callback to the request and capture the timestamp
    OV_ASSERT_NO_THROW(req.set_callback([&](std::exception_ptr exception_ptr) {
        if (exception_ptr) {
            callbackTimeStamp.set_exception(exception_ptr);
        } else {
            callbackTimeStamp.set_value(std::chrono::system_clock::now());
        }
    }));
    OV_ASSERT_NO_THROW(req.start_async());
    bool ready = false;
    OV_ASSERT_NO_THROW(ready = req.wait_for({}));
    // get timestamp taken AFTER return from the wait(STATUS_ONLY)
    const auto afterWaitTimeStamp = std::chrono::system_clock::now();
    // IF the callback timestamp is larger than the afterWaitTimeStamp
    // then we should observe false ready result
    if (afterWaitTimeStamp < callbackTimeStampFuture.get()) {
        ASSERT_FALSE(ready);
    }
    OV_ASSERT_NO_THROW(req.wait());
}

TEST_P(OVInferRequestCallbackTests, ImplDoesNotCopyCallback) {
    ov::InferRequest req;
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    {
        auto somePtr = std::make_shared<int>(42);
        OV_ASSERT_NO_THROW(req.set_callback([somePtr](std::exception_ptr exception_ptr) {
            ASSERT_EQ(nullptr, exception_ptr);
            ASSERT_EQ(1, somePtr.use_count());
        }));
    }
    OV_ASSERT_NO_THROW(req.start_async());
    OV_ASSERT_NO_THROW(req.wait());
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
