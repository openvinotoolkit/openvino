// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>
#include <memory>
#include <future>
#include "ie_extension.h"
#include <condition_variable>
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"
#include <ie_core.hpp>
#include <base/behavior_test_utils.hpp>
#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "ngraph_functions/pass/convert_prc.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include "behavior/infer_request_callback.hpp"

namespace BehaviorTestsDefinitions {
using CallbackTests = BehaviorTestsUtils::BehaviorTestsBasic;

TEST_P(CallbackTests, canCallSyncAndAsyncWithCompletionCallback) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    InferenceEngine::InferRequest req = execNet.CreateInferRequest();
    bool isCalled = false;
    req.SetCompletionCallback<std::function<void(InferenceEngine::InferRequest, InferenceEngine::StatusCode)>>(
            [&](InferenceEngine::InferRequest request, InferenceEngine::StatusCode status) {
                // HSD_1805940120: Wait on starting callback return HDDL_ERROR_INVAL_TASK_HANDLE
                if (targetDevice != CommonTestUtils::DEVICE_HDDL) {
                    ASSERT_EQ(static_cast<int>(InferenceEngine::StatusCode::OK), status);
                }
                isCalled = true;
            });

    req.StartAsync();
    InferenceEngine::StatusCode waitStatus = req.Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);

    ASSERT_EQ(static_cast<int>(InferenceEngine::StatusCode::OK), waitStatus);
    ASSERT_TRUE(isCalled);
}

// test that can wait all callbacks on dtor
TEST_P(CallbackTests, canStartSeveralAsyncInsideCompletionCallbackWithSafeDtor) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    const int NUM_ITER = 10;
    struct TestUserData {
        std::atomic<int> numIter = {0};
        std::promise<InferenceEngine::StatusCode> promise;
    };
    TestUserData data;

    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    InferenceEngine::InferRequest req = execNet.CreateInferRequest();
    req.SetCompletionCallback<std::function<void(InferenceEngine::InferRequest, InferenceEngine::StatusCode)>>(
            [&](InferenceEngine::IInferRequest::Ptr request, InferenceEngine::StatusCode status) {
                if (status != InferenceEngine::StatusCode::OK) {
                    data.promise.set_value(status);
                } else {
                    if (data.numIter.fetch_add(1) != NUM_ITER) {
                        auto sts = request->StartAsync(nullptr);
                        if (sts != InferenceEngine::StatusCode::OK) {
                            data.promise.set_value(sts);
                        }
                    } else {
                        data.promise.set_value(InferenceEngine::StatusCode::OK);
                    }
                }
            });
    auto future = data.promise.get_future();
    req.StartAsync();
    InferenceEngine::StatusCode waitStatus = req.Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
    ASSERT_EQ((int) InferenceEngine::StatusCode::OK, waitStatus);
    future.wait();
    auto callbackStatus = future.get();
    ASSERT_EQ((int) InferenceEngine::StatusCode::OK, callbackStatus);
    auto dataNumIter = data.numIter - 1;
    ASSERT_EQ(NUM_ITER, dataNumIter);
}

TEST_P(CallbackTests, inferDoesNotCallCompletionCallback) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    InferenceEngine::InferRequest req = execNet.CreateInferRequest();
    bool isCalled = false;
    req.SetCompletionCallback<std::function<void(InferenceEngine::InferRequest, InferenceEngine::StatusCode)>>(
            [&](InferenceEngine::IInferRequest::Ptr request, InferenceEngine::StatusCode status) {
                isCalled = true;
            });
    req.Infer();
    ASSERT_FALSE(isCalled);
}

TEST_P(CallbackTests, returnGeneralErrorIfCallbackThrowException) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    InferenceEngine::IInferRequest::Ptr req = static_cast<InferenceEngine::IInferRequest::Ptr &>(execNet.CreateInferRequest());
    req->SetCompletionCallback(
            [](InferenceEngine::IInferRequest::Ptr, InferenceEngine::StatusCode status) {
                THROW_IE_EXCEPTION << "returnGeneralErrorIfCallbackThrowException";
            });

    InferenceEngine::ResponseDesc resp;
    req->StartAsync(&resp);
    InferenceEngine::StatusCode waitStatus = InferenceEngine::StatusCode::INFER_NOT_STARTED;
    while (InferenceEngine::StatusCode::RESULT_NOT_READY == waitStatus ||
           InferenceEngine::StatusCode::INFER_NOT_STARTED == waitStatus) {
        waitStatus = req->Wait(InferenceEngine::IInferRequest::WaitMode::STATUS_ONLY, &resp);
    }
    ASSERT_EQ(InferenceEngine::StatusCode::GENERAL_ERROR, waitStatus);
    ASSERT_NE(std::string(resp.msg).find("returnGeneralErrorIfCallbackThrowException"), std::string::npos);
}
}  // namespace BehaviorTestsDefinitions