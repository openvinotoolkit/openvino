// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>
#include <memory>
#include "ie_extension.h"
#include <condition_variable>
#include "functional_test_utils/layer_test_utils.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"
#include <ie_core.hpp>
#include <functional_test_utils/behavior_test_utils.hpp>
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
TEST_P(CallbackTests, canStartAsyncInsideCompletionCallback) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    struct TestUserData {
        bool startAsyncOK = false;
        int numIsCalled = 0;
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
                // HSD_1805940120: Wait on starting callback return HDDL_ERROR_INVAL_TASK_HANDLE
                if (targetDevice != CommonTestUtils::DEVICE_HDDL) {
                    ASSERT_EQ(static_cast<int>(InferenceEngine::StatusCode::OK), status);
                }
                data.numIsCalled++;
                // WA for deadlock
                request->SetCompletionCallback(nullptr);
                InferenceEngine::StatusCode sts = request->StartAsync(nullptr);
                if (sts == InferenceEngine::StatusCode::OK) {
                    data.startAsyncOK = true;
                }
            });

    req.StartAsync();
    InferenceEngine::ResponseDesc responseWait;
    InferenceEngine::StatusCode waitStatus = req.Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);

    ASSERT_EQ(static_cast<int>(InferenceEngine::StatusCode::OK), waitStatus) << responseWait.msg;
    ASSERT_EQ(1, data.numIsCalled);
    ASSERT_TRUE(data.startAsyncOK);
}

// test that can wait all callbacks on dtor
TEST_P(CallbackTests, canStartSeveralAsyncInsideCompletionCallbackWithSafeDtor) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    const int NUM_ITER = 10;
    struct TestUserData {
        int numIter = NUM_ITER;
        bool startAsyncOK = true;
        std::atomic<int> numIsCalled{0};
        std::mutex mutex_block_emulation;
        std::condition_variable cv_block_emulation;
        bool isBlocked = true;
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
                // HSD_1805940120: Wait on starting callback return HDDL_ERROR_INVAL_TASK_HANDLE
                if (targetDevice != CommonTestUtils::DEVICE_HDDL) {
                    ASSERT_EQ(static_cast<int>(InferenceEngine::StatusCode::OK), status);
                }
                if (--data.numIter) {
                    InferenceEngine::StatusCode sts = request->StartAsync(nullptr);
                    if (sts != InferenceEngine::StatusCode::OK) {
                        data.startAsyncOK = false;
                    }
                }
                data.numIsCalled++;
                if (!data.numIter) {
                    data.isBlocked = false;
                    data.cv_block_emulation.notify_all();
                }
            });

    req.StartAsync();
    InferenceEngine::ResponseDesc responseWait;
    InferenceEngine::StatusCode waitStatus = req.Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
    // intentionally block until notification from callback
    std::unique_lock<std::mutex> lock(data.mutex_block_emulation);
    data.cv_block_emulation.wait(lock, [&]() { return !data.isBlocked; });

    ASSERT_EQ((int) InferenceEngine::StatusCode::OK, waitStatus) << responseWait.msg;
    ASSERT_EQ(NUM_ITER, data.numIsCalled);
    ASSERT_TRUE(data.startAsyncOK);
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

TEST_P(CallbackTests, canStartAsyncInsideCompletionCallbackNoSafeDtor) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    struct TestUserData {
        int numIter = 0;
        bool startAsyncOK = true;
        bool getDataOK = true;
        std::atomic<int> numIsCalled{0};
        std::mutex mutex_block_emulation;
        std::condition_variable cv_block_emulation;
        bool isBlocked = true;

        TestUserData(int i) : numIter(i) {}
    };
    TestUserData data(1);

    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    InferenceEngine::InferRequest req = execNet.CreateInferRequest();

    req.SetCompletionCallback<std::function<void(InferenceEngine::InferRequest, InferenceEngine::StatusCode)>>(
            [&](InferenceEngine::IInferRequest::Ptr request, InferenceEngine::StatusCode status) {
                // WA for deadlock
                if (!--data.numIter) {
                    request->SetCompletionCallback(nullptr);
                }
                InferenceEngine::StatusCode sts = request->StartAsync(nullptr);
                if (sts != InferenceEngine::StatusCode::OK) {
                    data.startAsyncOK = false;
                }
                data.numIsCalled++;
                if (!data.numIter) {
                    data.isBlocked = false;
                    data.cv_block_emulation.notify_one();
                }
            });
    req.StartAsync();
    InferenceEngine::ResponseDesc responseWait;
    InferenceEngine::StatusCode waitStatus = req.Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
    // intentionally block until notification from callback
    std::unique_lock<std::mutex> lock(data.mutex_block_emulation);
    data.cv_block_emulation.wait(lock, [&]() { return !data.isBlocked; });

    ASSERT_EQ(static_cast<int>(InferenceEngine::StatusCode::OK), waitStatus);

    ASSERT_EQ(1, data.numIsCalled);
    ASSERT_TRUE(data.startAsyncOK);
    ASSERT_TRUE(data.getDataOK);
}

TEST_P(CallbackTests, canStartSeveralAsyncInsideCompletionCallbackNoSafeDtor) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    struct TestUserData {
        int numIter = 0;
        bool startAsyncOK = true;
        bool getDataOK = true;
        std::atomic<int> numIsCalled{0};
        std::mutex mutex_block_emulation;
        std::condition_variable cv_block_emulation;
        bool isBlocked = true;

        TestUserData(int i) : numIter(i) {}
    };
    TestUserData data(10);

    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    InferenceEngine::InferRequest req = execNet.CreateInferRequest();

    req.SetCompletionCallback<std::function<void(InferenceEngine::InferRequest, InferenceEngine::StatusCode)>>(
            [&](InferenceEngine::IInferRequest::Ptr request, InferenceEngine::StatusCode status) {
                // WA for deadlock
                if (!--data.numIter) {
                    request->SetCompletionCallback(nullptr);
                }
                InferenceEngine::StatusCode sts = request->StartAsync(nullptr);
                if (sts != InferenceEngine::StatusCode::OK) {
                    data.startAsyncOK = false;
                }
                data.numIsCalled++;
                if (!data.numIter) {
                    data.isBlocked = false;
                    data.cv_block_emulation.notify_one();
                }
            });
    req.StartAsync();
    InferenceEngine::ResponseDesc responseWait;
    InferenceEngine::StatusCode waitStatus = req.Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
    // intentionally block until notification from callback
    std::unique_lock<std::mutex> lock(data.mutex_block_emulation);
    data.cv_block_emulation.wait(lock, [&]() { return !data.isBlocked; });

    ASSERT_EQ(static_cast<int>(InferenceEngine::StatusCode::OK), waitStatus);

    ASSERT_EQ(10, data.numIsCalled);
    ASSERT_TRUE(data.startAsyncOK);
    ASSERT_TRUE(data.getDataOK);
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