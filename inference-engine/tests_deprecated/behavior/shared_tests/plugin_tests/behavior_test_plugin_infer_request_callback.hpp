// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior_test_plugin.h"
#include <mutex>
#include <condition_variable>

using namespace std;
using namespace ::testing;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

namespace {
std::string getTestCaseName(testing::TestParamInfo<BehTestParams> obj) {
    std::string config;
    for (auto&& cfg : obj.param.config) {
        config += "_" + cfg.first + "_" + cfg.second;
    }
    return obj.param.device + "_" + obj.param.input_blob_precision.name() + config;
}
}

TEST_P(BehaviorPluginTestInferRequestCallback, canGetWithNullptr) {
    TestEnv::Ptr testEnv;
    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(GetParam(), testEnv));
    ASSERT_NO_FATAL_FAILURE(testEnv->inferRequest->GetUserData(nullptr, nullptr));
}

TEST_P(BehaviorPluginTestInferRequestCallback, canSetAndGetUserData) {
    TestEnv::Ptr testEnv;
    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(GetParam(), testEnv));
    bool setBool = true;
    bool *getBool = nullptr;

    auto set_sts = testEnv->inferRequest->SetUserData(&setBool, nullptr);
    auto get_sts = testEnv->inferRequest->GetUserData((void **) &getBool, nullptr);
    ASSERT_NE(getBool, nullptr);
    ASSERT_TRUE(*getBool);
    ASSERT_EQ((int) StatusCode::OK, get_sts);
    ASSERT_EQ((int) StatusCode::OK, set_sts);
}

TEST_P(BehaviorPluginTestInferRequestCallback, canCallSyncAndAsyncWithCompletionCallback) {
    TestEnv::Ptr testEnv;
    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(GetParam(), testEnv));

    sts = testEnv->inferRequest->Infer(&response);
    ASSERT_EQ((int) StatusCode::OK, sts) << response.msg;

    bool isCalled = false;
    InferRequest cppRequest(testEnv->inferRequest);
    cppRequest.SetCompletionCallback<std::function<void(InferRequest, StatusCode)>>([&](InferRequest request, StatusCode status) {
        // HSD_1805940120: Wait on starting callback return HDDL_ERROR_INVAL_TASK_HANDLE
        if (GetParam().device != CommonTestUtils::DEVICE_HDDL) {
            ASSERT_EQ((int) StatusCode::OK, status);
        }
        isCalled = true;
    });

    sts = testEnv->inferRequest->StartAsync(nullptr);
    StatusCode waitStatus = testEnv->inferRequest->Wait(IInferRequest::WaitMode::RESULT_READY, nullptr);

    ASSERT_EQ((int) StatusCode::OK, sts);
    ASSERT_EQ((int) StatusCode::OK, waitStatus);
    ASSERT_TRUE(isCalled);
}

// test that can wait all callbacks on dtor
// TODO: check that is able to wait and to not callback tasks! now it isn't !
TEST_P(BehaviorPluginTestInferRequestCallback, canStartAsyncInsideCompletionCallback) {
    TestEnv::Ptr testEnv;
    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(GetParam(), testEnv));
    struct TestUserData {
        bool startAsyncOK = false;
        bool getUserDataOK = false;
        int numIsCalled = 0;
        string device;
    };
    TestUserData data;
    data.device = GetParam().device;
    testEnv->inferRequest->SetUserData(&data, nullptr);
    testEnv->inferRequest->SetCompletionCallback(
            [](InferenceEngine::IInferRequest::Ptr request, StatusCode status) {
                TestUserData *userData = nullptr;
                ResponseDesc desc;
                StatusCode sts = request->GetUserData((void **) &userData, &desc);
                ASSERT_EQ((int) StatusCode::OK, sts) << desc.msg;
                if (sts == StatusCode::OK) {
                    userData->getUserDataOK = true;
                }
                // HSD_1805940120: Wait on starting callback return HDDL_ERROR_INVAL_TASK_HANDLE
                if (userData->device != CommonTestUtils::DEVICE_HDDL) {
                    ASSERT_EQ((int) StatusCode::OK, status);
                }
                userData->numIsCalled++;
                // WA for deadlock
                request->SetCompletionCallback(nullptr);
                sts = request->StartAsync(nullptr);
                if (sts == StatusCode::OK) {
                    userData->startAsyncOK = true;
                }
            });

    sts = testEnv->inferRequest->StartAsync(&response);
    ResponseDesc responseWait;
    StatusCode waitStatus = testEnv->inferRequest->Wait(IInferRequest::WaitMode::RESULT_READY, &responseWait);

    ASSERT_EQ((int) StatusCode::OK, sts) << response.msg;
    ASSERT_EQ((int) StatusCode::OK, waitStatus) << responseWait.msg;
    ASSERT_EQ(1, data.numIsCalled);
    ASSERT_TRUE(data.startAsyncOK);
    ASSERT_TRUE(data.getUserDataOK);
}

// TODO: test that callback tasks not dtor while someone wait them

// test that can wait all callbacks on dtor
TEST_P(BehaviorPluginTestInferRequestCallback, canStartSeveralAsyncInsideCompletionCallbackWithSafeDtor) {
    TestEnv::Ptr testEnv;
    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(GetParam(), testEnv));
    const int NUM_ITER = 10;
    struct TestUserData {
        int numIter = NUM_ITER;
        bool startAsyncOK = true;
        bool getDataOK = true;
        int numIsCalled = 0;
        std::mutex mutex_block_emulation;
        std::condition_variable cv_block_emulation;
        bool isBlocked = true;
        string device;
    };
    TestUserData data;
    data.device = GetParam().device;
    testEnv->inferRequest->SetUserData(&data, nullptr);
    testEnv->inferRequest->SetCompletionCallback(
            [](InferenceEngine::IInferRequest::Ptr request, StatusCode status) {
                TestUserData *userData = nullptr;
                StatusCode sts = request->GetUserData((void **) &userData, nullptr);
                if (sts != StatusCode::OK) {
                    userData->getDataOK = false;
                }
                // HSD_1805940120: Wait on starting callback return HDDL_ERROR_INVAL_TASK_HANDLE
                if (userData->device != CommonTestUtils::DEVICE_HDDL) {
                    ASSERT_EQ((int) StatusCode::OK, status);
                }
                if (--userData->numIter) {
                    sts = request->StartAsync(nullptr);
                    if (sts != StatusCode::OK) {
                        userData->startAsyncOK = false;
                    }
                }
                userData->numIsCalled++;
                if (!userData->numIter) {
                    userData->isBlocked = false;
                    userData->cv_block_emulation.notify_all();
                }
            });

    sts = testEnv->inferRequest->StartAsync(nullptr);
    StatusCode waitStatus = testEnv->inferRequest->Wait(IInferRequest::WaitMode::RESULT_READY, nullptr);
    // intentionally block until notification from callback
    std::unique_lock<std::mutex> lock(data.mutex_block_emulation);
    data.cv_block_emulation.wait(lock, [&]() { return !data.isBlocked; });

    ASSERT_EQ((int) StatusCode::OK, sts);
    ASSERT_EQ((int) StatusCode::OK, waitStatus);


    ASSERT_EQ(NUM_ITER, data.numIsCalled);
    ASSERT_TRUE(data.startAsyncOK);
    ASSERT_TRUE(data.getDataOK);
}

// test that can wait all callbacks on dtor
// FIXME: CVS-8956, dll is unloaded before finishing infer request
TEST_P(BehaviorPluginTestInferRequestCallback, DISABLED_canStartSeveralAsyncInsideCompletionCallbackNoSafeDtor) {
    TestEnv::Ptr testEnv;
    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(GetParam(), testEnv));
    const int NUM_ITER = 10;
    struct TestUserData {
        int numIter = NUM_ITER;
        bool startAsyncOK = true;
        bool getDataOK = true;
        int numIsCalled = 0;
        std::mutex mutex_block_emulation;
        std::condition_variable cv_block_emulation;
        bool isBlocked = true;
    };
    TestUserData data;
    testEnv->inferRequest->SetUserData(&data, nullptr);
    testEnv->inferRequest->SetCompletionCallback(
            [](InferenceEngine::IInferRequest::Ptr request, StatusCode status) {
                TestUserData *userData = nullptr;
                StatusCode sts = request->GetUserData((void **) &userData, nullptr);
                if (sts != StatusCode::OK) {
                    userData->getDataOK = false;
                }
                // WA for deadlock
                if (!--userData->numIter) {
                    request->SetCompletionCallback(nullptr);
                }
                sts = request->StartAsync(nullptr);
                if (sts != StatusCode::OK) {
                    userData->startAsyncOK = false;
                }
                userData->numIsCalled++;
                if (!userData->numIter) {
                    userData->isBlocked = false;
                    userData->cv_block_emulation.notify_all();
                }
            });

    sts = testEnv->inferRequest->StartAsync(nullptr);
    StatusCode waitStatus = testEnv->inferRequest->Wait(IInferRequest::WaitMode::RESULT_READY, nullptr);
    testEnv->inferRequest = nullptr;

    // intentionally block until notification from callback
    std::unique_lock<std::mutex> lock(data.mutex_block_emulation);
    data.cv_block_emulation.wait(lock, [&]() { return !data.isBlocked; });

    ASSERT_EQ((int) StatusCode::OK, sts);
    ASSERT_EQ((int) StatusCode::OK, waitStatus);

    ASSERT_EQ(NUM_ITER, data.numIsCalled);
    ASSERT_TRUE(data.startAsyncOK);
    ASSERT_TRUE(data.getDataOK);
}

// test that can wait all callbacks on dtor
// FIXME: CVS-8956, dll is unloaded before finishing infer request
TEST_P(BehaviorPluginTestInferRequest, DISABLED_canStartSeveralAsyncInsideCompletionCallbackNoSafeDtorWithoutWait) {
    TestEnv::Ptr testEnv;
    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(GetParam(), testEnv));
    const int NUM_ITER = 1;
    struct TestUserData {
        int numIter = NUM_ITER;
        bool startAsyncOK = true;
        bool getDataOK = true;
        int numIsCalled = 0;
        std::mutex mutex_block_emulation;
        std::condition_variable cv_block_emulation;
        bool isBlocked = true;
    };
    TestUserData data;
    testEnv->inferRequest->SetUserData(&data, nullptr);
    testEnv->inferRequest->SetCompletionCallback(
            [](InferenceEngine::IInferRequest::Ptr request, StatusCode status) {
                TestUserData *userData = nullptr;
                StatusCode sts = request->GetUserData((void **) &userData, nullptr);
                if (sts != StatusCode::OK) {
                    userData->getDataOK = false;
                }
                // WA for deadlock
                if (!--userData->numIter) {
                    request->SetCompletionCallback(nullptr);
                }
                sts = request->StartAsync(nullptr);
                if (sts != StatusCode::OK) {
                    userData->startAsyncOK = false;
                }
                userData->numIsCalled++;
                if (!userData->numIter) {
                    userData->isBlocked = false;
                    userData->cv_block_emulation.notify_all();
                }
            });

    sts = testEnv->inferRequest->StartAsync(nullptr);
    testEnv->inferRequest = nullptr;
    testEnv = nullptr;

    // intentionally block until notification from callback
    std::unique_lock<std::mutex> lock(data.mutex_block_emulation);
    data.cv_block_emulation.wait(lock, [&]() { return !data.isBlocked; });

    ASSERT_EQ((int) StatusCode::OK, sts);

    ASSERT_EQ(NUM_ITER, data.numIsCalled);
    ASSERT_TRUE(data.startAsyncOK);
    ASSERT_TRUE(data.getDataOK);
}

// DEAD LOCK with Wait
TEST_P(BehaviorPluginTestInferRequestCallback, DISABLED_canStartSeveralAsyncInsideCompletionCallbackWithWaitInside) {
    TestEnv::Ptr testEnv;
    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(GetParam(), testEnv));
    const int NUM_ITER = 10;
    struct TestUserData {
        int numIter = NUM_ITER;
        bool startAsyncOK = true;
        bool waitOK = true;
        int numIsCalled = 0;
        std::mutex mutex_block_emulation;
        std::condition_variable cv_block_emulation;
        bool isBlocked = true;
    };
    TestUserData data;
    testEnv->inferRequest->SetUserData(&data, nullptr);
    testEnv->inferRequest->SetCompletionCallback(
            [](InferenceEngine::IInferRequest::Ptr request, StatusCode status) {
                TestUserData *userData = nullptr;
                StatusCode sts = request->GetUserData((void **) &userData, nullptr);
                if (sts == StatusCode::OK) {
                    userData->numIsCalled++;
                }
                // WA for deadlock
                if (!--userData->numIter) {
                    request->SetCompletionCallback(nullptr);
                    userData->isBlocked = false;
                    userData->cv_block_emulation.notify_all();
                }
                sts = request->StartAsync(nullptr);
                if (sts != StatusCode::OK) {
                    userData->startAsyncOK = false;
                }
                if (userData->numIter % 2) {
                    sts = request->Wait(IInferRequest::WaitMode::RESULT_READY, nullptr);
                    if (sts != StatusCode::OK) {
                        userData->waitOK = false;
                    }
                }
            });

    sts = testEnv->inferRequest->StartAsync(nullptr);
    testEnv->inferRequest = nullptr;

    // intentionally block until notification from callback
    std::unique_lock<std::mutex> lock(data.mutex_block_emulation);
    data.cv_block_emulation.wait(lock, [&]() { return !data.isBlocked; });

    ASSERT_EQ((int) StatusCode::OK, sts);

    ASSERT_EQ(NUM_ITER, data.numIsCalled);
    ASSERT_TRUE(data.startAsyncOK);
    ASSERT_TRUE(data.waitOK);
}

// TODO: no, this is not correct test. callback throw exception and plugin shouldn't fail? user have to process this by himself.
TEST_P(BehaviorPluginTestInferRequestCallback, DISABLED_returnGeneralErrorIfCallbackThrowException) {
    TestEnv::Ptr testEnv;
    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(GetParam(), testEnv));
    testEnv->inferRequest->SetCompletionCallback(
            [](InferenceEngine::IInferRequest::Ptr, StatusCode status) {
                THROW_IE_EXCEPTION << "returnGeneralErrorIfCallbackThrowException";
            });

    sts = testEnv->inferRequest->StartAsync(nullptr);
    StatusCode waitStatus = INFER_NOT_STARTED;
    while (StatusCode::RESULT_NOT_READY == waitStatus || StatusCode::INFER_NOT_STARTED == waitStatus) {
        waitStatus = testEnv->inferRequest->Wait(IInferRequest::WaitMode::STATUS_ONLY, &response);
    }

    ASSERT_EQ((int) StatusCode::OK, sts);
    ASSERT_EQ(StatusCode::GENERAL_ERROR, waitStatus);
    string refError = "returnGeneralErrorIfCallbackThrowException";
    response.msg[refError.length()] = '\0';
    ASSERT_EQ(refError, response.msg);
}

TEST_P(BehaviorPluginTestInferRequestCallback, inferDoesNotCallCompletionCallback) {
    TestEnv::Ptr testEnv;
    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(GetParam(), testEnv));
    struct TestUserData {
        bool isCalled = false;
    };
    TestUserData data;
    testEnv->inferRequest->SetUserData(&data, nullptr);
    testEnv->inferRequest->SetCompletionCallback(
            [](InferenceEngine::IInferRequest::Ptr request, StatusCode status) {
                TestUserData *userData = nullptr;
                request->GetUserData((void **) &userData, nullptr);
                userData->isCalled = true;
            });
    sts = testEnv->inferRequest->Infer(&response);
    ASSERT_EQ((int) StatusCode::OK, sts);
    ASSERT_FALSE(data.isCalled);
}

// TODO: develop test that request not released until request is done itself? (to check wait in dtor?)
TEST_P(BehaviorPluginTestInferRequestCallback, DISABLED_requestNotReleasedUntilCallbackAreDone) {
    TestEnv::Ptr testEnv;
    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(GetParam(), testEnv));

    struct SyncEnv {
        std::mutex mutex_block_emulation;
        std::condition_variable cv_block_emulation;
        bool isBlocked = true;
        bool isCalled = false;
        typedef std::shared_ptr<SyncEnv> Ptr;
    };
    SyncEnv::Ptr syncEnv = std::make_shared<SyncEnv>();
    testEnv->inferRequest->SetUserData(static_cast<void *>(syncEnv.get()), &response);
    testEnv->inferRequest->SetCompletionCallback(
            [](InferenceEngine::IInferRequest::Ptr request, StatusCode status) {
                SyncEnv *userData = nullptr;
                StatusCode sts = request->GetUserData((void **) &userData, nullptr);
                if (sts == StatusCode::OK) {
                    userData->isCalled = true;
                }
                // intentionally block task for launching tasks after calling dtor for TaskExecutor
                std::unique_lock<std::mutex> lock(userData->mutex_block_emulation);
                userData->cv_block_emulation.wait(lock, [&]() { return userData->isBlocked; });

                // TODO: notify that everything is called
            });

    sts = testEnv->inferRequest->StartAsync(nullptr);
    testEnv->inferRequest = nullptr; //Release();
    syncEnv->isBlocked = false;
    syncEnv->cv_block_emulation.notify_all();

    // TODO: wait until notification from callback
    ASSERT_EQ((int) StatusCode::OK, sts);
    ASSERT_TRUE(syncEnv->isCalled);
}
