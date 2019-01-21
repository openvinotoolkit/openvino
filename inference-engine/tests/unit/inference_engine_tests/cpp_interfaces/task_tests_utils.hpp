// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>
#include <gmock/gmock-spec-builders.h>
#include <thread>
#include <mutex>
#include <condition_variable>

using namespace InferenceEngine;
using namespace ::testing;
using namespace std;

class MetaThread {
    bool _isThreadStarted;
    std::mutex _isThreadStartedMutex;
    std::condition_variable _isThreadStartedCV;

    bool _isThreadFinished;
    std::mutex _isThreadFinishedMutex;
    std::condition_variable _isThreadFinishedCV;

    std::thread _thread;
    std::function<void()> _function;
public:
    bool exceptionWasThrown;

    MetaThread(std::function<void()> function)
            : _function(function), _isThreadStarted(false), exceptionWasThrown(false), _isThreadFinished(false) {
        _thread = std::thread([this]() {
            _isThreadStarted = true;
            _isThreadStartedCV.notify_all();
            try {
                _function();
            } catch (...) {
                exceptionWasThrown = true;
            }
            _isThreadFinished = true;
            _isThreadFinishedCV.notify_all();
        });
    }

    ~MetaThread() {
        join();
    }

    void waitUntilThreadStarted() {
        std::unique_lock<std::mutex> lock(_isThreadStartedMutex);
        _isThreadStartedCV.wait(lock, [this]() { return _isThreadStarted; });
    }

    void waitUntilThreadFinished() {
        std::unique_lock<std::mutex> lock(_isThreadFinishedMutex);
        _isThreadFinishedCV.wait(lock, [this]() { return _isThreadFinished; });
    }

    void join() {
        if (_thread.joinable()) _thread.join();
    }

    typedef std::shared_ptr<MetaThread> Ptr;
};
