// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <gmock/gmock.h>
#include "utils/log.hpp"
using namespace MockMultiDevicePlugin;
namespace MockMultiDevice {

class MockLog : public Log {
public:
    MOCK_METHOD(void, print, (std::stringstream& stream),  (override));
    MockLog(std::string unittest):Log(unittest) {
    }
    static MockLog* GetInstance() {
        if (_mockLog == NULL) {
            _mockLog = new MockLog("unittest");
        }
        return _mockLog;
    }
    static void Release() {
        if (_mockLog) {
            delete _mockLog;
            _mockLog = NULL;
        }
    }
    static MockLog* _mockLog;
};
}// namespace MockMultiDevice
