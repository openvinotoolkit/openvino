// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <gmock/gmock.h>

#include "utils/log.hpp"
namespace ov {
namespace mock_auto_plugin {

class MockLog : public Log {
public:
    MOCK_METHOD(void, print, (std::stringstream & stream), (override));
    MockLog(std::string unittest) : Log(unittest) {}
    static MockLog* get_instance() {
        if (m_mocklog == NULL) {
            m_mocklog = new MockLog("unittest");
        }
        return m_mocklog;
    }
    static void release() {
        if (m_mocklog) {
            delete m_mocklog;
            m_mocklog = NULL;
        }
    }
    static MockLog* m_mocklog;
};
}  // namespace mock_auto_plugin
}  // namespace ov
