// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/common_utils.hpp"

#include "gtest/gtest.h"
#include "openvino/core/except.hpp"
#include "precomp.hpp"

namespace ov {
namespace test {
namespace utils {

std::ostream& operator<<(std::ostream& os, OpType type) {
    switch (type) {
    case OpType::SCALAR:
        os << "SCALAR";
        break;
    case OpType::VECTOR:
        os << "VECTOR";
        break;
    default:
        OPENVINO_THROW("NOT_SUPPORTED_OP_TYPE");
    }
    return os;
}

std::string generateTestFilePrefix() {
    // Generate unique file names based on test name, thread id and timestamp
    // This allows execution of tests in parallel (stress mode)
    auto testInfo = ::testing::UnitTest::GetInstance()->current_test_info();
    std::string testName = testInfo->test_case_name();
    testName += testInfo->name();
    testName = std::to_string(std::hash<std::string>()(testName));
    std::stringstream ss;
    auto ts = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now().time_since_epoch());
    ss << testName << "_" << std::this_thread::get_id() << "_" << ts.count();
    testName = ss.str();
    return testName;
}

}  // namespace utils
}  // namespace test
}  // namespace ov
