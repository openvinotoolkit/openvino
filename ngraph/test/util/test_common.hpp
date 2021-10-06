// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "openvino/core/function.hpp"

namespace InferenceEngine {
class IExtension;
}

namespace ov {
namespace test {

class TestsCommon : virtual public ::testing::Test {
protected:
    TestsCommon();
    ~TestsCommon() override;

    static std::string GetTimestamp();
    std::string GetTestName() const;
};

}  // namespace test
}  // namespace ov
