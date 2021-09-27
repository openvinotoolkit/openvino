// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "ie_iextension.h"
#include "openvino/core/function.hpp"

namespace ov {
namespace test {

class TestsCommon : virtual public ::testing::Test {
protected:
    TestsCommon();
    ~TestsCommon() override;

    static std::string GetTimestamp();
    std::string GetTestName() const;
    std::shared_ptr<ov::Function> read(const std::string& model,
                                       const std::string& weights = "",
                                       const std::vector<InferenceEngine::IExtensionPtr>& exts = {});
};

}  // namespace test
}  // namespace ov
