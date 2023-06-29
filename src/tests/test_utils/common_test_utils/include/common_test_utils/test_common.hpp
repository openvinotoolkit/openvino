// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>
#include <string>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/core/model.hpp"


namespace ov {
namespace test {
class TestsCommon;
}  // namespace test
}  // namespace ov

// Temporal WA for applications.ai.vpu-accelerators.vpux-plugin repo
// It requires to have TestCommon name in CommonTestUtils namespace
// Need to change namespace in VPU repo and then remove this WA
namespace CommonTestUtils {
using ov::test::TestsCommon;
}  // namespace CommonTestUtils

namespace ov {
namespace test {
class TestsCommon : virtual public ::testing::Test {
protected:
    TestsCommon();
    ~TestsCommon() override;

    static std::string GetTimestamp();
    std::string GetTestName() const;
    std::string GetFullTestName() const;
};

}  // namespace test

class SharedRTInfo;

class ModelAccessor {
    std::weak_ptr<ov::Model> m_function;

public:
    ModelAccessor(std::weak_ptr<ov::Model> f) : m_function(std::move(f)) {}

    std::shared_ptr<SharedRTInfo> get_shared_info() const;
};

class NodeAccessor {
    std::weak_ptr<ov::Node> m_node;

public:
    NodeAccessor(std::weak_ptr<ov::Node> node) : m_node(std::move(node)) {}

    std::set<std::shared_ptr<SharedRTInfo>> get_shared_info() const;
};

}  // namespace ov
