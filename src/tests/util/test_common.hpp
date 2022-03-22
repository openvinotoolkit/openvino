// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <memory>
#include <openvino/core/model.hpp>
#include <string>
#include <utility>

namespace ov {

class SharedRTInfo;

namespace test {

class TestsCommon : virtual public ::testing::Test {
protected:
    TestsCommon();
    ~TestsCommon() override;

    static std::string GetTimestamp();
    std::string GetTestName() const;
};

}  // namespace test

class ModelAccessor {
    std::weak_ptr<Model> m_function;

public:
    ModelAccessor(std::weak_ptr<Model> f) : m_function(std::move(f)) {}

    std::shared_ptr<SharedRTInfo> get_shared_info() const;
};

class NodeAccessor {
    std::weak_ptr<Node> m_node;

public:
    NodeAccessor(std::weak_ptr<Node> node) : m_node(std::move(node)) {}

    std::set<std::shared_ptr<SharedRTInfo>> get_shared_info() const;
};
}  // namespace ov
