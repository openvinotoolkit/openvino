// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

using SimpleIfParamsTuple = typename std::tuple<std::vector<ov::test::InputShape>,  // Input shapes
                                                ov::test::ElementType,              // Network precision
                                                bool,                               // If condition
                                                std::string                         // Device name
                                                >;

class SimpleIfTest : public testing::WithParamInterface<SimpleIfParamsTuple>,
                     virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<SimpleIfParamsTuple>& obj);

protected:
    void SetUp() override;
    void compare(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual) override;

    size_t inferNum = 0;
};

class SimpleIf2OutTest : public SimpleIfTest {
protected:
    void SetUp() override;
};

class SimpleIfNotConstConditionTest : public SimpleIfTest {
public:
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;

protected:
    void SetUp() override;

    bool condition;
};

class SimpleIfNotConstConditionAndInternalDynamismTest : public SimpleIfNotConstConditionTest {
protected:
    void SetUp() override;
};

class SimpleIfNotConstConditionAndDimsIncreaseTest : public SimpleIfNotConstConditionTest {
protected:
    void SetUp() override;
    void compare(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual) override;
};

class SimpleIfNotConstConditionUnusedOutputPortsTest : public SimpleIfNotConstConditionTest {
protected:
    void SetUp() override;
};
}  // namespace test
}  // namespace ov
