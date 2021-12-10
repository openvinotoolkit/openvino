// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"

namespace SubgraphTestsDefinitions {

using SimpleIfParamsTuple = typename std::tuple<
        std::vector<ov::test::InputShape>,   // Input shapes
        ov::test::ElementType,               // Network precision
        bool,                                // If condition
        std::string                          // Device name
>;

class SimpleIfTest:
        public testing::WithParamInterface<SimpleIfParamsTuple>,
        virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<SimpleIfParamsTuple> &obj);
protected:
    void SetUp() override;
};

class SimpleIf2OutTest : public SimpleIfTest {
protected:
    void SetUp() override;
};

class SimpleIfNotConstConditionTest : public SimpleIfTest {
public:
    void generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) override;

protected:
    void SetUp() override;

    bool condition;
};

class SimpleIfNotConstConditionAndInternalDynamismTest : public SimpleIfNotConstConditionTest {
protected:
    void SetUp() override;
};

}  // namespace SubgraphTestsDefinitions
