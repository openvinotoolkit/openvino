// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace SubgraphTestsDefinitions {

using SplitConcatMemoryParamsTuple = typename std::tuple<
    CommonTestUtils::SizeVector, // input shapes
    ov::element::Type_t,         // precision
    int,                         // axis of split
    std::string                  // device name
>;

class SplitConcatMemory : public testing::WithParamInterface<SplitConcatMemoryParamsTuple>,
                          virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ParamType>& obj);

protected:
    void SetUp() override;

    ov::element::Type_t netPrecision;
    int axis;
};

}  // namespace SubgraphTestsDefinitions
