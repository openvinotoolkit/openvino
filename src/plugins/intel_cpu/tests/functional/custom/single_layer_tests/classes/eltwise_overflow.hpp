// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

enum class EltwiseOverflowKind { UNDERFLOW, OVERFLOW };

typedef std::tuple<EltwiseOverflowKind, ov::Shape> EltwiseOverflowTestParams;

class EltwiseOverflowLayerCPUTest : public testing::WithParamInterface<EltwiseOverflowTestParams>,
                                    virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<EltwiseOverflowTestParams>& obj);

protected:
    void SetUp() override;
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;

private:
    EltwiseOverflowKind overflowKind;
};

}  // namespace test
}  // namespace ov
