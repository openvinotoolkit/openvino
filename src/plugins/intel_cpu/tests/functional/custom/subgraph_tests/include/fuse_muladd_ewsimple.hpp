// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

namespace ov {
namespace test {

using FuseMulAddAndEwSimpleParams = std::tuple<ov::Shape,         // Input shape
                                               ov::element::Type  // Input precision
                                               >;

class FuseMulAddAndEwSimpleTest : public testing::WithParamInterface<FuseMulAddAndEwSimpleParams>,
                                  public CPUTestUtils::CPUTestsBase,
                                  virtual public SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<FuseMulAddAndEwSimpleParams> obj);

protected:
    void SetUp() override;
    virtual void CreateGraph() = 0;

    ov::Shape inputShape;
    ov::element::Type inPrec;
};

class FuseMulAddAndEwSimpleTest1 : public FuseMulAddAndEwSimpleTest {
protected:
    void CreateGraph() override;
};

class FuseMulAddAndEwSimpleTest2 : public FuseMulAddAndEwSimpleTest {
protected:
    void CreateGraph() override;
};

class FuseMulAddAndEwSimpleTest3 : public FuseMulAddAndEwSimpleTest {
protected:
    void CreateGraph() override;
};

}  // namespace test
}  // namespace ov
