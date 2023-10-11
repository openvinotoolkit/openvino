// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// NOTE: WILL BE REWORKED (31905)

#pragma once

#include "ov_models/utils/ov_helpers.hpp"
#include "common_test_utils/common_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
namespace subgraph {

typedef std::tuple<
    std::vector<InputShape>,           // input shapes
    ngraph::helpers::EltwiseTypes,     // eltwise op type
    ngraph::helpers::InputLayerType,   // secondary input type
    ov::test::utils::OpType,           // op type
    ElementType,                       // Net precision
    ElementType,                       // In precision
    ElementType,                       // Out precision
    TargetDevice,                      // Device name
    ov::AnyMap                         // Additional network configuration
> EltwiseTestParams;

class EltwiseLayerTest : public testing::WithParamInterface<EltwiseTestParams>,
                         virtual public SubgraphBaseTest {
protected:
    void SetUp() override;

public:
    static std::string getTestCaseName(const testing::TestParamInfo<EltwiseTestParams>& obj);

private:
    void transformInputShapesAccordingEltwise(const ov::PartialShape& secondInputShape);
};
} // namespace subgraph
} // namespace test
} // namespace ov
