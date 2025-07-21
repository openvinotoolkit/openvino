// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// NOTE: WILL BE REWORKED (31905)

#pragma once

#include "common_test_utils/test_enums.hpp"
#include "common_test_utils/common_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
typedef std::tuple<
    std::vector<InputShape>,           // input shapes
    ov::test::utils::EltwiseTypes,     // eltwise op type
    ov::test::utils::InputLayerType,   // secondary input type
    ov::test::utils::OpType,           // op type
    ElementType,                       // Model type
    ElementType,                       // In type
    ElementType,                       // Out type
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
} // namespace test
} // namespace ov
