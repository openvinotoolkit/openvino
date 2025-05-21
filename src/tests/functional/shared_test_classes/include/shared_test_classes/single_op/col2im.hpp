// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/file_utils.hpp"
#include "openvino/op/col2im.hpp"

namespace ov {
namespace test {
namespace Col2Im {
using Col2ImOpsSpecificParams =  std::tuple<
    ov::Shape,                                          // data shape
    std::vector<int64_t>,                               // output size values
    std::vector<int64_t>,                               // kernel size values
    ov::Strides,                                        // strides
    ov::Strides,                                        // dilations
    ov::Shape,                                          // pads_begin
    ov::Shape                                           // pads_end
>;

using Col2ImLayerSharedTestParams = std::tuple<
    Col2ImOpsSpecificParams,
    ov::element::Type,                                  // data precision
    ov::element::Type,                                  // size precision

    ov::test::TargetDevice                              // device name
>;

class Col2ImLayerSharedTest : public testing::WithParamInterface<Col2ImLayerSharedTestParams>,
                              virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<Col2ImLayerSharedTestParams> &obj);

protected:
    void SetUp() override;
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
};
}  // namespace Col2Im
}  // namespace test
}  // namespace ov
