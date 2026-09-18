// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>
#include <tuple>

#include "openvino/core/type/element_type.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

// Activation Parameter shape (dynamic + static test shapes) plus the Convolution output channels Cout.
// The input channels and spatial size are derived from data_shape and the input decoration.
struct Conv1x1WeightCompressedShapeParams {
    InputShape data_shape;
    size_t channels_out;
};

// Expected runtime op-type counts (ov::exec_model_info::LAYER_TYPE) after compilation; only the listed
// types are asserted. Runtime op names are plugin specific, so the map is supplied per instantiation.
using Conv1x1ExpectedOpCounts = std::map<std::string, size_t>;

// A decoration is one of "None", "Transpose", "Reshape" ("Reshape" on the input side requires H*W == 1).
using Conv1x1WeightCompressedToMatmulParams = std::tuple<
    Conv1x1WeightCompressedShapeParams,  // geometry
    std::string,                         // input decoration
    std::string,                         // output decoration
    ov::element::Type,                   // activation precision
    ov::element::Type,                   // compressed weights precision
    Conv1x1ExpectedOpCounts,             // expected runtime op-type counts
    ov::AnyMap,                          // additional plugin configuration
    std::string                          // target device
>;

class Conv1x1WeightCompressedToMatmulTest
    : public testing::WithParamInterface<Conv1x1WeightCompressedToMatmulParams>,
      virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<Conv1x1WeightCompressedToMatmulParams>& obj);

protected:
    void SetUp() override;
    void validate() override;

private:
    Conv1x1ExpectedOpCounts expected_op_counts_;
};

}  // namespace test
}  // namespace ov
