// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvertU4WeightsZeroPointToScalar;
class TRANSFORMATIONS_API ConvertU4WeightsFloatZeroPointToScalar;
class TRANSFORMATIONS_API ConvertU4WeightsU4ZeroPointToScalar;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief Converts U4 weights zero point to scalar if all values are equal
 */
class ov::pass::ConvertU4WeightsZeroPointToScalar : public ov::pass::GraphRewrite {
public:
    OPENVINO_RTTI("ConvertU4WeightsZeroPointToScalar", "0");
    ConvertU4WeightsZeroPointToScalar();
};

class ov::pass::ConvertU4WeightsFloatZeroPointToScalar : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertU4WeightsFloatZeroPointToScalar", "0");
    ConvertU4WeightsFloatZeroPointToScalar();
};

class ov::pass::ConvertU4WeightsU4ZeroPointToScalar : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertU4WeightsU4ZeroPointToScalar", "0");
    ConvertU4WeightsU4ZeroPointToScalar();
};
