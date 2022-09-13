// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/op/fake_quantize.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API FakeQuantizeDecomposition;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief FakeQuantizeDecomposition transformation decomposes FakeQuantize layer.
 *
 * Expression from specification:
 * if x <= min(input_low, input_high):
 *   output = output_low
 * elif x > max(input_low, input_high):
 *   output = output_high
 * else:
 *   output = round((x - input_low) / (input_high - input_low) * (levels-1)) / (levels-1) * (output_high - output_low) +
 * output_low
 *
 * expand brackets into round:
 * round(x * (levels-1) / (input_high - input_low) - input_low * (levels-1) / (input_high - input_low))
 * div on (levels-1) and mult on (output_high - output_low) => mult on (output_high - output_low) / (levels-1)
 *
 *  =>
 * round(x * (levels-1) / (input_high - input_low) - input_low * (levels-1) / (input_high - input_low)) * (output_high -
 * output_low) / (levels-1) + output_low
 *
 * This transformation doesn't support following cases:
 * 1. At least one 'range' input is not Constant
 * 2. At least one 'input_low' input value greater or equal than 'input_high' input value
 *
 */

class ngraph::pass::FakeQuantizeDecomposition : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("FakeQuantizeDecomposition", "0");
    FakeQuantizeDecomposition(const bool constant_weights = true);

    static bool isAllScalarConstant(const std::shared_ptr<const ngraph::Node>& node);
    static bool getScalesAndShifts(const std::shared_ptr<const ngraph::op::v0::FakeQuantize>& fq_node,
                                   std::vector<float>& cl,
                                   std::vector<float>& ch,
                                   std::vector<float>& isc,
                                   std::vector<float>& ish,
                                   std::vector<float>& osc,
                                   std::vector<float>& osh);
    static std::vector<float> calculateScales(const ngraph::element::Type& out_type,
                                              const std::vector<float>& cl,
                                              const std::vector<float>& ch,
                                              const std::vector<float>& isc,
                                              const std::vector<float>& ish,
                                              const std::vector<float>& osc,
                                              const std::vector<float>& osh);
};
