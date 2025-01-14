// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/fake_quantize.hpp"
#include "openvino/pass/matcher_pass.hpp"

namespace ov {
namespace snippets {
namespace pass {

/**
 * @interface FakeQuantizeDecomposition
 * @ingroup snippets
 * @brief FakeQuantizeDecomposition transformation decomposes FakeQuantize layer.
 *
 * Expression from specification:
 * if x <= min(il, ih):
 *   output = ol
 * elif x > max(il, ih):
 *   output = oh
 * else:
 *   output = round((x - il) / (ih - il) * (levels-1)) / (levels-1) * (oh - ol) + ol
 *
 * Expand brackets:
 *   round(x * (levels-1) / (ih - il) - il * (levels-1) / (ih - il)) * (oh - ol) / (levels-1) + ol
 *
 * Marking:
 *   - isc := (levels-1) / (ih - il)
 *   - ish := -il * isc
 *   - osc := (oh - ol) / (levels-1)
 *   - osh := ol
 * Final expression:
 *   round(x * isc + ish) * osc + osh
 *
 * Some optimizations (example for scalars):
 * 1. If output element type of FQ is U8 and il = 0, ish = 0, osc = 1, osh = 0, there is enough expression: x * isc
 * 2. If output element type of FQ is I8 and ish ~= 128, osc = 1, osh ~= -128, il * isc ~= -128, ih * isc ~= 127 there is enough expression: x * isc
 * 3. If osc = 1, osh = 0, there isn't dequantization
 * 4. If there isn't dequantization and output element type of FQ isn't FP32, there isn't rounding
 *
 * This transformation doesn't support following cases:
 * 1. At least one 'range' input is not Constant
 * 2. At least one 'il' input value greater or equal than 'ih' input value
 *
 */

class FakeQuantizeDecomposition : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("snippets::pass::FakeQuantizeDecomposition");
    FakeQuantizeDecomposition();

    static bool getScalesAndShifts(const std::shared_ptr<const ov::op::v0::FakeQuantize>& fq_node,
                                   std::vector<float>& cl,
                                   std::vector<float>& ch,
                                   std::vector<float>& isc,
                                   std::vector<float>& ish,
                                   std::vector<float>& osc,
                                   std::vector<float>& osh);
    static std::vector<float> calculateScales(const ov::element::Type& out_type,
                                              const std::vector<float>& cl,
                                              const std::vector<float>& ch,
                                              const std::vector<float>& isc,
                                              const std::vector<float>& ish,
                                              const std::vector<float>& osc,
                                              const std::vector<float>& osh);
};

/**
 * @interface CommonFakeQuantizeDecomposition
 * @ingroup snippets
 * @brief CommonFakeQuantizeDecomposition pass applies all needed transformations for
 *        correct FQ Decomposition:
 *          0. Disable Validate() pass after each transformations
 *          1. FakeQuantization decomposition
 *          2. ConstantFolding
 *          3. Validate
 */
class CommonFakeQuantizeDecomposition: public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("snippets::pass::CommonFakeQuantizeDecomposition");

    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;

    static bool is_supported_fq(const std::shared_ptr<const ov::op::v0::FakeQuantize>& fq);
};

}  // namespace pass
}  // namespace snippets
}  // namespace ov
