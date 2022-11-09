// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pattern/matcher.hpp>

namespace ngraph {
namespace snippets {
namespace pass {

/**
 * @interface AlignElementType
 * @brief Wrap sequence of operations which doesn't support execution on original element type by ConvertSaturation
 *        and reset element type for type relaxed nodes inside body to align element type between nodes.
 *        Example 1:
 *          - After FQ decomposition there may be Convert[U8/I8]. If after the Convert there are other operations
 *            that don't support U8/I8, new ConvertSaturation[exec_type] will be inserted after the FQ decomposition
 *            to execute these operations on supported element type
 *        Example 2:
 *          - Input[I8] -> Unsupported I8 op -> Movement op -> Output[I8]. There will be inserted two ConvertSaturation:
 *              * ConvertSatiration[exec_type] before op which is unsupported I8
 *              * ConvertSaturation[I8] before Movement op to return original low precision.
 *        Note: We cannot just remove original Convert[I8/U8] in Example 1 because we should cover two things:
 *              * allow execution of operations on supported element type for them
 *              * keep computations mathematically equivalent to the original function
 *              Thus, for these cases we should have the following pipeline: FP32 -> Convert[I8/U8] -> Convert[FP32] -> FP32
 *        Note: We shouldn't call validate_and_infer_type() after Convert insertions to avoid element type conflicts on inputs of ops
 * @ingroup snippets
 */
class AlignElementType: public ngraph::pass::FunctionPass {
public:
    OPENVINO_RTTI("AlignElementType", "0");
    AlignElementType(const ov::element::Type exec_type = ov::element::f32);
    bool run_on_model(const std::shared_ptr<ngraph::Function>& m) override;

    static bool opNeedsAlignElementType(const std::shared_ptr<ov::Node>& n, const ov::element::Type exec_type = ov::element::f32);
private:
    ov::element::Type exec_type;
};

}  // namespace pass
}  // namespace snippets
}  // namespace ngraph
