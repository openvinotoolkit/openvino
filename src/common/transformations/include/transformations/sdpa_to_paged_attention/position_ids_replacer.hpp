// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/add.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"
#include "transformations_visibility.hpp"

#include "openvino/op/range.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/einsum.hpp"

#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/einsum.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/utils/utils.hpp"

static ov::Output<ov::Node> insert_identity(const ov::Output<ov::Node>& in_node) {
    auto axis_1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto identity_1 = std::make_shared<ov::op::v0::Unsqueeze>(in_node, axis_1);
    return std::make_shared<ov::op::v15::Squeeze>(identity_1, axis_1);
}

using ResultVector = std::vector<std::shared_ptr<ov::op::v0::Result>>;
namespace ov {
namespace pass {

class TRANSFORMATIONS_API PositionIDsReplacer;
class TRANSFORMATIONS_API PositionIDsReplacerQwen;
class TRANSFORMATIONS_API ReplaceSliceStartRangeCodegen2;

}  // namespace pass
}  // namespace ov

class ov::pass::PositionIDsReplacer : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("PositionIDsReplacer");
    explicit PositionIDsReplacer(const Output<Node>& position_ids);
};

/**
 * @brief Qwen model expects data processing in order, the "position ids" input is detached and
 * is not explicitly used in the model. The model uses implicitly defined "position ids" based
 * on the past KV cache size.
 *
 * To use this model in Continuous batching mode, we need to apply position_ids and
 * use the corresponding rotary_emb_cos/rotary_emb_sin.
 * For this, we replace
 *      rotary_emb_cos/rotary_emb_sin -> Slice -> Slice
 * With
 *      rotary_emb_cos/rotary_emb_sin -> Gather(by position_ids)
 * Which enables applying RoPE for each token independently of their order in the input tensor.
 */
class ov::pass::PositionIDsReplacerQwen : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("PositionIDsReplacerQwen");
    explicit PositionIDsReplacerQwen(const Output<Node>& position_ids);
};

class ov::pass::ReplaceSliceStartRangeCodegen2 : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ReplaceSliceStartRangeCodegen2");
    explicit ReplaceSliceStartRangeCodegen2(const std::shared_ptr<ov::op::v0::Unsqueeze>& unsqueezed_position_ids,
                                            const std::shared_ptr<ov::op::v0::Parameter>& max_context_len);
};