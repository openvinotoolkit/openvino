// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/add.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API PositionIDsReplacer;
class TRANSFORMATIONS_API PositionIDsReplacerQwen;
class TRANSFORMATIONS_API PositionIDsReplacerCodeGen2;

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

/**
 * @brief Codegen2 model doesn't use the position_ids input explicitly.
 * Instead, the model infers them from the max_context_len value by generating
 * a range from 0 to max_context_len, applying RoPE and only then Slicing the
 * last token which is not correct in case of 0th iteration (prompt iteration)
 * when values for the entire sequence need to be sliced.
 *
 * We change from this:
 *
 *  ┌─────┐
 *  │Range│
 *  └──┬──┘
 *     │
 *  ┌──┴──┐     ┌─────────┐    ┌─────────┐
 *  │RoPE │     │  Start  │    │   End   │
 *  │Block│     │(prev.seq|    │(cur.seq │
 *  └──┬──┘     │ len)    │    │   len)  │
 *     │        └────┬────┘    └────┬────┘
 *  ┌──┴──┐──────────┘              │
 *  |Slice├─────────────────────────┘
 *  └─────┘
 *
 * To this to Gather by position_ids
 *
 *  ┌─────┐
 *  │Range│
 *  └──┬──┘
 *     │
 *  ┌──┴──┐
 *  │RoPE │
 *  │Block│    ┌──────────────┐
 *  └──┬──┘    │ position_ids │
 *     │       └──────┬───────┘
 *  ┌──┴───┐          │
 *  │Gather├──────────┘
 *  └──────┘
 */
class ov::pass::PositionIDsReplacerCodeGen2 : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("PositionIDsReplacerCodeGen2");
    explicit PositionIDsReplacerCodeGen2(const std::shared_ptr<ov::op::v0::Parameter>& position_ids);
};
