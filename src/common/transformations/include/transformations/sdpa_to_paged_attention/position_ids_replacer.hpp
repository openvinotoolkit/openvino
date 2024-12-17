// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/add.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API PositionIDsReplacer;
class TRANSFORMATIONS_API PositionIDsReplacerQwen;

}  // namespace pass
}  // namespace ov

class ov::pass::PositionIDsReplacer : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("PositionIDsReplacer", "0");
    explicit PositionIDsReplacer(const Output<Node>& position_ids);
};

/**
 * @brief Qwen model has a specific feature in the model structure not to use position_ids input,
 * this input is detached. The model expects data processing in order.
 *
 * To use this model in Continuous batching mode, we need to apply position_ids and
 * use the corresponding rotary_emb_cos/rotary_emb_sin.
 * For this, we replace
 *      rotary_emb_cos/rotary_emb_sin -> Slice -> Slice
 * With
 *      rotary_emb_cos/rotary_emb_sin -> Slice -> Gather(by position_ids)
 */
class ov::pass::PositionIDsReplacerQwen : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("PositionIDsReplacerQwen", "0");
    explicit PositionIDsReplacerQwen(const Output<Node>& position_ids);
};
