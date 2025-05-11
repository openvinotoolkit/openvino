// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/concat.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API TotalSequenceLengthPattern;
class TRANSFORMATIONS_API TotalSequenceLengthPatternQwen;

}  // namespace pass
}  // namespace ov

class ov::pass::TotalSequenceLengthPattern : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("TotalSequenceLengthPattern");
    explicit TotalSequenceLengthPattern(const std::shared_ptr<ov::op::v0::Parameter>& max_context_len);
};

/**
 * @brief Qwen model has a specific pattern for TotalSequenceLen place detection.
 *
 * common pattern: Add (PrevSeqLen, CurrentSeqLen)
 *
 * The CurrentSeqLen is presented in this form:
 * CurrentSeqLen: Parameter(name: input_ids) -> ShapeOf -> Gather
 *
 * Before applying this transformation, we already detected the PrevSeqLen place in the PrevSequenceLengthPattern
 * and replaced it with the next subgraph:
 * PrevSeqLen: Subtract (in: Parameter(name: max_context_len), in: CurrentSeqLen)
 *
 **/
class ov::pass::TotalSequenceLengthPatternQwen : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("TotalSequenceLengthPattern", "0");
    explicit TotalSequenceLengthPatternQwen(const std::shared_ptr<ov::op::v0::Parameter>& max_context_len);
};
