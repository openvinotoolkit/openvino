// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/shape_of.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API PrevSequenceLengthPattern;

}  // namespace pass
}  // namespace ov

class ov::pass::PrevSequenceLengthPattern : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("PrevSequenceLengthPattern", "0");
    explicit PrevSequenceLengthPattern(const std::shared_ptr<ov::Node>& unsqueezed_input_ids,
                                       const std::shared_ptr<ov::Node>& max_context_len,
                                       const std::shared_ptr<ov::Node>& position_ids);
};
