// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvertMulticlassNmsToMulticlassNmsIE;

}  // namespace pass
}  // namespace ov

class ov::pass::ConvertMulticlassNmsToMulticlassNmsIE : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertMulticlassNmsToMulticlassNmsIE");
    ConvertMulticlassNmsToMulticlassNmsIE(bool force_i32_output_type = true);
};
