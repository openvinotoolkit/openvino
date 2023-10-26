// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvertMatrixNmsToMatrixNmsIE;

}  // namespace pass
}  // namespace ov

class ov::pass::ConvertMatrixNmsToMatrixNmsIE : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertMatrixNmsToMatrixNmsIE", "0");
    ConvertMatrixNmsToMatrixNmsIE(bool force_i32_output_type = true);
};
