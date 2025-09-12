// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov::snippets::pass {

/**
 * @interface MatMulToBrgemm
 * @brief Replaces ov::MatMul with snippets::op::Brgemm operation (only non-trasposing MatMuls are currently supported)
 * @ingroup snippets
 */
class MatMulToBrgemm : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("snippets::pass::MatMulToBrgemm");
    MatMulToBrgemm();
};

}  // namespace ov::snippets::pass
