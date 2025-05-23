// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

#include "openvino/op/transpose.hpp"

#include "snippets/lowered/port_descriptor.hpp"

namespace ov {
namespace snippets {
namespace pass {

/**
 * @interface FuseTransposeBrgemm
 * @brief Fuses Transpose with Brgemm node, fusing on both Brgemm inputs and output is supported. Applicable to
 *        Transposes that don't change the position of the last dimension (since Brgemm supports strided rows i/o).
 *        Supported any Transpose order where last index is equal to [rank - 1] - it means that last dimension isn't moved.
 * @ingroup snippets
 */
class FuseTransposeBrgemm: public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("snippets::pass::FuseTransposeBrgemm");
    FuseTransposeBrgemm();

    static bool is_supported_transpose(const Output<Node>& transpose_out);
    static bool is_supported_transpose_order(const std::vector<int32_t>& order);
};

}  // namespace pass
}  // namespace snippets
}  // namespace ov
