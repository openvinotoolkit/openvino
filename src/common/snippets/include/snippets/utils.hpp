// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A file contains public utilities.
 * @file utils.hpp
 */
#pragma once

#include "snippets_isa.hpp"
#include "emitter.hpp"

namespace ngraph {
namespace snippets {
namespace utils {

// Get non-scalar Constant count that will be created after FakeQuantize decomposition.
// This count is needed to know exact count of non-scalar Constants during tokenization.
auto get_non_scalar_constant_count_for_fq(const std::shared_ptr<ngraph::opset1::FakeQuantize>& fq) -> size_t;

// Check if operation is Parameter/Constant/Result
auto is_special_op(const std::shared_ptr<ov::Node>& n) -> bool;

// Check if operation supports only execution element type f32
// NOTE: In the future this check should be updated by addition of new operations: Movement ops, MatMul, etc.
auto op_supports_only_exec_type(const std::shared_ptr<ov::Node>& n) -> bool;

// Check if executable operation supports only execution element type f32
// NOTE: Executable op is node that isn't Parameter/Constant/Result
inline auto is_executable_op_only_on_exec_type(const std::shared_ptr<ov::Node>& n) -> bool {
    return ngraph::snippets::utils::op_supports_only_exec_type(n) && !ngraph::snippets::utils::is_special_op(n);
}

inline auto is_scalar_constant(const std::shared_ptr<ngraph::Node>& source_output_node) -> bool {
    return ngraph::is_type<ngraph::opset1::Constant>(source_output_node) && ngraph::shape_size(source_output_node->get_shape()) == 1;
}

} // namespace utils
} // namespace snippets
} // namespace ngraph