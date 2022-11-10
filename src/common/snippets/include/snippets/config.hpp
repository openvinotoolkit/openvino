// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace ngraph {
namespace snippets {


/**
 * @interface SubgraphConfig
 * @brief Config to know which transformations should be called.
 *        It helps to avoid overheads of extra transformation calls
 * @ingroup snippets
 */

struct SubgraphConfig {
    // True if Subgraph contains FakeQuantize -> FQ decomposition should be called
    bool m_is_quantized = false;
    // True if we should align element types indise body
    bool m_is_needed_to_align_precision = false;
    // True if Subgraph contains TypeRelaxed nodes -> for several streams in tp mode we should copy body using mutexes
    // because TypeRelaxed::copy_with_new_inputs() isn't save-thread method
    bool m_has_type_relaxed_ops = false;
    // True if we should check runtime info for nodes to call specific needed transformations
    bool m_need_fill_tail_register = false;
    // True if we should go through whole body to check for where loops should be explicitly inserted.
    // Otherwise, we insert Loops on Parameters and Results - for example, it's optimized out for subgraph with only Eltwise ops
    bool m_explicit_loop_insertion = false;
    // True if body has operations that don't support plugin-side domain optimizations
    // (e.g. Transpose, Softmax, MatMul in general doesn't support dimensions collapsing)
    bool m_has_domain_sensitive_ops = false;
    // True if one evaluation optimizations are enabled
    bool m_one_evaluation_optimizations = true;
};

}  // namespace snippets
}  // namespace ngraph
