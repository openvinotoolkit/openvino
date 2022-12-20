// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pattern/matcher.hpp>

namespace ngraph {
namespace snippets {
namespace pass {

/**
 * @interface PropagateBufferOffset
 * @brief All buffers in body have one common memory pointer. To correct work with them each buffer has own offset for common memory ptr
 *        The pass consistently set offset in the corresponding for Buffer MemoryAccess nodes: Load, Store, MatMul.
 * @ingroup snippets
 */
class PropagateBufferOffset: public ngraph::pass::MatcherPass {
public:
    PropagateBufferOffset();

private:
    size_t current_offset = 0lu;
};

} // namespace pass
} // namespace snippets
} // namespace ngraph
