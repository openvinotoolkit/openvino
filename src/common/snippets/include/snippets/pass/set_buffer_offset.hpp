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
 * @interface SetBufferOffset
 * @brief All buffers in body have one common memory pointer. To correct work with them each buffer has own offset for common memory ptr
 *        The pass consistently set offset in buffers.
 *        NOTE: Should be called after Load/Store insertion and before LoadMoveBroadcastToBroadcastLoad because
 *              we cannot fuse Load with non-zero offset and MoveBroadcast
 * @ingroup snippets
 */
class SetBufferOffset: public ngraph::pass::MatcherPass {
public:
    SetBufferOffset();

private:
    size_t current_offset = 0lu;
};

} // namespace pass
} // namespace snippets
} // namespace ngraph
