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
 * @interface BufferIdentification
 * @brief The pass set identifiers for Buffers in common Buffer system
 *        Note: should be called before ResetBuffer() pass to have correct offsets
 * @ingroup snippets
 */
class BufferIdentification: public ngraph::pass::FunctionPass {
public:
    OPENVINO_RTTI("InsertLoops", "0");
    BufferIdentification() = default;

    bool run_on_model(const std::shared_ptr<ngraph::Function>& m) override;
};

} // namespace pass
} // namespace snippets
} // namespace ngraph
