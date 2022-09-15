// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pattern/matcher.hpp>

namespace ngraph {
namespace snippets {
namespace pass {

/**
 * @interface InsertLoops
 * @brief Insert explicit Loop operations into the body to process multiple data entities during one kernel execution
 * @ingroup snippets
 */
class InsertLoops: public ngraph::pass::FunctionPass {
public:
    OPENVINO_RTTI("InsertLoops", "0");
    InsertLoops(ov::PartialShape master_shape, size_t vector_size);
    bool run_on_model(const std::shared_ptr<ngraph::Function>& m) override;

private:
    ov::PartialShape master_shape;
    size_t vector_size;
};

}  // namespace pass
}  // namespace snippets
}  // namespace ngraph
