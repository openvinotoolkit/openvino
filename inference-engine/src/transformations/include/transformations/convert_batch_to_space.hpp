// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/ops.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ConvertBatchToSpace;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertBatchToSpace: public ngraph::pass::GraphRewrite {
public:
    ConvertBatchToSpace() : GraphRewrite() {
        // convert_batch_to_space();
        convert_batch_to_space_ie_side();
    }

private:
    void convert_batch_to_space();
    void convert_batch_to_space_ie_side();
};
