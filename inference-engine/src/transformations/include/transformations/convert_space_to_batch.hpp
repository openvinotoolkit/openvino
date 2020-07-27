// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ConvertSpaceToBatch;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertSpaceToBatch: public ngraph::pass::GraphRewrite {
public:
    ConvertSpaceToBatch() : GraphRewrite() {
        // convert_space_to_batch();
        convert_space_to_batch_by_elements();
    }

private:
    void convert_space_to_batch();
    void convert_space_to_batch_by_elements();
};
