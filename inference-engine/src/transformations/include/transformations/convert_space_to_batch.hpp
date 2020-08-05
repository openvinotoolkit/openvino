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

/**
 * @ingroup ie_transformation_common_api
 * @brief ConvertSpaceToBatch transformation decomposes SpaceToBatch layer to Pad->Reshape->Transpose->Reshape.
 *
 * @param convert_by_elements - reduces the maximum number of dimensions that arise during the transformation
 * if enabled. Default value: true.
 *  false - SpaceToBatch decomposes to Pad->Reshape->Transpose->Reshape. During transformation, the number of
 *  tensor dimensions can be increased by length of block_shape input of SpaceToBatch layer.
 *  true - SpaceToBatch decomposes to Pad-> N x (Reshape->Transpose->Reshape), where N = length of block_shape input
 *  of SpaceToBatch layer. During transformation, the number of tensor dimensions can be increased by 1.
 *
 */

class ngraph::pass::ConvertSpaceToBatch: public ngraph::pass::MatcherPass {
public:
    explicit ConvertSpaceToBatch(bool convert_by_elements = true) : MatcherPass() {
        if (convert_by_elements)
            convert_space_to_batch_by_elements();
        else
            convert_space_to_batch();
    }

private:
    void convert_space_to_batch();
    void convert_space_to_batch_by_elements();
};
