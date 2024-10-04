// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvertSpaceToBatch;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
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

class ov::pass::ConvertSpaceToBatch : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertSpaceToBatch", "0");
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
