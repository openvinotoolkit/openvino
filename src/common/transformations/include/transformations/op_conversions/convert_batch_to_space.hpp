// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvertBatchToSpace;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief ConvertBatchToSpace transformation decomposes BatchToSpace layer to Reshape->Transpose->Reshape->Crop.
 *
 * @param convert_by_elements - reduces the maximum number of dimensions that arise during the transformation
 * if enabled. Default value: true.
 *  false - BatchToSpace decomposes to Reshape->Transpose->Reshape->Crop. During transformation, the number of
 *  tensor dimensions can be increased by length of block_shape input of BatchToSpace layer.
 *  true - BatchToSpace decomposes to N x (Reshape->Transpose->Reshape)->Crop, where N = length of block_shape input
 *  of BatchToSpace layer. During transformation, the number of tensor dimensions can be increased by 1.
 *
 */

class ov::pass::ConvertBatchToSpace : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertBatchToSpace");
    explicit ConvertBatchToSpace(bool convert_by_elements = true) : MatcherPass() {
        if (convert_by_elements)
            convert_batch_to_space_by_elements();
        else
            convert_batch_to_space();
    }

private:
    void convert_batch_to_space();
    void convert_batch_to_space_by_elements();
};
