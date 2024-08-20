// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

/**
 * @ingroup ov_transformation_common_api
 * @brief This transformation marks whether the input and output of "ReadValue" need to be checked to execute.
 *     intput1  
 *        |
 *     MatMul
 *        |
 *     ReadValue
 *        |     \
 *     Assign   others
 */

class TRANSFORMATIONS_API MarkReadValueInputs : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("MarkReadValueInputs", "0");
    MarkReadValueInputs();
};

}  // namespace pass
}  // namespace ov