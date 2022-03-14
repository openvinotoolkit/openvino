// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <openvino/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>
#include <vector>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API RemoveConcatZeroDimInput;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief RemoveConcatZeroDimInput transformation
 * removes input of Concat if the tensor size is equal to 0
 */

class ov::pass::RemoveConcatZeroDimInput : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("RemoveConcatZeroDimInput", "0");
    RemoveConcatZeroDimInput();
};
