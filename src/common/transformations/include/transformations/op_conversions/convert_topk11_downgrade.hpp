// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

namespace ov {
namespace pass {
/**
 * @ingroup ie_transformation_common_api
 * @brief Converts TopK version 11 to TopK version 3 if TopK 11 stable attribute is set to false
 */
class TRANSFORMATIONS_API ConvertTopk11ToTopk3 : public MatcherPass {
public:
    OPENVINO_RTTI("ConvertTopk11ToTopk3", "0");
    ConvertTopk11ToTopk3();
};

}  // namespace pass
}  // namespace ov
