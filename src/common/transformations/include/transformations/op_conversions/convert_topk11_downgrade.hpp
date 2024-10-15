// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {
/**
 * @ingroup ov_transformation_common_api
 * @brief Converts TopK version 11 to TopK version 3 if TopK 11 stable attribute is set to false
 */
class TRANSFORMATIONS_API ConvertTopK11ToTopK3 : public MatcherPass {
public:
    OPENVINO_RTTI("ConvertTopK11ToTopK3", "0");
    ConvertTopK11ToTopK3();
};

}  // namespace pass
}  // namespace ov
