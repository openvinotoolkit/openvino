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
 * @brief Converts AvgPool v14 to AvgPool v1
 */
class TRANSFORMATIONS_API ConvertAvgPool14ToAvgPool1 : public MatcherPass {
public:
    OPENVINO_RTTI("ConvertAvgPool14ToAvgPool1", "0");
    ConvertAvgPool14ToAvgPool1();
};
}  // namespace pass
}  // namespace ov
