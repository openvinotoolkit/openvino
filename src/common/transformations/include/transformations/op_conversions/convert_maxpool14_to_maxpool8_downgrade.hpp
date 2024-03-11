// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

namespace ov {
namespace pass {
/**
 * @ingroup ie_transformation_common_api
 * @brief Converts MaxPool v14 to MaxPool v8
 */
class TRANSFORMATIONS_API ConvertMaxPool14ToMaxPool8 : public MatcherPass {
public:
    OPENVINO_RTTI("ConvertMaxPool14ToMaxPool8", "0");
    ConvertMaxPool14ToMaxPool8();
};
}  // namespace pass
}  // namespace ov
