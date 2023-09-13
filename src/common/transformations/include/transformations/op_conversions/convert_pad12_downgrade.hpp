// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {
/**
 * @ingroup ie_transformation_common_api
 * @brief Converts Pad v12 to Pad v1
 */
class TRANSFORMATIONS_API ConvertPad12ToPad1 : public MatcherPass {
public:
    OPENVINO_RTTI("ConvertPad12ToPad1", "0");
    ConvertPad12ToPad1();
};

}  // namespace pass
}  // namespace ov
