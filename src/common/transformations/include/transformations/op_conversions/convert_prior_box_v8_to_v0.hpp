// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvertPriorBox8To0;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief ConvertPriorBox8To1 converts v8::PriorBox into v0::PriorBox.
 */
class ov::pass::ConvertPriorBox8To0 : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertPriorBox8To0", "0");
    ConvertPriorBox8To0();
};
