// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API TransposeToReshape;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief TransposeToReshape transformation replaces suitable Transposes with Reshape operation or optimizes them out
 */
class ov::pass::TransposeToReshape : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("TransposeToReshape", "0");
    TransposeToReshape();
};
