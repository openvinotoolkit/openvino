// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {
/**
 * @ingroup ov_transformation_common_api
 * @brief Converts ScatterNDUpdate version 15 to ScatterNDUpdate version 3 if ScatterNDUpdate reduction attribute is set
 * to None.
 */
class TRANSFORMATIONS_API ConvertScatterNDUpdate15ToScatterNDUpdate3 : public MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertScatterNDUpdate15ToScatterNDUpdate3");
    ConvertScatterNDUpdate15ToScatterNDUpdate3();
};

}  // namespace pass
}  // namespace ov
