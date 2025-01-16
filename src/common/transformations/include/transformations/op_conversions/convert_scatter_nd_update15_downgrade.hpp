// Copyright (C) 2018-2023 Intel Corporation
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
    OPENVINO_RTTI("ConvertScatterNDUpdate15ToScatterNDUpdate3", "0");
    ConvertScatterNDUpdate15ToScatterNDUpdate3();
};

}  // namespace pass
}  // namespace ov
