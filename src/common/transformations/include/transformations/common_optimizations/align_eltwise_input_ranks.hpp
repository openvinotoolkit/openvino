// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

/**
 * @ingroup ie_transformation_common_api
 * @brief transformation aligns elementwise constant inputs ranks with its output rank
 */

namespace ov {
namespace pass {

class TRANSFORMATIONS_API AlignEltwiseInputRanks : public MatcherPass {
public:
    OPENVINO_RTTI("TRANSFORMATIONS_API", "0");
    AlignEltwiseInputRanks();
};

}  // namespace pass
}  // namespace ov

namespace ngraph {
namespace pass {
using ov::pass::AlignEltwiseInputRanks;
}  // namespace pass
}  // namespace ngraph
