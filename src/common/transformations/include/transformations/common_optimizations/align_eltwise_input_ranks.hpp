// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

/**
 * @ingroup ie_transformation_common_api
 * @brief transformation aligns elementwise constant inputs ranks with its output rank
 */

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API AlignEltwiseInputRanks: public MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    AlignEltwiseInputRanks();
};

}  // namespace pass
}  // namespace ngraph
