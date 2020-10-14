// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ConvertExtractImagePatchesToReorgYolo;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief ConvertExtractImagePatchesToReorgYolo transformation replaces ExtractImagePatches with a ReorgYolo op.
 */
class ngraph::pass::ConvertExtractImagePatchesToReorgYolo : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertExtractImagePatchesToReorgYolo();
};
