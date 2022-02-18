// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API SliceToStridedSlice;

}  // namespace pass
}  // namespace ngraph


/**
 * @ingroup ie_transformation_common_api
 * @brief SliceToStridedSlice transformation convert v8::Slice to v1::StridedSlice
 */
class ngraph::pass::SliceToStridedSlice: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    SliceToStridedSlice(bool use_shapes);
};
