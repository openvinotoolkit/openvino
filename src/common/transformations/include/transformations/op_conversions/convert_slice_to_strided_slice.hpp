// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API SliceToStridedSlice;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief SliceToStridedSlice transformation convert v8::Slice to v1::StridedSlice
 */
class ngraph::pass::SliceToStridedSlice : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("SliceToStridedSlice", "0");
    SliceToStridedSlice(bool use_shapes);
};
