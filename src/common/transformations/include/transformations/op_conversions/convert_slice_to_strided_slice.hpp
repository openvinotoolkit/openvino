// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API SliceToStridedSlice;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief SliceToStridedSlice transformation convert v8::Slice to v1::StridedSlice
 */
class ov::pass::SliceToStridedSlice : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("SliceToStridedSlice", "0");
    SliceToStridedSlice(bool use_shapes);
};
