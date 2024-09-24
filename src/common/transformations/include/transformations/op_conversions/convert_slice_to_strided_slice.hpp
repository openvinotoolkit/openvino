// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API SliceToStridedSlice;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief SliceToStridedSlice transformation convert v8::Slice to v1::StridedSlice
 */
class ov::pass::SliceToStridedSlice : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("SliceToStridedSlice", "0");
    SliceToStridedSlice(bool use_shapes);
};
