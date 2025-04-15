// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvertSoftMax8ToSoftMax1;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief ConvertSoftMax8ToSoftMax1 converts v8::SoftMax into v1::SoftMax.
 */
class ov::pass::ConvertSoftMax8ToSoftMax1 : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertSoftMax8ToSoftMax1");
    ConvertSoftMax8ToSoftMax1();
};
