// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvertSoftMax1ToSoftMax8;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief ConvertSoftMax1ToSoftMax8 converts v1::SoftMax into v8::SoftMax.
 */

class ov::pass::ConvertSoftMax1ToSoftMax8 : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertSoftMax1ToSoftMax8", "0");
    ConvertSoftMax1ToSoftMax8();
};
