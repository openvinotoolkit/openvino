// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvertMaxPool1ToMaxPool8;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief ConvertMaxPool1ToMaxPool8 converts v1::MaxPool into v8::MaxPool.
 */

class ov::pass::ConvertMaxPool1ToMaxPool8 : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertMaxPool1ToMaxPool8");
    ConvertMaxPool1ToMaxPool8();
};
