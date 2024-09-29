// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API Gelu7Downgrade;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief Gelu7Downgrade converts v7::Gelu operation to v2::Gelu unconditionally. This is done because only limited
 * set of plugins support v7::Gelu which has an attribute specifying approximation mode. For other plugins the
 * behaviour is to use v2 version of the operation which does not support the approximation mode.
 */
class ov::pass::Gelu7Downgrade : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("Gelu7Downgrade", "0");
    Gelu7Downgrade();
};
