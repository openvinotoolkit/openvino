// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {
class TRANSFORMATIONS_API NopBroadcast;
}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief Optimizes out Broadcast(data, Maximum(shape, ones)) if labels on data and shape are equal
 * Use case with data being empty should not be considered here since original graph has Maximum with ones
 */
class ov::pass::NopBroadcast : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("NopBroadcast", "0");
    NopBroadcast();
};