// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

namespace ov {
namespace pass {
class TRANSFORMATIONS_API NopBroadcast;
}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief Optimizes out Broadcast(input, Maximum(shape, ones)) if labels on input and shape are equal
 * Use case with input being empty should not be considered here since original graph has Maximum with ones
 */
class ov::pass::NopBroadcast : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("NopBroadcast", "0");
    NopBroadcast();
};