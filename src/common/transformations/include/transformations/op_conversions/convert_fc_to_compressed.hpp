// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "ov_ops/fully_connected.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvertFullyConnectedToFullyConnectedCompressed;

}  // namespace pass
}  // namespace ov

class ov::pass::ConvertFullyConnectedToFullyConnectedCompressed : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertFullyConnectedToFullyConnectedCompressed");

    using SupportsPredicate =
        std::function<bool(const std::shared_ptr<ov::op::internal::FullyConnected>&, size_t, size_t, size_t)>;

    ConvertFullyConnectedToFullyConnectedCompressed(const std::vector<ov::element::Type>& supported_activation_types,
                                                    const std::vector<ov::element::Type>& supported_weights_types,
                                                    SupportsPredicate supports_config = nullptr,
                                                    bool convert_u4zp_to_u8 = false);
};
