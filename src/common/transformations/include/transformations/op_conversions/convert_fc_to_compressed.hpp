// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvertFullyConnectedToFullyConnectedCompressed;

}  // namespace pass
}  // namespace ov

class ov::pass::ConvertFullyConnectedToFullyConnectedCompressed : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertFullyConnectedToFullyConnectedCompressed", "0");
    ConvertFullyConnectedToFullyConnectedCompressed(
        const std::vector<ov::element::Type>& supported_compression_types,
        std::function<bool(size_t, size_t, size_t)> supports_config = nullptr,
        bool convert_u4zp_to_u8 = false);
};
