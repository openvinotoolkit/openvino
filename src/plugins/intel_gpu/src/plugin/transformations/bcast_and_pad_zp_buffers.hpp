// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace intel_gpu {

// Makes zero-points and compensation parameters of internal::Convolution op broadcasted to corresponding channels size
// and adds optional padding to align elements count to `alignment` value
class BroadcastAndPadZeroPointBuffers : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("BroadcastAndPadZeroPointBuffers", "0");
    BroadcastAndPadZeroPointBuffers(size_t alignment = 1, bool supports_immad = false);
};

}  // namespace intel_gpu
}  // namespace ov
