// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "openvino/frontend/gguf/builder/graph_context.hpp"
#include "openvino/frontend/gguf/builder/model_builder.hpp"

namespace ov::frontend::gguf {
std::shared_ptr<ModelBuilder> make_lfm_builder(const BuildContext& context);
std::shared_ptr<ModelBuilder> make_lfm2_vocoder_builder(const BuildContext& context);

// State: prepend a recurrent state input and retain its trailing taps.
// Causal: prepend zeros. Centered: pad both sides with zeros for an odd kernel.
enum class LfmConvPadding { STATE, CAUSAL, CENTERED };

// Returns C * conv(B * x) from the [B, C, x] input projection; state_name also names padding constants.
GgufValue build_lfm_short_conv(GgufGraphContext& g,
                               const GgufValue& projected,
                               const GgufValue& conv_weight,
                               int64_t channels,
                               const std::string& state_name,
                               bool batch_major = true,
                               LfmConvPadding padding = LfmConvPadding::STATE);
}  // namespace ov::frontend::gguf
