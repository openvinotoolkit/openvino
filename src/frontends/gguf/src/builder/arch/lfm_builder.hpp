// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

#include "openvino/frontend/gguf/builder/graph_context.hpp"

namespace ov::frontend::gguf {
struct LfmDecoderLayer {
    bool recurrent = false;
    bool sandwich_norm = false;
};

struct LfmDecoderConfig {
    std::vector<LfmDecoderLayer> layers;
    GgufValue output_norm;
    GgufValue output_weight;
    GgufValue output_bias;
    std::string hidden_output;
    bool select_output_rows = true;
};

std::shared_ptr<GgufGraph> build_lfm_decoder(GgufGraphContext& g,
                                             const DecoderDimensions& dims,
                                             const LfmDecoderConfig& config);

// Returns C * conv(B * x), retaining trailing taps from the [B, C, x] input projection.
GgufValue build_lfm_short_conv(GgufGraphContext& g,
                               const GgufValue& projected,
                               const GgufValue& conv_weight,
                               int64_t channels,
                               const std::string& state_name,
                               bool batch_major = true);
}  // namespace ov::frontend::gguf
