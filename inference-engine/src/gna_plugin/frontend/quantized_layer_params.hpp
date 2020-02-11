// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace GNAPluginNS {

struct Quantization {
    float scale = 1.0f;
    float offset = 0.0f;
    int shift = 0.0f;
};

struct QuantizedLayerParams {
    Quantization _src_quant;
    Quantization _dst_quant;
    Quantization _weights_quant;
    Quantization _bias_quant;
    float _o_shift = 0.0f;
    float _b_shift = 0.0f;
};

}  // namespace GNAPluginNS