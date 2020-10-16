// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace GNAPluginNS {

struct Quantization {
public:
    float GetScale() const {
        return scale;
    }
    void SetScale(float s) {
        scale = s;
        scale_set = true;
    }
    bool IsScaleSet() const {
        return scale_set;
    }
    float GetMaxValue() const {
        return max_value;
    }
    void SetMaxValue(float value) {
        max_value = value;
        max_value_set = true;
    }
    bool IsMaxValueSet() const {
        return max_value_set;
    }

private:
    float scale = 1.0f;
    bool scale_set = false;
    float max_value = 0.0f;
    bool max_value_set = false;
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
