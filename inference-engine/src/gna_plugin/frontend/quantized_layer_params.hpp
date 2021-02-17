// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace GNAPluginNS {

class Quantization {
public:
    void SetScale(float s) {
        scale = s;
        scale_set = true;
    }
    float GetScale() const {
        return scale;
    }
    bool IsScaleSet() const {
        return scale_set;
    }
    void SetLevels(int32_t l) {
        levels = l;
    }
    int32_t GetLevels() const {
        return levels;
    }
    void SetMinValues(const std::vector<float> &min) {
        min_values.clear();
        min_values.insert(min_values.end(), min.begin(), min.end());
    }
    const std::vector<float>& GetMinValues() const {
        return min_values;
    }
    void SetMaxValues(const std::vector<float>& max) {
        max_values.clear();
        max_values.insert(max_values.end(), max.begin(), max.end());
    }
    const std::vector<float>& GetMaxValues() const {
        return max_values;
    }

private:
    float scale = 1.0f;
    bool scale_set = false;
    int32_t levels = 0;
    std::vector<float> min_values;
    std::vector<float> max_values;
};

struct QuantizedLayerParams {
    Quantization _src_quant;
    Quantization _dst_quant;

    // deprecate this
    Quantization _weights_quant;
    bool _weights_quantized = false;
    Quantization _bias_quant;
    float _o_shift = 0.0f;
    float _b_shift = 0.0f;
};

}  // namespace GNAPluginNS
