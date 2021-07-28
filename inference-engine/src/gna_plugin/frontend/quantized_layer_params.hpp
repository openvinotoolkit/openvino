// Copyright (C) 2018-2021 Intel Corporation
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
    void SetLevels(size_t l) {
        levels = l;
    }
    size_t GetLevels() const {
        return levels;
    }
    bool IsStatsSet() const {
        return !input_min_values.empty() && !input_max_values.empty();
    }
    void SetMinValues(const std::vector<float> &min, bool input = true) {
        if (input) {
            input_min_values.clear();
            input_min_values.insert(input_min_values.end(), min.begin(), min.end());
        } else {
            output_min_values.clear();
            output_min_values.insert(output_min_values.end(), min.begin(), min.end());
        }
    }
    std::vector<float>& GetMinValues(bool input = true) {
        if (input) {
            return input_min_values;
        }

        return output_min_values;
    }
    void SetMaxValues(const std::vector<float>& max, bool input = true) {
        if (input) {
            input_max_values.clear();
            input_max_values.insert(input_max_values.end(), max.begin(), max.end());
        } else {
            output_max_values.clear();
            output_max_values.insert(output_max_values.end(), max.begin(), max.end());
        }
    }
    std::vector<float>& GetMaxValues(bool input = true) {
        if (input) {
            return input_max_values;
        }

        return output_max_values;
    }
    void CopyStats(Quantization &src) {
        levels = src.GetLevels();
        SetMinValues(src.GetMinValues(true), true);
        SetMaxValues(src.GetMaxValues(true), true);
        SetMinValues(src.GetMinValues(false), false);
        SetMaxValues(src.GetMaxValues(false), false);
    }

private:
    float scale = 1.0f;
    bool scale_set = false;
    size_t levels = 0;
    std::vector<float> input_min_values;
    std::vector<float> input_max_values;
    std::vector<float> output_min_values;
    std::vector<float> output_max_values;
};

struct QuantizedLayerParams {
    Quantization _src_quant;
    Quantization _dst_quant;

    // deprecate this
    Quantization _weights_quant;
    Quantization _bias_quant;

    bool lowPrecision = false;
};

}  // namespace GNAPluginNS
