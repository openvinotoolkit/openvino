// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace ov {
namespace intel_gna {
namespace frontend {

class QuantizationParams {
    float scale = 1.0f;
    bool scale_set = false;
    size_t levels = 0;
    std::vector<float> input_min_values;
    std::vector<float> input_max_values;
    std::vector<float> output_min_values;
    std::vector<float> output_max_values;

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
    void CopyStats(QuantizationParams &src) {
        levels = src.GetLevels();
        SetMinValues(src.GetMinValues(true), true);
        SetMaxValues(src.GetMaxValues(true), true);
        SetMinValues(src.GetMinValues(false), false);
        SetMaxValues(src.GetMaxValues(false), false);
    }
};

struct QuantizedLayerParams {
    QuantizationParams _src_quant;
    QuantizationParams _dst_quant;

    // deprecate this
    QuantizationParams _weights_quant;
    QuantizationParams _bias_quant;
};

}  // namespace frontend
}  // namespace intel_gna
}  // namespace ov
