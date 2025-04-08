// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <unordered_set>
#include <ostream>
#include <vector>

#include "low_precision/lpt_visibility.hpp"
#include "openvino/opsets/opset1.hpp"

namespace ov {
namespace pass {
namespace low_precision {

enum levels : size_t {
    int4 = 16,
    int4_narrow_range = 15,
    int8 = 256,
    int8_narrow_range = 255,
    int16 = 65536,
    int16_narrow_range = 65535,
    int32 = size_t(4294967296),  // for ARM and ia32 platforms where this number bigger than size_t but never used
    int32_narrow_range = 4294967295
};

static std::set<levels> all_levels = {
    levels::int4,  levels::int4_narrow_range,
    levels::int8,  levels::int8_narrow_range,
    levels::int16, levels::int16_narrow_range,
    levels::int32, levels::int32_narrow_range
};

class LP_TRANSFORMATIONS_API QuantizationDetails {
public:
    QuantizationDetails();
    QuantizationDetails(const QuantizationDetails& quantizationDetails);
    QuantizationDetails(
            const size_t levels,
            const std::vector<float>& inputLowValues,
            const std::vector<float>& inputHighValues,
            const std::vector<float>& outputLowValues,
            const std::vector<float>& outputHighValues);

    static bool outputLayoutIsSupported(std::shared_ptr<ov::opset1::FakeQuantize> quantize, bool isConvertExpected = false);

    static void getInputIntervals(
            std::shared_ptr<ov::opset1::FakeQuantize> quantize,
            std::vector<float>& inputLowValues,
            std::vector<float>& inputHighValues);

    static void getOutputIntervals(
            std::shared_ptr<ov::opset1::FakeQuantize> quantize,
            std::vector<float>& outputLowValues,
            std::vector<float>& outputHighValues);

    static QuantizationDetails getDetails(std::shared_ptr<ov::opset1::FakeQuantize>);
    bool hasNegativeOutput() const;
    float maxOutput(const size_t channel) const;
    float maxInput(const size_t channel) const;

    float getInputLowValue(const size_t channel) const;
    float getInputHighValue(const size_t channel) const;
    float getOutputLowValue(const size_t channel) const;
    float getOutputHighValue(const size_t channel) const;

    bool empty() const noexcept;

    static bool isSupportedLevel(
        const size_t level,
        const std::set<levels>& supported_levels = all_levels);

    const size_t levels;
    const std::vector<float> inputLowValues;
    const std::vector<float> inputHighValues;
    const std::vector<float> outputLowValues;
    const std::vector<float> outputHighValues;

private:
    static std::vector<float> getBlobValue(std::shared_ptr<Node> constantLayer);
};

inline std::ostream &operator << (std::ostream &os, const QuantizationDetails& value) {
    os << "levels: " << value.levels <<
       ", input 1/" << value.inputLowValues.size() << ": [" << value.getInputLowValue(0) << " : " << value.getInputHighValue(0) << "], " <<
       ", output 1/" << value.outputLowValues.size() << ": [" << value.getOutputLowValue(0) << " : " << value.getOutputHighValue(0) << "]";
    return os;
}


} // namespace low_precision
} // namespace pass
} // namespace ov
