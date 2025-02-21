// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cassert>
#include <limits>
#include <list>
#include <memory>
#include <vector>

#include "openvino/pass/matcher_pass.hpp"
#include "quantization_details.hpp"
#include "low_precision/common/ie_lpt_exception.hpp"
#include "common/fake_quantize_dequantization.hpp"

/*****************************************************
 * Debug capability
 *  - LPT_PRINT_DEQUANTIZATION_INFO : Define it to enable dequantization info printing: scales, shifts, etc.
 *****************************************************/
// #define LPT_PRINT_DEQUANTIZATION_INFO

namespace ov {
namespace pass {
namespace low_precision {
namespace precision_set {
    LP_TRANSFORMATIONS_API const std::vector<element::Type>& get_int8_support();
    LP_TRANSFORMATIONS_API const std::vector<element::Type>& get_int8_int16_int32_support();
} // namespace precision_set

class LP_TRANSFORMATIONS_API DataPrecision {
public:
    DataPrecision() : precision(element::dynamic), min(0.f), max(0.f), hasZeroPoint(false) {}

    explicit DataPrecision(const element::Type& precision) {
        this->precision = precision;
        min = getMinValue(precision, levels::int8);
        max = getMaxValue(precision, levels::int8);
        hasZeroPoint = false;
    }

    DataPrecision(const element::Type precision, const float min, const float max, const bool hasZeroPoint) :
            precision(precision),
            min(min),
            max(max),
            hasZeroPoint(hasZeroPoint) {}

    bool empty() const noexcept {
        assert(((precision == element::dynamic) && (min == 0.f) && (max == 0.f) && (!hasZeroPoint)) ||
               ((precision != element::dynamic) && (max != 0.f)));
        return (precision == element::dynamic) && (min == 0.f) && (max == 0.f) && (!hasZeroPoint);
    }

    static bool isSupported(const element::Type& precision) {
        static const std::set<element::Type_t> lowPrecision = {
                element::i8, element::u8,
                element::i16, element::u16,
                element::i32, element::u32
        };
        return lowPrecision.find(precision) != lowPrecision.end();
    }

    static bool check(const element::Type precision, const size_t levels) {
        switch (precision) {
            case element::i4:
            case element::u4:
            case element::nf4:
                return (levels == low_precision::levels::int4) || (levels == low_precision::levels::int4_narrow_range);
            case element::i8:
            case element::u8:
                return (levels == low_precision::levels::int8) || (levels == low_precision::levels::int8_narrow_range);
            case element::i16:
            case element::u16:
                return (levels == low_precision::levels::int16) || (levels == low_precision::levels::int16_narrow_range);
            case element::i32:
            case element::u32:
                return (levels == low_precision::levels::int32) || (levels == low_precision::levels::int32_narrow_range);
            default:
                return false;
        }
    }

    // the lowest value (example, for signed symetric types: -max)
    static float getMinValue(const element::Type precision, const size_t levels) {
        switch (precision) {
            case element::u4:
            case element::u8:
            case element::u16:
            case element::u32:
                return 0.f;
            case element::i4:
                return -8.f;
            case element::i8:
                switch (levels) {
                    case low_precision::levels::int4:
                        return -8.f;
                    case low_precision::levels::int4_narrow_range:
                        return -7.f;
                    case low_precision::levels::int8:
                        return -128.f;
                    case low_precision::levels::int8_narrow_range:
                        return -127.f;
                }
                break;
            case element::i16:
                switch (levels) {
                    case low_precision::levels::int16:
                        return -32768.f;
                    case low_precision::levels::int16_narrow_range:
                        return -32767.f;
                }
                break;
            case element::i32:
                switch (levels) {
                    case low_precision::levels::int32:
                        return -2147483648.f;
                    case low_precision::levels::int32_narrow_range:
                        return -2147483647.f;
                }
                break;
            case element::f16:
                return -1.0e15f;
            case element::bf16:
                return -3.38953139e38f;
            case element::f32:
                return std::numeric_limits<float>::lowest();
            default:
                OPENVINO_ASSERT(false, "unexpected precision ", precision);
        }
        OPENVINO_ASSERT(false, "unexpected levels ", levels, " for precision ", precision);
    }

    static float getMaxValue(const element::Type precision, const size_t levels) {
        switch (precision) {
            case element::u4:
                return 15.f;
            case element::u8:
                switch (levels) {
                    case 16:
                        return 15.f;
                    default:
                        return 255.f;
                }
            case element::u16:
                return 65535.f;
            case element::u32:
                return 4294967296.f; // 4294967296.f == 4294967295.f
            case element::i4:
                return 7.f;
            case element::i8:
                switch (levels) {
                    case 16:
                        return 7.f;
                    default:
                        return 127.f;
                }
            case element::i16:
                return 32767.f;
            case element::i32:
                return 2147483648.f;  // 2147483648.f == 2147483647.f
            case element::f16:
                return 1.0e15f;
            case element::bf16:
                return 3.38953139e38f;
            case element::f32:
                return std::numeric_limits<float>::max();
            default:
                OPENVINO_ASSERT(false, "unexpected precision ", precision);
        }
    }

    // Return maximum value for quantization level. Quantization level is maximum value for precision.
    static float getMaxValue(const size_t maxLevelsForPrecision) {
        std::set<size_t> validLevels = {
            levels::int4,  levels::int4_narrow_range,
            levels::int8,  levels::int8_narrow_range,
            levels::int16, levels::int16_narrow_range,
            levels::int32, levels::int32_narrow_range
        };
        if (validLevels.find(maxLevelsForPrecision) != validLevels.end()) {
            return maxLevelsForPrecision - 1.f;
        } else {
            THROW_TRANSFORMATION_EXCEPTION << "unexpected quantization level " << maxLevelsForPrecision;
        }
    }

    static bool hasNegativeValues(const std::vector<float>& values) {
        for (const float value : values) {
            if (value < 0.0) {
                return true;
            }
        }
        return false;
    }

    element::Type precision;
    float min;
    float max;
    bool hasZeroPoint;

    static element::Type getPrecision(const std::vector<float>& outputLowValues, const std::vector<float>& outputHighValues) {
        return (hasNegativeValues(outputLowValues) || hasNegativeValues(outputHighValues)) ? element::i8 : element::u8;
    }

    static element::Type getPrecision(const size_t /* quantizationLevels */, const bool signedInterval) {
        return signedInterval ? element::i8 : element::u8;
    }
};

inline bool operator==(const DataPrecision& value1, const DataPrecision& value2) {
    return
            (value1.precision == value2.precision) &&
            (value1.min == value1.min) &&
            (value1.max == value1.max);
}

inline bool operator!=(const DataPrecision& value1, const DataPrecision& value2) {
    return !(value1 == value2);
}

inline std::ostream &operator << (std::ostream &os, const DataPrecision& value) {
    os << value.precision << ", min: " << value.min << ", max: " << value.max;
    return os;
}

/**
 * @ingroup ov_transformation_common_api
 * @brief Base class for low precision transformation.
 */
class LP_TRANSFORMATIONS_API LayerTransformation : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("low_precision::LayerTransformation");
    class Params {
    public:
        Params(
            const bool updatePrecisions = true,
            element::Type deqPrecision = element::f32,
            const std::vector<ov::element::Type> defaultPrecisions =
            { ov::element::u8,  ov::element::i8 },
            const bool reshapeIgnorePerTensorQuantizationCheck = false,
            const bool scalingMode = false) :
            updatePrecisions(updatePrecisions),
            deqPrecision(deqPrecision),
            defaultPrecisions(defaultPrecisions),
            reshapeIgnorePerTensorQuantizationCheck(reshapeIgnorePerTensorQuantizationCheck),
            scalingMode(scalingMode) {}

        Params& setUpdatePrecisions(const bool updatePrecisions) {
            this->updatePrecisions = updatePrecisions;
            return *this;
        }

        Params& setDeqPrecision(const element::Type& deqPrecision) {
            this->deqPrecision = deqPrecision;
            return *this;
        }

        Params& setDefaultPrecisions(const std::vector<ov::element::Type>& defaultPrecisions) {
            this->defaultPrecisions = defaultPrecisions;
            return *this;
        }

        bool updatePrecisions;
        // Use deqPrecision only for FakeQuantize operation decomposition,
        // use existing precisions in other cases.
        // In this case if FakeQuantize operations were decomposed then original precision will be used.
        element::Type deqPrecision;
        std::vector<ov::element::Type> defaultPrecisions;
        // to support GPU workarround to keep Reshape and MatMul in FP32
        bool reshapeIgnorePerTensorQuantizationCheck;
        // to support Activations Scaling
        bool scalingMode;
    };

    class PrecisionDetails {
    public:
        PrecisionDetails(const element::Type& precision, const bool hasNegativeOutput, const bool hasZeroPoint) :
                precision(precision),
                hasNegativeOutput(hasNegativeOutput),
                hasZeroPoint(hasZeroPoint) {}

        element::Type precision;
        bool hasNegativeOutput;
        bool hasZeroPoint;
    };

    LayerTransformation(const Params& params);
    virtual bool transform(ov::pass::pattern::Matcher &m) = 0;

    virtual bool canBeTransformed(const std::shared_ptr<Node>& layer) const;
    static bool canBeTransformedStatic(const std::shared_ptr<Node>& layer,
        const std::vector<ov::element::Type>& defaultPrecisions = precision_set::get_int8_support());

    bool canSubtractBeHandled(const std::shared_ptr<Node>& op, const FakeQuantizeDequantization& dequantization) const;

    // Get precision based on FakeQuantize operation.
    // Undefined value is expected. In this case the accuracy has to be defined by the calling code.
    // TODO: LPT: INT8 specific here
    static PrecisionDetails getPrecisionDetails(
        const size_t quantizationLevels,
        const std::vector<float>& outputLowValues,
        const std::vector<float>& outputHighValues);
    static PrecisionDetails getPrecisionDetails(const QuantizationDetails& quantizationDetails);

    static bool isAsymmetricQuantization(const std::shared_ptr<const Node>& node,
        const std::vector<ov::element::Type>& defaultPrecisions = precision_set::get_int8_support());

    // return true if operation can be quantized and false otherwise
    // for example: if convolution operation weights are not quantized, then isQuantize returns false and true otherwise
    // note: dequantization operations on activations are absent during method execution
    virtual bool isQuantized(const std::shared_ptr<const Node>& layer,
        const std::vector<ov::element::Type>& defaultPrecisions) const;

    // return true if operation can be preserved for precision
    // note: dequantization operations on activations are absent during method execution
    virtual bool isPrecisionPreserved(std::shared_ptr<Node> layer) const = 0;

    // weights specific
    static DataPrecision getDataPrecision(
            const std::shared_ptr<Node>& layer,
            const QuantizationDetails& quantizationDetails,
            const std::vector<element::Type>& requiredPrecisions);

protected:
#ifdef LPT_PRINT_DEQUANTIZATION_INFO
    static void printDequantizationInfo(const std::shared_ptr<Node>& layer);
    static void printDequantizationInfo(const DataPrecision& dataPrecision);
    static void printDequantizationValues(
        const std::vector<float>& dequantizationScales,
        const std::vector<float>& dequantizationShifts);
#endif

    const bool updatePrecisions;
    const element::Type deqPrecision;
    const std::vector<ov::element::Type> defaultPrecisions;
    const bool reshapeIgnorePerTensorQuantizationCheck;
    const bool scalingMode;

    static constexpr char originalLayerPostfix[] = "_original";

protected:
    std::shared_ptr<ov::Node> moveDequantizationAfter(
        const std::shared_ptr<ov::Node>& operation,
        const FakeQuantizeDequantization& dequantization,
        const bool updateOutputPrecision = true,
        const bool moveSubtract = true) const;

    std::shared_ptr<ov::Node> moveDequantizationBefore(
        const std::shared_ptr<ov::Node>& operation,
        const FakeQuantizeDequantization& dequantization,
        const bool moveSubtract = true) const;

    bool updateOutput(const std::shared_ptr<ov::Node>& lastNode, const std::shared_ptr<ov::Node>& originalNode) const;

    // TODO: replace with canBeTransformed when quantization by special dimension is supported for all transformations
    bool canBeTransformedSpatialDimension(const std::shared_ptr<Node>& layer) const;
};

}  // namespace low_precision
}  // namespace pass
}  // namespace ov
