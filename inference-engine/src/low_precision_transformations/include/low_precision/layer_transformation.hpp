// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <limits>
#include <list>
#include <memory>
#include <vector>

#include <ngraph/ngraph.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

#include "transformation_context.hpp"
#include "quantization_details.hpp"
#include "low_precision/common/ie_lpt_exception.hpp"
#include "common/fake_quantize_dequantization.hpp"

/*****************************************************
 * Debug capability
 *  - ORIGINAL_MODEL_PATH : Specify with existing folder name
 *    to serialize original model into it (XML & BIN extensions were added)
 *  - TRANSFORMED_MODEL_PATH : Specify with existing folder name
 *    to serialize original model into it (XML & BIN extensions were added)
 *  - LPT_PRINT_DEQUANTIZATION_INFO : Define it to enable
 *    dequantization layers printing
 *  - LPT_DISPLAY_PRECISION : Define it to to display precision info
 *    during low precision transformations
 *
 *****************************************************/
// #define LPT_ORIGINAL_MODEL_PATH "/localdisk/orig.model"
// #define LPT_TRANSFORMED_MODEL_PATH "/localdisk/transformed.model"
// #define LPT_PRINT_DEQUANTIZATION_INFO
// #define LPT_DISPLAY_PRECISION

namespace ngraph {
namespace pass {
namespace low_precision {

class LP_TRANSFORMATIONS_API DataPrecision {
public:
    DataPrecision() : precision(element::undefined), min(0.f), max(0.f), hasZeroPoint(false) {}

    explicit DataPrecision(const element::Type& precision) {
        this->precision = precision;
        min = getMinValue(precision, 256);
        max = getMaxValue(precision, 256);
        hasZeroPoint = false;
    }

    DataPrecision(const element::Type precision, const float min, const float max, const bool hasZeroPoint) :
            precision(precision),
            min(min),
            max(max),
            hasZeroPoint(hasZeroPoint) {}

    static bool isSupported(const element::Type& precision) {
        return (precision == element::u8) || (precision == element::i8);
    }

    static float getMinValue(const element::Type precision, const size_t levels) {
        if (precision == element::i8) {
            if (levels == 255) {
                return static_cast<float>(std::numeric_limits<signed char>::lowest()) + 1.f;
            } else if (levels == 256) {
                return static_cast<float>(std::numeric_limits<signed char>::lowest());
            } else {
                NGRAPH_CHECK(false, "unexpected levels ", levels, " for precision ", precision);
            }
        } else if (precision == element::u8) {
            return static_cast<float>(std::numeric_limits<unsigned char>::lowest());
        } else if (precision == element::f16) {
            return -1.0e15f;
        } else if (precision == element::f32) {
            return std::numeric_limits<float>::lowest();
        } else if (precision == element::i4) {
            return -8.f;
        } else if (precision == element::u4) {
            return 0.f;
        } else {
            NGRAPH_CHECK(false, "unexpected precision ", precision);
        }
    }

    static float getMaxValue(const element::Type precision, const size_t levels) {
        if ((levels != 255ul) && (levels != 256ul)) {
            THROW_TRANSFORMATION_EXCEPTION << "unexpected levels " << levels;
        }

        if (precision == element::i8) {
            return static_cast<float>(std::numeric_limits<signed char>::max());
        } else if (precision == element::u8) {
            return static_cast<float>(std::numeric_limits<unsigned char>::max()) - (256 - levels);
        } else if (precision == element::f16) {
            return 1.0e15f;
        } else if (precision == element::f32) {
            return std::numeric_limits<float>::max();
        } else if (precision == element::i4) {
            return 7.f;
        } else if (precision == element::u4) {
            return 15.f;
        } else {
            THROW_TRANSFORMATION_EXCEPTION << "unexpected precision " << precision;
        }
    }

    // Return maximum value for quantization level. Quantization level is maximum value for precision.
    static float getMaxValue(const size_t maxLevelsForPrecision) {
        if (maxLevelsForPrecision == 255ul) {
            return 254.f;
        } else if (maxLevelsForPrecision == 256ul) {
            return 255.f;
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

// Base class for all LP transformations, holds some common data structures
class LP_TRANSFORMATIONS_API LayerTransformation : public ngraph::pass::MatcherPass {
public:
    class Params {
    public:
        Params(
            const bool updatePrecisions = true,
            element::Type deqPrecision = element::f32) :
            updatePrecisions(updatePrecisions),
            deqPrecision(deqPrecision) {}

        Params& setUpdatePrecisions(const bool updatePrecisions) {
            this->updatePrecisions = updatePrecisions;
            return *this;
        }

        Params& setDeqPrecision(const element::Type& deqPrecision) {
            this->deqPrecision = deqPrecision;
            return *this;
        }

        bool updatePrecisions;
        element::Type deqPrecision;
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
    virtual ~LayerTransformation() = default;
    virtual bool transform(TransformationContext& context, ngraph::pattern::Matcher &m) = 0;

    void setContext(TransformationContext* context) noexcept;

    void setUpdatePrecisions(const bool updatePrecisions);

    virtual bool canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> layer) const;
    static bool canBeTransformedStatic(const std::shared_ptr<Node>& layer);

    bool canSubtractBeHandled(const std::shared_ptr<Node>& op, const FakeQuantizeDequantization& dequantization) const;

    // Get precision based on FakeQuantize operation.
    // Undefined value is expected. In this case the accuracy has to be defined by the calling code.
    // TODO: LPT: INT8 specific here
    static PrecisionDetails getPrecisionDetails(
        const size_t quantizationLevels,
        const std::vector<float>& outputLowValues,
        const std::vector<float>& outputHighValues);
    static PrecisionDetails getPrecisionDetails(const QuantizationDetails& quantizationDetails);

    static bool isAsymmetricQuantization(const std::shared_ptr<const Node>& node);

    // return true if operation can be quantized and false otherwise
    // for example: if convolution operation weights are not quantized, then isQuantize returns false and true otherwise
    // note: dequantization operations on activations are absent during method execution
    virtual bool isQuantized(const std::shared_ptr<const Node>& layer) const noexcept;

    // return true if operation can be preserved for precision
    // note: dequantization operations on activations are absent during method execution
    virtual bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept = 0;

    // weights specific
    static DataPrecision getDataPrecision(
            const std::shared_ptr<Node>& layer,
            const QuantizationDetails& quantizationDetails,
            const std::vector<element::Type>& precisions);

protected:
#ifdef LPT_PRINT_DEQUANTIZATION_INFO
    static void printDequantizationInfo(const std::shared_ptr<Node>& layer);
    static void printDequantizationInfo(const DataPrecision& dataPrecision);
    static void printDequantizationValues(
        const std::vector<float>& dequantizationScales,
        const std::vector<float>& dequantizationShifts);
#endif

    bool updatePrecisions;
    element::Type deqPrecision;

    static const char originalLayerPostfix[];
    TransformationContext* context;

protected:
    std::shared_ptr<ngraph::Node> moveDequantizationAfter(
        TransformationContext &context,
        const std::shared_ptr<ngraph::Node>& operation,
        const FakeQuantizeDequantization& dequantization,
        const bool updatePrecision,
        const bool moveSubtract = true) const;

    void updateOutput(
        TransformationContext &context,
        std::shared_ptr<ngraph::Node> lastNode,
        std::shared_ptr<ngraph::Node> originalNode) const;

    void updateOutput(
        TransformationContext& context,
        std::shared_ptr<ngraph::Node> lastNode,
        std::string originalName) const;

    void addPattern(ngraph::pass::GraphRewrite& pass, TransformationContext& context, std::shared_ptr<Node> patternRoot);

    //TODO: replace with canBeTransformed when quantization by special dimension is supported for all transformations
    bool canBeTransformedSpatialDimension(const TransformationContext& context, std::shared_ptr<Node> layer) const;

    template <typename Operation>
    void addSingleNodePattern(ngraph::pass::GraphRewrite& pass, TransformationContext& context) const {
        using namespace ngraph;

        auto is_op_type = [](std::shared_ptr<Node> n) {
            return !!as_type_ptr<Operation>(n);
        };
        auto p_node = std::make_shared<pattern::op::Label>(element::f32, Shape{}, is_op_type);

        addPattern(pass, context, p_node);
    }
};

typedef std::shared_ptr<LayerTransformation> LayerTransformationPtr;

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph
