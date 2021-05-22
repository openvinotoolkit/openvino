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

#include "iparams_manager.hpp"
#include "ilayer_transformations_manager.hpp"
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

class TRANSFORMATIONS_API DataPrecision {
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
class TRANSFORMATIONS_API LayerTransformation {
public:
    enum QuantizedTensorAlignment {
        None,
        UpdateLevel
    };

    class Params {
    public:
        Params(
                const bool updatePrecisions = true,
                const QuantizedTensorAlignment quantizedTensorAlignmentOnActivations = QuantizedTensorAlignment::UpdateLevel,
                const QuantizedTensorAlignment quantizedTensorAlignmentOnWeights = QuantizedTensorAlignment::None,
                bool supportAsymmetricQuantization = false,
                std::vector<element::Type> precisionsOnActivations = { element::u8, element::i8 },
                std::vector<element::Type> precisionsOnWeights = { element::i8 },
                element::Type deqPrecision = element::f32,
                bool support3DTensorOnActivations = true,
                bool deconvolutionSpecificChannelsRatio = false) :
                updatePrecisions(updatePrecisions),
                quantizedTensorAlignmentOnActivations(quantizedTensorAlignmentOnActivations),
                quantizedTensorAlignmentOnWeights(quantizedTensorAlignmentOnWeights),
                supportAsymmetricQuantization(supportAsymmetricQuantization),
                precisionsOnActivations(precisionsOnActivations),
                precisionsOnWeights(precisionsOnWeights),
                deqPrecision(deqPrecision),
                support3DTensorOnActivations(support3DTensorOnActivations),
                deconvolutionSpecificChannelsRatio(deconvolutionSpecificChannelsRatio) {
            if (precisionsOnActivations.size() == 0ul) {
                THROW_TRANSFORMATION_EXCEPTION << "precisions on activations are not specisifed";
            }

            if (precisionsOnWeights.size() == 0ul) {
                THROW_TRANSFORMATION_EXCEPTION << "precisions on weights are not specisifed";
            }
        }

        Params& setUpdatePrecisions(const bool updatePrecisions) {
            this->updatePrecisions = updatePrecisions;
            return *this;
        }

        Params& setQuantizedTensorAlignmentOnActivations(const QuantizedTensorAlignment quantizedTensorAlignmentOnActivations) {
            this->quantizedTensorAlignmentOnActivations = quantizedTensorAlignmentOnActivations;
            return *this;
        }

        Params& setQuantizedTensorAlignmentOnWeights(const QuantizedTensorAlignment quantizedTensorAlignmentOnWeights) {
            this->quantizedTensorAlignmentOnWeights = quantizedTensorAlignmentOnWeights;
            return *this;
        }

        Params& setSupportAsymmetricQuantization(const bool supportAsymmetricQuantization) {
            this->supportAsymmetricQuantization = supportAsymmetricQuantization;
            return *this;
        }

        Params& setPrecisionsOnActivations(const std::vector<element::Type>& precisionsOnActivations) {
            this->precisionsOnActivations = precisionsOnActivations;
            return *this;
        }

        Params& setPrecisionsOnWeights(const std::vector<element::Type>& precisionsOnWeights) {
            this->precisionsOnWeights = precisionsOnWeights;
            return *this;
        }

        Params& setSupport3DTensorOnActivations(const bool support3DTensorOnActivations) {
            this->support3DTensorOnActivations = support3DTensorOnActivations;
            return *this;
        }

        Params& setDeconvolutionSpecificChannelsRatio(const bool deconvolutionSpecificChannelsRatio) {
            this->deconvolutionSpecificChannelsRatio = deconvolutionSpecificChannelsRatio;
            return *this;
        }

        bool updatePrecisions;
        QuantizedTensorAlignment quantizedTensorAlignmentOnActivations;
        QuantizedTensorAlignment quantizedTensorAlignmentOnWeights;
        bool supportAsymmetricQuantization;
        std::vector<element::Type> precisionsOnActivations;
        std::vector<element::Type> precisionsOnWeights;
        element::Type deqPrecision;
        bool support3DTensorOnActivations;
        bool deconvolutionSpecificChannelsRatio;
    };

    class PrecisionDetails {
    public:
        PrecisionDetails(const element::Type& precision, const bool hasNegativeOutput, const bool hasZeroPoint) :
                precision(precision),
                hasNegativeOutput(hasNegativeOutput),
                hasZeroPoint(hasZeroPoint) {}

        const element::Type precision;
        const bool hasNegativeOutput;
        const bool hasZeroPoint;
    };

    LayerTransformation(const Params& params);
    virtual ~LayerTransformation() = default;
    virtual void registerMatcherIn(ngraph::pass::GraphRewrite& pass, TransformationContext& context) const = 0;
    virtual bool transform(TransformationContext& context, ngraph::pattern::Matcher &m) const = 0;

    void setParamsManager(IParamsManager* paramsManager) noexcept;
    void setLayerTransformationsManager(ILayerTransformationsManager* layerTransformationsManager) noexcept;

    void setUpdatePrecisions(const bool updatePrecisions);
    void setQuantizedTensorAlignmentOnActivations(const QuantizedTensorAlignment quantizedTensorAlignmentOnActivations);
    void setQuantizedTensorAlignmentOnWeights(const QuantizedTensorAlignment quantizedTensorAlignmentOnWeights);

    void setQuantizationIntervalAsymmetryThreshold(const float value);
    void setZeroThreshold(const float value);
    void setMinQuantizationLevels(const size_t levels);

    const std::vector<element::Type>& getPrecisionsOnActivations() const;
    const std::vector<element::Type>& getPrecisionsOnWeights() const;

    virtual bool canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> layer) const;

    bool canSubtractBeHandled(const std::shared_ptr<Node>& op, const size_t parentIndex = 0ul) const;

    bool canSubtractBeHandled(const std::shared_ptr<Node>& op, const FakeQuantizeDequantization& dequantization) const;

    PrecisionDetails getPrecisionDetails(const QuantizationDetails& quantizationDetails) const;

    // return true if operation can be quantized and false otherwise
    // for example: if convolution operation weights are not quantized, then isQuantize returns false and true otherwise
    // note: dequantization operations on activations are absent during method execution
    virtual bool isQuantized(std::shared_ptr<Node> layer) const noexcept;

    // return true if operation can be preserved for precision
    // note: dequantization operations on activations are absent during method execution
    virtual bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept = 0;

    DataPrecision getDataPrecision(
            std::shared_ptr<Node> layer,
            const QuantizationDetails& quantizationDetails,
            const bool onWeights) const;

    void fillAvailablePrecisions(std::shared_ptr<Node> layer, std::vector<element::Type>& availablePrecisions) const;

    std::vector<std::shared_ptr<Node>> getChildrenRecursivelyExceptPrecisionPreserved(const std::shared_ptr<Node>& op) const noexcept;

protected:
#ifdef LPT_PRINT_DEQUANTIZATION_INFO
    static void printDequantizationInfo(const std::shared_ptr<Node>& layer);
    static void printDequantizationInfo(const DataPrecision& dataPrecision);
    static void printDequantizationValues(
        const std::vector<float>& dequantizationScales,
        const std::vector<float>& dequantizationShifts);
#endif

    bool updatePrecisions;
    QuantizedTensorAlignment quantizedTensorAlignmentOnActivations;
    QuantizedTensorAlignment quantizedTensorAlignmentOnWeights;
    bool supportAsymmetricQuantization;
    std::vector<element::Type> precisionsOnActivations;
    std::vector<element::Type> precisionsOnWeights;
    element::Type deqPrecision;
    bool support3DTensorOnActivations;
    bool deconvolutionSpecificChannelsRatio;

    // absolute value, used to determine quantization interval asymmetry
    float quantizationIntervalAsymmetryThreshold;
    // absolute value, used to determine zero
    float zeroThreshold;
    size_t minQuantizationLevels;

    static const char originalLayerPostfix[];
    IParamsManager* paramsManager;
    ILayerTransformationsManager* layerTransformationsManager;

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

    void addPattern(ngraph::pass::GraphRewrite& pass, TransformationContext& context, std::shared_ptr<Node> patternRoot) const;

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

inline std::ostream &operator << (std::ostream &os, const LayerTransformation::QuantizedTensorAlignment& value) {
    switch (value) {
        case LayerTransformation::QuantizedTensorAlignment::None: {
            os << "None";
            break;
        }
        case LayerTransformation::QuantizedTensorAlignment::UpdateLevel: {
            os << "UpdateLevel";
            break;
        }
        default: {
            os << static_cast<int>(value);
            break;
        }
    }
    return os;
}

inline std::ostream &operator << (std::ostream &os, const std::vector<element::Type>& values) {
    os << "{";
    for (size_t i = 0; i < values.size(); ++i) {
        const element::Type& value = values[i];
        if (i > 0) {
            os << value;
        } else {
            os << ", " << value;
        }
    }
    os << "}";
    return os;
}

typedef std::shared_ptr<LayerTransformation> LayerTransformationPtr;

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph
