// Copyright (C) 2018-2020 Intel Corporation
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
#include <transformations/low_precision/common/ie_lpt_exception.hpp>
#include "common/fake_quantize_dequantization.hpp"

/*****************************************************
 * Debug capability
 *  - ORIGINAL_MODEL_PATH : Specify with existing folder name
 *    to serialize original model into it (XML & BIN extensions were added)
 *  - TRANSFORMED_MODEL_PATH : Specify with existing folder name
 *    to serialize original model into it (XML & BIN extensions were added)
 *  - LPT_PRINT_DEQUANTIZATION_INFO : Define it to enable
 *    dequantization layers printing
 *
 *****************************************************/
//#define LPT_ORIGINAL_MODEL_PATH "/localdisk/slyalin/orig.model"
//#define LPT_TRANSFORMED_MODEL_PATH "/localdisk/slyalin/transformed.model"
// #define LPT_PRINT_DEQUANTIZATION_INFO

namespace ngraph {
namespace pass {
namespace low_precision {


class TRANSFORMATIONS_API DataPrecision {
public:
    DataPrecision() : precision(element::undefined), min(0.f), max(0.f), hasZeroPoint(false) {}

    DataPrecision(const element::Type precision, const float min, const float max, const bool hasZeroPoint) :
            precision(precision),
            min(min),
            max(max),
            hasZeroPoint(hasZeroPoint) {}

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
        } else {
            NGRAPH_CHECK(false, "unexpected precision ", precision);
        }
    }

    static float getMaxValue(const element::Type precision) {
        if (precision == element::i8) {
            return static_cast<float>(std::numeric_limits<signed char>::max());
        } else if (precision == element::u8) {
            return static_cast<float>(std::numeric_limits<unsigned char>::max());
        } else if (precision == element::f16) {
            return 1.0e15f;
        } else if (precision == element::f32) {
            return std::numeric_limits<float>::max();
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

    static float getMin(const size_t quantizationLevels, const bool signedInterval) {
        if (quantizationLevels == 255) {
            return signedInterval  ? -127.0 : 0.0;
        } else if (quantizationLevels == 256) {
            return signedInterval ? -128.0 : 0.0;
        } else {
            // THROW_TRANSFORMATION_EXCEPTION << "quantization level " << quantizationLevels << " is not supported";
            // FIXME: not completed
            return signedInterval ? -128.0 : 0.0;
        }
    }

    static float getMax(const size_t quantizationLevels, const bool signedInterval) {
        if ((quantizationLevels == 255) || (quantizationLevels == 256)) {
            return signedInterval ? 127.0 : 255.0;
        } else {
            // THROW_TRANSFORMATION_EXCEPTION << "quantization level " << quantizationLevels << " is not supported";
            // FIXME: not completed
            // return quantizationLevels - 1.0;
            return signedInterval ? 127.0 : 255.0;
        }
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
        UpdateIntervals,
        UpdateLevel,
        // UpdateIntervals & UpdateLevel & ...
                Mixed
    };

    class Params {
    public:
        Params(
                const bool updatePrecisions = true,
                const bool quantizeOutputs = false,
                const bool weightsToConst = true,
                const QuantizedTensorAlignment quantizedTensorAlignmentOnActivations = QuantizedTensorAlignment::UpdateLevel,
                const QuantizedTensorAlignment quantizedTensorAlignmentOnWeights = QuantizedTensorAlignment::None,
                const bool roundQuantizedValues = true,
                const bool updateBiases = true,
                bool supportAsymmetricQuantization = true,
                std::vector<element::Type> precisionsOnActivations = { element::u8, element::i8 },
                std::vector<element::Type> precisionsOnWeights = { element::i8 }) :
                updatePrecisions(updatePrecisions),
                quantizeOutputs(quantizeOutputs),
                weightsToConst(weightsToConst),
                quantizedTensorAlignmentOnActivations(quantizedTensorAlignmentOnActivations),
                quantizedTensorAlignmentOnWeights(quantizedTensorAlignmentOnWeights),
                roundQuantizedValues(roundQuantizedValues),
                updateBiases(updateBiases),
                supportAsymmetricQuantization(supportAsymmetricQuantization),
                precisionsOnActivations(precisionsOnActivations),
                precisionsOnWeights(precisionsOnWeights) {
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

        Params& setQuantizeOutputs(const bool quantizeOutputs) {
            this->quantizeOutputs = quantizeOutputs;
            return *this;
        }

        Params& setWeightsToConst(const bool weightsToConst) {
            this->weightsToConst = weightsToConst;
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

        Params& setRoundQuantizedValues(const bool roundQuantizedValues) {
            this->roundQuantizedValues = roundQuantizedValues;
            return *this;
        }

        Params& setUpdateBiases(const bool updateBiases) {
            this->updateBiases = updateBiases;
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

        bool updatePrecisions;
        bool quantizeOutputs;
        bool weightsToConst;
        QuantizedTensorAlignment quantizedTensorAlignmentOnActivations;
        QuantizedTensorAlignment quantizedTensorAlignmentOnWeights;
        bool roundQuantizedValues;
        bool updateBiases;
        bool supportAsymmetricQuantization;
        std::vector<element::Type> precisionsOnActivations;
        std::vector<element::Type> precisionsOnWeights;
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
    virtual void transform(TransformationContext& context, ngraph::pattern::Matcher &m) const = 0;

    void setParamsManager(IParamsManager* paramsManager) noexcept;
    void setLayerTransformationsManager(ILayerTransformationsManager* layerTransformationsManager) noexcept;

    void setUpdatePrecisions(const bool updatePrecisions);
    void setQuantizeOutputs(const bool quantizeOutputs);
    void setWeightsToConst(const bool weightsToConst);
    void setQuantizedTensorAlignmentOnActivations(const QuantizedTensorAlignment quantizedTensorAlignmentOnActivations);
    void setQuantizedTensorAlignmentOnWeights(const QuantizedTensorAlignment quantizedTensorAlignmentOnWeights);

    void setQuantizationIntervalAsymmetryThreshold(const float value);
    void setZeroThreshold(const float value);
    void setDequantizationShiftToZeroRatioTreshold(const float value);
    void setMinQuantizationLevels(const size_t levels);

    const std::vector<element::Type>& getPrecisionsOnActivations() const;
    const std::vector<element::Type>& getPrecisionsOnWeights() const;

    virtual bool canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> layer) const;

    PrecisionDetails getPrecisionDetails(const QuantizationDetails& quantizationDetails) const;

    virtual bool isQuantized(std::shared_ptr<Node> layer) const noexcept;
    virtual bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept;

    DataPrecision getDataPrecision(
            std::shared_ptr<Node> layer,
            const QuantizationDetails& quantizationDetails,
            const bool onWeights,
            const bool supportAsymmetricQuantization) const;

    void fillAvailablePrecisions(std::shared_ptr<Node> layer, std::vector<element::Type>& availablePrecisions) const;

protected:
#ifdef LPT_PRINT_DEQUANTIZATION_INFO
            static void printDequantizationInfo(const CNNLayer& layer);
    static void printDequantizationInfo(const DataPrecision& dataPrecision);
    static void printDequantizationValues(
        const std::vector<float>& dequantizationScales,
        const std::vector<float>& dequantizationShifts);
#endif
    void addDequantizationLayer(
        TransformationContext& context,
        const std::shared_ptr<Node> layer,
        const FakeQuantizeDequantization& dequantization) const;

    void fillFromQuantizationDetails(
            const QuantizationDetails& quantizationDetails,
            const DataPrecision& dataPrecision,
            std::vector<float>& dequantizationScales,
            std::vector<float>& dequantizationShifts) const;

    void checkAndUpdateDequantizationShiftWithZero(
            const QuantizationDetails& quantizationDetails,
            std::vector<float>& dequantizationShifts) const;

    void fillFromDequantizationLayer(
            std::shared_ptr<Node> dequantizationLayer,
            std::vector<float>& dequantizationScales,
            std::vector<float>& dequantizationShifts) const;

    bool updatePrecisions;
    bool quantizeOutputs;
    bool weightsToConst;
    QuantizedTensorAlignment quantizedTensorAlignmentOnActivations;
    QuantizedTensorAlignment quantizedTensorAlignmentOnWeights;
    bool roundQuantizedValues;
    bool updateBiases;
    bool supportAsymmetricQuantization;
    std::vector<element::Type> precisionsOnActivations;
    std::vector<element::Type> precisionsOnWeights;

    // absolute value, used to determine quantization interval asymmetry
    float quantizationIntervalAsymmetryThreshold;
    // absolute value, used to determine zero
    float zeroThreshold;
    // relative value, used to replace quantization shift to zero
    float dequantizationShiftToZeroRatioTreshold;
    size_t minQuantizationLevels;

    static const char lastLayerPrefix[];
    IParamsManager* paramsManager;
    ILayerTransformationsManager* layerTransformationsManager;

protected:
    std::shared_ptr<ngraph::Node> separateInStandaloneBranch(std::shared_ptr<ngraph::Node> node) const;

    void updateOutput(
        TransformationContext &context,
        std::shared_ptr<ngraph::Node> lastNode,
        std::shared_ptr<ngraph::Node> originalNode) const;

    void addPattern(ngraph::pass::GraphRewrite& pass, TransformationContext& context, std::shared_ptr<Node> patternRoot) const;

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
        case LayerTransformation::QuantizedTensorAlignment::UpdateIntervals: {
            os << "UpdateIntervals";
            break;
        }
        case LayerTransformation::QuantizedTensorAlignment::UpdateLevel: {
            os << "UpdateLevel";
            break;
        }
        case LayerTransformation::QuantizedTensorAlignment::Mixed: {
            os << "Mixed";
            break;
        }
        default: {
            os << static_cast<int>(value);
            break;
        }
    }
    return os;
}

typedef std::shared_ptr<LayerTransformation> LayerTransformationPtr;

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph
