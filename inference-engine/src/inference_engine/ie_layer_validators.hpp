// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ie_layers.h"
#include "details/caseless.hpp"
#include <memory>
#include <string>
#include <map>
#include <vector>

namespace InferenceEngine {
namespace details {

struct InOutDims {
    std::vector<std::vector<size_t>> inDims;
    std::vector<std::vector<size_t>> outDims;
};

/**
 * @brief Contains methods to validate layer of specific type
 */
class INFERENCE_ENGINE_API_CLASS(LayerValidator) {
public:
    using Ptr = std::shared_ptr<LayerValidator>;

    explicit LayerValidator(const std::string& _type) : _type(_type) {}

    /**
     * @brief It parses map of params <string,string> and applies to the layer's fields.
     * This checks for presence of all required attributes, and that there's no extraneous parameters only.
     * Throws exception in case of parsing error
     */
    virtual void parseParams(CNNLayer* layer) {}

    /**
     * @brief Validates layer parameters separately from blobs and shapes
     * This is semantic check, like height and width more than kernel sizes, stride > 0, beta > 0, axis is correct and etc
     * Throws exception if the check fails
     */
    virtual void checkParams(const CNNLayer* layer) {}

    /**
     * @brief Checks correspondence of input shapes and layer parameters.
     * @note: This function doesn't touch ins and out Data of the layer.
     * Throws exception if the check fails
     */
    virtual void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const {}

    /**
     * @brief Checks correspondence of all parameters in the aggregate, except output shapes.
     * @note: This function doesn't touch ins and out Data of the layer.
     * Throws exception if the check fails
     */
    virtual void checkCorrespondence(const CNNLayer* layer,
                                     const std::map<std::string, Blob::Ptr>& blobs,
                                     const std::vector<SizeVector>& inShapes) const {}

protected:
    std::string _type;
};

/**
 * @brief Contains all validators, registered for specific layer type
 */
class INFERENCE_ENGINE_API_CLASS(LayerValidators) {
public:
    static LayerValidators* getInstance();

    LayerValidators(LayerValidators const&) = delete;

    void operator=(LayerValidators const&)  = delete;

    LayerValidator::Ptr getValidator(const std::string& type);

    void addImpl(const std::string& type, const LayerValidator::Ptr& validator);

private:
    LayerValidators() = default;

private:
    static LayerValidators* _instance;
    InferenceEngine::details::caseless_unordered_map<std::string, LayerValidator::Ptr> _validators;
};

static void getInOutShapes(const CNNLayer* layer, InOutDims& inOutShapes) {
    inOutShapes.inDims.clear();
    inOutShapes.outDims.clear();
    if (layer) {
        for (const auto& inData : layer->insData) {
            auto locked = inData.lock();
            if (locked) {
                inOutShapes.inDims.push_back(locked->getDims());
            }
        }
        for (const auto& outData : layer->outData) {
            if (outData) {
                inOutShapes.outDims.push_back(outData->getDims());
            }
        }
    }
}

class GeneralValidator : public LayerValidator {
public:
    explicit GeneralValidator(const std::string& _type);
};

class INFERENCE_ENGINE_API_CLASS(ConvolutionValidator) : public LayerValidator {
public:
    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    explicit ConvolutionValidator(const std::string& _type);

    void checkCorrespondence(const CNNLayer* layer,
                             const std::map<std::string, Blob::Ptr>& blobs,
                             const std::vector<SizeVector>& inShapes) const override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class INFERENCE_ENGINE_API_CLASS(DeconvolutionValidator) : public ConvolutionValidator {
public:
    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    explicit DeconvolutionValidator(const std::string& _type);

    void checkCorrespondence(const CNNLayer* layer,
                             const std::map<std::string, Blob::Ptr>& blobs,
                             const std::vector<SizeVector>& inShapes) const override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};


class INFERENCE_ENGINE_API_CLASS(PoolingValidator) : public LayerValidator {
public:
    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;

    explicit PoolingValidator(const std::string& _type);
};

class INFERENCE_ENGINE_API_CLASS(FullyConnectedValidator) : public LayerValidator {
public:
    explicit FullyConnectedValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkCorrespondence(const CNNLayer* layer,
                             const std::map<std::string, Blob::Ptr>& blobs,
                             const std::vector<SizeVector>& inShapes) const override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class INFERENCE_ENGINE_API_CLASS(CropValidator) : public LayerValidator {
public:
    explicit CropValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class INFERENCE_ENGINE_API_CLASS(TileValidator) : public LayerValidator {
public:
    explicit TileValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class INFERENCE_ENGINE_API_CLASS(BatchNormalizationValidator) : public LayerValidator {
public:
    explicit BatchNormalizationValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class INFERENCE_ENGINE_API_CLASS(PowerValidator) : public LayerValidator {
public:
    explicit PowerValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class INFERENCE_ENGINE_API_CLASS(PReLUValidator) : public LayerValidator {
public:
    explicit PReLUValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class INFERENCE_ENGINE_API_CLASS(ScaleShiftValidator) : public LayerValidator {
public:
    explicit ScaleShiftValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class INFERENCE_ENGINE_API_CLASS(ReshapeValidator) : public LayerValidator {
public:
    explicit ReshapeValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;
};

class INFERENCE_ENGINE_API_CLASS(EltwiseValidator) : public LayerValidator {
public:
    explicit EltwiseValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class INFERENCE_ENGINE_API_CLASS(ClampValidator) : public LayerValidator {
public:
    explicit ClampValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class INFERENCE_ENGINE_API_CLASS(ReLUValidator) : public LayerValidator {
public:
    explicit ReLUValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class INFERENCE_ENGINE_API_CLASS(MVNValidator) : public LayerValidator {
public:
    explicit MVNValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class INFERENCE_ENGINE_API_CLASS(GRNValidator) : public LayerValidator {
public:
    explicit GRNValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class INFERENCE_ENGINE_API_CLASS(SoftMaxValidator) : public LayerValidator {
public:
    explicit SoftMaxValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class INFERENCE_ENGINE_API_CLASS(NormValidator) : public LayerValidator {
public:
    explicit NormValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class INFERENCE_ENGINE_API_CLASS(SplitValidator) : public LayerValidator {
public:
    explicit SplitValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class INFERENCE_ENGINE_API_CLASS(ConcatValidator) : public LayerValidator {
public:
    explicit ConcatValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class INFERENCE_ENGINE_API_CLASS(GemmValidator) : public LayerValidator {
public:
    explicit GemmValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class INFERENCE_ENGINE_API_CLASS(PadValidator) : public LayerValidator {
public:
    explicit PadValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class INFERENCE_ENGINE_API_CLASS(GatherValidator) : public LayerValidator {
public:
    explicit GatherValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class INFERENCE_ENGINE_API_CLASS(StridedSliceValidator) : public LayerValidator {
public:
    explicit StridedSliceValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class INFERENCE_ENGINE_API_CLASS(ShuffleChannelsValidator) : public LayerValidator {
public:
    explicit ShuffleChannelsValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class INFERENCE_ENGINE_API_CLASS(DepthToSpaceValidator) : public LayerValidator {
public:
    explicit DepthToSpaceValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class INFERENCE_ENGINE_API_CLASS(SpaceToDepthValidator) : public LayerValidator {
public:
    explicit SpaceToDepthValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class INFERENCE_ENGINE_API_CLASS(ReverseSequenceValidator) : public LayerValidator {
public:
    explicit ReverseSequenceValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class INFERENCE_ENGINE_API_CLASS(SqueezeValidator) : public LayerValidator {
public:
    explicit SqueezeValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class INFERENCE_ENGINE_API_CLASS(UnsqueezeValidator) : public LayerValidator {
public:
    explicit UnsqueezeValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class INFERENCE_ENGINE_API_CLASS(RangeValidator) : public LayerValidator {
public:
    explicit RangeValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class INFERENCE_ENGINE_API_CLASS(FillValidator) : public LayerValidator {
public:
    explicit FillValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class INFERENCE_ENGINE_API_CLASS(ExpandValidator) : public LayerValidator {
public:
    explicit ExpandValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

template<RNNSequenceLayer::CellType CELL>
class INFERENCE_ENGINE_API_CLASS(RNNBaseValidator) : public LayerValidator {
public:
    explicit RNNBaseValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkCorrespondence(const CNNLayer* layer,
                             const std::map<std::string, Blob::Ptr>& blobs,
                             const std::vector<SizeVector>& inShapes) const override;

protected:
    static std::vector<std::string> def_acts;  // Default values for cell gate activations
    static std::vector<float> def_alpha;  // Default activation alpha parameter
    static std::vector<float> def_beta;   // Default activation beta parameter
    static size_t G;   // gate number
    static size_t NS;  // state number
};

template<RNNSequenceLayer::CellType CELL>
class INFERENCE_ENGINE_API_CLASS(RNNCellValidator) : public RNNBaseValidator<CELL> {
public:
    explicit RNNCellValidator(const std::string& _type);

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

extern template class INFERENCE_ENGINE_API_CLASS(RNNCellValidator)<RNNSequenceLayer::LSTM>;
extern template class INFERENCE_ENGINE_API_CLASS(RNNCellValidator)<RNNSequenceLayer::GRU>;
extern template class INFERENCE_ENGINE_API_CLASS(RNNCellValidator)<RNNSequenceLayer::RNN>;

template<RNNSequenceLayer::CellType CELL>
class INFERENCE_ENGINE_API_CLASS(RNNSequenceValidator) : public RNNBaseValidator<CELL> {
public:
    explicit RNNSequenceValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

extern template class INFERENCE_ENGINE_API_CLASS(RNNSequenceValidator)<RNNSequenceLayer::LSTM>;
extern template class INFERENCE_ENGINE_API_CLASS(RNNSequenceValidator)<RNNSequenceLayer::GRU>;
extern template class INFERENCE_ENGINE_API_CLASS(RNNSequenceValidator)<RNNSequenceLayer::RNN>;

class INFERENCE_ENGINE_API_CLASS(ArgMaxValidator) : public LayerValidator {
public:
    explicit ArgMaxValidator(const std::string& _type);

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class INFERENCE_ENGINE_API_CLASS(CTCGreedyDecoderValidator) : public LayerValidator {
public:
    explicit CTCGreedyDecoderValidator(const std::string& _type);

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class INFERENCE_ENGINE_API_CLASS(DetectionOutputValidator) : public LayerValidator {
public:
    explicit DetectionOutputValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class INFERENCE_ENGINE_API_CLASS(InterpValidator) : public LayerValidator {
public:
    explicit InterpValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class INFERENCE_ENGINE_API_CLASS(PermuteValidator) : public LayerValidator {
public:
    explicit PermuteValidator(const std::string& _type);

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class INFERENCE_ENGINE_API_CLASS(PriorBoxValidator) : public LayerValidator {
public:
    explicit PriorBoxValidator(const std::string& _type);

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class INFERENCE_ENGINE_API_CLASS(PriorBoxClusteredValidator) : public LayerValidator {
public:
    explicit PriorBoxClusteredValidator(const std::string& _type);

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class INFERENCE_ENGINE_API_CLASS(ProposalValidator) : public LayerValidator {
public:
    explicit ProposalValidator(const std::string& _type);

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class INFERENCE_ENGINE_API_CLASS(PSROIPoolingValidator) : public LayerValidator {
public:
    explicit PSROIPoolingValidator(const std::string& _type);

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class INFERENCE_ENGINE_API_CLASS(RegionYoloValidator) : public LayerValidator {
public:
    explicit RegionYoloValidator(const std::string& _type);

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class INFERENCE_ENGINE_API_CLASS(ReorgYoloValidator) : public LayerValidator {
public:
    explicit ReorgYoloValidator(const std::string& _type);

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class INFERENCE_ENGINE_API_CLASS(ResampleValidator) : public LayerValidator {
public:
    explicit ResampleValidator(const std::string& _type);

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class INFERENCE_ENGINE_API_CLASS(ROIPoolingValidator) : public LayerValidator {
public:
    explicit ROIPoolingValidator(const std::string& _type);

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class INFERENCE_ENGINE_API_CLASS(SimplerNMSValidator) : public LayerValidator {
public:
    explicit SimplerNMSValidator(const std::string& _type);

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class INFERENCE_ENGINE_API_CLASS(SpatialTransformerValidator) : public LayerValidator {
public:
    explicit SpatialTransformerValidator(const std::string& _type);

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class INFERENCE_ENGINE_API_CLASS(UpsamplingValidator) : public LayerValidator {
public:
    explicit UpsamplingValidator(const std::string& _type);

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class INFERENCE_ENGINE_API_CLASS(ActivationValidator) : public LayerValidator {
public:
    explicit ActivationValidator(const std::string& _type);

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class INFERENCE_ENGINE_API_CLASS(ConstValidator) : public LayerValidator {
public:
    explicit ConstValidator(const std::string& _type);

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class INFERENCE_ENGINE_API_CLASS(ELUValidator) : public LayerValidator {
public:
    explicit ELUValidator(const std::string& _type);

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class INFERENCE_ENGINE_API_CLASS(InputValidator) : public LayerValidator {
public:
    explicit InputValidator(const std::string& _type);

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class INFERENCE_ENGINE_API_CLASS(MemoryValidator) : public LayerValidator {
public:
    explicit MemoryValidator(const std::string& _type);

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class INFERENCE_ENGINE_API_CLASS(NormalizeValidator) : public LayerValidator {
public:
    explicit NormalizeValidator(const std::string& _type);

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class INFERENCE_ENGINE_API_CLASS(CopyValidator) : public LayerValidator {
public:
    explicit CopyValidator(const std::string& _type);

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class INFERENCE_ENGINE_API_CLASS(PowerFileValidator) : public LayerValidator {
public:
    explicit PowerFileValidator(const std::string& _type);

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class INFERENCE_ENGINE_API_CLASS(ReLU6Validator) : public LayerValidator {
public:
    explicit ReLU6Validator(const std::string& _type);

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class INFERENCE_ENGINE_API_CLASS(SigmoidValidator) : public LayerValidator {
public:
    explicit SigmoidValidator(const std::string& _type);

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class INFERENCE_ENGINE_API_CLASS(TanHValidator) : public LayerValidator {
public:
    explicit TanHValidator(const std::string& _type);

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class INFERENCE_ENGINE_API_CLASS(UnpoolingValidator) : public LayerValidator {
public:
    explicit UnpoolingValidator(const std::string& _type);

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class INFERENCE_ENGINE_API_CLASS(QuantizeValidator) : public LayerValidator {
public:
    explicit QuantizeValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class INFERENCE_ENGINE_API_CLASS(BinaryConvolutionValidator) : public LayerValidator {
public:
    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    explicit BinaryConvolutionValidator(const std::string& _type);

    void checkCorrespondence(const CNNLayer* layer,
                             const std::map<std::string, Blob::Ptr>& blobs,
                             const std::vector<SizeVector>& inShapes) const override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

template<typename Validator>
class ValidatorRegisterBase {
public:
    explicit ValidatorRegisterBase(const std::string& type) {
        LayerValidators::getInstance()->addImpl(type, std::make_shared<Validator>(type));
    }
};

#define REG_LAYER_VALIDATOR_FOR_TYPE(__validator, __type) \
static ValidatorRegisterBase<__validator> __reg__##__type(#__type)

REG_LAYER_VALIDATOR_FOR_TYPE(ActivationValidator, Activation);
REG_LAYER_VALIDATOR_FOR_TYPE(ArgMaxValidator, ArgMax);
REG_LAYER_VALIDATOR_FOR_TYPE(BatchNormalizationValidator, BatchNormalization);
REG_LAYER_VALIDATOR_FOR_TYPE(CTCGreedyDecoderValidator, CTCGreedyDecoder);
REG_LAYER_VALIDATOR_FOR_TYPE(ClampValidator, Clamp);
REG_LAYER_VALIDATOR_FOR_TYPE(ConcatValidator, Concat);
REG_LAYER_VALIDATOR_FOR_TYPE(ConstValidator, Const);
REG_LAYER_VALIDATOR_FOR_TYPE(ConvolutionValidator, Convolution);
REG_LAYER_VALIDATOR_FOR_TYPE(CopyValidator, Copy);
REG_LAYER_VALIDATOR_FOR_TYPE(CropValidator, Crop);
REG_LAYER_VALIDATOR_FOR_TYPE(DeconvolutionValidator, Deconvolution);
REG_LAYER_VALIDATOR_FOR_TYPE(DetectionOutputValidator, DetectionOutput);
REG_LAYER_VALIDATOR_FOR_TYPE(ELUValidator, ELU);
REG_LAYER_VALIDATOR_FOR_TYPE(EltwiseValidator, Eltwise);
REG_LAYER_VALIDATOR_FOR_TYPE(FullyConnectedValidator, InnerProduct);
REG_LAYER_VALIDATOR_FOR_TYPE(FullyConnectedValidator, FullyConnected);
REG_LAYER_VALIDATOR_FOR_TYPE(GRNValidator, GRN);
REG_LAYER_VALIDATOR_FOR_TYPE(InputValidator, Input);
REG_LAYER_VALIDATOR_FOR_TYPE(InterpValidator, Interp);
REG_LAYER_VALIDATOR_FOR_TYPE(MVNValidator, MVN);
REG_LAYER_VALIDATOR_FOR_TYPE(MemoryValidator, Memory);
REG_LAYER_VALIDATOR_FOR_TYPE(NormValidator, Norm);
REG_LAYER_VALIDATOR_FOR_TYPE(NormValidator, LRN);
REG_LAYER_VALIDATOR_FOR_TYPE(NormalizeValidator, Normalize);
REG_LAYER_VALIDATOR_FOR_TYPE(PReLUValidator, PReLU);
REG_LAYER_VALIDATOR_FOR_TYPE(PSROIPoolingValidator, PSROIPooling);
REG_LAYER_VALIDATOR_FOR_TYPE(PermuteValidator, Permute);
REG_LAYER_VALIDATOR_FOR_TYPE(PoolingValidator, Pooling);
REG_LAYER_VALIDATOR_FOR_TYPE(PowerValidator, Power);
REG_LAYER_VALIDATOR_FOR_TYPE(PowerFileValidator, PowerFile);
REG_LAYER_VALIDATOR_FOR_TYPE(PriorBoxClusteredValidator, PriorBoxClustered);
REG_LAYER_VALIDATOR_FOR_TYPE(PriorBoxValidator, PriorBox);
REG_LAYER_VALIDATOR_FOR_TYPE(ProposalValidator, Proposal);
REG_LAYER_VALIDATOR_FOR_TYPE(ROIPoolingValidator, ROIPooling);
REG_LAYER_VALIDATOR_FOR_TYPE(ReLUValidator, ReLU);
REG_LAYER_VALIDATOR_FOR_TYPE(ReLU6Validator, ReLU6);
REG_LAYER_VALIDATOR_FOR_TYPE(RegionYoloValidator, RegionYolo);
REG_LAYER_VALIDATOR_FOR_TYPE(ReorgYoloValidator, ReorgYolo);
REG_LAYER_VALIDATOR_FOR_TYPE(ResampleValidator, Resample);
REG_LAYER_VALIDATOR_FOR_TYPE(ReshapeValidator, Reshape);
REG_LAYER_VALIDATOR_FOR_TYPE(ReshapeValidator, Flatten);
REG_LAYER_VALIDATOR_FOR_TYPE(ScaleShiftValidator, ScaleShift);
REG_LAYER_VALIDATOR_FOR_TYPE(SigmoidValidator, Sigmoid);
REG_LAYER_VALIDATOR_FOR_TYPE(SigmoidValidator, Logistic);
REG_LAYER_VALIDATOR_FOR_TYPE(SimplerNMSValidator, SimplerNMS);
REG_LAYER_VALIDATOR_FOR_TYPE(SoftMaxValidator, SoftMax);
REG_LAYER_VALIDATOR_FOR_TYPE(SpatialTransformerValidator, SpatialTransformer);
REG_LAYER_VALIDATOR_FOR_TYPE(SplitValidator, Split);
REG_LAYER_VALIDATOR_FOR_TYPE(SplitValidator, Slice);
REG_LAYER_VALIDATOR_FOR_TYPE(GemmValidator, Gemm);
REG_LAYER_VALIDATOR_FOR_TYPE(PadValidator, Pad);
REG_LAYER_VALIDATOR_FOR_TYPE(GatherValidator, Gather);
REG_LAYER_VALIDATOR_FOR_TYPE(StridedSliceValidator, StridedSlice);
REG_LAYER_VALIDATOR_FOR_TYPE(ShuffleChannelsValidator, ShuffleChannels);
REG_LAYER_VALIDATOR_FOR_TYPE(DepthToSpaceValidator, DepthToSpace);
REG_LAYER_VALIDATOR_FOR_TYPE(SpaceToDepthValidator, SpaceToDepth);
REG_LAYER_VALIDATOR_FOR_TYPE(ReverseSequenceValidator, ReverseSequence);
REG_LAYER_VALIDATOR_FOR_TYPE(RNNCellValidator<RNNSequenceLayer::RNN>, RNNCell);
REG_LAYER_VALIDATOR_FOR_TYPE(RNNCellValidator<RNNSequenceLayer::GRU>, GRUCell);
REG_LAYER_VALIDATOR_FOR_TYPE(RNNCellValidator<RNNSequenceLayer::LSTM>, LSTMCell);
REG_LAYER_VALIDATOR_FOR_TYPE(RNNSequenceValidator<RNNSequenceLayer::RNN>, RNNSequence);
REG_LAYER_VALIDATOR_FOR_TYPE(RNNSequenceValidator<RNNSequenceLayer::GRU>, GRUSequence);
REG_LAYER_VALIDATOR_FOR_TYPE(RNNSequenceValidator<RNNSequenceLayer::LSTM>, LSTMSequence);
REG_LAYER_VALIDATOR_FOR_TYPE(SqueezeValidator, Squeeze);
REG_LAYER_VALIDATOR_FOR_TYPE(UnsqueezeValidator, Unsqueeze);
REG_LAYER_VALIDATOR_FOR_TYPE(RangeValidator, Range);
REG_LAYER_VALIDATOR_FOR_TYPE(FillValidator, Fill);
REG_LAYER_VALIDATOR_FOR_TYPE(ExpandValidator, Expand);
REG_LAYER_VALIDATOR_FOR_TYPE(TanHValidator, TanH);
REG_LAYER_VALIDATOR_FOR_TYPE(TileValidator, Tile);
REG_LAYER_VALIDATOR_FOR_TYPE(UnpoolingValidator, Unpooling);
REG_LAYER_VALIDATOR_FOR_TYPE(UpsamplingValidator, Upsampling);
REG_LAYER_VALIDATOR_FOR_TYPE(QuantizeValidator, Quantize);
REG_LAYER_VALIDATOR_FOR_TYPE(BinaryConvolutionValidator, BinaryConvolution);
}  // namespace details
}  // namespace InferenceEngine
