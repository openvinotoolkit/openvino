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
class LayerValidator {
public:
    using Ptr = std::shared_ptr<LayerValidator>;

    explicit LayerValidator(const std::string& _type) : _type(_type) {}
    virtual ~LayerValidator() = default;

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
class LayerValidators {
public:
    static LayerValidators* getInstance();

    LayerValidators(LayerValidators const&) = delete;

    void operator=(LayerValidators const&)  = delete;

    LayerValidator::Ptr getValidator(const std::string& type);

private:
    LayerValidators();

private:
    static LayerValidators* _instance;
    InferenceEngine::details::caseless_unordered_map<std::string, LayerValidator::Ptr> _validators;
};

inline static void getInOutShapes(const CNNLayer* layer, InOutDims& inOutShapes) {
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

class ConvolutionValidator : public LayerValidator {
public:
    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    explicit ConvolutionValidator(const std::string& _type);

    void checkCorrespondence(const CNNLayer* layer,
                             const std::map<std::string, Blob::Ptr>& blobs,
                             const std::vector<SizeVector>& inShapes) const override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class DeconvolutionValidator : public ConvolutionValidator {
public:
    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    explicit DeconvolutionValidator(const std::string& _type);

    void checkCorrespondence(const CNNLayer* layer,
                             const std::map<std::string, Blob::Ptr>& blobs,
                             const std::vector<SizeVector>& inShapes) const override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class DeformableConvolutionValidator : public ConvolutionValidator {
public:
    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    explicit DeformableConvolutionValidator(const std::string& _type);

    void checkCorrespondence(const CNNLayer* layer,
                             const std::map<std::string, Blob::Ptr>& blobs,
                             const std::vector<SizeVector>& inShapes) const override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class PoolingValidator : public LayerValidator {
public:
    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;

    explicit PoolingValidator(const std::string& _type);
};

class FullyConnectedValidator : public LayerValidator {
public:
    explicit FullyConnectedValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkCorrespondence(const CNNLayer* layer,
                             const std::map<std::string, Blob::Ptr>& blobs,
                             const std::vector<SizeVector>& inShapes) const override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class CropValidator : public LayerValidator {
public:
    explicit CropValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class TileValidator : public LayerValidator {
public:
    explicit TileValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class BatchNormalizationValidator : public LayerValidator {
public:
    explicit BatchNormalizationValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class PowerValidator : public LayerValidator {
public:
    explicit PowerValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class PReLUValidator : public LayerValidator {
public:
    explicit PReLUValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class ScaleShiftValidator : public LayerValidator {
public:
    explicit ScaleShiftValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class ReshapeValidator : public LayerValidator {
public:
    explicit ReshapeValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;
};

class EltwiseValidator : public LayerValidator {
public:
    explicit EltwiseValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class ClampValidator : public LayerValidator {
public:
    explicit ClampValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class ReLUValidator : public LayerValidator {
public:
    explicit ReLUValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class MVNValidator : public LayerValidator {
public:
    explicit MVNValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class GRNValidator : public LayerValidator {
public:
    explicit GRNValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class SoftMaxValidator : public LayerValidator {
public:
    explicit SoftMaxValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class NormValidator : public LayerValidator {
public:
    explicit NormValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class SplitValidator : public LayerValidator {
public:
    explicit SplitValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class ConcatValidator : public LayerValidator {
public:
    explicit ConcatValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class GemmValidator : public LayerValidator {
public:
    explicit GemmValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class PadValidator : public LayerValidator {
public:
    explicit PadValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class GatherValidator : public LayerValidator {
public:
    explicit GatherValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class StridedSliceValidator : public LayerValidator {
public:
    explicit StridedSliceValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class ShuffleChannelsValidator : public LayerValidator {
public:
    explicit ShuffleChannelsValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class DepthToSpaceValidator : public LayerValidator {
public:
    explicit DepthToSpaceValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class SpaceToDepthValidator : public LayerValidator {
public:
    explicit SpaceToDepthValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class SparseFillEmptyRowsValidator : public LayerValidator {
public:
    explicit SparseFillEmptyRowsValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class SparseSegmentReduceValidator : public LayerValidator {
public:
    explicit SparseSegmentReduceValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class ReverseSequenceValidator : public LayerValidator {
public:
    explicit ReverseSequenceValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class SqueezeValidator : public LayerValidator {
public:
    explicit SqueezeValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class UnsqueezeValidator : public LayerValidator {
public:
    explicit UnsqueezeValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class RangeValidator : public LayerValidator {
public:
    explicit RangeValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class FillValidator : public LayerValidator {
public:
    explicit FillValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class BroadcastValidator : public LayerValidator {
public:
    explicit BroadcastValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class RNNBaseValidator : public LayerValidator {
public:
    RNNBaseValidator(const std::string& _type, RNNSequenceLayer::CellType CELL);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkCorrespondence(const CNNLayer* layer,
                             const std::map<std::string, Blob::Ptr>& blobs,
                             const std::vector<SizeVector>& inShapes) const override;

protected:
    std::vector<std::string> def_acts;  // Default values for cell gate activations
    std::vector<float> def_alpha;  // Default activation alpha parameter
    std::vector<float> def_beta;   // Default activation beta parameter
    size_t G;   // gate number
    size_t NS;  // state number
};

template<RNNSequenceLayer::CellType CELL>
class RNNCellValidator : public RNNBaseValidator {
public:
    explicit RNNCellValidator(const std::string& _type);

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

extern template class RNNCellValidator<RNNSequenceLayer::LSTM>;
extern template class RNNCellValidator<RNNSequenceLayer::GRU>;
extern template class RNNCellValidator<RNNSequenceLayer::RNN>;

template<RNNSequenceLayer::CellType CELL>
class RNNSequenceValidator : public RNNBaseValidator {
public:
    explicit RNNSequenceValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

extern template class RNNSequenceValidator<RNNSequenceLayer::LSTM>;
extern template class RNNSequenceValidator<RNNSequenceLayer::GRU>;
extern template class RNNSequenceValidator<RNNSequenceLayer::RNN>;

class ArgMaxValidator : public LayerValidator {
public:
    explicit ArgMaxValidator(const std::string& _type);

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class CTCGreedyDecoderValidator : public LayerValidator {
public:
    explicit CTCGreedyDecoderValidator(const std::string& _type);

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class DetectionOutputValidator : public LayerValidator {
public:
    explicit DetectionOutputValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class InterpValidator : public LayerValidator {
public:
    explicit InterpValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class PermuteValidator : public LayerValidator {
public:
    explicit PermuteValidator(const std::string& _type);

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class PriorBoxValidator : public LayerValidator {
public:
    explicit PriorBoxValidator(const std::string& _type);

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class PriorBoxClusteredValidator : public LayerValidator {
public:
    explicit PriorBoxClusteredValidator(const std::string& _type);

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class ProposalValidator : public LayerValidator {
public:
    explicit ProposalValidator(const std::string& _type);

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class PSROIPoolingValidator : public LayerValidator {
public:
    explicit PSROIPoolingValidator(const std::string& _type);

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class RegionYoloValidator : public LayerValidator {
public:
    explicit RegionYoloValidator(const std::string& _type);

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class ReorgYoloValidator : public LayerValidator {
public:
    explicit ReorgYoloValidator(const std::string& _type);

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class ResampleValidator : public LayerValidator {
public:
    explicit ResampleValidator(const std::string& _type);

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class ROIPoolingValidator : public LayerValidator {
public:
    explicit ROIPoolingValidator(const std::string& _type);

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class SimplerNMSValidator : public LayerValidator {
public:
    explicit SimplerNMSValidator(const std::string& _type);

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class SpatialTransformerValidator : public LayerValidator {
public:
    explicit SpatialTransformerValidator(const std::string& _type);

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class OneHotValidator : public LayerValidator {
public:
    explicit OneHotValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class UpsamplingValidator : public LayerValidator {
public:
    explicit UpsamplingValidator(const std::string& _type);

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class ActivationValidator : public LayerValidator {
public:
    explicit ActivationValidator(const std::string& _type);

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class ConstValidator : public LayerValidator {
public:
    explicit ConstValidator(const std::string& _type);

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class ELUValidator : public LayerValidator {
public:
    explicit ELUValidator(const std::string& _type);

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class InputValidator : public LayerValidator {
public:
    explicit InputValidator(const std::string& _type);

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class MemoryValidator : public LayerValidator {
public:
    explicit MemoryValidator(const std::string& _type);

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class NormalizeValidator : public LayerValidator {
public:
    explicit NormalizeValidator(const std::string& _type);

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class CopyValidator : public LayerValidator {
public:
    explicit CopyValidator(const std::string& _type);

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class PowerFileValidator : public LayerValidator {
public:
    explicit PowerFileValidator(const std::string& _type);

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class ReLU6Validator : public LayerValidator {
public:
    explicit ReLU6Validator(const std::string& _type);

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class SigmoidValidator : public LayerValidator {
public:
    explicit SigmoidValidator(const std::string& _type);

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class TanHValidator : public LayerValidator {
public:
    explicit TanHValidator(const std::string& _type);

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class UnpoolingValidator : public LayerValidator {
public:
    explicit UnpoolingValidator(const std::string& _type);

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class QuantizeValidator : public LayerValidator {
public:
    explicit QuantizeValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class BinaryConvolutionValidator : public LayerValidator {
public:
    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    explicit BinaryConvolutionValidator(const std::string& _type);

    void checkCorrespondence(const CNNLayer* layer,
                             const std::map<std::string, Blob::Ptr>& blobs,
                             const std::vector<SizeVector>& inShapes) const override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class SelectValidator : public LayerValidator {
public:
    explicit SelectValidator(const std::string& _type);

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class MathValidator : public LayerValidator {
public:
    explicit MathValidator(const std::string& _type);

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class ReduceValidator : public LayerValidator {
public:
    explicit ReduceValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class GatherTreeValidator : public LayerValidator {
public:
    explicit GatherTreeValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class TopKValidator : public LayerValidator {
public:
    explicit TopKValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class UniqueValidator : public LayerValidator {
public:
    explicit UniqueValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class NMSValidator : public LayerValidator {
public:
    explicit NMSValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class ScatterValidator : public LayerValidator {
public:
    explicit ScatterValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

}  // namespace details
}  // namespace InferenceEngine
