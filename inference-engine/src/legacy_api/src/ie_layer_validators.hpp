// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "caseless.hpp"
#include <legacy/ie_layers.h>

namespace InferenceEngine {
namespace details {

/**
 * @brief Contains methods to validate layer of specific type
 */
class LayerValidator {
public:
    using Ptr = std::shared_ptr<LayerValidator>;

    explicit LayerValidator(const std::string& _type): _type(_type) {}
    virtual ~LayerValidator() = default;

    /**
     * @brief It parses map of params <string,string> and applies to the layer's fields.
     * This checks for presence of all required attributes, and that there's no extraneous parameters only.
     * Throws exception in case of parsing error
     */
    virtual void parseParams(CNNLayer* layer) {}

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

    void operator=(LayerValidators const&) = delete;

    LayerValidator::Ptr getValidator(const std::string& type);

private:
    LayerValidators();

private:
    InferenceEngine::details::caseless_unordered_map<std::string, LayerValidator::Ptr> _validators;
};

class GeneralValidator : public LayerValidator {
public:
    explicit GeneralValidator(const std::string& _type);
};

class ConvolutionValidator : public LayerValidator {
public:
    void parseParams(CNNLayer* layer) override;

    explicit ConvolutionValidator(const std::string& _type);
};

class DeconvolutionValidator : public ConvolutionValidator {
public:
    void parseParams(CNNLayer* layer) override;

    explicit DeconvolutionValidator(const std::string& _type);
};

class DeformableConvolutionValidator : public ConvolutionValidator {
public:
    void parseParams(CNNLayer* layer) override;

    explicit DeformableConvolutionValidator(const std::string& _type);
};

class PoolingValidator : public LayerValidator {
public:
    void parseParams(CNNLayer* layer) override;

    explicit PoolingValidator(const std::string& _type);
};

class FullyConnectedValidator : public LayerValidator {
public:
    explicit FullyConnectedValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;
};

class CropValidator : public LayerValidator {
public:
    explicit CropValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;
};

class TileValidator : public LayerValidator {
public:
    explicit TileValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;
};

class BatchNormalizationValidator : public LayerValidator {
public:
    explicit BatchNormalizationValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;
};

class PowerValidator : public LayerValidator {
public:
    explicit PowerValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;
};

class PReLUValidator : public LayerValidator {
public:
    explicit PReLUValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;
};

class ScaleShiftValidator : public LayerValidator {
public:
    explicit ScaleShiftValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;};

class ReshapeValidator : public LayerValidator {
public:
    explicit ReshapeValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;
};

class EltwiseValidator : public LayerValidator {
public:
    explicit EltwiseValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;
};

class ClampValidator : public LayerValidator {
public:
    explicit ClampValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;
};

class ReLUValidator : public LayerValidator {
public:
    explicit ReLUValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;
};

class MVNValidator : public LayerValidator {
public:
    explicit MVNValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;
};

class GRNValidator : public LayerValidator {
public:
    explicit GRNValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;
};

class SoftMaxValidator : public LayerValidator {
public:
    explicit SoftMaxValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;
};

class NormValidator : public LayerValidator {
public:
    explicit NormValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;
};

class SplitValidator : public LayerValidator {
public:
    explicit SplitValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;
};

class ConcatValidator : public LayerValidator {
public:
    explicit ConcatValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;
};

class GemmValidator : public LayerValidator {
public:
    explicit GemmValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;
};

class PadValidator : public LayerValidator {
public:
    explicit PadValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;
};

class GatherValidator : public LayerValidator {
public:
    explicit GatherValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;
};

class StridedSliceValidator : public LayerValidator {
public:
    explicit StridedSliceValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;
};

class ShuffleChannelsValidator : public LayerValidator {
public:
    explicit ShuffleChannelsValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;
};

class DepthToSpaceValidator : public LayerValidator {
public:
    explicit DepthToSpaceValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;
};

class SpaceToDepthValidator : public LayerValidator {
public:
    explicit SpaceToDepthValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;
};

class SpaceToBatchValidator : public LayerValidator {
public:
    explicit SpaceToBatchValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;
};

class BatchToSpaceValidator : public LayerValidator {
public:
    explicit BatchToSpaceValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;
};

class SparseFillEmptyRowsValidator : public LayerValidator {
public:
    explicit SparseFillEmptyRowsValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;
};

class BucketizeValidator : public LayerValidator {
public:
    explicit BucketizeValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;
};

class ReverseSequenceValidator : public LayerValidator {
public:
    explicit ReverseSequenceValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;
};

class RNNBaseValidator : public LayerValidator {
public:
    RNNBaseValidator(const std::string& _type, RNNSequenceLayer::CellType CELL);

    void parseParams(CNNLayer* layer) override;
protected:
    std::vector<std::string> def_acts;  // Default values for cell gate activations
    std::vector<float> def_alpha;       // Default activation alpha parameter
    std::vector<float> def_beta;        // Default activation beta parameter
    size_t G;                           // gate number
    size_t NS;                          // state number
};

template <RNNSequenceLayer::CellType CELL>
class RNNCellValidator : public RNNBaseValidator {
public:
    explicit RNNCellValidator(const std::string& _type);
};

extern template class RNNCellValidator<RNNSequenceLayer::LSTM>;
extern template class RNNCellValidator<RNNSequenceLayer::GRU>;
extern template class RNNCellValidator<RNNSequenceLayer::RNN>;

template <RNNSequenceLayer::CellType CELL>
class RNNSequenceValidator : public RNNBaseValidator {
public:
    explicit RNNSequenceValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;
};

extern template class RNNSequenceValidator<RNNSequenceLayer::LSTM>;
extern template class RNNSequenceValidator<RNNSequenceLayer::GRU>;
extern template class RNNSequenceValidator<RNNSequenceLayer::RNN>;

class DetectionOutputValidator : public LayerValidator {
public:
    explicit DetectionOutputValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;
};

class ProposalValidator : public LayerValidator {
public:
    explicit ProposalValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;
};

class OneHotValidator : public LayerValidator {
public:
    explicit OneHotValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;
};

class QuantizeValidator : public LayerValidator {
public:
    explicit QuantizeValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;
};

class BinaryConvolutionValidator : public LayerValidator {
public:
    void parseParams(CNNLayer* layer) override;

    explicit BinaryConvolutionValidator(const std::string& _type);
};

class ReduceValidator : public LayerValidator {
public:
    explicit ReduceValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;
};

class TopKValidator : public LayerValidator {
public:
    explicit TopKValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;
};

class UniqueValidator : public LayerValidator {
public:
    explicit UniqueValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;
};

class NMSValidator : public LayerValidator {
public:
    explicit NMSValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;
};

class ScatterUpdateValidator : public LayerValidator {
public:
    explicit ScatterUpdateValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;
};

class ScatterElementsUpdateValidator : public LayerValidator {
public:
    explicit ScatterElementsUpdateValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;
};

}  // namespace details
}  // namespace InferenceEngine
