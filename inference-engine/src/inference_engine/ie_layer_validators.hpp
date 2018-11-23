// Copyright (C) 2018 Intel Corporation
//
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
    virtual void checkShapes(const CNNLayer* layer,
                             const std::vector<SizeVector>& inShapes) const {}

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

static void checkWeakData(const DataWeakPtr& data) {
}

static void checkData(const DataPtr& data) {
}


/**
 * @brief Checks that input Data is not empty and pointers are not null, number of inputs correspond number of input shapes, dimensions in Data are not empty
 */
static void checkInputs(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) {
    // TODO: not finished implementation
    if (layer->insData.size() != inShapes.size())
        THROW_IE_EXCEPTION << "Number of layer's inputs don't correspond number of new input shapes";

    auto inData = layer->insData[0].lock();
    bool isCorrect = false;
    SizeVector inDims, inShape;
    if (inData) {
        inDims = inData->getDims();
        inShape = inShapes[0];
        isCorrect = inShape.size() == inDims.size() && !inShape.empty() && !inDims.empty();
    }

    if (!isCorrect)
        THROW_IE_EXCEPTION << " Failed with invalid shapes: shapes are empty"
                                  << "new input shape size=" << inShape.size() << ", input shape size in IR="
                                  << inDims.size();
}

/**
 * @brief Checks that output Data is not empty and pointers are not null, number of outputs correspond number of output shapes, dimensions in Data are not empty
 */
static void checkOutputs(const CNNLayer* layer, const std::vector<SizeVector>& outShapes) {}

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
};

class INFERENCE_ENGINE_API_CLASS(DeconvolutionValidator) : public LayerValidator {
public:
    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    explicit DeconvolutionValidator(const std::string& _type);

    void checkCorrespondence(const CNNLayer* layer,
                             const std::map<std::string, Blob::Ptr>& blobs,
                             const std::vector<SizeVector>& inShapes) const override;
};


class INFERENCE_ENGINE_API_CLASS(PoolingValidator) : public LayerValidator {
public:
    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

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
};

class INFERENCE_ENGINE_API_CLASS(BatchNormalizationValidator) : public LayerValidator {
public:
    explicit BatchNormalizationValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;
};

class INFERENCE_ENGINE_API_CLASS(PowerValidator) : public LayerValidator {
public:
    explicit PowerValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;
};

class INFERENCE_ENGINE_API_CLASS(PReLUValidator) : public LayerValidator {
public:
    explicit PReLUValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;
};

class INFERENCE_ENGINE_API_CLASS(ScaleShiftValidator) : public LayerValidator {
public:
    explicit ScaleShiftValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;
};

class INFERENCE_ENGINE_API_CLASS(ReshapeValidator) : public LayerValidator {
public:
    explicit ReshapeValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

protected:
    void calculateIn2Out(ReshapeLayer* layer);
};

class INFERENCE_ENGINE_API_CLASS(EltwiseValidator) : public LayerValidator {
public:
    explicit EltwiseValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;
};

class INFERENCE_ENGINE_API_CLASS(ClampValidator) : public LayerValidator {
public:
    explicit ClampValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;
};

class INFERENCE_ENGINE_API_CLASS(ReLUValidator) : public LayerValidator {
public:
    explicit ReLUValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;
};

class INFERENCE_ENGINE_API_CLASS(MVNValidator) : public LayerValidator {
public:
    explicit MVNValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;
};

class INFERENCE_ENGINE_API_CLASS(GRNValidator) : public LayerValidator {
public:
    explicit GRNValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;
};

class INFERENCE_ENGINE_API_CLASS(SoftMaxValidator) : public LayerValidator {
public:
    explicit SoftMaxValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;
};

class INFERENCE_ENGINE_API_CLASS(NormValidator) : public LayerValidator {
public:
    explicit NormValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;
};

class INFERENCE_ENGINE_API_CLASS(SplitValidator) : public LayerValidator {
public:
    explicit SplitValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;
};

class INFERENCE_ENGINE_API_CLASS(ConcatValidator) : public LayerValidator {
public:
    explicit ConcatValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;
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

REG_LAYER_VALIDATOR_FOR_TYPE(ConvolutionValidator, Convolution);
REG_LAYER_VALIDATOR_FOR_TYPE(DeconvolutionValidator, Deconvolution);
REG_LAYER_VALIDATOR_FOR_TYPE(PoolingValidator, Pooling);
REG_LAYER_VALIDATOR_FOR_TYPE(FullyConnectedValidator, InnerProduct);
REG_LAYER_VALIDATOR_FOR_TYPE(FullyConnectedValidator, FullyConnected);
REG_LAYER_VALIDATOR_FOR_TYPE(CropValidator, Crop);
REG_LAYER_VALIDATOR_FOR_TYPE(BatchNormalizationValidator, BatchNormalization);
REG_LAYER_VALIDATOR_FOR_TYPE(PowerValidator, Power);
REG_LAYER_VALIDATOR_FOR_TYPE(PReLUValidator, PReLU);
REG_LAYER_VALIDATOR_FOR_TYPE(ScaleShiftValidator, ScaleShift);
REG_LAYER_VALIDATOR_FOR_TYPE(TileValidator, Tile);
REG_LAYER_VALIDATOR_FOR_TYPE(ReshapeValidator, Reshape);
REG_LAYER_VALIDATOR_FOR_TYPE(ReshapeValidator, Flatten);
REG_LAYER_VALIDATOR_FOR_TYPE(EltwiseValidator, Eltwise);
REG_LAYER_VALIDATOR_FOR_TYPE(ClampValidator, Clamp);
REG_LAYER_VALIDATOR_FOR_TYPE(ReLUValidator, ReLU);
REG_LAYER_VALIDATOR_FOR_TYPE(MVNValidator, MVN);
REG_LAYER_VALIDATOR_FOR_TYPE(GRNValidator, GRN);
REG_LAYER_VALIDATOR_FOR_TYPE(SoftMaxValidator, SoftMax);
REG_LAYER_VALIDATOR_FOR_TYPE(NormValidator, Norm);
REG_LAYER_VALIDATOR_FOR_TYPE(NormValidator, LRN);
REG_LAYER_VALIDATOR_FOR_TYPE(SplitValidator, Split);
REG_LAYER_VALIDATOR_FOR_TYPE(SplitValidator, Slice);
REG_LAYER_VALIDATOR_FOR_TYPE(ConcatValidator, Concat);

}  // namespace details
}  // namespace InferenceEngine
