// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "details/caseless.hpp"
#include "ie_layers.h"

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

    explicit LayerValidator(const std::string& _type): _type(_type) {}
    virtual ~LayerValidator() = default;

    /**
     * @brief It parses map of params <string,string> and applies to the layer's fields.
     * This checks for presence of all required attributes, and that there's no extraneous parameters only.
     * Throws exception in case of parsing error
     */
    virtual void parseParams(CNNLayer* layer) {}

    /**
     * @brief Validates layer parameters separately from blobs and shapes
     * This is semantic check, like height and width more than kernel sizes, stride > 0, beta > 0, axis is correct and
     * etc Throws exception if the check fails
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
    virtual void checkCorrespondence(const CNNLayer* layer, const std::map<std::string, Blob::Ptr>& blobs,
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

    void operator=(LayerValidators const&) = delete;

    LayerValidator::Ptr getValidator(const std::string& type);

private:
    LayerValidators();

private:
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

class SparseToDenseValidator : public LayerValidator {
public:
    explicit SparseToDenseValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class RNNBaseValidator : public LayerValidator {
public:
    RNNBaseValidator(const std::string& _type, RNNSequenceLayer::CellType CELL);

    void parseParams(CNNLayer* layer) override;

    void checkParams(const CNNLayer* layer) override;

    void checkCorrespondence(const CNNLayer* layer, const std::map<std::string, Blob::Ptr>& blobs,
                             const std::vector<SizeVector>& inShapes) const override;

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

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

extern template class RNNCellValidator<RNNSequenceLayer::GRU>;
extern template class RNNCellValidator<RNNSequenceLayer::RNN>;


class ProposalValidator : public LayerValidator {
public:
    explicit ProposalValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

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

class PowerFileValidator : public LayerValidator {
public:
    explicit PowerFileValidator(const std::string& _type);

    void checkParams(const CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

class UniqueValidator : public LayerValidator {
public:
    explicit UniqueValidator(const std::string& _type);

    void parseParams(CNNLayer* layer) override;

    void checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const override;
};

}  // namespace details
}  // namespace InferenceEngine
