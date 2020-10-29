// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ie_layer_validators.hpp"

namespace InferenceEngine {
namespace ShapeInfer {

/**
 * @experimental
 * @class IConstInferImpl
 * @brief This class provides interface for the layer's implementation to propagate const
 */
class IConstInferImpl {
public:
    using Ptr = std::shared_ptr<IConstInferImpl>;

    virtual ~IConstInferImpl() = default;

    /**
     * @brief all shapes are valid, blobs are allocated
     *
     */
    virtual void infer(const std::vector<Blob::CPtr>& inData, const std::map<std::string, std::string>& params,
                       const std::map<std::string, Blob::Ptr>& blobs, std::vector<Blob::Ptr>& outData) = 0;
};

class ConstInferImpl : public IConstInferImpl {
public:
    explicit ConstInferImpl(const std::string& type): _type(type) {
        _validator = details::LayerValidators::getInstance()->getValidator(_type);
        if (!_validator)
            THROW_IE_EXCEPTION << "Internal error: failed to find validator for layer with type: " << _type;
    }

    virtual void inferImpl(const std::vector<Blob::CPtr>& inData, const std::map<std::string, std::string>& params,
                           const std::map<std::string, Blob::Ptr>& blobs, std::vector<Blob::Ptr>& outData) = 0;

    void infer(const std::vector<Blob::CPtr>& inData, const std::map<std::string, std::string>& params,
               const std::map<std::string, Blob::Ptr>& blobs, std::vector<Blob::Ptr>& outData) override;

protected:
    std::string _type;
    // to get parsed descendant CNNLayer from map<string,string>
    details::LayerValidator::Ptr _validator;
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
