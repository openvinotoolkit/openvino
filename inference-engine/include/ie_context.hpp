// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for the IE Context class
 * @file ie_context.hpp
 */
#pragma once

#include <details/caseless.hpp>
#include <ie_iextension.h>
#include <string>
#include <vector>
#include <map>

namespace InferenceEngine {

/**
 * @brief This class implements object
 */
class INFERENCE_ENGINE_API_CLASS(Context) {
public:
    Context();

    /**
     * @brief Registers extension within the context
     * @param ext Pointer to already loaded extension
     */
    void addExtension(const IShapeInferExtensionPtr& ext);

    /**
     * @brief Registers Shape Infer implementation within the Context
     * @param type Layer type
     * @param impl Shape Infer implementation
     */
    void addShapeInferImpl(const std::string& type, const IShapeInferImpl::Ptr& impl);

    /**
     * @brief Returns the shape infer implementation by layer type
     * @param type Layer type
     * @return Shape Infer implementation
     */
    IShapeInferImpl::Ptr getShapeInferImpl(const std::string& type);

private:
    details::caseless_map<std::string, IShapeInferImpl::Ptr> shapeInferImpls;
};

}  // namespace InferenceEngine
