// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for the IE Context class
 *
 * @file ie_context.hpp
 */
#pragma once

#include <ie_iextension.h>

#include <details/caseless.hpp>
#include <map>
#include <string>
#include <vector>

namespace InferenceEngine {

/**
 * @deprecated Use ngraph API instead.
 * @brief This class implements object
 */
class INFERENCE_ENGINE_NN_BUILDER_API_CLASS(Context) {
public:
    Context();

    /**
     * @brief Registers extension within the context
     *
     * @param ext Pointer to already loaded extension
     */
    void addExtension(const IShapeInferExtensionPtr& ext);

    /**
     * @brief Registers Shape Infer implementation within the Context
     *
     * @param type Layer type
     * @param impl Shape Infer implementation
     */
    IE_SUPPRESS_DEPRECATED_START
    void addShapeInferImpl(const std::string& type, const IShapeInferImpl::Ptr& impl);

    /**
     * @brief Returns the shape infer implementation by layer type
     *
     * @param type Layer type
     * @return Shape Infer implementation
     */
    IShapeInferImpl::Ptr getShapeInferImpl(const std::string& type);

private:
    details::caseless_map<std::string, IShapeInferImpl::Ptr> shapeInferImpls;
    IE_SUPPRESS_DEPRECATED_END
};

}  // namespace InferenceEngine
