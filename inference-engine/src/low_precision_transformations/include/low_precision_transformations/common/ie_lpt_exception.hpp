// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "details/ie_exception.hpp"
#include <string>
#include <legacy/ie_layers.h>

/**
* @def THROW_IE_EXCEPTION_LPT
* @brief A macro used to throw the exception with a notable description for low precision transformations
*/
#define THROW_IE_LPT_EXCEPTION(layer) throw InferenceEngine::details::InferenceEngineLptException(__FILE__, __LINE__, layer)

namespace InferenceEngine {
namespace details {

class INFERENCE_ENGINE_API_CLASS(InferenceEngineLptException) : public InferenceEngineException {
public:
    InferenceEngineLptException(const std::string& filename, const int line, const CNNLayer& layer) : InferenceEngineException(filename, line) {
        *this << "Exception during low precision transformation for " << layer.type << " layer '" << layer.name << "'. ";
    }
};

}  // namespace details
}  // namespace InferenceEngine
