// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ie_layers.h"

namespace InferenceEngine {
namespace details {

/**
 * @brief low precision transformation component interface.
  */
class INFERENCE_ENGINE_API_CLASS(ILayerTransformationsManager) {
public:
    virtual bool isQuantized(const CNNLayer& layer) const noexcept = 0;
    virtual bool isPrecisionPreserved(const CNNLayer& layer) const noexcept = 0;
};

}  // namespace details
}  // namespace InferenceEngine
