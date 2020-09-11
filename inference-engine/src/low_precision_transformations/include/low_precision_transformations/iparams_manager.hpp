// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ie_precision.hpp"

namespace InferenceEngine {
namespace details {

/**
 * @brief low precision transformation component interface.
  */
class INFERENCE_ENGINE_API_CLASS(IParamsManager) {
public:
    virtual std::vector<Precision> getPrecisionsOnActivations(const std::string& layerName) const noexcept = 0;
};

}  // namespace details
}  // namespace InferenceEngine
