// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This header file defines the list of public transformations.
 *
 * @file ie_transformations.hpp
 */

#pragma once

#include <ie_api.h>
#include <cpp/ie_cnn_network.h>

namespace InferenceEngine {

INFERENCE_ENGINE_API_CPP(void) LowLatency(InferenceEngine::CNNNetwork& network);

}
