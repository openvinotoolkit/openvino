// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

namespace InferenceEngine {
namespace details {

/**
* Class for serialization of model been presented as ICNNNetwork to the disk
*/
class NetworkSerializer {
public:
    static void serialize(const std::string &xmlPath, const std::string &binPath, const InferenceEngine::ICNNNetwork& network);

private:
    static void updateStdLayerParams(InferenceEngine::CNNLayer::Ptr layer);
};

}  // namespace details
}  // namespace InferenceEngine
