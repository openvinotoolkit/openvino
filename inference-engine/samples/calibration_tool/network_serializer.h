// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "inference_engine.hpp"
#include <pugixml/pugixml.hpp>
#include <string>

/** Class for serialization of model been presented as ICNNNetwork to the disk
 */
class CNNNetworkSerializer {
public:
    void Serialize(const std::string &xmlPath, const std::string &binPath,
                   InferenceEngine::ICNNNetwork& network);

protected:
    void updateStdLayerParams(InferenceEngine::CNNLayer::Ptr layer);
};
