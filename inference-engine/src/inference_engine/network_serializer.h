// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "xml_parse_utils.h"

namespace InferenceEngine {
namespace details {

/**
* Class for serialization of model been presented as ICNNNetwork to the disk
*/
class NetworkSerializer {
public:
    static void serialize(const std::string &xmlPath, const std::string &binPath, const InferenceEngine::ICNNNetwork& network);

private:
    static void updateStdLayerParams(const InferenceEngine::CNNLayer::Ptr &layer);
    static void updatePreProcInfo(const InferenceEngine::ICNNNetwork& network, pugi::xml_node &netXml);
    static void updateStatisticsInfo(const InferenceEngine::ICNNNetwork& network, pugi::xml_node &netXml);
};

}  // namespace details
}  // namespace InferenceEngine
