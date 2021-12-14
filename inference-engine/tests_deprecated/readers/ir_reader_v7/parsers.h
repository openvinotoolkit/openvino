// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <legacy/cnn_network_impl.hpp>

namespace pugi {
class xml_node;
}  // namespace pugi

namespace InferenceEngine {
namespace details {
struct IFormatParser {
    virtual ~IFormatParser() {}

    virtual CNNNetworkImplPtr Parse(pugi::xml_node& root) = 0;

    virtual void SetWeights(const TBlob<uint8_t>::Ptr& weights) = 0;
};
}  // namespace details
}  // namespace InferenceEngine
