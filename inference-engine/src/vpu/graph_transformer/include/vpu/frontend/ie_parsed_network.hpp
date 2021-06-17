// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <unordered_map>
#include <legacy/ie_layers.h>
#include <cpp/ie_cnn_network.h>
#include <ngraph/node_output.hpp>
#include <ngraph/output_vector.hpp>

namespace vpu {
using OutNode = ngraph::Output<ngraph::Node>;
using NodePtr = std::shared_ptr<ngraph::Node>;
namespace ie = InferenceEngine;

struct IeParsedNetwork final {
    ie::InputsDataMap networkInputs;
    ie::OutputsDataMap networkOutputs;

    // Nodes that shouldn't be executed
    ngraph::NodeVector networkParameters;
    ngraph::NodeVector networkConstants;
    ngraph::NodeVector networkResults;

    // Nodes that should be executed
    ngraph::NodeVector orderedOps;
};

IeParsedNetwork parseNetwork(const ie::CNNNetwork& network);

}  // namespace vpu
