// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_transformations.hpp"
#include <ngraph/pass/low_latency.hpp>
#include <ngraph/pass/manager.hpp>

using namespace InferenceEngine;

void InferenceEngine::LowLatency(InferenceEngine::CNNNetwork &network) {
    auto function = network.getFunction();
    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::LowLatency>();
    manager.run_passes(function);
}
