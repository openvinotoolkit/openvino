// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_transformations.hpp"
#include <ngraph/pass/low_latency.hpp>
#include <ngraph/pass/manager.hpp>

using namespace InferenceEngine;

void InferenceEngine::LowLatency(InferenceEngine::CNNNetwork &network) {
    auto function = network.getFunction();
    ngraph::pass::Manager manager;
    NGRAPH_SUPPRESS_DEPRECATED_START
    manager.register_pass<ngraph::pass::LowLatency>();
    NGRAPH_SUPPRESS_DEPRECATED_END
    manager.run_passes(function);
}

void InferenceEngine::lowLatency2(InferenceEngine::CNNNetwork &network,
                                  bool use_const_initializer) {
    auto function = network.getFunction();
    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::LowLatency2>(use_const_initializer);
    manager.run_passes(function);
}
