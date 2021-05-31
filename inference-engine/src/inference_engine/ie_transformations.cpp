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
    manager.register_pass<ngraph::pass::LowLatency>();
    manager.run_passes(function);
}

void InferenceEngine::LowLatency2(InferenceEngine::CNNNetwork &network,
                                    bool use_const_initializer,
                                    const ngraph::pass::LowLatency2::SubGraphIterations& sub_graph_iterations) {
    auto function = network.getFunction();
    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::LowLatency2>(use_const_initializer, sub_graph_iterations);
    manager.run_passes(function);
}
