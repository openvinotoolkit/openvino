// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_transformations.hpp"

#include "ngraph/pass/low_latency.hpp"
#include "ngraph/pass/manager.hpp"
#include "openvino/pass/make_stateful_test.hpp"

using namespace InferenceEngine;

void InferenceEngine::LowLatency(InferenceEngine::CNNNetwork& network) {
    auto function = network.getFunction();
    ngraph::pass::Manager manager;
    NGRAPH_SUPPRESS_DEPRECATED_START
    manager.register_pass<ngraph::pass::LowLatency>();
    NGRAPH_SUPPRESS_DEPRECATED_END
    manager.run_passes(function);
}

void InferenceEngine::lowLatency2(InferenceEngine::CNNNetwork& network, bool use_const_initializer) {
    auto function = network.getFunction();
    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::LowLatency2>(use_const_initializer);
    manager.run_passes(function);
}

INFERENCE_ENGINE_API_CPP(void)
InferenceEngine::makeStateful(InferenceEngine::CNNNetwork& network,
                              std::vector<std::pair<std::string, std::string>>& in_out_names) {
    auto function = network.getFunction();
    ngraph::pass::Manager manager;
    manager.register_pass<ov::pass::MakeStateful>(
        ov::pass::MakeStateful::find_param_results_by_names(function, in_out_names));
    manager.run_passes(function);
}
