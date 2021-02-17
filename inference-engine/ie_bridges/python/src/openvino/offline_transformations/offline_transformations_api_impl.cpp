// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "offline_transformations_api_impl.hpp"

#include <moc_transformations.hpp>

#include <transformations/control_flow/unroll_tensor_iterator.hpp>

#include <ngraph/pass/low_latency.hpp>
#include <ngraph/pass/manager.hpp>

void InferenceEnginePython::ApplyMOCTransformations(InferenceEnginePython::IENetwork network, bool cf) {
    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::MOCTransformations>(cf);
    manager.run_passes(network.actual->getFunction());
}

void InferenceEnginePython::ApplyLowLatencyTransformation(InferenceEnginePython::IENetwork network) {
    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::LowLatency>();
    manager.register_pass<ngraph::pass::UnrollTensorIterator>();

    auto pass_config = manager.get_pass_config();
    pass_config->set_callback<ngraph::pass::UnrollTensorIterator>([](const std::shared_ptr<const ngraph::Node> &node) -> bool {
        return node->get_rt_info().count("UNROLL_TI") == 0;
    });
    manager.run_passes(network.actual->getFunction());
}
