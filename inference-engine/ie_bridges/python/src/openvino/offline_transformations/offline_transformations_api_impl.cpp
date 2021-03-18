// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "offline_transformations_api_impl.hpp"

#include <moc_transformations.hpp>

#include <transformations/control_flow/unroll_tensor_iterator.hpp>

#include <ngraph/pass/constant_folding.hpp>
#include <ngraph/pass/low_latency.hpp>
#include <ngraph/pass/manager.hpp>

#include <ngraph/opsets/opset6.hpp>

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

void InferenceEnginePython::CheckAPI() {
    std::shared_ptr<ngraph::Function> f;
    {
        auto input = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{1, 1000, 4});
        auto reshape = std::make_shared<ngraph::opset6::Reshape>(input, std::make_shared<ngraph::opset6::ShapeOf>(input), true);
        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{reshape}, ngraph::ParameterVector{input});
    }
    ngraph::pass::Manager m;
    m.register_pass<ngraph::pass::ConstantFolding>();
    m.run_passes(f);

    assert(f->get_results().size() == 1);
    auto reshape = f->get_result()->input_value(0).get_node_shared_ptr();
    assert(std::dynamic_pointer_cast<ngraph::opset6::Parameter>(reshape->input_value(0).get_node_shared_ptr()));
    assert(std::dynamic_pointer_cast<ngraph::opset6::Constant>(reshape->input_value(1).get_node_shared_ptr()));
}