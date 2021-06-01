// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "offline_transformations_api_impl.hpp"

#include <generate_mapping_file.hpp>
#include <moc_transformations.hpp>
#include <ngraph/opsets/opset6.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <ngraph/pass/low_latency.hpp>
#include <ngraph/pass/manager.hpp>
#include <pot_transformations.hpp>
#include <pruning.hpp>
#include <transformations/control_flow/unroll_tensor_iterator.hpp>

void InferenceEnginePython::ApplyMOCTransformations(InferenceEnginePython::IENetwork network, bool cf) {
    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::MOCTransformations>(cf);
    manager.run_passes(network.actual->getFunction());
}

void InferenceEnginePython::ApplyPOTTransformations(InferenceEnginePython::IENetwork network, std::string device) {
    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::POTTransformations>(std::move(device));
    manager.run_passes(network.actual->getFunction());
}

void InferenceEnginePython::ApplyLowLatencyTransformation(InferenceEnginePython::IENetwork network, bool use_const_initializer) {
    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::LowLatency2>(use_const_initializer);
    manager.run_passes(network.actual->getFunction());
}

void InferenceEnginePython::ApplyPruningTransformation(InferenceEnginePython::IENetwork network) {
    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::Pruning>();
    manager.run_passes(network.actual->getFunction());
}

void InferenceEnginePython::GenerateMappingFile(InferenceEnginePython::IENetwork network, std::string path, bool extract_names) {
    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::GenerateMappingFile>(path, extract_names);
    manager.run_passes(network.actual->getFunction());
}

void InferenceEnginePython::CheckAPI() {
    std::shared_ptr<ngraph::Function> f;
    {
        auto input = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape {1, 1000, 4});
        auto reshape = std::make_shared<ngraph::opset6::Reshape>(input, std::make_shared<ngraph::opset6::ShapeOf>(input), true);
        f = std::make_shared<ngraph::Function>(ngraph::NodeVector {reshape}, ngraph::ParameterVector {input});
    }
    ngraph::pass::Manager m;
    m.register_pass<ngraph::pass::ConstantFolding>();
    m.run_passes(f);

    assert(f->get_results().size() == 1);
    auto reshape = f->get_result()->input_value(0).get_node_shared_ptr();
    assert(std::dynamic_pointer_cast<ngraph::opset6::Parameter>(reshape->input_value(0).get_node_shared_ptr()));
    assert(std::dynamic_pointer_cast<ngraph::opset6::Constant>(reshape->input_value(1).get_node_shared_ptr()));
}