// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <moc_transformations.hpp>
#include <ngraph/pass/manager.hpp>

#include "offline_api_impl.hpp"

void InferenceEnginePython::ApplyMOCTransformations(InferenceEnginePython::IENetwork network, bool cf) {
    ngraph::pass::Manager manager;
    manager.register_pass<MOCTransformations>(cf);
    manager.run_passes(network.actual->getFunction());
}

void InferenceEnginePython::ApplyMOCTransformationsCPU(InferenceEnginePython::IENetwork network) {
    ngraph::pass::Manager manager;
    manager.register_pass<MOCTransformationsCPU>();
    manager.run_passes(network.actual->getFunction());
}