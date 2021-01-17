// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "offline_transformations_api_impl.hpp"

#include <moc_transformations.hpp>
#include <ngraph/pass/manager.hpp>

void InferenceEnginePython::ApplyMOCTransformations(InferenceEnginePython::IENetwork network, bool cf) {
    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::MOCTransformations>(cf);
    manager.run_passes(network.actual->getFunction());
}