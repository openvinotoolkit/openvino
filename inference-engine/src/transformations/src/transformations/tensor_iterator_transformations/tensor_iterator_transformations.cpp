// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/tensor_iterator_transformations/tensor_iterator_transformations.hpp"

#include <transformations/tensor_iterator_transformations/apply_transformations_to_ti_body.hpp>
#include <transformations/tensor_iterator_transformations/unroll_tensor_iterator.hpp>

#include <ngraph/pass/constant_folding.hpp>
#include <ngraph/pass/manager.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

#include <memory>
#include <vector>

bool ngraph::pass::TensorIteratorTransformations::run_on_function(std::shared_ptr<ngraph::Function> f) {
    ngraph::pass::Manager manager;

    manager.register_pass<ngraph::pass::ConstantFolding>();
    auto fusion = manager.register_pass<ngraph::pass::ApplyTransformationsToTIBody>(m_external_manager);
    auto anchor = manager.register_pass<ngraph::pass::UnrollTensorIterator>();
    anchor->set_name("ngraph::pass::TensorIteratorTransformations");


    manager.set_callback(m_transformation_callback);
    manager.run_passes(f);
    return true;
}