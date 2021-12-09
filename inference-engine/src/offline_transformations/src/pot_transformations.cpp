// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include <ngraph/pass/manager.hpp>

#include <transformations/op_conversions/lstm_cell_decomposition.hpp>

#include "pot_transformations.hpp"

NGRAPH_RTTI_DEFINITION(ngraph::pass::POTTransformations, "POTTransformations", 0);

bool ngraph::pass::POTTransformations::run_on_function(std::shared_ptr<ngraph::Function> f) {
    ngraph::pass::Manager manager(get_pass_config());
    if (m_device == "GNA") {
        manager.register_pass<ngraph::pass::LSTMCellDecomposition>();
    }
    manager.run_passes(f);
    return false;
}