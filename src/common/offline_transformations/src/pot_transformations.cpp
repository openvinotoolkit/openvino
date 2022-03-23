// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include <ngraph/pass/manager.hpp>

#include <transformations/op_conversions/lstm_cell_decomposition.hpp>

#include "pot_transformations.hpp"

bool ngraph::pass::POTTransformations::run_on_model(const std::shared_ptr<ngraph::Function>& f) {
    ngraph::pass::Manager manager(get_pass_config());
    if (m_device == "GNA") {
        manager.register_pass<ngraph::pass::LSTMCellDecomposition>();
    }
    manager.run_passes(f);
    return false;
}
