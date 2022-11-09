// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pot_transformations.hpp"

#include <memory>
#include <ngraph/pass/manager.hpp>
#include <transformations/op_conversions/bidirectional_sequences_decomposition.hpp>
#include <transformations/op_conversions/convert_sequences_to_tensor_iterator.hpp>
#include <transformations/op_conversions/gru_cell_decomposition.hpp>
#include <transformations/op_conversions/lstm_cell_decomposition.hpp>

bool ngraph::pass::POTTransformations::run_on_model(const std::shared_ptr<ngraph::Function>& f) {
    ngraph::pass::Manager manager(get_pass_config());
    if (m_device == "GNA") {
        manager.register_pass<ngraph::pass::BidirectionalSequenceDecomposition>();
        manager.register_pass<ngraph::pass::ConvertSequenceToTensorIterator>();
        manager.register_pass<ngraph::pass::GRUCellDecomposition>();
        manager.register_pass<ngraph::pass::LSTMCellDecomposition>();
    }
    manager.run_passes(f);
    return false;
}
