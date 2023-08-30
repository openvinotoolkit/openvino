// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pot_transformations.hpp"

#include <memory>

#include "openvino/pass/manager.hpp"
#include "transformations/op_conversions/bidirectional_sequences_decomposition.hpp"
#include "transformations/op_conversions/convert_sequences_to_tensor_iterator.hpp"
#include "transformations/op_conversions/gru_cell_decomposition.hpp"
#include "transformations/op_conversions/lstm_cell_decomposition.hpp"

bool ov::pass::POTTransformations::run_on_model(const std::shared_ptr<ov::Model>& f) {
    ov::pass::Manager manager(get_pass_config());
    manager.register_pass<ov::pass::BidirectionalSequenceDecomposition>();
    manager.register_pass<ov::pass::ConvertSequenceToTensorIterator>();
    manager.register_pass<ov::pass::GRUCellDecomposition>();
    manager.register_pass<ov::pass::LSTMCellDecomposition>();
    manager.run_passes(f);
    return false;
}
