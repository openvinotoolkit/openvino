// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include <ngraph/pass/manager.hpp>

#include <transformations/op_conversions/lstm_cell_decomposition.hpp>
#include <transformations/common_optimizations/split_squeeze_concat_fusion.hpp>
#include <transformations/common_optimizations/transpose_to_reshape.hpp>
#include <transformations/common_optimizations/transpose_sinking.hpp>

#include "pot_transformations.hpp"

NGRAPH_RTTI_DEFINITION(ngraph::pass::POTTransformations, "POTTransformations", 0);

bool ngraph::pass::POTTransformations::run_on_function(std::shared_ptr<ngraph::Function> f) {
    ngraph::pass::Manager manager(get_pass_config());
    if (m_device == "GNA") {
        manager.register_pass<ngraph::pass::LSTMCellDecomposition>();
    }
    manager.register_pass<ngraph::pass::TransposeToReshape>();
    manager.register_pass<ngraph::pass::TransposeReduction>();
    manager.register_pass<ngraph::pass::TransposeSinking>();
    manager.register_pass<ngraph::pass::TransposeFQReduction>();
    manager.register_pass<ngraph::pass::TransposeFuse>();
    manager.register_pass<ngraph::pass::SplitSqueezeConcatFusion>();
    manager.run_passes(f);
    return false;
}
