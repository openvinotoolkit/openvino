// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/transpose_sinking_general.hpp"
#include "transformations/common_optimizations/transpose_sinking.hpp"
#include "transformations/common_optimizations/transpose_sinking_unary.hpp"
#include "transformations/common_optimizations/transpose_sinking_binary.hpp"
#include "transformations/common_optimizations/transpose_sinking_concat.hpp"
#include "transformations/common_optimizations/transpose_sinking_split.hpp"

#include <ngraph/pass/constant_folding.hpp>

#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>

#include "itt.hpp"
#include "transformations/utils/utils.hpp"

#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

ov::pass::TransposeSinkingGeneralForward::TransposeSinkingGeneralForward() {
    MATCHER_SCOPE(TransposeSinkingGeneralForward);
    add_matcher<ov::pass::TransposeSinkingUnaryForward>();
    add_matcher<ov::pass::TransposeSinkingBinaryElementwiseForward>();
    add_matcher<ov::pass::TransposeSinkingConcatForward>();
    add_matcher<ov::pass::TransposeSinkingSplitForward>();
    add_matcher<ngraph::pass::TransposeFuse>();
}

ov::pass::TransposeSinkingGeneralBackward::TransposeSinkingGeneralBackward() {

    MATCHER_SCOPE(TransposeSinkingGeneralBackward);
    add_matcher<ov::pass::TransposeSinkingUnaryBackward>();
    add_matcher<ov::pass::TransposeSinkingBinaryElementwiseBackward>();
    add_matcher<ov::pass::TransposeSinkingConcatBackward>();
    add_matcher<ov::pass::TransposeSinkingSplitBackward>();
    add_matcher<ngraph::pass::TransposeFuse>();
}

bool ov::pass::TransposeSinkingGeneral::run_on_model(const std::shared_ptr<ov::Model>& f) {    
    {
        ngraph::pass::Manager manager(get_pass_config());
        manager.register_pass<ov::pass::TransposeSinkingGeneralForward>();
        manager.register_pass<ngraph::pass::ConstantFolding>();
        manager.run_passes(f);
    }

    {
        ngraph::pass::Manager manager(get_pass_config());
        manager.register_pass<ov::pass::TransposeSinkingGeneralBackward>();
        manager.register_pass<ngraph::pass::ConstantFolding>();
        manager.run_passes(f);
    }

    // Returning value is false because pass::Manager always apply Validation pass
    // if function was changed. This helps to avoid excess Validations after applying
    // this pass. In future when we will return more meaningful status code it will be
    // replaced with real status reported by manager.run_passes() method call.
    return false;
}
