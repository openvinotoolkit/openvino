// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/transpose_sinking_general.hpp"
#include "transformations/common_optimizations/transpose_sinking.hpp"
#include "transformations/common_optimizations/transpose_sinking_unary.hpp"
#include "transformations/common_optimizations/transpose_sinking_binary.hpp"

#include <ngraph/pass/constant_folding.hpp>

#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>

#include "itt.hpp"
#include "transformations/utils/utils.hpp"

#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

ngraph::pass::TransposeSinkingGeneralForward::TransposeSinkingGeneralForward() {
    MATCHER_SCOPE(TransposeSinkingGeneralForward);
    add_matcher<ngraph::pass::TransposeSinkingUnaryForward>();
    add_matcher<ngraph::pass::TransposeSinkingBinaryForward>();
    add_matcher<ngraph::pass::TransposeSinkingConcatForward>();
    add_matcher<ngraph::pass::TransposeSinkingSplitForward>();
    add_matcher<ngraph::pass::TransposeFuse>();
    add_matcher<ngraph::pass::ConstantFolding>();
}

ngraph::pass::TransposeSinkingGeneralBackward::TransposeSinkingGeneralBackward() {

    MATCHER_SCOPE(TransposeSinkingGeneralBackward);
    add_matcher<ngraph::pass::TransposeSinkingUnaryBackward>();
    add_matcher<ngraph::pass::TransposeSinkingBinaryBackward>();
    add_matcher<ngraph::pass::TransposeSinkingConcatBackward>();
    // WANTFIX add_matcher<ngraph::pass::TransposeSinkingSplitBackward>();
    add_matcher<ngraph::pass::TransposeFuse>();
    add_matcher<ngraph::pass::ConstantFolding>();
}

bool ngraph::pass::TransposeSinkingGeneral::run_on_model(const std::shared_ptr<ov::Model>& f) {    
    {
        ngraph::pass::Manager manager(get_pass_config());
        manager.register_pass<ngraph::pass::TransposeSinkingGeneralForward>();
        manager.run_passes(f);
    }

    {
        ngraph::pass::Manager manager(get_pass_config());
        manager.register_pass<ngraph::pass::TransposeSinkingGeneralBackward>();
        manager.run_passes(f);
    }

    // Returning value is false because pass::Manager always apply Validation pass
    // if function was changed. This helps to avoid excess Validations after applying
    // this pass. In future when we will return more meaningful status code it will be
    // replaced with real status reported by manager.run_passes() method call.
    return false;
}
