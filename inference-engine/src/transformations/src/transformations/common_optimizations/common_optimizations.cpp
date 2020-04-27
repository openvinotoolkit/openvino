// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/common_optimizations.hpp"

#include <memory>

#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/nop_elimination.hpp>

bool ngraph::pass::CommonOptimizations::run_on_function(std::shared_ptr<ngraph::Function> f) {
    ngraph::pass::Manager CommonOptimizations;

#define NGRAPH_PASS(NAME, NAMESPACE) CommonOptimizations.register_pass<NAMESPACE::NAME>();
#include <transformations/common_optimizations/common_optimizations_tbl.hpp>
#undef NGRAPH_PASS

    CommonOptimizations.run_passes(f);
    return true;
}
