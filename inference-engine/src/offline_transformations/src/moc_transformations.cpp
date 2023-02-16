// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "moc_transformations.hpp"
#include "pruning.hpp"

#include <ngraph/pass/manager.hpp>
#include <transformations/common_optimizations/split_squeeze_concat_fusion.hpp>
#include <transformations/common_optimizations/transpose_sinking.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::MOCTransformations, "MOCTransformations", 0);

bool ngraph::pass::MOCTransformations::run_on_function(std::shared_ptr<ngraph::Function> f) {
    ngraph::pass::Manager manager(get_pass_config());
    auto transpose_sinking = manager.register_pass<ngraph::pass::GraphRewrite>();
    transpose_sinking->add_matcher<ngraph::pass::TransposeSinking>();
    // SplitSqueezeConcatFusion should work in same GraphRewrite as TransposesSinking,
    // because it replaces pattern that may contain Transposes which must be optimized before
    // the transformation and it also inserts Transpose that can be optimized by TransposeSinking
    transpose_sinking->add_matcher<ngraph::pass::SplitSqueezeConcatFusion>();
    manager.run_passes(f);
    return false;
}