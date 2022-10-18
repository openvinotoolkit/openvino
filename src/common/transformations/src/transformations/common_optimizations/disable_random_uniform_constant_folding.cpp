// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/disable_random_uniform_constant_folding.hpp"

#include <memory>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <transformations/rt_info/disable_constant_folding.hpp>

ngraph::pass::DisableRandomUniformConstantFolding::DisableRandomUniformConstantFolding() {
    auto random_uniform = pattern::wrap_type<opset8::RandomUniform>();

    ngraph::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        disable_constant_folding(m.get_match_root());
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(random_uniform, "DisableRandomUniformConstantFolding");
    this->register_matcher(m, callback);
}
