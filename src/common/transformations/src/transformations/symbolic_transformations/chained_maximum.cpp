// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/symbolic_transformations/chained_maximum.hpp"

#include "itt.hpp"
#include "openvino/core/dimension.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/symbolic_transformations/utils.hpp"

using namespace ov::symbol::util;

ov::pass::ChainedMaximumOptimization::ChainedMaximumOptimization() {
    MATCHER_SCOPE(ChainedMaximumOptimization);
    auto A_input = pattern::any_input();
    auto B_input = pattern::any_input();
    auto C_input = pattern::any_input();
    auto first_maximum = pattern::wrap_type<op::v1::Maximum>({A_input, B_input});
    auto maximum = pattern::wrap_type<op::v1::Maximum>({first_maximum, C_input});

    ov::matcher_pass_callback matcher_pass_callback = [=](pattern::Matcher& m) {
        const auto& vm = m.get_pattern_value_map();

        auto A = vm.at(A_input), B = vm.at(B_input), C = vm.at(C_input);
        auto output_to_replace = vm.at(first_maximum);

        ov::TensorSymbol A_symbols, B_symbols, C_symbols;
        bool A_read = get_symbols(A, A_symbols);
        bool B_read = get_symbols(B, B_symbols);
        bool C_read = get_symbols(C, C_symbols);

        if (!A_read && !B_read && !C_read)
            return false;

        if (are_unique_and_equal_symbols(A_symbols, C_symbols)) {
            // Matched Maximum(Maximum(A, B), C) with A == C -> Maximum(B, C)
            return ov::replace_output_update_name(output_to_replace, B);
        } else if (are_unique_and_equal_symbols(B_symbols, C_symbols)) {
            // Matched Maximum(Maximum(A, B), C) with B == C -> Maximum(A, C)
            return ov::replace_output_update_name(output_to_replace, A);
        }
        return false;
    };

    auto m = std::make_shared<pattern::Matcher>(maximum, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
