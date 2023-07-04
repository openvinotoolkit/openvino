// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/big_transpose.hpp"

#include "common/graph_utils.hpp"
#include "openvino/cc/ngraph/itt.hpp"
#include "openvino/opsets/opset11.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/transformation_helper.hpp"

using namespace ov::opset11;
using namespace ov::pass;
using namespace ov::intel_gna::pass;

namespace {

inline std::vector<size_t> find_primes(size_t n) {
    std::vector<size_t> factors = {};
    size_t n_tmp = n;
    size_t p = 2;
    while (n_tmp * n_tmp >= p * p) {
        if ((n_tmp % p) == 0) {
            factors.push_back(p);
            n_tmp = n_tmp / p;
        } else {
            p++;
        }
    }
    if (n_tmp > 1) {
        factors.push_back(n_tmp);
    }
    return factors;
}

inline bool is_factored_transpose_feasible(const std::vector<size_t>& factors) {
    // check if there are any factors too large for GNA transpose
    for (size_t i = 0; i < factors.size(); i++) {
        if (factors[i] > 8) {
            return false;
        }
    }
    return true;
}

inline std::vector<size_t> combine_factors(std::vector<size_t> factors) {
    // combine prime factors if possible
    std::vector<size_t> combined_factors = {};
    size_t new_factor = 1;
    for (size_t i = 0; i < factors.size(); i++) {
        size_t product = new_factor * factors[i];
        if (product > 8) {
            combined_factors.push_back(new_factor);
            new_factor = factors[i];
        } else {
            new_factor = product;
        }
    }
    combined_factors.push_back(new_factor);

    return (combined_factors);
}

}  // namespace

ReplaceBigTranspose::ReplaceBigTranspose() {
    MATCHER_SCOPE(ReplaceBigTranspose);

    auto transpose_const = pattern::wrap_type<Constant>();
    auto transpose =
        pattern::wrap_type<Transpose>({pattern::any_input(), transpose_const}, [](const ov::Output<ov::Node>& node) {
            return !limitations::Limitations::is_transpose_supported(node.get_node_shared_ptr()) &&
                   graph_utils::is_shape_2d(node.get_shape());
        });

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto transpose_order = as_type_ptr<Constant>(pattern_map.at(transpose_const).get_node_shared_ptr());
        const auto transpose_node = pattern_map.at(transpose).get_node_shared_ptr();
        const ov::Shape& shape_in = transpose_node->get_input_shape(0);
        ov::AxisVector axis = transpose_order->get_axis_vector_val();

        ov::Shape shape_in_sq = graph_utils::squeeze_shape(shape_in);
        size_t h = shape_in_sq[0];
        size_t w = shape_in_sq[1];
        bool is_dim_factor_w = false;

        size_t dim_factor = 0;
        if (h % 8 == 0) {
            dim_factor = h;
        } else if (w % 8 == 0) {
            dim_factor = w;
            is_dim_factor_w = true;
        } else {
            return false;
        }

        std::vector<size_t> factors = find_primes(dim_factor);
        if (!is_factored_transpose_feasible(factors)) {
            return false;
        }

        std::vector<size_t> factors_combined = combine_factors(factors);

        // generate transpose transformation
        ov::NodeVector new_nodes = ov::NodeVector{transpose_node->get_input_node_shared_ptr(0)};
        for (const size_t& factor : factors_combined) {
            // New Reshape
            ov::Shape shape_new = {h * w / factor, factor};
            if (is_dim_factor_w) {
                std::swap(shape_new[0], shape_new[1]);
            }

            if (!limitations::Limitations::is_transpose_supported(shape_new))
                return false;

            auto reshape_new_const =
                std::make_shared<Constant>(ov::element::i32, ov::Shape{shape_new.size()}, shape_new);
            auto reshape_new = std::make_shared<Reshape>(new_nodes.back(), reshape_new_const, false);

            // New Transpose
            ov::AxisVector transpose_order = {1, 0};
            auto transpose_const =
                std::make_shared<Constant>(ov::element::i8, ov::Shape{transpose_order.size()}, transpose_order);
            auto transpose_new = std::make_shared<Transpose>(reshape_new, transpose_const);

            new_nodes.push_back(transpose_new);
        }

        const ov::Shape& shape_out = transpose_node->get_output_shape(0);
        auto reshape_out_const = std::make_shared<Constant>(ov::element::i32, ov::Shape{shape_out.size()}, shape_out);
        auto reshape_out = std::make_shared<Reshape>(new_nodes.back(), reshape_out_const, false);

        ov::replace_output_update_name(transpose_node->output(0), reshape_out->output(0));
        // ov::replace_node(transpose, new_nodes.back());
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(transpose, matcher_name);
    this->register_matcher(m, callback);
}
