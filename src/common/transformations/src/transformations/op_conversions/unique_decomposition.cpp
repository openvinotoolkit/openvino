// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/unique_decomposition.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/non_zero.hpp"
#include "openvino/op/not_equal.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/unique.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::pass;

namespace {
Output<Node> get_elements_number_1d(const ov::Output<ov::Node>& output,
                                    ov::element::Type output_type,
                                    ov::pass::NodeRegistry& rg) {
    auto output_rank = output.get_partial_shape().rank();
    auto shape = rg.make<ov::op::v3::ShapeOf>(output, output_type);
    auto num_elements = rg.make<ov::op::v0::Squeeze>(shape);
    return num_elements;
}
}  // namespace

ov::pass::UniqueDecomposition::UniqueDecomposition() {
    MATCHER_SCOPE(UniqueDecomposition);

    auto unique = pattern::wrap_type<ov::op::v10::Unique>();

    matcher_pass_callback callback = [=](pattern::Matcher& matcher) {
        NodeRegistry rg;

        auto unique_node = ov::as_type_ptr<ov::op::v10::Unique>(matcher.get_match_root());
        if (!unique_node) {
            return false;
        }

        // currently, the transformation supports only the first and the third outputs
        if (!unique_node->get_output_target_inputs(1).empty() || !unique_node->get_output_target_inputs(3).empty()) {
            return false;
        }

        // the second input to Unique is optional and this an axis parameter. When it has the single parameter we can
        // decompose. Otherwise, we need to check a rank of the input tensor. Currently, the transformation supports
        // only searching for unique elements among scalar elements, not vectors.
        auto input_rank = unique_node->get_input_partial_shape(0).rank();
        if (unique_node->get_input_size() > 1 && (input_rank.is_dynamic() || input_rank.get_length() > 1)) {
            return false;
        }

        auto x_unflatten = unique_node->input_value(0);

        // make input tensor flatten
        auto minus_one_const = rg.make<ov::op::v0::Constant>(element::i32, Shape{1}, -1);
        auto x = rg.make<ov::op::v1::Reshape>(x_unflatten, minus_one_const, false);

        auto output_indices_type = unique_node->get_index_element_type();
        auto x_type = x_unflatten.get_element_type();
        if (!x_type.is_real() && !x_type.is_integral_number()) {
            return false;
        }

        // in case the input with the single input, there is no transformation required
        // this separate pass needed since now there is no handler for empty tensors
        // that should be converted to empty Constant nodes
        // x is already of rank equal to 1 after Reshape
        auto x_shape = x->get_input_partial_shape(0);
        if (x_shape[0].is_static() && x_shape[0] == 1) {
            if (!unique_node->get_output_target_inputs(0).empty()) {
                x->set_friendly_name(unique_node->get_friendly_name() + ".0");
                unique_node->output(0).replace(x->output(0));
            }

            if (!unique_node->get_output_target_inputs(2).empty()) {
                auto zero_const = rg.make<ov::op::v0::Constant>(output_indices_type, Shape{1}, 0);
                zero_const->set_friendly_name(unique_node->get_friendly_name() + ".2");
                unique_node->output(2).replace(zero_const->output(0));
            }

            copy_runtime_info(unique_node, rg.get());
            return true;
        }

        // denote a number of elements in x as n
        auto n = get_elements_number_1d(x, element::i32, rg);

        // create auxiliry constants to be re-used by different operations
        auto zero_const = rg.make<ov::op::v0::Constant>(element::i32, Shape{1}, 0);
        auto one_const = rg.make<ov::op::v0::Constant>(element::i32, Shape{1}, 1);
        auto one_const_scalar = rg.make<ov::op::v0::Constant>(element::i32, Shape{}, 1);
        auto true_const = rg.make<ov::op::v0::Constant>(element::boolean, Shape{1}, true);
        auto one_const_out_idx = rg.make<ov::op::v0::Constant>(output_indices_type, Shape{1}, 1);
        auto zero_const_out_idx = rg.make<ov::op::v0::Constant>(output_indices_type, Shape{1}, 0);

        // compute unique elements but not in the original order
        // 1. sort elements in x in order to compute unique elements
        auto x_sorted = rg.make<ov::op::v3::TopK>(x,
                                                  n,
                                                  0,
                                                  ov::op::v3::TopK::Mode::MIN,
                                                  ov::op::v3::TopK::SortType::SORT_VALUES,
                                                  element::i32);
        // 2. generate two vectors from x_sorted vector by padding in the beginning and in the end:
        // x1 = [0, x0, x1, ..., xn]
        // x2 = [x0, x1, ..., xn, 0]
        auto pad = rg.make<ov::op::v0::Constant>(x_type, Shape{1}, 0);
        auto x1 = rg.make<ov::op::v0::Concat>(OutputVector{pad, x_sorted->output(0)}, 0);
        auto x2 = rg.make<ov::op::v0::Concat>(OutputVector{x_sorted->output(0), pad}, 0);
        // 3. compare two vectors to see where unique elements are placed
        // and correct a mask because the first element is always unique
        // because the latest boolean element must be removed from the mask since
        // the vectors are padded
        auto mask1 = rg.make<ov::op::v1::NotEqual>(x1, x2);
        auto mask1_part = rg.make<ov::op::v8::Slice>(mask1, one_const, minus_one_const, one_const, zero_const);
        auto is_unique = rg.make<ov::op::v0::Concat>(OutputVector{true_const, mask1_part}, 0);
        // 5. compute positions where unique elements are placed in the sorted x
        auto is_unique_01 = rg.make<ov::op::v1::Select>(is_unique, one_const, zero_const);
        auto indices = rg.make<ov::op::v3::NonZero>(is_unique_01, element::i64);
        auto unique_element_indices = rg.make<ov::op::v0::Squeeze>(indices, zero_const);
        // 6. collect unique elements but currently they are not in the original order
        auto unique_elements = rg.make<ov::op::v8::Gather>(x_sorted->output(0), unique_element_indices, zero_const);

        // compute unique elements in the original order
        auto unsqueeze_x = rg.make<ov::op::v0::Unsqueeze>(x, zero_const);
        auto unsqueeze_unique_elements = rg.make<ov::op::v0::Unsqueeze>(unique_elements, one_const);
        // 1. compute a mask of pair comparison where each unique element is placed in the original
        auto nplus1 = rg.make<ov::op::v1::Add>(n, one_const_scalar);
        auto unique_vs_x = rg.make<ov::op::v1::Equal>(unsqueeze_unique_elements, unsqueeze_x);
        auto unique_vs_x_01 = rg.make<ov::op::v1::Select>(unique_vs_x, one_const_scalar, nplus1);
        auto range_1nplus1 = rg.make<ov::op::v4::Range>(one_const_scalar, nplus1, one_const_scalar, element::i32);
        auto unsqueeze_range_1nplus1 = rg.make<ov::op::v0::Unsqueeze>(range_1nplus1, zero_const);
        // 2. compute a mask with indices counting from one
        auto unique_vs_x_ind = rg.make<ov::op::v1::Multiply>(unique_vs_x_01, unsqueeze_range_1nplus1);
        // 3. compute positions of the first occurrence for each unique element
        // or these are positions of unique elements in the original order
        auto minimum_indices_plus1 = rg.make<ov::op::v1::ReduceMin>(unique_vs_x_ind, one_const);
        auto minimum_indices = rg.make<ov::op::v1::Subtract>(minimum_indices_plus1, one_const);
        // denote a number of unique elements as m
        auto m = get_elements_number_1d(minimum_indices, element::i32, rg);
        auto sorted_minumum_indices = rg.make<ov::op::v3::TopK>(minimum_indices,
                                                                m,
                                                                0,
                                                                ov::op::v3::TopK::Mode::MIN,
                                                                ov::op::v3::TopK::SortType::SORT_VALUES,
                                                                element::i32);
        auto output_unique_elements = rg.make<ov::op::v8::Gather>(x, sorted_minumum_indices->output(0), zero_const);

        if (!unique_node->get_output_target_inputs(0).empty()) {
            output_unique_elements->set_friendly_name(unique_node->get_friendly_name() + ".0");
            unique_node->output(0).replace(output_unique_elements->output(0));
        }

        if (!unique_node->get_output_target_inputs(2).empty()) {
            // compute the second output
            // indices of elements of x in the vector of unique elements
            // 1. compute a mask for unique elements in the original order
            auto unsqueeze_output_unique_elements = rg.make<ov::op::v0::Unsqueeze>(output_unique_elements, one_const);
            auto unique_vs_x_orig = rg.make<ov::op::v1::Equal>(unsqueeze_output_unique_elements, unsqueeze_x);
            auto mplus1 = rg.make<ov::op::v1::Add>(m, one_const_scalar);
            auto unique_vs_x_orig_01 =
                rg.make<ov::op::v1::Select>(unique_vs_x_orig, one_const_out_idx, zero_const_out_idx);
            // 2. compute positions where each element from x is located in unique elements vector
            // the position counts from 1
            auto range_1mplus1 =
                rg.make<ov::op::v4::Range>(one_const_scalar, mplus1, one_const_scalar, output_indices_type);
            auto unsqueeze_range_1mplus1 = rg.make<ov::op::v0::Unsqueeze>(range_1mplus1, one_const);
            auto unique_vs_x_ind_orig = rg.make<ov::op::v1::Multiply>(unique_vs_x_orig_01, unsqueeze_range_1mplus1);
            auto output_idx_plus1 = rg.make<ov::op::v1::ReduceMax>(unique_vs_x_ind_orig, zero_const);
            auto output_idx = rg.make<ov::op::v1::Subtract>(output_idx_plus1, one_const_out_idx);

            output_idx->set_friendly_name(unique_node->get_friendly_name() + ".2");
            unique_node->output(2).replace(output_idx->output(0));
        }

        copy_runtime_info(unique_node, rg.get());

        return true;
    };

    auto m = make_shared<pattern::Matcher>(unique, matcher_name);
    this->register_matcher(m, callback);
}
