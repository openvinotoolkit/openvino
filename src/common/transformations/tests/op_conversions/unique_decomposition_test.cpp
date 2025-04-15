// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/unique_decomposition.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "gtest/gtest.h"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/pass/manager.hpp"

using namespace std;
using namespace ov;
using namespace opset10;
using namespace element;

namespace {
Output<Node> get_elements_number_1d(const Output<Node>& output, element::Type output_type) {
    auto shape = make_shared<ShapeOf>(output, output_type);
    auto num_elements = make_shared<Squeeze>(shape);
    return num_elements;
}

shared_ptr<Model> gen_model(PartialShape input_shape, element::Type out_idx) {
    auto x = make_shared<Parameter>(f32, input_shape);
    auto unique = make_shared<Unique>(x, false, out_idx);

    return make_shared<Model>(OutputVector{unique->output(0), unique->output(2)}, ParameterVector{x});
}

shared_ptr<Model> gen_model_ref(PartialShape input_shape, element::Type out_idx) {
    auto x_unflatten = make_shared<Parameter>(f32, input_shape);

    auto minus_one_const = make_shared<Constant>(element::i32, Shape{1}, -1);
    // flatten input
    auto x = make_shared<Reshape>(x_unflatten, minus_one_const, false);

    // denote a number of elements in x as n
    auto n = get_elements_number_1d(x, element::i32);

    // create auxiliry constants to be re-used by different operations
    auto zero_const = make_shared<Constant>(element::i32, Shape{1}, 0);
    auto one_const = make_shared<Constant>(element::i32, Shape{1}, 1);
    auto one_const_scalar = make_shared<Constant>(element::i32, Shape{}, 1);
    auto true_const = make_shared<Constant>(element::boolean, Shape{1}, true);
    auto one_const_out_idx = make_shared<Constant>(out_idx, Shape{1}, 1);
    auto zero_const_out_idx = make_shared<Constant>(out_idx, Shape{1}, 0);

    // compute unique elements but not in the original order
    // 1. sort elements in x in order to compute unique elements
    auto x_sorted = make_shared<TopK>(x, n, 0, TopK::Mode::MIN, TopK::SortType::SORT_VALUES, element::i32);
    // 2. generate two vectors from x_sorted vector by padding in the beginning and in the end:
    // x1 = [0, x0, x1, ..., xn]
    // x2 = [x0, x1, ..., xn, 0]
    auto pad = make_shared<Constant>(x->get_element_type(), Shape{1}, 0);
    auto x1 = make_shared<Concat>(OutputVector{pad, x_sorted->output(0)}, 0);
    auto x2 = make_shared<Concat>(OutputVector{x_sorted->output(0), pad}, 0);
    // 3. compare two vectors to see where unique elements are placed
    // and correct a mask because the first element is always unique
    // because the latest boolean element must be removed from the mask since
    // the vectors are padded
    auto mask1 = make_shared<NotEqual>(x1, x2);
    auto mask1_part = make_shared<Slice>(mask1, one_const, minus_one_const, one_const, zero_const);
    auto is_unique = make_shared<Concat>(OutputVector{true_const, mask1_part}, 0);
    // 5. compute positions where unique elements are placed in the sorted x
    auto is_unique_01 = make_shared<Select>(is_unique, one_const, zero_const);
    auto indices = make_shared<NonZero>(is_unique_01, element::i64);
    auto unique_element_indices = make_shared<Squeeze>(indices, zero_const);
    // 6. collect unique elements but currently they are not in the original order
    auto unique_elements = make_shared<Gather>(x_sorted->output(0), unique_element_indices, zero_const);

    // compute unique elements in the original order
    auto unsqueeze_x = make_shared<Unsqueeze>(x, zero_const);
    auto unsqueeze_unique_elements = make_shared<Unsqueeze>(unique_elements, one_const);
    // 1. compute a mask of pair comparison where each unique element is placed in the original
    auto nplus1 = make_shared<Add>(n, one_const_scalar);
    auto unique_vs_x = make_shared<Equal>(unsqueeze_unique_elements, unsqueeze_x);
    auto unique_vs_x_01 = make_shared<Select>(unique_vs_x, one_const_scalar, nplus1);
    auto range_1nplus1 = make_shared<Range>(one_const_scalar, nplus1, one_const_scalar, element::i32);
    auto unsqueeze_range_1nplus1 = make_shared<Unsqueeze>(range_1nplus1, zero_const);
    // 2. compute a mask with indices counting from one
    auto unique_vs_x_ind = make_shared<Multiply>(unique_vs_x_01, unsqueeze_range_1nplus1);
    // 3. compute positions of the first occurrence for each unique element
    // or these are positions of unique elements in the original order
    auto minimum_indices_plus1 = make_shared<ReduceMin>(unique_vs_x_ind, one_const);
    auto minimum_indices = make_shared<Subtract>(minimum_indices_plus1, one_const);
    // denote a number of unique elements as m
    auto m = get_elements_number_1d(minimum_indices, element::i32);
    auto sorted_minumum_indices =
        make_shared<TopK>(minimum_indices, m, 0, TopK::Mode::MIN, TopK::SortType::SORT_VALUES, element::i32);
    auto output_unique_elements = make_shared<Gather>(x, sorted_minumum_indices->output(0), zero_const);

    // compute the second output
    // indices of elements of x in the vector of unique elements
    // 1. compute a mask for unique elements in the original order
    auto unsqueeze_output_unique_elements = make_shared<Unsqueeze>(unique_elements, one_const);
    auto unique_vs_x_orig = make_shared<Equal>(unsqueeze_output_unique_elements, unsqueeze_x);
    auto unique_vs_x_orig_01 = make_shared<Select>(unique_vs_x_orig, one_const_out_idx, zero_const_out_idx);
    // 2. compute positions where each element from x is located in unique elements vector
    // the position counts from 1
    auto mplus1 = make_shared<Add>(m, one_const_scalar);
    auto range_1mplus1 = make_shared<Range>(one_const_scalar, mplus1, one_const_scalar, out_idx);
    auto unsqueeze_range_1mplus1 = make_shared<Unsqueeze>(range_1mplus1, one_const);
    auto unique_vs_x_ind_orig = make_shared<Multiply>(unique_vs_x_orig_01, unsqueeze_range_1mplus1);
    auto output_idx_plus1 = make_shared<ReduceMax>(unique_vs_x_ind_orig, zero_const);
    auto output_idx = make_shared<Subtract>(output_idx_plus1, one_const_out_idx);

    return make_shared<Model>(OutputVector{output_unique_elements->output(0), output_idx->output(0)},
                              ParameterVector{x_unflatten});
}

}  // namespace

TEST_F(TransformationTestsF, UniqueDecompositionInt32) {
    {
        model = gen_model(PartialShape{10}, element::i32);
        manager.register_pass<ov::pass::UniqueDecomposition>();
    }
    { model_ref = gen_model_ref(PartialShape{10}, element::i32); }
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, UniqueDecompositionInt64) {
    {
        model = gen_model(PartialShape{42}, element::i64);
        manager.register_pass<ov::pass::UniqueDecomposition>();
    }
    { model_ref = gen_model_ref(PartialShape{42}, element::i64); }
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}
