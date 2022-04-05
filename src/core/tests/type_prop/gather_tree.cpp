// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <array>
#include <utility>

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

namespace {
constexpr size_t step_ids_input_idx = 0;
constexpr size_t parent_idx_input_idx = 1;
constexpr size_t max_seq_len_input_idx = 2;
constexpr size_t end_token_input_idx = 3;
constexpr size_t gather_tree_required_inputs = 4;
struct GatherTreeInputInfo {
    element::Type in_et;
    PartialShape in_pshape;
};

using GatherTreeInputParams = std::array<GatherTreeInputInfo, gather_tree_required_inputs>;

std::shared_ptr<Node> makeGatherTreeOp(const GatherTreeInputParams& p) {
    if (p.size() != gather_tree_required_inputs) {
        throw runtime_error("GatherTree requires 4 inputs");
    }
    auto step_ids = make_shared<op::Parameter>(p.at(step_ids_input_idx).in_et, p.at(step_ids_input_idx).in_pshape);
    auto parent_idx =
        make_shared<op::Parameter>(p.at(parent_idx_input_idx).in_et, p.at(parent_idx_input_idx).in_pshape);
    auto max_seq_len =
        make_shared<op::Parameter>(p.at(max_seq_len_input_idx).in_et, p.at(max_seq_len_input_idx).in_pshape);
    auto end_token = make_shared<op::Parameter>(p.at(end_token_input_idx).in_et, p.at(end_token_input_idx).in_pshape);
    return make_shared<op::v1::GatherTree>(step_ids, parent_idx, max_seq_len, end_token);
}
}  // namespace

TEST(type_prop, gather_tree_invalid_input_element_type) {
    Shape scalar_shape{};
    Shape vector_shape{2};
    Shape tensor_shape{1, 2, 3};

    element::Type input_et = element::boolean;
    GatherTreeInputParams params{GatherTreeInputInfo{input_et, tensor_shape},
                                 GatherTreeInputInfo{input_et, tensor_shape},
                                 GatherTreeInputInfo{input_et, vector_shape},
                                 GatherTreeInputInfo{input_et, scalar_shape}};
    try {
        auto gather_tree = makeGatherTreeOp(params);
        FAIL() << "Invalid element types for inputs not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Element type of inputs must be numeric.");
    } catch (...) {
        FAIL() << "Element type check for inputs failed for unexpected reason";
    }
}

TEST(type_prop, gather_tree_incompatible_input_element_types) {
    element::Type float_et = element::f32;
    element::Type integer_et = element::i32;

    Shape scalar_shape{};
    Shape vector_shape{2};
    Shape tensor_shape{1, 2, 3};

    vector<GatherTreeInputParams> test_cases = {// step_ids input has incompatible element type
                                                GatherTreeInputParams{GatherTreeInputInfo{integer_et, tensor_shape},
                                                                      GatherTreeInputInfo{float_et, tensor_shape},
                                                                      GatherTreeInputInfo{float_et, vector_shape},
                                                                      GatherTreeInputInfo{float_et, scalar_shape}},
                                                // parent_idx input has incompatible element type
                                                GatherTreeInputParams{GatherTreeInputInfo{float_et, tensor_shape},
                                                                      GatherTreeInputInfo{integer_et, tensor_shape},
                                                                      GatherTreeInputInfo{float_et, vector_shape},
                                                                      GatherTreeInputInfo{float_et, scalar_shape}},
                                                // max_seq_len input has incompatible element type
                                                GatherTreeInputParams{GatherTreeInputInfo{float_et, tensor_shape},
                                                                      GatherTreeInputInfo{float_et, tensor_shape},
                                                                      GatherTreeInputInfo{integer_et, vector_shape},
                                                                      GatherTreeInputInfo{float_et, scalar_shape}},
                                                // end_token input has incompatible element type
                                                GatherTreeInputParams{GatherTreeInputInfo{float_et, tensor_shape},
                                                                      GatherTreeInputInfo{float_et, tensor_shape},
                                                                      GatherTreeInputInfo{float_et, vector_shape},
                                                                      GatherTreeInputInfo{integer_et, scalar_shape}}};

    for (const auto& test_case : test_cases) {
        try {
            auto gather_tree = makeGatherTreeOp(test_case);
            FAIL() << "Incompatible element types for inputs not detected";
        } catch (const NodeValidationFailure& error) {
            EXPECT_HAS_SUBSTRING(error.what(), "Inputs must have the same element type.");
        } catch (...) {
            FAIL() << "Element type check for inputs failed for unexpected reason";
        }
    }
}

TEST(type_prop, gather_tree_input_element_types) {
    Shape scalar_shape{};
    Shape vector_shape{2};
    Shape tensor_shape{1, 2, 3};

    std::vector<element::Type> element_types{element::u4,
                                             element::u8,
                                             element::u16,
                                             element::u32,
                                             element::i8,
                                             element::i16,
                                             element::i32,
                                             element::i64,
                                             element::f32,
                                             element::f64,
                                             element::u32};
    std::vector<GatherTreeInputParams> test_cases;
    std::for_each(std::begin(element_types), std::end(element_types), [&](element::Type et) {
        GatherTreeInputParams params{GatherTreeInputInfo{et, tensor_shape},
                                     GatherTreeInputInfo{et, tensor_shape},
                                     GatherTreeInputInfo{et, vector_shape},
                                     GatherTreeInputInfo{et, scalar_shape}};
        test_cases.insert(test_cases.end(), params);
    });
    for (const auto& test_case : test_cases) {
        try {
            EXPECT_NO_THROW(makeGatherTreeOp(test_case));
        } catch (...) {
            FAIL() << "Inputs element type validation check failed for unexpected reason";
        }
    }
}

TEST(type_prop, gather_tree_invalid_step_ids_and_parent_idx_input_shapes) {
    element::Type et = element::f32;

    Shape scalar_shape{};
    PartialShape vector_shape{Dimension()};

    std::vector<std::pair<PartialShape, PartialShape>> input_shapes = {
        {PartialShape{1}, PartialShape{1, 2, 3}},
        {PartialShape{1, 2, 3}, PartialShape{3, 3, 3, 3}},
        {PartialShape{Dimension(), Dimension(), 3}, PartialShape::dynamic(4)},
        {PartialShape::dynamic(2), PartialShape::dynamic()},
        {PartialShape{1, 2, 3}, PartialShape{Dimension(), Dimension(3, 5), 3}}};
    std::vector<GatherTreeInputParams> test_cases;
    std::for_each(std::begin(input_shapes), std::end(input_shapes), [&](std::pair<PartialShape, PartialShape> shapes) {
        GatherTreeInputParams params{GatherTreeInputInfo{et, shapes.first},
                                     GatherTreeInputInfo{et, shapes.second},
                                     GatherTreeInputInfo{et, vector_shape},
                                     GatherTreeInputInfo{et, scalar_shape}};
        test_cases.insert(test_cases.end(), params);
    });
    for (const auto& test_case : test_cases) {
        try {
            auto gather_tree = makeGatherTreeOp(test_case);
            FAIL() << "Incompatible shapes for inputs step_ids and parent_idx not detected";
        } catch (const NodeValidationFailure& error) {
            EXPECT_HAS_SUBSTRING(error.what(), "step_ids and parent_idx inputs must have the same shape with rank 3.");
        } catch (...) {
            FAIL() << "Shape check for step_ids and parent_idx inputs failed for unexpected reason";
        }
    }
}

TEST(type_prop, gather_tree_invalid_max_seq_len_rank) {
    element::Type et = element::f32;

    Shape tensor_shape{1, 2, 3};
    Shape scalar_shape{};

    std::vector<PartialShape> max_seq_len_shapes = {{}, {Dimension(), 1}, PartialShape::dynamic(3), {1, 2, 3, 4}};

    std::vector<GatherTreeInputParams> test_cases;
    std::for_each(std::begin(max_seq_len_shapes), std::end(max_seq_len_shapes), [&](PartialShape shape) {
        GatherTreeInputParams params{GatherTreeInputInfo{et, tensor_shape},
                                     GatherTreeInputInfo{et, tensor_shape},
                                     GatherTreeInputInfo{et, shape},
                                     GatherTreeInputInfo{et, scalar_shape}};
        test_cases.insert(test_cases.end(), params);
    });
    for (const auto& test_case : test_cases) {
        try {
            auto gather_tree = makeGatherTreeOp(test_case);
            FAIL() << "Invalid shapes for max_seq_len input not detected";
        } catch (const NodeValidationFailure& error) {
            EXPECT_HAS_SUBSTRING(error.what(), "max_seq_len input must have rank 1.");
        } catch (...) {
            FAIL() << "Shape check for max_seq_len input failed for unexpected reason";
        }
    }
}

TEST(type_prop, gather_tree_incompatible_step_ids_and_max_seq_len_shapes) {
    element::Type et = element::f32;

    Shape scalar_shape{};

    std::vector<std::pair<PartialShape, PartialShape>> input_shapes = {
        {PartialShape{1, 2, 3}, PartialShape{4}},
        {PartialShape{Dimension(), 2, 3}, PartialShape{Dimension(3, 6)}}};
    std::vector<GatherTreeInputParams> test_cases;
    std::for_each(std::begin(input_shapes), std::end(input_shapes), [&](std::pair<PartialShape, PartialShape> shapes) {
        GatherTreeInputParams params{GatherTreeInputInfo{et, shapes.first},
                                     GatherTreeInputInfo{et, shapes.first},
                                     GatherTreeInputInfo{et, shapes.second},
                                     GatherTreeInputInfo{et, scalar_shape}};
        test_cases.insert(test_cases.end(), params);
    });
    for (const auto& test_case : test_cases) {
        try {
            auto gather_tree = makeGatherTreeOp(test_case);
            FAIL() << "Incompatible shapes for inputs step_ids and max_seq_len not detected";
        } catch (const NodeValidationFailure& error) {
            EXPECT_HAS_SUBSTRING(error.what(),
                                 "Number of elements of max_seq_len input must match BATCH_SIZE dimension of "
                                 "step_ids/parent_idx inputs.");
        } catch (...) {
            FAIL() << "Shape check for step_ids and max_seq_len inputs failed for unexpected reason";
        }
    }
}

TEST(type_prop, gather_tree_output_shape) {
    element::Type et = element::f32;
    Shape scalar_shape{};

    std::vector<std::pair<PartialShape, PartialShape>> input_shapes = {
        {PartialShape{1, 2, 3}, PartialShape{2}},
        {PartialShape{1, 2, 3}, PartialShape::dynamic(1)},
        {PartialShape{Dimension(), 2, Dimension()}, PartialShape{2}},
        {
            PartialShape::dynamic(3),
            PartialShape{4},
        },
        {PartialShape{Dimension(), Dimension(3, 5), Dimension()}, PartialShape{Dimension(1, 3)}},
        {PartialShape::dynamic(), PartialShape::dynamic()}};
    std::vector<GatherTreeInputParams> test_cases;
    std::for_each(std::begin(input_shapes), std::end(input_shapes), [&](std::pair<PartialShape, PartialShape> shapes) {
        GatherTreeInputParams params{GatherTreeInputInfo{et, shapes.first},
                                     GatherTreeInputInfo{et, shapes.first},
                                     GatherTreeInputInfo{et, shapes.second},
                                     GatherTreeInputInfo{et, scalar_shape}};
        test_cases.insert(test_cases.end(), params);
    });
    for (const auto& test_case : test_cases) {
        try {
            auto gather_tree = makeGatherTreeOp(test_case);

            PartialShape result_shape{test_case.at(step_ids_input_idx).in_pshape};
            PartialShape max_seq_len_shape{test_case.at(max_seq_len_input_idx).in_pshape};
            if (result_shape.rank().is_static() && max_seq_len_shape.rank().is_static()) {
                result_shape[1] = result_shape[1] & max_seq_len_shape[0];
            }
            ASSERT_EQ(gather_tree->get_output_partial_shape(0), result_shape);
            ASSERT_EQ(gather_tree->get_output_element_type(0), et);
        } catch (...) {
            FAIL() << "Output shape check failed for unexpected reason";
        }
    }
}

TEST(type_prop, gather_tree_invalid_end_token_rank) {
    element::Type et = element::f32;

    Shape tensor_shape{1, 2, 3};
    Shape vector_shape{2};

    std::vector<PartialShape> end_token_shapes = {{3}, {Dimension(), 1}, PartialShape::dynamic(3), {1, 2, 3, 4}};

    std::vector<GatherTreeInputParams> test_cases;
    std::for_each(std::begin(end_token_shapes), std::end(end_token_shapes), [&](PartialShape shape) {
        GatherTreeInputParams params{GatherTreeInputInfo{et, tensor_shape},
                                     GatherTreeInputInfo{et, tensor_shape},
                                     GatherTreeInputInfo{et, vector_shape},
                                     GatherTreeInputInfo{et, shape}};
        test_cases.insert(test_cases.end(), params);
    });
    for (const auto& test_case : test_cases) {
        try {
            auto gather_tree = makeGatherTreeOp(test_case);
            FAIL() << "Invalid shapes for end_token input not detected";
        } catch (const NodeValidationFailure& error) {
            EXPECT_HAS_SUBSTRING(error.what(), "end_token input must be scalar.");
        } catch (...) {
            FAIL() << "Shape check for end_token input failed for unexpected reason";
        }
    }
}
