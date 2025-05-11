// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/batch_to_space.hpp"

#include <gtest/gtest.h>

#include <array>

#include "common_test_utils/type_prop.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/space_to_batch.hpp"
#include "openvino/util/common_util.hpp"

using namespace std;
using namespace testing;

namespace {
constexpr size_t data_input_idx = 0;
constexpr size_t block_shape_input_idx = 1;
constexpr size_t crops_begin_input_idx = 2;
constexpr size_t crops_end_input_idx = 3;
constexpr size_t batch_to_space_required_inputs = 4;
struct InputInfo {
    ov::element::Type in_et;
    ov::PartialShape in_pshape;
};

using BatchToSpaceInputParams = std::array<InputInfo, batch_to_space_required_inputs>;

std::shared_ptr<ov::Node> makeBatchToSpaceOp(const BatchToSpaceInputParams& p) {
    if (p.size() != batch_to_space_required_inputs) {
        throw runtime_error("BatchToSpace requires 4 inputs");
    }
    auto data = make_shared<ov::op::v0::Parameter>(p.at(data_input_idx).in_et, p.at(data_input_idx).in_pshape);
    auto block_shape =
        make_shared<ov::op::v0::Parameter>(p.at(block_shape_input_idx).in_et, p.at(block_shape_input_idx).in_pshape);
    auto crops_begin =
        make_shared<ov::op::v0::Parameter>(p.at(crops_begin_input_idx).in_et, p.at(crops_begin_input_idx).in_pshape);
    auto crops_end =
        make_shared<ov::op::v0::Parameter>(p.at(crops_end_input_idx).in_et, p.at(crops_end_input_idx).in_pshape);
    return make_shared<ov::op::v1::BatchToSpace>(data, block_shape, crops_begin, crops_end);
}
}  // namespace

TEST(type_prop, batch_to_space_incompatible_input_element_types) {
    ov::element::Type float_et = ov::element::f32;
    ov::element::Type integer64_et = ov::element::i64;
    ov::element::Type integer32_et = ov::element::i32;

    ov::Shape data_sshape{10, 26};
    ov::Shape inputs_sshape{2};

    vector<BatchToSpaceInputParams> test_cases;
    test_cases.push_back(BatchToSpaceInputParams{InputInfo{float_et, data_sshape},
                                                 InputInfo{integer64_et, inputs_sshape},
                                                 InputInfo{integer32_et, inputs_sshape},
                                                 InputInfo{integer32_et, inputs_sshape}});

    test_cases.push_back(BatchToSpaceInputParams{InputInfo{float_et, data_sshape},
                                                 InputInfo{integer32_et, inputs_sshape},
                                                 InputInfo{integer64_et, inputs_sshape},
                                                 InputInfo{integer32_et, inputs_sshape}});

    test_cases.push_back(BatchToSpaceInputParams{InputInfo{float_et, data_sshape},
                                                 InputInfo{integer64_et, inputs_sshape},
                                                 InputInfo{float_et, inputs_sshape},
                                                 InputInfo{float_et, inputs_sshape}});

    for (const auto& test_case : test_cases) {
        try {
            auto batch_to_space = makeBatchToSpaceOp(test_case);
            FAIL() << "Incompatible element types for block_shape/crops_begin/crops_end inputs not detected";
        } catch (const ov::NodeValidationFailure& error) {
            EXPECT_HAS_SUBSTRING(error.what(),
                                 "block_shape, crops_begin and crops_end inputs must have same element type.");
        } catch (...) {
            FAIL() << "Element type check for block_shape/crops_begin/crops_end inputs failed for unexpected reason";
        }
    }
}

TEST(type_prop, batch_to_space_invalid_input_element_types) {
    ov::element::Type float_et = ov::element::f32;

    ov::Shape data_sshape{10, 26};
    ov::Shape inputs_sshape{2};

    const BatchToSpaceInputParams params{InputInfo{float_et, data_sshape},
                                         InputInfo{float_et, inputs_sshape},
                                         InputInfo{float_et, inputs_sshape},
                                         InputInfo{float_et, inputs_sshape}};

    try {
        auto batch_to_space = makeBatchToSpaceOp(params);
        FAIL() << "Invalid non-integer element type for block_shape/crops_begin/crops_end inputs not detected";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "block_shape and crops inputs must have integer element type.");
    } catch (...) {
        FAIL() << "Element type check for block_shape/crops_begin/crops_end inputs failed for unexpected reason";
    }
}

TEST(type_prop, batch_to_space_invalid_data_input_rank) {
    ov::Shape data_sshape{4};
    ov::element::Type data_et = ov::element::f32;

    ov::Shape inputs_sshape{2};
    ov::element::Type inputs_et = ov::element::i64;

    const BatchToSpaceInputParams params{InputInfo{data_et, data_sshape},
                                         InputInfo{inputs_et, inputs_sshape},
                                         InputInfo{inputs_et, inputs_sshape},
                                         InputInfo{inputs_et, inputs_sshape}};

    try {
        auto batch_to_space = makeBatchToSpaceOp(params);
        FAIL() << "Invalid rank of data input not detected";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "data input must have rank greater or equal than 2.");
    } catch (...) {
        FAIL() << "Rank check for data input failed for unexpected reason";
    }
}

TEST(type_prop, batch_to_space_incompatible_secondary_inputs_shapes) {
    ov::Shape data_sshape{10, 26};
    ov::element::Type data_et = ov::element::f32;

    ov::Shape inputs_sshape_1D{2};
    ov::Shape inputs_sshape_2D{2, 1};
    ov::element::Type inputs_et = ov::element::i64;

    vector<BatchToSpaceInputParams> test_cases;
    test_cases.push_back(BatchToSpaceInputParams{InputInfo{data_et, data_sshape},
                                                 InputInfo{inputs_et, inputs_sshape_2D},
                                                 InputInfo{inputs_et, inputs_sshape_1D},
                                                 InputInfo{inputs_et, inputs_sshape_1D}});

    test_cases.push_back(BatchToSpaceInputParams{InputInfo{data_et, data_sshape},
                                                 InputInfo{inputs_et, inputs_sshape_1D},
                                                 InputInfo{inputs_et, inputs_sshape_2D},
                                                 InputInfo{inputs_et, inputs_sshape_1D}});

    test_cases.push_back(BatchToSpaceInputParams{InputInfo{data_et, data_sshape},
                                                 InputInfo{inputs_et, inputs_sshape_1D},
                                                 InputInfo{inputs_et, inputs_sshape_2D},
                                                 InputInfo{inputs_et, inputs_sshape_2D}});

    for (const auto& test_case : test_cases) {
        try {
            auto batch_to_space = makeBatchToSpaceOp(test_case);
            FAIL() << "Incompatible shapes for block_shape/crops_begin/crops_end inputs not detected";
        } catch (const ov::NodeValidationFailure& error) {
            EXPECT_HAS_SUBSTRING(error.what(),
                                 "block_shape, crops_begin and crops_end inputs must have the same shape.");
        } catch (...) {
            FAIL() << "Shapes check for block_shape/crops_begin/crops_end inputs failed for unexpected reason";
        }
    }
}

TEST(type_prop, batch_to_space_invalid_secondary_inputs_rank) {
    ov::Shape data_sshape{10, 26};
    ov::element::Type data_et = ov::element::f32;

    ov::Shape inputs_sshape_2D{2, 1};
    ov::element::Type inputs_et = ov::element::i64;

    const BatchToSpaceInputParams params{InputInfo{data_et, data_sshape},
                                         InputInfo{inputs_et, inputs_sshape_2D},
                                         InputInfo{inputs_et, inputs_sshape_2D},
                                         InputInfo{inputs_et, inputs_sshape_2D}};

    try {
        auto batch_to_space = makeBatchToSpaceOp(params);
        FAIL() << "Invalid rank for block_shape/crops_begin/crops_end inputs not detected";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "block_shape and crops inputs must have rank 1.");
    } catch (...) {
        FAIL() << "Rank check for block_shape/crops_begin/crops_end inputs failed for unexpected reason";
    }
}

TEST(type_prop, batch_to_space_incompatible_data_and_secondary_inputs_shapes) {
    ov::Shape data_sshape{10, 26};
    ov::element::Type data_et = ov::element::f32;

    ov::Shape inputs_sshape{5};
    ov::element::Type inputs_et = ov::element::i64;

    const BatchToSpaceInputParams params{InputInfo{data_et, data_sshape},
                                         InputInfo{inputs_et, inputs_sshape},
                                         InputInfo{inputs_et, inputs_sshape},
                                         InputInfo{inputs_et, inputs_sshape}};

    try {
        auto batch_to_space = makeBatchToSpaceOp(params);
        FAIL() << "Incompatible shapes for data and block_shape/crops_begin/crops_end inputs not detected";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "block_shape and crop inputs must have same number of elements "
                             "as data input rank.");
    } catch (...) {
        FAIL() << "Compatibility shape check for data and block_shape/crops_begin/crops_end inputs failed for "
                  "unexpected reason";
    }
}

TEST(type_prop, batch_to_space_invalid_block_shape_input) {
    ov::Shape data_sshape{100, 7, 13, 3};
    ov::element::Type data_et = ov::element::f32;

    ov::Shape inputs_sshape{4};
    ov::element::Type inputs_et = ov::element::i64;

    auto data = make_shared<ov::op::v0::Parameter>(data_et, data_sshape);
    auto block_shape = make_shared<ov::op::v0::Constant>(inputs_et, inputs_sshape, vector<int64_t>{0, 10, 5, 1});
    auto crops_begin = make_shared<ov::op::v0::Constant>(inputs_et, inputs_sshape, vector<int64_t>{0, 3, 1, 0});
    auto crops_end = make_shared<ov::op::v0::Constant>(inputs_et, inputs_sshape, vector<int64_t>{0, 3, 0, 0});

    try {
        auto batch_to_space = make_shared<ov::op::v1::BatchToSpace>(data, block_shape, crops_begin, crops_end);
        FAIL() << "Invalid elements of block_shape input not detected";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Elements of block_shape input must be greater or equal to one.");
    } catch (...) {
        FAIL() << "Greater than zero elements of block_shape input check failed for unexpected reason";
    }
}

TEST(type_prop, batch_to_space_invalid_crops_input_values) {
    ov::Shape data_sshape{100, 7, 13, 3};
    ov::element::Type data_et = ov::element::f32;

    ov::Shape inputs_sshape{4};
    ov::element::Type inputs_et = ov::element::i64;

    try {
        auto data = make_shared<ov::op::v0::Parameter>(data_et, data_sshape);
        auto block_shape = make_shared<ov::op::v0::Constant>(inputs_et, inputs_sshape, vector<int64_t>{1, 10, 5, 1});
        auto crops_begin = make_shared<ov::op::v0::Constant>(inputs_et, inputs_sshape, vector<int64_t>{0, 3, 1, -1});
        auto crops_end = make_shared<ov::op::v0::Constant>(inputs_et, inputs_sshape, vector<int64_t>{0, 3, 0, 0});
        auto batch_to_space = make_shared<ov::op::v1::BatchToSpace>(data, block_shape, crops_begin, crops_end);
        FAIL() << "Invalid crops_begin input values not detected";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Elements of crops_begin and crops_end inputs must be greater or equal to zero.");
    } catch (...) {
        FAIL() << "Non-negative element check of crops_begin input values failed for unexpected reason";
    }

    try {
        auto data = make_shared<ov::op::v0::Parameter>(data_et, data_sshape);
        auto block_shape = make_shared<ov::op::v0::Constant>(inputs_et, inputs_sshape, vector<int64_t>{1, 10, 5, 1});
        auto crops_begin = make_shared<ov::op::v0::Constant>(inputs_et, inputs_sshape, vector<int64_t>{0, 3, 1, 0});
        auto crops_end = make_shared<ov::op::v0::Constant>(inputs_et, inputs_sshape, vector<int64_t>{0, 3, -1, 0});
        auto batch_to_space = make_shared<ov::op::v1::BatchToSpace>(data, block_shape, crops_begin, crops_end);
        FAIL() << "Invalid crops_end input values not detected";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Elements of crops_begin and crops_end inputs must be greater or equal to zero.");
    } catch (...) {
        FAIL() << "Non-negative element check of crops_end input values failed for unexpected reason";
    }
}

TEST(type_prop, batch_to_space_incompatible_block_shape_input_values_with_data_shape) {
    ov::Shape data_sshape{80, 7, 13, 3};
    ov::element::Type data_et = ov::element::f32;

    ov::Shape inputs_sshape{4};
    ov::element::Type inputs_et = ov::element::i64;

    auto data = make_shared<ov::op::v0::Parameter>(data_et, data_sshape);
    auto block_shape = make_shared<ov::op::v0::Constant>(inputs_et, inputs_sshape, vector<int64_t>{1, 10, 5, 1});
    auto crops_begin = make_shared<ov::op::v0::Constant>(inputs_et, inputs_sshape, vector<int64_t>{0, 3, 1, 0});
    auto crops_end = make_shared<ov::op::v0::Constant>(inputs_et, inputs_sshape, vector<int64_t>{0, 3, 0, 0});

    try {
        auto batch_to_space = make_shared<ov::op::v1::BatchToSpace>(data, block_shape, crops_begin, crops_end);
        FAIL() << "Incompatible data shape and block_shape input values not detected";
    } catch (const ov::Exception& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "[ 80, 80] must be a multiple of divisor: 50");
    } catch (...) {
        FAIL() << "Data shape and block_shape input values check failed for unexpected reason";
    }
}

TEST(type_prop, batch_to_space_invalid_crops_out_of_bounds) {
    ov::Shape data_sshape{32, 4, 1, 3};
    ov::element::Type data_et = ov::element::f32;

    ov::Shape inputs_sshape{4};
    ov::element::Type inputs_et = ov::element::i64;

    auto data = make_shared<ov::op::v0::Parameter>(data_et, data_sshape);
    auto block_shape = make_shared<ov::op::v0::Constant>(inputs_et, inputs_sshape, vector<int64_t>{1, 2, 2, 1});
    auto crops_begin = make_shared<ov::op::v0::Constant>(inputs_et, inputs_sshape, vector<int64_t>{0, 3, 1, 2});
    auto crops_end = make_shared<ov::op::v0::Constant>(inputs_et, inputs_sshape, vector<int64_t>{0, 3, 0, 2});

    try {
        auto batch_to_space = make_shared<ov::op::v1::BatchToSpace>(data, block_shape, crops_begin, crops_end);
        FAIL() << "Invalid out of bound crops values not detected";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "crops_begin[i] + crops_end[i] must be less or equal to block_shape[i] * input_shape[i]");
    } catch (...) {
        FAIL() << "Crops values check failed for unexpected reason";
    }
}

TEST(type_prop, batch_to_space_output_shape_2D) {
    auto data = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{10, 26});
    auto block_shape = make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{2}, vector<int64_t>{1, 5});
    auto crops_begin = make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{2}, vector<int64_t>{0, 2});
    auto crops_end = make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{2}, vector<int64_t>{0, 0});
    auto batch_to_space = make_shared<ov::op::v1::BatchToSpace>(data, block_shape, crops_begin, crops_end);

    ASSERT_EQ(batch_to_space->get_element_type(), ov::element::f32);
    ASSERT_EQ(batch_to_space->get_shape(), (ov::Shape{10 / 5, 26 * 5 - 2}));
}

TEST(type_prop, batch_to_space_output_shape_4D) {
    auto data = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{100, 7, 13, 3});
    auto block_shape = make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, vector<int64_t>{1, 10, 5, 1});
    auto crops_begin = make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, vector<int64_t>{0, 3, 1, 0});
    auto crops_end = make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, vector<int64_t>{0, 3, 0, 0});
    auto batch_to_space = make_shared<ov::op::v1::BatchToSpace>(data, block_shape, crops_begin, crops_end);

    ASSERT_EQ(batch_to_space->get_element_type(), ov::element::f32);
    ASSERT_EQ(batch_to_space->get_shape(), (ov::Shape{100 / (10 * 5), 7 * 10 - 3 - 3, 13 * 5 - 1, 3}));
}

TEST(type_prop, batch_to_space_output_shape_5D) {
    auto data = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{960, 6, 13, 128, 16});
    auto block_shape =
        make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{5}, vector<int64_t>{1, 6, 5, 1, 16});
    auto crops_begin =
        make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{5}, vector<int64_t>{0, 2, 0, 0, 0});
    auto crops_end = make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{5}, vector<int64_t>{0, 2, 1, 0, 0});
    auto batch_to_space = make_shared<ov::op::v1::BatchToSpace>(data, block_shape, crops_begin, crops_end);

    ASSERT_EQ(batch_to_space->get_element_type(), ov::element::f32);
    ASSERT_EQ(batch_to_space->get_shape(), (ov::Shape{960 / (6 * 5 * 16), 6 * 6 - 2 - 2, 13 * 5 - 1, 128, 16 * 16}));
}

TEST(type_prop, batch_to_space_output_dynamic_shape_5D_when_batch_is_static) {
    auto data = make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                   ov::PartialShape{960, {2, 20}, {12, 14}, {100, 150}, {10, 20}});
    auto block_shape =
        make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{5}, vector<int64_t>{1, 6, 5, 1, 16});
    auto crops_begin =
        make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{5}, vector<int64_t>{0, 2, 0, 0, 0});
    auto crops_end = make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{5}, vector<int64_t>{0, 2, 1, 0, 0});
    auto batch_to_space = make_shared<ov::op::v1::BatchToSpace>(data, block_shape, crops_begin, crops_end);

    ASSERT_EQ(batch_to_space->get_output_partial_shape(0),
              (ov::PartialShape{960 / (6 * 5 * 16),
                                {2 * 6 - 2 - 2, 20 * 6 - 2 - 2},
                                {12 * 5 - 1, 14 * 5 - 1},
                                {100, 150},
                                {10 * 16, 20 * 16}}));
}

TEST(type_prop, batch_to_space_output_dynamic_shape_5D_when_batch_is_dynamic) {
    auto data_shape = ov::PartialShape{{959, 962}, {2, 34}, {9, 21}, {100, 162}, {1, 1999}};
    auto symbols = set_shape_symbols(data_shape);
    auto data = make_shared<ov::op::v0::Parameter>(ov::element::f32, data_shape);
    auto block_shape =
        make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{5}, vector<int64_t>{1, 6, 5, 1, 16});
    auto crops_begin =
        make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{5}, vector<int64_t>{0, 2, 0, 0, 0});
    auto crops_end = make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{5}, vector<int64_t>{0, 2, 1, 0, 0});
    auto batch_to_space = make_shared<ov::op::v1::BatchToSpace>(data, block_shape, crops_begin, crops_end);

    EXPECT_EQ(batch_to_space->get_output_partial_shape(0),
              (ov::PartialShape{{ov::util::ceil_div(959, (6 * 5 * 16)), 962 / (6 * 5 * 16)},
                                {2 * 6 - 2 - 2, 34 * 6 - 2 - 2},
                                {9 * 5 - 1, 21 * 5 - 1},
                                {100, 162},
                                {1 * 16, 1999 * 16}}));
    EXPECT_THAT(get_shape_symbols(batch_to_space->get_output_partial_shape(0)),
                ElementsAre(nullptr, nullptr, nullptr, symbols[3], nullptr));
}

TEST(type_prop, batch_to_space_input_interval_shape_block_one) {
    auto data_shape = ov::PartialShape{{959, 962}, {2, 34}, {9, 21}};
    auto symbols = set_shape_symbols(data_shape);
    auto data = make_shared<ov::op::v0::Parameter>(ov::element::f32, data_shape);
    auto block_shape = make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{3}, vector<int64_t>{1, 1, 1});
    auto crops_begin = make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{3}, vector<int64_t>{0, 0, 0});
    auto crops_end = make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{3}, vector<int64_t>{0, 0, 1});
    auto batch_to_space = make_shared<ov::op::v1::BatchToSpace>(data, block_shape, crops_begin, crops_end);

    EXPECT_EQ(batch_to_space->get_output_partial_shape(0),
              ov::PartialShape({{959, 962}, {2, 34}, {9 * 1 - 1, 21 * 1 - 1}}));
    EXPECT_THAT(get_shape_symbols(batch_to_space->get_output_partial_shape(0)),
                ElementsAre(symbols[0], symbols[1], nullptr));
}

TEST(type_prop, batch_to_space_and_space_to_batch) {
    auto data = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{4800, 9, {11, -1}, 2});
    auto block_shape =
        make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, vector<int64_t>{1, 12, 100, 2});
    auto crops_begin = make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, vector<int64_t>{0, 3, 38, 1});
    auto crops_end = make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, vector<int64_t>{0, 5, 38, 0});
    auto batch_to_space = make_shared<ov::op::v1::BatchToSpace>(data, block_shape, crops_begin, crops_end);

    ASSERT_EQ(batch_to_space->get_element_type(), ov::element::f32);
    ASSERT_EQ(batch_to_space->get_output_partial_shape(0),
              (ov::PartialShape{4800 / (12 * 100 * 2), 9 * 12 - 3 - 5, {11 * 100 - 38 - 38, -1}, 2 * 2 - 1}));

    auto space_to_batch = make_shared<ov::op::v1::SpaceToBatch>(batch_to_space, block_shape, crops_begin, crops_end);
    ASSERT_EQ(space_to_batch->get_element_type(), ov::element::f32);
    ASSERT_EQ(space_to_batch->get_output_partial_shape(0), (ov::PartialShape{4800, 9, {11, -1}, 2}));
}

TEST(type_prop, batch_to_space_dynamic_shape_static_rank) {
    auto data = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
    auto block_shape = make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, vector<int64_t>{1, 10, 5, 1});
    auto crops_begin = make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, vector<int64_t>{0, 3, 1, 0});
    auto crops_end = make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, vector<int64_t>{0, 3, 0, 0});
    auto batch_to_space = make_shared<ov::op::v1::BatchToSpace>(data, block_shape, crops_begin, crops_end);

    ASSERT_EQ(batch_to_space->get_element_type(), ov::element::f32);
    ASSERT_EQ(batch_to_space->get_output_partial_shape(0), ov::PartialShape::dynamic(4));
}

TEST(type_prop, batch_to_space_dynamic_shape_dynamic_rank) {
    auto data = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    auto block_shape = make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, vector<int64_t>{1, 10, 5, 1});
    auto crops_begin = make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, vector<int64_t>{0, 3, 1, 0});
    auto crops_end = make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, vector<int64_t>{0, 3, 0, 0});
    auto batch_to_space = make_shared<ov::op::v1::BatchToSpace>(data, block_shape, crops_begin, crops_end);

    ASSERT_EQ(batch_to_space->get_element_type(), ov::element::f32);
    ASSERT_EQ(batch_to_space->get_output_partial_shape(0), ov::PartialShape::dynamic());
}

TEST(type_prop, batch_to_space_default_ctor) {
    auto data = make_shared<ov::op::v0::Parameter>(ov::element::i16, ov::Shape{100, 7, 13, 3});
    auto block_shape = make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, vector<int64_t>{1, 10, 5, 1});
    auto crops_begin = make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, vector<int64_t>{0, 3, 1, 0});
    auto crops_end = make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, vector<int64_t>{0, 3, 0, 0});

    auto batch_to_space = make_shared<ov::op::v1::BatchToSpace>();

    batch_to_space->set_arguments(ov::OutputVector{data, block_shape, crops_begin, crops_end});
    batch_to_space->validate_and_infer_types();

    EXPECT_EQ(batch_to_space->get_input_size(), 4);
    EXPECT_EQ(batch_to_space->get_output_size(), 1);
    EXPECT_EQ(batch_to_space->get_element_type(), ov::element::i16);
    EXPECT_EQ(batch_to_space->get_shape(), (ov::Shape{100 / (10 * 5), 7 * 10 - 3 - 3, 13 * 5 - 1, 3}));
}

TEST(type_prop, batch_to_space_non_const_inputs) {
    auto data = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{100, 7, 13, 3});

    auto block_shape = make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{4});
    auto crops_begin = make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{4});
    auto crops_end = make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{4});
    auto batch_to_space = make_shared<ov::op::v1::BatchToSpace>(data, block_shape, crops_begin, crops_end);

    EXPECT_EQ(batch_to_space->get_element_type(), ov::element::f32);
    EXPECT_EQ(batch_to_space->get_output_partial_shape(0), ov::PartialShape::dynamic(4));
}

TEST(type_prop, batch_to_space_block_non_constant_only) {
    auto data = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{100, 7, 13, 3});
    auto block_shape = make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{4});
    auto crops_begin = make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, vector<int64_t>{0, 3, 1, 0});
    auto crops_end = make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, vector<int64_t>{0, 3, 0, 0});
    auto batch_to_space = make_shared<ov::op::v1::BatchToSpace>(data, block_shape, crops_begin, crops_end);

    EXPECT_EQ(batch_to_space->get_element_type(), ov::element::f32);
    EXPECT_EQ(batch_to_space->get_output_partial_shape(0), ov::PartialShape({-1, {1, -1}, {12, -1}, {3, -1}}));
}

TEST(type_prop, batch_to_space_crops_non_constant_only) {
    auto data = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{100, 7, 13, 3});
    auto block_shape = make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, vector<int64_t>{1, 2, 5, 1});
    auto crops_begin = make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{4});
    auto crops_end = make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{4});
    auto batch_to_space = make_shared<ov::op::v1::BatchToSpace>(data, block_shape, crops_begin, crops_end);

    EXPECT_EQ(batch_to_space->get_element_type(), ov::element::f32);
    EXPECT_EQ(batch_to_space->get_output_partial_shape(0), ov::PartialShape({10, -1, -1, -1}));
}
