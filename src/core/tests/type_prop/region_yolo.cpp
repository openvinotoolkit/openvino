// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/region_yolo.hpp"

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"

using namespace testing;
using namespace ov;

class TypePropRegionYoloV0Test : public TypePropOpTest<op::v0::RegionYolo> {};

TEST_F(TypePropRegionYoloV0Test, default_ctor_do_softmax) {
    const std::vector<int64_t> mask{0, 1, 2};

    const auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{10, 5, 16, 16});
    const auto op = make_op();
    op->set_argument(0, data);
    op->set_do_softmax(true);
    op->set_axis(-1);
    op->set_end_axis(2);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_input_size(), 1);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({10, 5, 16, 16}));
}

TEST_F(TypePropRegionYoloV0Test, default_ctor_no_softmax) {
    const std::vector<int64_t> mask{0, 1, 2};

    const auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{10, 5, 11, 12});
    const auto op = make_op();
    op->set_argument(0, data);
    op->set_do_softmax(false);
    op->set_num_classes(5);
    op->set_num_coords(2);
    op->set_mask({1, 2});
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_input_size(), 1);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({10, 16, 11, 12}));
}

TEST_F(TypePropRegionYoloV0Test, data_input_dynamic_rank_do_not_softmax) {
    constexpr size_t num = 5, coords = 4, classes = 20;
    constexpr int axis = 1, end_axis = 3;
    const std::vector<int64_t> mask{0, 1, 2};

    const auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto op = make_op(data, coords, classes, num, false, mask, axis, end_axis);

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape::dynamic());
}

TEST_F(TypePropRegionYoloV0Test, data_input_dynamic_rank_do_softmax) {
    constexpr size_t num = 5, coords = 4, classes = 20;
    constexpr int axis = 1, end_axis = 3;
    const std::vector<int64_t> mask{0, 1, 2};

    const auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto op = make_op(data, coords, classes, num, true, mask, axis, end_axis);

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape::dynamic());
}

TEST_F(TypePropRegionYoloV0Test, data_input_static_rank_do_softmax) {
    constexpr size_t num = 5, coords = 4, classes = 20;
    constexpr int axis = 1, end_axis = 3;
    const std::vector<int64_t> mask{0, 1, 2};

    auto data_shape = PartialShape::dynamic(4);
    auto symbols = set_shape_symbols(data_shape);

    const auto data = std::make_shared<op::v0::Parameter>(element::f64, data_shape);
    const auto op = make_op(data, coords, classes, num, true, mask, axis, end_axis);

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f64);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape::dynamic(2));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), ElementsAre(symbols[0], nullptr));
}

TEST_F(TypePropRegionYoloV0Test, do_softmax_end_axis_is_negative) {
    constexpr size_t num = 5, coords = 4, classes = 20;
    constexpr int axis = 1, end_axis = -1;
    const std::vector<int64_t> mask{0, 1, 2};

    auto data_shape = PartialShape::dynamic(4);
    auto symbols = set_shape_symbols(data_shape);

    const auto data = std::make_shared<op::v0::Parameter>(element::f32, data_shape);
    const auto op = make_op(data, coords, classes, num, true, mask, axis, end_axis);

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape::dynamic(2));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), ElementsAre(symbols[0], nullptr));
}

TEST_F(TypePropRegionYoloV0Test, do_softmax_axis_eq_end_axis) {
    constexpr size_t num = 5, coords = 4, classes = 20;
    constexpr int axis = 2, end_axis = 2;
    const std::vector<int64_t> mask{0, 1, 2};

    auto data_shape = PartialShape{5, 4, 10, 11};
    auto symbols = set_shape_symbols(data_shape);

    const auto data = std::make_shared<op::v0::Parameter>(element::f32, data_shape);
    const auto op = make_op(data, coords, classes, num, true, mask, axis, end_axis);

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({5, 4, 10, 11}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), symbols);
}

TEST_F(TypePropRegionYoloV0Test, do_softmax_axis_gt_end_axis) {
    constexpr size_t num = 5, coords = 4, classes = 20;
    constexpr int axis = 3, end_axis = 1;
    const std::vector<int64_t> mask{0, 1, 2};

    auto data_shape = PartialShape{5, 4, 10, 11};
    auto symbols = set_shape_symbols(data_shape);

    const auto data = std::make_shared<op::v0::Parameter>(element::f32, data_shape);
    const auto op = make_op(data, coords, classes, num, true, mask, axis, end_axis);

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({5, 4, 10, 11}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), symbols);
}

TEST_F(TypePropRegionYoloV0Test, do_softmax_axis_end_axis_on_last_dim) {
    constexpr size_t num = 5, coords = 4, classes = 20;
    constexpr int axis = -1, end_axis = -1;
    const std::vector<int64_t> mask{0, 1, 2};

    auto data_shape = PartialShape{5, 4, 10, 11};
    auto symbols = set_shape_symbols(data_shape);

    const auto data = std::make_shared<op::v0::Parameter>(element::f32, data_shape);
    const auto op = make_op(data, coords, classes, num, true, mask, axis, end_axis);

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({5, 4, 10, 11}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), symbols);
}

TEST_F(TypePropRegionYoloV0Test, data_input_interval_shape_with_symbols_do_softmax) {
    constexpr size_t num = 5, coords = 4, classes = 20;
    constexpr int axis = 2, end_axis = 3;
    const std::vector<int64_t> mask{0, 1, 2};

    auto data_shape = PartialShape{{2, 4}, {5, 8}, -1, {0, 10}};
    auto symbols = set_shape_symbols(data_shape);

    const auto data = std::make_shared<op::v0::Parameter>(element::f16, data_shape);
    const auto op = make_op(data, coords, classes, num, true, mask, axis, end_axis);

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f16);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({{2, 4}, {5, 8}, -1}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), ElementsAre(symbols[0], symbols[1], nullptr));
}

TEST_F(TypePropRegionYoloV0Test, do_softmax_start_axis_negative) {
    constexpr size_t num = 5, coords = 4, classes = 20;
    constexpr int axis = -2, end_axis = 3;
    const std::vector<int64_t> mask{0, 1, 2};

    auto data_shape = PartialShape{{2, 4}, {5, 8}, -1, {0, 10}};
    auto symbols = set_shape_symbols(data_shape);

    const auto data = std::make_shared<op::v0::Parameter>(element::f16, data_shape);
    const auto op = make_op(data, coords, classes, num, true, mask, axis, end_axis);

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f16);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({{2, 4}, {5, 8}, -1}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), ElementsAre(symbols[0], symbols[1], nullptr));
}

TEST_F(TypePropRegionYoloV0Test, data_input_interval_shape_with_symbols_no_softmax) {
    constexpr size_t num = 5, coords = 4, classes = 20;
    constexpr int axis = 2, end_axis = 3;
    const std::vector<int64_t> mask{0, 1, 2};

    auto data_shape = PartialShape{{2, 4}, {5, 8}, -1, {0, 10}};
    auto symbols = set_shape_symbols(data_shape);

    const auto data = std::make_shared<op::v0::Parameter>(element::f16, data_shape);
    const auto op = make_op(data, coords, classes, num, false, mask, axis, end_axis);

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f16);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({{2, 4}, 75, -1, {0, 10}}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)),
                ElementsAre(symbols[0], nullptr, symbols[2], symbols[3]));
}

TEST_F(TypePropRegionYoloV0Test, data_input_not_4d) {
    constexpr size_t num = 5, coords = 4, classes = 20;
    constexpr int axis = 1, end_axis = 5;
    const std::vector<int64_t> mask{0, 1, 2};

    OV_EXPECT_THROW(std::ignore = make_op(std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(3)),
                                          coords,
                                          classes,
                                          num,
                                          true,
                                          mask,
                                          axis,
                                          end_axis),
                    NodeValidationFailure,
                    HasSubstr("Input must be a tensor of rank 4"));

    OV_EXPECT_THROW(std::ignore = make_op(std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(5)),
                                          coords,
                                          classes,
                                          num,
                                          true,
                                          mask,
                                          axis,
                                          end_axis),
                    NodeValidationFailure,
                    HasSubstr("Input must be a tensor of rank 4"));
}

TEST_F(TypePropRegionYoloV0Test, do_softmax_axis_not_valid_value) {
    constexpr size_t num = 5, coords = 4, classes = 20;
    const std::vector<int64_t> mask{0, 1, 2};
    const auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(4));

    OV_EXPECT_THROW(std::ignore = make_op(data, coords, classes, num, true, mask, 4, 2),
                    AssertFailure,
                    HasSubstr("out of the tensor rank range"));

    OV_EXPECT_THROW(std::ignore = make_op(data, coords, classes, num, true, mask, -5, 2),
                    AssertFailure,
                    HasSubstr("out of the tensor rank range"));
}

TEST_F(TypePropRegionYoloV0Test, do_softmax_end_axis_not_valid_value) {
    constexpr size_t num = 5, coords = 4, classes = 20;
    const std::vector<int64_t> mask{0, 1, 2};
    const auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(4));

    OV_EXPECT_THROW(std::ignore = make_op(data, coords, classes, num, true, mask, 1, 4),
                    AssertFailure,
                    HasSubstr("out of the tensor rank range"));

    OV_EXPECT_THROW(std::ignore = make_op(data, coords, classes, num, true, mask, 1, -5),
                    AssertFailure,
                    HasSubstr("out of the tensor rank range"));
}
