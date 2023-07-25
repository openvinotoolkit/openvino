// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/extractimagepatches.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"

using namespace std;
using namespace ov;
using namespace testing;

TEST(type_prop, extractimagepatches_default_ctor) {
    auto data = make_shared<op::v0::Parameter>(element::i32, Shape{64, 3, 10, 10});
    auto padtype_padding = op::PadType::VALID;

    auto op = make_shared<op::v3::ExtractImagePatches>();
    op->set_argument(0, data);
    op->set_sizes({3, 3});
    op->set_strides({5, 5});
    op->set_rates({1, 1});
    op->set_auto_pad(padtype_padding);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_input_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::i32);
    EXPECT_EQ(op->get_output_shape(0), (Shape{64, 27, 2, 2}));
}

TEST(type_prop, extractimagepatches_i32) {
    auto data = make_shared<op::v0::Parameter>(element::i32, Shape{64, 3, 10, 10});
    auto sizes = Shape{3, 3};
    auto strides = Strides{5, 5};
    auto rates = Shape{1, 1};
    auto padtype_padding = op::PadType::VALID;
    auto extractimagepatches = make_shared<op::v3::ExtractImagePatches>(data, sizes, strides, rates, padtype_padding);

    EXPECT_EQ(extractimagepatches->get_output_element_type(0), element::i32);
    EXPECT_EQ(extractimagepatches->get_output_shape(0), (Shape{64, 27, 2, 2}));
}

TEST(type_prop, extractimagepatches_i64) {
    auto data = make_shared<op::v0::Parameter>(element::i64, Shape{64, 3, 12, 12});
    auto sizes = Shape{3, 3};
    auto strides = Strides{5, 5};
    auto rates = Shape{1, 1};
    auto padtype_padding = op::PadType::VALID;
    auto extractimagepatches = make_shared<op::v3::ExtractImagePatches>(data, sizes, strides, rates, padtype_padding);

    EXPECT_EQ(extractimagepatches->get_output_element_type(0), element::i64);
    EXPECT_EQ(extractimagepatches->get_output_shape(0), (Shape{64, 27, 2, 2}));
}

TEST(type_prop, extractimagepatches_rates_change) {
    auto data = make_shared<op::v0::Parameter>(element::i32, Shape{64, 3, 33, 43});
    auto sizes = Shape{3, 3};
    auto strides = Strides{5, 5};
    auto rates = Shape{2, 2};
    auto padtype_padding = op::PadType::VALID;
    auto extractimagepatches = make_shared<op::v3::ExtractImagePatches>(data, sizes, strides, rates, padtype_padding);

    EXPECT_EQ(extractimagepatches->get_output_element_type(0), element::i32);
    EXPECT_EQ(extractimagepatches->get_output_shape(0), (Shape{64, 27, 6, 8}));
}

TEST(type_prop, extractimagepatches_input_shape_change) {
    auto data_shape = PartialShape{64, 3, 9, 9};
    set_shape_labels(data_shape, 10);
    auto data = make_shared<op::v0::Parameter>(element::i32, data_shape);
    auto sizes = Shape{3, 3};
    auto strides = Strides{5, 5};
    auto rates = Shape{2, 2};
    auto padtype_padding = op::PadType::VALID;
    auto extractimagepatches = make_shared<op::v3::ExtractImagePatches>(data, sizes, strides, rates, padtype_padding);

    EXPECT_EQ(extractimagepatches->get_output_element_type(0), element::i32);
    EXPECT_EQ(extractimagepatches->get_output_shape(0), (Shape{64, 27, 1, 1}));
    EXPECT_THAT(get_shape_labels(extractimagepatches->get_output_partial_shape(0)), ElementsAre(10, 0, 0, 0));
}

TEST(type_prop, extractimagepatches_dynamic_shape) {
    auto data_shape = PartialShape::dynamic(4);
    set_shape_labels(data_shape, 10);

    auto data = make_shared<op::v0::Parameter>(element::i32, data_shape);
    auto sizes = Shape{3, 3};
    auto strides = Strides{5, 5};
    auto rates = Shape{2, 2};
    auto padtype_padding = op::PadType::VALID;
    auto extractimagepatches = make_shared<op::v3::ExtractImagePatches>(data, sizes, strides, rates, padtype_padding);

    EXPECT_EQ(extractimagepatches->get_output_element_type(0), element::i32);
    EXPECT_EQ(extractimagepatches->get_output_partial_shape(0), PartialShape::dynamic(4));
    EXPECT_THAT(get_shape_labels(extractimagepatches->get_output_partial_shape(0)), ElementsAre(10, 0, 0, 0));
}

TEST(type_prop, extractimagepatches_dynamic_batch_shape) {
    auto data = make_shared<op::v0::Parameter>(element::i32, PartialShape{Dimension::dynamic(), 3, 14, 23});
    auto sizes = Shape{3, 3};
    auto strides = Strides{5, 5};
    auto rates = Shape{1, 1};
    auto padtype_padding = op::PadType::VALID;
    auto extractimagepatches = make_shared<op::v3::ExtractImagePatches>(data, sizes, strides, rates, padtype_padding);

    EXPECT_EQ(extractimagepatches->get_output_element_type(0), element::i32);
    EXPECT_EQ(extractimagepatches->get_output_partial_shape(0), PartialShape({-1, 27, 3, 5}));
}

TEST(type_prop, extractimagepatches_interval_shape_and_labels) {
    auto data_shape = PartialShape{{1, 10}, {3, 4}, {10, 51}, {13, 71}};
    set_shape_labels(data_shape, 10);
    auto data = make_shared<op::v0::Parameter>(element::i32, data_shape);
    auto sizes = Shape{3, 3};
    auto strides = Strides{5, 5};
    auto rates = Shape{1, 1};
    auto padtype_padding = op::PadType::VALID;
    auto extractimagepatches = make_shared<op::v3::ExtractImagePatches>(data, sizes, strides, rates, padtype_padding);

    EXPECT_EQ(extractimagepatches->get_output_element_type(0), element::i32);
    EXPECT_EQ(extractimagepatches->get_output_partial_shape(0), PartialShape({{1, 10}, {27, 36}, {2, 10}, {3, 14}}));
    EXPECT_THAT(get_shape_labels(extractimagepatches->get_output_partial_shape(0)), ElementsAre(10, 0, 0, 0));
}

TEST(type_prop, extractimagepatches_padding_same_lower1) {
    auto data = make_shared<op::v0::Parameter>(element::i32, Shape{64, 3, 31, 27});
    auto sizes = Shape{3, 3};
    auto strides = Strides{5, 5};
    auto rates = Shape{1, 1};
    auto padtype_padding = op::PadType::SAME_LOWER;
    auto extractimagepatches = make_shared<op::v3::ExtractImagePatches>(data, sizes, strides, rates, padtype_padding);

    EXPECT_EQ(extractimagepatches->get_output_element_type(0), element::i32);
    EXPECT_EQ(extractimagepatches->get_output_shape(0), (Shape{64, 27, 7, 6}));
}

TEST(type_prop, extractimagepatches_padding_same_lower2) {
    auto data = make_shared<op::v0::Parameter>(element::i32, Shape{64, 3, 9, 9});
    auto sizes = Shape{3, 3};
    auto strides = Strides{5, 5};
    auto rates = Shape{1, 1};
    auto padtype_padding = op::PadType::SAME_LOWER;
    auto extractimagepatches = make_shared<op::v3::ExtractImagePatches>(data, sizes, strides, rates, padtype_padding);

    EXPECT_EQ(extractimagepatches->get_output_element_type(0), element::i32);
    EXPECT_EQ(extractimagepatches->get_output_shape(0), (Shape{64, 27, 2, 2}));
}
TEST(type_prop, extractimagepatches_padding_same_upper) {
    auto data = make_shared<op::v0::Parameter>(element::i32, Shape{64, 3, 23, 11});
    auto sizes = Shape{3, 3};
    auto strides = Strides{5, 5};
    auto rates = Shape{1, 1};
    auto padtype_padding = op::PadType::SAME_UPPER;
    auto extractimagepatches = make_shared<op::v3::ExtractImagePatches>(data, sizes, strides, rates, padtype_padding);

    EXPECT_EQ(extractimagepatches->get_output_element_type(0), element::i32);
    EXPECT_EQ(extractimagepatches->get_output_shape(0), (Shape{64, 27, 5, 3}));
}

TEST(type_prop, extractimagepatches_padding_same_upper2) {
    auto data = make_shared<op::v0::Parameter>(element::i32, Shape{64, 3, 6, 11});
    auto sizes = Shape{3, 3};
    auto strides = Strides{5, 5};
    auto rates = Shape{1, 1};
    auto padtype_padding = op::PadType::SAME_UPPER;
    auto extractimagepatches = make_shared<op::v3::ExtractImagePatches>(data, sizes, strides, rates, padtype_padding);

    EXPECT_EQ(extractimagepatches->get_output_element_type(0), element::i32);
    EXPECT_EQ(extractimagepatches->get_output_shape(0), (Shape{64, 27, 2, 3}));
}

TEST(type_prop, extractimagepatches_zero_dim_inputs) {
    auto data = make_shared<op::v0::Parameter>(element::i32, Shape{64, 0, 0, 0});
    auto sizes = Shape{3, 3};
    auto strides = Strides{5, 5};
    auto rates = Shape{1, 1};
    auto padtype_padding = op::PadType::VALID;
    auto extractimagepatches = make_shared<op::v3::ExtractImagePatches>(data, sizes, strides, rates, padtype_padding);

    EXPECT_EQ(extractimagepatches->get_output_element_type(0), element::i32);
    EXPECT_EQ(extractimagepatches->get_output_shape(0), (Shape{64, 0, 0, 0}));
}

TEST(type_prop, extractimagepatches_large_stride_valid_padding) {
    auto data = make_shared<op::v0::Parameter>(element::i32, Shape{64, 3, 10, 10});
    auto sizes = Shape{3, 3};
    auto strides = Strides{15, 15};
    auto rates = Shape{1, 1};
    auto padtype_padding = op::PadType::VALID;
    auto extractimagepatches = make_shared<op::v3::ExtractImagePatches>(data, sizes, strides, rates, padtype_padding);

    EXPECT_EQ(extractimagepatches->get_output_element_type(0), element::i32);
    EXPECT_EQ(extractimagepatches->get_output_shape(0), (Shape{64, 27, 1, 1}));
}

TEST(type_prop, extractimagepatches_large_stride_same_padding) {
    auto data = make_shared<op::v0::Parameter>(element::i32, Shape{64, 3, 10, 10});
    auto sizes = Shape{3, 3};
    auto strides = Strides{15, 15};
    auto rates = Shape{1, 1};
    auto padtype_padding = op::PadType::SAME_UPPER;
    auto extractimagepatches = make_shared<op::v3::ExtractImagePatches>(data, sizes, strides, rates, padtype_padding);

    EXPECT_EQ(extractimagepatches->get_output_element_type(0), element::i32);
    EXPECT_EQ(extractimagepatches->get_output_shape(0), (Shape{64, 27, 1, 1}));
}

TEST(type_prop, extractimagepatches_dyn) {
    auto data = make_shared<op::v0::Parameter>(element::i64, PartialShape::dynamic());
    auto sizes = Shape{3, 3};
    auto strides = Strides{5, 5};
    auto rates = Shape{1, 1};
    auto padtype_padding = op::PadType::VALID;
    auto extractimagepatches = make_shared<op::v3::ExtractImagePatches>(data, sizes, strides, rates, padtype_padding);

    EXPECT_EQ(extractimagepatches->get_output_element_type(0), element::i64);
    EXPECT_EQ(extractimagepatches->get_output_partial_shape(0), PartialShape::dynamic(4));
}

TEST(type_prop, extractimagepatches_interval_shape_and_labels_padding_same_upper) {
    auto data_shape = PartialShape{{1, 10}, {3, 4}, {10, 51}, {13, 71}};
    set_shape_labels(data_shape, 10);
    auto data = make_shared<op::v0::Parameter>(element::i32, data_shape);
    auto sizes = Shape{1, 1};
    auto strides = Strides{5, 5};
    auto rates = Shape{1, 1};
    auto padtype_padding = op::PadType::SAME_LOWER;
    auto extractimagepatches = make_shared<op::v3::ExtractImagePatches>(data, sizes, strides, rates, padtype_padding);

    EXPECT_EQ(extractimagepatches->get_output_element_type(0), element::i32);
    EXPECT_EQ(extractimagepatches->get_output_partial_shape(0), PartialShape({{1, 10}, {3, 4}, {2, 11}, {3, 15}}));
    EXPECT_THAT(get_shape_labels(extractimagepatches->get_output_partial_shape(0)), ElementsAre(10, 11, 0, 0));
}

TEST(type_prop, extractimagepatches_data_not_rank_4d) {
    auto sizes = Shape{3, 3};
    auto strides = Strides{5, 5};
    auto rates = Shape{1, 1};
    auto padtype_padding = op::PadType::VALID;

    OV_EXPECT_THROW(ignore = make_shared<op::v3::ExtractImagePatches>(
                        make_shared<op::v0::Parameter>(element::i64, PartialShape::dynamic(3)),
                        sizes,
                        strides,
                        rates,
                        padtype_padding),
                    NodeValidationFailure,
                    HasSubstr("input tensor must be 4D tensor"));

    OV_EXPECT_THROW(ignore = make_shared<op::v3::ExtractImagePatches>(
                        make_shared<op::v0::Parameter>(element::i64, PartialShape::dynamic(5)),
                        sizes,
                        strides,
                        rates,
                        padtype_padding),
                    NodeValidationFailure,
                    HasSubstr("input tensor must be 4D tensor"));
}

TEST(type_prop, extractimagepatches_sizes_has_no_two_elements) {
    auto data = make_shared<op::v0::Parameter>(element::i64, PartialShape::dynamic());
    auto sizes = Shape{3};
    auto strides = Strides{5, 5};
    auto rates = Shape{1, 1};
    auto padtype_padding = op::PadType::EXPLICIT;

    OV_EXPECT_THROW(ignore = make_shared<op::v3::ExtractImagePatches>(data, sizes, strides, rates, padtype_padding),
                    NodeValidationFailure,
                    HasSubstr("Attribute sizes should be in [size_rows, size_cols] format"));
}

TEST(type_prop, extractimagepatches_strides_has_no_two_elements) {
    auto data = make_shared<op::v0::Parameter>(element::i64, PartialShape::dynamic());
    auto sizes = Shape{3, 3};
    auto strides = Strides{5, 5, 1};
    auto rates = Shape{1, 1};
    auto padtype_padding = op::PadType::EXPLICIT;

    OV_EXPECT_THROW(ignore = make_shared<op::v3::ExtractImagePatches>(data, sizes, strides, rates, padtype_padding),
                    NodeValidationFailure,
                    HasSubstr("Attribute strides should be in [stride_rows, stride_cols] format"));
}

TEST(type_prop, extractimagepatches_strides_has_zero_value) {
    auto data = make_shared<op::v0::Parameter>(element::i64, PartialShape::dynamic());
    auto sizes = Shape{3, 3};
    auto strides = Strides{5, 0};
    auto rates = Shape{1, 1};
    auto padtype_padding = op::PadType::EXPLICIT;

    OV_EXPECT_THROW(ignore = make_shared<op::v3::ExtractImagePatches>(data, sizes, strides, rates, padtype_padding),
                    NodeValidationFailure,
                    HasSubstr("Attribute strides should be strictly greater than zeros in values"));
}

TEST(type_prop, extractimagepatches_rates_has_no_two_elements) {
    auto data = make_shared<op::v0::Parameter>(element::i64, PartialShape::dynamic());
    auto sizes = Shape{3, 3};
    auto strides = Strides{5, 5};
    auto rates = Shape{1};
    auto padtype_padding = op::PadType::EXPLICIT;

    OV_EXPECT_THROW(ignore = make_shared<op::v3::ExtractImagePatches>(data, sizes, strides, rates, padtype_padding),
                    NodeValidationFailure,
                    HasSubstr("Attribute rates should be in [rate_rows, rate_cols] format"));
}

TEST(type_prop, extractimagepatches_rates_has_zero_value) {
    auto data = make_shared<op::v0::Parameter>(element::i64, PartialShape::dynamic());
    auto sizes = Shape{3, 3};
    auto strides = Strides{5, 5};
    auto rates = Shape{0, 1};
    auto padtype_padding = op::PadType::EXPLICIT;

    OV_EXPECT_THROW(ignore = make_shared<op::v3::ExtractImagePatches>(data, sizes, strides, rates, padtype_padding),
                    NodeValidationFailure,
                    HasSubstr("Attribute rates should be strictly greater than zeros in values"));
}

TEST(type_prop, extractimagepatches_not_supported_padding) {
    auto data = make_shared<op::v0::Parameter>(element::i64, PartialShape::dynamic());
    auto sizes = Shape{3, 3};
    auto strides = Strides{5, 5};
    auto rates = Shape{1, 1};
    auto padtype_padding = op::PadType::EXPLICIT;

    OV_EXPECT_THROW(ignore = make_shared<op::v3::ExtractImagePatches>(data, sizes, strides, rates, padtype_padding),
                    NodeValidationFailure,
                    HasSubstr("Attribute padding should be in either valid or same_lower or same_upper"));
}
