// Copyright (C) 2018-2025 Intel Corporation
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
    auto symbols = set_shape_symbols(data_shape);
    auto data = make_shared<op::v0::Parameter>(element::i32, data_shape);
    auto sizes = Shape{3, 3};
    auto strides = Strides{5, 5};
    auto rates = Shape{2, 2};
    auto padtype_padding = op::PadType::VALID;
    auto extractimagepatches = make_shared<op::v3::ExtractImagePatches>(data, sizes, strides, rates, padtype_padding);

    EXPECT_EQ(extractimagepatches->get_output_element_type(0), element::i32);
    EXPECT_EQ(extractimagepatches->get_output_shape(0), (Shape{64, 27, 1, 1}));
    EXPECT_THAT(get_shape_symbols(extractimagepatches->get_output_partial_shape(0)),
                ElementsAre(symbols[0], nullptr, nullptr, nullptr));
}

TEST(type_prop, extractimagepatches_dynamic_shape) {
    auto data_shape = PartialShape::dynamic(4);
    auto symbols = set_shape_symbols(data_shape);

    auto data = make_shared<op::v0::Parameter>(element::i32, data_shape);
    auto sizes = Shape{3, 3};
    auto strides = Strides{5, 5};
    auto rates = Shape{2, 2};
    auto padtype_padding = op::PadType::VALID;
    auto extractimagepatches = make_shared<op::v3::ExtractImagePatches>(data, sizes, strides, rates, padtype_padding);

    EXPECT_EQ(extractimagepatches->get_output_element_type(0), element::i32);
    EXPECT_EQ(extractimagepatches->get_output_partial_shape(0), PartialShape::dynamic(4));
    EXPECT_THAT(get_shape_symbols(extractimagepatches->get_output_partial_shape(0)),
                ElementsAre(symbols[0], nullptr, nullptr, nullptr));
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
    auto symbols = set_shape_symbols(data_shape);
    auto data = make_shared<op::v0::Parameter>(element::i32, data_shape);
    auto sizes = Shape{3, 3};
    auto strides = Strides{5, 5};
    auto rates = Shape{1, 1};
    auto padtype_padding = op::PadType::VALID;
    auto extractimagepatches = make_shared<op::v3::ExtractImagePatches>(data, sizes, strides, rates, padtype_padding);

    EXPECT_EQ(extractimagepatches->get_output_element_type(0), element::i32);
    EXPECT_EQ(extractimagepatches->get_output_partial_shape(0), PartialShape({{1, 10}, {27, 36}, {2, 10}, {3, 14}}));
    EXPECT_THAT(get_shape_symbols(extractimagepatches->get_output_partial_shape(0)),
                ElementsAre(symbols[0], nullptr, nullptr, nullptr));
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
    auto symbols = set_shape_symbols(data_shape);
    auto data = make_shared<op::v0::Parameter>(element::i32, data_shape);
    auto sizes = Shape{1, 1};
    auto strides = Strides{5, 5};
    auto rates = Shape{1, 1};
    auto padtype_padding = op::PadType::SAME_LOWER;
    auto extractimagepatches = make_shared<op::v3::ExtractImagePatches>(data, sizes, strides, rates, padtype_padding);

    EXPECT_EQ(extractimagepatches->get_output_element_type(0), element::i32);
    EXPECT_EQ(extractimagepatches->get_output_partial_shape(0), PartialShape({{1, 10}, {3, 4}, {2, 11}, {3, 15}}));
    EXPECT_THAT(get_shape_symbols(extractimagepatches->get_output_partial_shape(0)),
                ElementsAre(symbols[0], symbols[1], nullptr, nullptr));
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

using ExtractImagesPatchesParams = std::tuple<op::PadType,
                                              size_t,  // size
                                              size_t,  // rate
                                              size_t   // stride
                                              >;

class ExtractImagesPatchesV3Test : public TypePropOpTest<op::v3::ExtractImagePatches>,
                                   public WithParamInterface<ExtractImagesPatchesParams> {
protected:
    void SetUp() override {
        const auto& p = GetParam();

        pad_type = std::get<0>(p);
        sizes = Shape(2, std::get<1>(p));
        rates = Shape(2, std::get<2>(p));
        strides = Shape(2, std::get<3>(p));
    }

    int64_t calc_shape_padding(const int64_t input,
                               const int64_t rate,
                               const int64_t stride,
                               const int64_t patch_size,
                               const op::PadType type) {
        int64_t out = 0;
        if (type == op::PadType::VALID) {
            out = (input - rate * (patch_size - 1) - 1) / stride + 1;
        } else {
            out = 1 + (input - 1) / stride;
        }
        return out < 0 ? 0 : out;
    }

    PartialShape exp_shape() {
        auto shape = PartialShape{dim0, dim1, -1, -1};
        shape[1] *= sizes[0] * sizes[1];
        shape[2] = calc_shape_padding(100,
                                      static_cast<int64_t>(rates[0]),
                                      static_cast<int64_t>(strides[0]),
                                      static_cast<int64_t>(sizes[0]),
                                      pad_type);
        shape[3] = calc_shape_padding(100,
                                      static_cast<int64_t>(rates[1]),
                                      static_cast<int64_t>(strides[1]),
                                      static_cast<int64_t>(sizes[1]),
                                      pad_type);

        return shape;
    }

    Shape sizes, rates;
    Strides strides;
    op::PadType pad_type;
    int64_t dim0 = 64, dim1 = 2;
};

INSTANTIATE_TEST_SUITE_P(type_prop,
                         ExtractImagesPatchesV3Test,
                         Combine(Values(op::PadType::VALID, op::PadType::SAME_LOWER),
                                 Range<size_t>(1, 11),
                                 Range<size_t>(1, 11),
                                 Range<size_t>(1, 11)),
                         PrintToStringParamName());

TEST_P(ExtractImagesPatchesV3Test, check_calc_padding) {
    auto data = make_shared<op::v0::Parameter>(element::i32, PartialShape{dim0, dim1, 100, 100});
    auto extractimagepatches = make_shared<op::v3::ExtractImagePatches>(data, sizes, strides, rates, pad_type);

    EXPECT_EQ(extractimagepatches->get_output_element_type(0), element::i32);
    EXPECT_EQ(extractimagepatches->get_output_partial_shape(0), exp_shape());
}
