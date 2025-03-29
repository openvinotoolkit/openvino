// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/opsets/opset11.hpp"
#include "openvino/opsets/opset14.hpp"

using namespace std;
using namespace ov;
using namespace ov::opset11;
using namespace testing;

template <typename TOp>
class ROIAlignTest : public testing::Test {
protected:
    std::shared_ptr<TOp> make_roi_op(const Output<Node>& input,
                                     const Output<Node>& rois,
                                     const Output<Node>& batch_indices,
                                     const int pooled_h,
                                     const int pooled_w) {
        auto ret = std::make_shared<TOp>();

        ret->set_arguments(OutputVector{input, rois, batch_indices});
        ret->set_pooled_h(pooled_h);
        ret->set_pooled_w(pooled_w);
        ret->validate_and_infer_types();

        return ret;
    }

    ov::Dimension::value_type GetROISecondDimSizeForOp() const {
        // Those magic numbers comes from definition of ROIAlign ops.
        if (std::is_same<TOp, op::v15::ROIAlignRotated>::value)
            return 5;
        return 4;
    }
};

TYPED_TEST_SUITE_P(ROIAlignTest);

TYPED_TEST_P(ROIAlignTest, default_ctor) {
    const size_t second_roi_dim = static_cast<size_t>(this->GetROISecondDimSizeForOp());
    const auto data = make_shared<Parameter>(element::f32, PartialShape{2, 3, 5, 5});
    const auto rois = make_shared<Parameter>(element::f32, Shape{7, second_roi_dim});
    const auto batch_indices = make_shared<Parameter>(element::i32, Shape{7});

    const auto op = this->make_roi_op(data, rois, batch_indices, 2, 2);

    EXPECT_EQ(op->get_input_size(), 3);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({7, 3, 2, 2}));
}

TYPED_TEST_P(ROIAlignTest, simple_shape_inference) {
    auto data_shape = PartialShape{2, 3, 5, 5};
    auto rois_shape = PartialShape{7, this->GetROISecondDimSizeForOp()};
    auto batch_shape = PartialShape{7};

    auto data_symbols = set_shape_symbols(data_shape);
    set_shape_symbols(rois_shape);
    auto batch_symbols = set_shape_symbols(batch_shape);

    const auto data = make_shared<Parameter>(element::f16, data_shape);
    const auto rois = make_shared<Parameter>(element::f16, rois_shape);
    const auto batch_indices = make_shared<Parameter>(element::i16, batch_shape);

    const auto op = this->make_roi_op(data, rois, batch_indices, 2, 2);

    EXPECT_EQ(op->get_element_type(), element::f16);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({7, 3, 2, 2}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)),
                ElementsAre(batch_symbols[0], data_symbols[1], nullptr, nullptr));
}

TYPED_TEST_P(ROIAlignTest, dynamic_channels_dim) {
    auto data_shape = PartialShape{10, -1, 5, 5};
    auto rois_shape = PartialShape{7, this->GetROISecondDimSizeForOp()};
    auto batch_shape = PartialShape{7};

    auto data_symbols = set_shape_symbols(data_shape);
    auto batch_symbols = set_shape_symbols(batch_shape);

    const auto data = make_shared<Parameter>(element::f64, data_shape);
    const auto rois = make_shared<Parameter>(element::f64, rois_shape);
    const auto batch_indices = make_shared<Parameter>(element::i64, batch_shape);

    const auto op = this->make_roi_op(data, rois, batch_indices, 3, 4);

    EXPECT_EQ(op->get_element_type(), element::f64);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({7, -1, 3, 4}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)),
                ElementsAre(batch_symbols[0], data_symbols[1], nullptr, nullptr));
}

TYPED_TEST_P(ROIAlignTest, num_rois_from_batch_indices) {
    const auto data = make_shared<Parameter>(element::f32, PartialShape{10, 3, 5, 5});
    const auto rois = make_shared<Parameter>(element::f32, PartialShape::dynamic(2));
    const auto batch_indices = make_shared<Parameter>(element::i32, PartialShape{9});

    const auto op = this->make_roi_op(data, rois, batch_indices, 4, 2);

    EXPECT_EQ(op->get_element_type(), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({9, 3, 4, 2}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), Each(nullptr));
}

TYPED_TEST_P(ROIAlignTest, all_inputs_dynamic_rank) {
    const auto data = make_shared<Parameter>(element::bf16, PartialShape::dynamic());
    const auto rois = make_shared<Parameter>(element::bf16, PartialShape::dynamic());
    const auto batch_indices = make_shared<Parameter>(element::i8, PartialShape::dynamic());

    const auto op = this->make_roi_op(data, rois, batch_indices, 40, 12);

    EXPECT_EQ(op->get_element_type(), element::bf16);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({-1, -1, 40, 12}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), Each(nullptr));
}

TYPED_TEST_P(ROIAlignTest, all_inputs_static_rank_dynamic_dims) {
    auto data_shape = PartialShape::dynamic(4);
    auto rois_shape = PartialShape::dynamic(2);
    auto batch_shape = PartialShape::dynamic(1);

    auto data_symbols = set_shape_symbols(data_shape);
    set_shape_symbols(rois_shape);
    auto batch_symbols = set_shape_symbols(batch_shape);

    const auto data = make_shared<Parameter>(element::f16, data_shape);
    const auto rois = make_shared<Parameter>(element::f16, rois_shape);
    const auto batch_indices = make_shared<Parameter>(element::u16, batch_shape);

    const auto op = this->make_roi_op(data, rois, batch_indices, 8, 8);

    EXPECT_EQ(op->get_element_type(), element::f16);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({-1, -1, 8, 8}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)),
                ElementsAre(batch_symbols[0], data_symbols[1], nullptr, nullptr));
}

TYPED_TEST_P(ROIAlignTest, interval_shapes) {
    auto data_shape = PartialShape{{5, 10}, {2, 4}, {0, 100}, {10, -1}};
    auto rois_shape = PartialShape{{2, 10}, {2, -1}};
    auto batch_shape = PartialShape{{3, 6}};

    auto data_symbols = set_shape_symbols(data_shape);
    set_shape_symbols(rois_shape);
    auto batch_symbols = set_shape_symbols(batch_shape);

    const auto data = make_shared<Parameter>(element::f32, data_shape);
    const auto rois = make_shared<Parameter>(element::f32, rois_shape);
    const auto batch_indices = make_shared<Parameter>(element::u64, batch_shape);

    const auto op = this->make_roi_op(data, rois, batch_indices, 8, 18);

    EXPECT_EQ(op->get_element_type(), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({{3, 6}, {2, 4}, 8, 18}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)),
                ElementsAre(batch_symbols[0], data_symbols[1], nullptr, nullptr));
}

TYPED_TEST_P(ROIAlignTest, incompatible_num_rois) {
    const auto data = make_shared<Parameter>(element::f32, PartialShape{10, 3, 5, 5});
    const auto rois = make_shared<Parameter>(element::f32, PartialShape{1, -1});
    const auto batch_indices = make_shared<Parameter>(element::i32, PartialShape{9});

    OV_EXPECT_THROW(
        std::ignore = this->make_roi_op(data, rois, batch_indices, 8, 8),
        NodeValidationFailure,
        HasSubstr("The first dimension of ROIs input must be equal to the first dimension of the batch indices input"));
}

TYPED_TEST_P(ROIAlignTest, incompatible_input_rank) {
    const auto data = make_shared<Parameter>(element::f32, PartialShape{10, 3, 5, 5, 5});
    const auto rois = make_shared<Parameter>(element::f32, PartialShape{1, -1});
    const auto batch_indices = make_shared<Parameter>(element::i32, PartialShape{9});

    OV_EXPECT_THROW(std::ignore = this->make_roi_op(data, rois, batch_indices, 8, 8),
                    NodeValidationFailure,
                    HasSubstr("Expected a 4D tensor for the input data"));
}

TYPED_TEST_P(ROIAlignTest, incompatible_rois_rank) {
    const auto data = make_shared<Parameter>(element::f32, PartialShape{10, 3, 5, 5});
    const auto rois = make_shared<Parameter>(element::f32, PartialShape{2});
    const auto batch_indices = make_shared<Parameter>(element::i32, PartialShape{9});

    OV_EXPECT_THROW(std::ignore = this->make_roi_op(data, rois, batch_indices, 8, 8),
                    NodeValidationFailure,
                    HasSubstr("Expected a 2D tensor for the ROIs input"));
}

TYPED_TEST_P(ROIAlignTest, incompatible_batch_indicies_rank) {
    const auto data = make_shared<Parameter>(element::f32, PartialShape{10, 3, 5, 5});
    const auto rois = make_shared<Parameter>(element::f32, PartialShape{2, this->GetROISecondDimSizeForOp()});
    const auto batch_indices = make_shared<Parameter>(element::i32, PartialShape{2, 1});

    OV_EXPECT_THROW(std::ignore = this->make_roi_op(data, rois, batch_indices, 8, 8),
                    NodeValidationFailure,
                    HasSubstr("Expected a 1D tensor for the batch indices input."));
}

TYPED_TEST_P(ROIAlignTest, incompatible_rois_2nd_dim) {
    const auto data = make_shared<Parameter>(element::f32, PartialShape{10, 3, 5, 5});
    const auto rois = make_shared<Parameter>(element::f32, PartialShape{2, {7, -1}});
    const auto batch_indices = make_shared<Parameter>(element::i32, PartialShape{2});

    OV_EXPECT_THROW(std::ignore = this->make_roi_op(data, rois, batch_indices, 8, 8),
                    NodeValidationFailure,
                    HasSubstr("op dimension is expected to be equal to"));
}

TYPED_TEST_P(ROIAlignTest, incompatible_1st_dim_of_rois_and_batch) {
    const auto data = make_shared<Parameter>(element::f32, PartialShape{10, 3, 5, 5});
    const auto rois = make_shared<Parameter>(element::f32, PartialShape{{11, -1}, this->GetROISecondDimSizeForOp()});
    const auto batch_indices = make_shared<Parameter>(element::i32, PartialShape{{0, 10}});

    OV_EXPECT_THROW(
        std::ignore = this->make_roi_op(data, rois, batch_indices, 8, 8),
        NodeValidationFailure,
        HasSubstr("The first dimension of ROIs input must be equal to the first dimension of the batch indices input"));
}

TYPED_TEST_P(ROIAlignTest, data_not_floating_point) {
    const auto data = make_shared<Parameter>(element::i32, PartialShape{10, 3, 5, 5});
    const auto rois = make_shared<Parameter>(element::f32, PartialShape{8, 4});
    const auto batch_indices = make_shared<Parameter>(element::i32, PartialShape{8});

    OV_EXPECT_THROW(std::ignore = this->make_roi_op(data, rois, batch_indices, 8, 8),
                    NodeValidationFailure,
                    HasSubstr("The data type for input and ROIs is expected to be a same floating point type"));
}

TYPED_TEST_P(ROIAlignTest, rois_not_floating_point) {
    const auto data = make_shared<Parameter>(element::f32, PartialShape{10, 3, 5, 5});
    const auto rois = make_shared<Parameter>(element::i32, PartialShape{8, this->GetROISecondDimSizeForOp()});
    const auto batch_indices = make_shared<Parameter>(element::i32, PartialShape{8});

    OV_EXPECT_THROW(std::ignore = this->make_roi_op(data, rois, batch_indices, 8, 8),
                    NodeValidationFailure,
                    HasSubstr("The data type for input and ROIs is expected to be a same floating point type"));
}

TYPED_TEST_P(ROIAlignTest, data_and_rois_not_same_type) {
    const auto data = make_shared<Parameter>(element::f32, PartialShape{10, 3, 5, 5});
    const auto rois = make_shared<Parameter>(element::f16, PartialShape{8, this->GetROISecondDimSizeForOp()});
    const auto batch_indices = make_shared<Parameter>(element::i32, PartialShape{8});

    OV_EXPECT_THROW(std::ignore = this->make_roi_op(data, rois, batch_indices, 8, 8),
                    NodeValidationFailure,
                    HasSubstr("The data type for input and ROIs is expected to be a same floating point type"));
}

TYPED_TEST_P(ROIAlignTest, batch_indicies_not_integer) {
    const auto data = make_shared<Parameter>(element::f32, PartialShape{10, 3, 5, 5});
    const auto rois = make_shared<Parameter>(element::f32, PartialShape{8, this->GetROISecondDimSizeForOp()});
    const auto batch_indices = make_shared<Parameter>(element::f32, PartialShape{8});

    OV_EXPECT_THROW(std::ignore = this->make_roi_op(data, rois, batch_indices, 8, 8),
                    NodeValidationFailure,
                    HasSubstr("The data type for batch indices is expected to be an integer"));
}

REGISTER_TYPED_TEST_SUITE_P(ROIAlignTest,
                            default_ctor,
                            simple_shape_inference,
                            dynamic_channels_dim,
                            num_rois_from_batch_indices,
                            all_inputs_dynamic_rank,
                            all_inputs_static_rank_dynamic_dims,
                            interval_shapes,
                            incompatible_num_rois,
                            incompatible_input_rank,
                            incompatible_rois_rank,
                            incompatible_batch_indicies_rank,
                            incompatible_rois_2nd_dim,
                            incompatible_1st_dim_of_rois_and_batch,
                            data_not_floating_point,
                            rois_not_floating_point,
                            data_and_rois_not_same_type,
                            batch_indicies_not_integer);

typedef Types<op::v3::ROIAlign, op::v9::ROIAlign, op::v15::ROIAlignRotated> ROIAlignTypes;
INSTANTIATE_TYPED_TEST_SUITE_P(type_prop, ROIAlignTest, ROIAlignTypes);
