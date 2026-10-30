// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/util/multi_subgraph_base.hpp"
#include "openvino/reference/if.hpp"
#include "openvino/reference/loop.hpp"
#include "openvino/reference/tensor_iterator.hpp"
#include "openvino/runtime/tensor.hpp"

using namespace ov;
using ov::op::util::MultiSubGraphOp;

namespace {
ov::Tensor make_tensor(const element::Type& type, const Shape& shape, float value) {
    ov::Tensor t(type, shape);
    t.data<float>()[0] = value;
    return t;
}

ov::Tensor make_bool_tensor(bool value) {
    ov::Tensor t(element::boolean, Shape{1});
    t.data<bool>()[0] = value;
    return t;
}

ov::Tensor make_trip_count_tensor(int64_t value) {
    ov::Tensor t(element::i64, Shape{1});
    t.data<int64_t>()[0] = value;
    return t;
}
}  // namespace

TEST(IfReferenceSecurityTest, body_parameter_index_beyond_descriptor_count_does_not_overflow) {
    ParameterVector then_params;
    for (int i = 0; i < 5; ++i)
        then_params.push_back(std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic()));
    auto then_body = std::make_shared<Model>(OutputVector{then_params[4]}, then_params);

    auto else_param = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto else_body = std::make_shared<Model>(OutputVector{else_param}, ParameterVector{else_param});

    MultiSubGraphOp::MultiSubgraphInputDescriptionVector then_inputs{
        std::make_shared<MultiSubGraphOp::InvariantInputDescription>(1, 4)};
    MultiSubGraphOp::MultiSubgraphInputDescriptionVector else_inputs{
        std::make_shared<MultiSubGraphOp::InvariantInputDescription>(1, 0)};
    MultiSubGraphOp::MultiSubgraphOutputDescriptionVector then_outputs{
        std::make_shared<MultiSubGraphOp::BodyOutputDescription>(0, 0)};
    MultiSubGraphOp::MultiSubgraphOutputDescriptionVector else_outputs{
        std::make_shared<MultiSubGraphOp::BodyOutputDescription>(0, 0)};

    ov::TensorVector args{make_bool_tensor(true), make_tensor(element::f32, Shape{1}, 42.0f)};
    ov::TensorVector out{ov::Tensor(element::f32, Shape{1})};

    OV_ASSERT_NO_THROW(ov::reference::if_reference({then_body, else_body},
                                                   {then_outputs, else_outputs},
                                                   {then_inputs, else_inputs},
                                                   out,
                                                   args));
    EXPECT_EQ(out[0].data<float>()[0], 42.0f);
}

TEST(IfReferenceSecurityTest, body_parameter_index_beyond_body_size_throws) {
    auto then_param = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto then_body = std::make_shared<Model>(OutputVector{then_param}, ParameterVector{then_param});
    auto else_param = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto else_body = std::make_shared<Model>(OutputVector{else_param}, ParameterVector{else_param});

    MultiSubGraphOp::MultiSubgraphInputDescriptionVector then_inputs{
        std::make_shared<MultiSubGraphOp::InvariantInputDescription>(1, 7)};
    MultiSubGraphOp::MultiSubgraphInputDescriptionVector else_inputs{
        std::make_shared<MultiSubGraphOp::InvariantInputDescription>(1, 0)};
    MultiSubGraphOp::MultiSubgraphOutputDescriptionVector then_outputs{
        std::make_shared<MultiSubGraphOp::BodyOutputDescription>(0, 0)};
    MultiSubGraphOp::MultiSubgraphOutputDescriptionVector else_outputs{
        std::make_shared<MultiSubGraphOp::BodyOutputDescription>(0, 0)};

    ov::TensorVector args{make_bool_tensor(true), make_tensor(element::f32, Shape{1}, 1.0f)};
    ov::TensorVector out{ov::Tensor(element::f32, Shape{1})};

    OV_EXPECT_THROW_HAS_SUBSTRING(ov::reference::if_reference({then_body, else_body},
                                                              {then_outputs, else_outputs},
                                                              {then_inputs, else_inputs},
                                                              out,
                                                              args),
                                  ov::Exception,
                                  "If body has not parameter");
}

TEST(TensorIteratorReferenceSecurityTest, body_parameter_index_beyond_descriptor_count_does_not_overflow) {
    ParameterVector params;
    for (int i = 0; i < 4; ++i)
        params.push_back(std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic()));
    auto body = std::make_shared<Model>(OutputVector{params[3]}, params);

    op::util::InputDescriptionVector input_descs{std::make_shared<MultiSubGraphOp::InvariantInputDescription>(0, 3)};
    op::util::OutputDescriptionVector output_descs{std::make_shared<MultiSubGraphOp::BodyOutputDescription>(0, 0)};

    ov::TensorVector args{make_tensor(element::f32, Shape{1}, 7.0f)};
    ov::TensorVector out{ov::Tensor(element::f32, Shape{1})};

    OV_ASSERT_NO_THROW(ov::reference::tensor_iterator(1, body, output_descs, input_descs, out, args));
    EXPECT_EQ(out[0].data<float>()[0], 7.0f);
}

TEST(TensorIteratorReferenceSecurityTest, body_parameter_index_beyond_body_size_throws) {
    auto param = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto body = std::make_shared<Model>(OutputVector{param}, ParameterVector{param});

    op::util::InputDescriptionVector input_descs{std::make_shared<MultiSubGraphOp::InvariantInputDescription>(0, 7)};
    op::util::OutputDescriptionVector output_descs{std::make_shared<MultiSubGraphOp::BodyOutputDescription>(0, 0)};

    ov::TensorVector args{make_tensor(element::f32, Shape{1}, 1.0f)};
    ov::TensorVector out{ov::Tensor(element::f32, Shape{1})};

    OV_EXPECT_THROW_HAS_SUBSTRING(ov::reference::tensor_iterator(1, body, output_descs, input_descs, out, args),
                                  ov::Exception,
                                  "TensorIterator/Loop body has not parameter");
}

TEST(TensorIteratorReferenceSecurityTest, zero_iterations_throws_instead_of_reading_empty_body_outputs) {
    auto param = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto body = std::make_shared<Model>(OutputVector{param}, ParameterVector{param});

    op::util::InputDescriptionVector input_descs{std::make_shared<MultiSubGraphOp::InvariantInputDescription>(0, 0)};
    op::util::OutputDescriptionVector output_descs{std::make_shared<MultiSubGraphOp::BodyOutputDescription>(0, 0)};

    ov::TensorVector args{make_tensor(element::f32, Shape{1}, 1.0f)};
    ov::TensorVector out{ov::Tensor(element::f32, Shape{1})};

    OV_EXPECT_THROW_HAS_SUBSTRING(ov::reference::tensor_iterator(0, body, output_descs, input_descs, out, args),
                                  ov::Exception,
                                  "produced no output");
}

TEST(LoopReferenceSecurityTest, body_parameter_index_beyond_body_size_throws) {
    auto param = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto body = std::make_shared<Model>(OutputVector{param}, ParameterVector{param});

    // input index 0 is trip_count, 1 is execution_condition; body inputs start at 2.
    op::util::InputDescriptionVector input_descs{std::make_shared<MultiSubGraphOp::InvariantInputDescription>(2, 7)};
    op::util::OutputDescriptionVector output_descs{std::make_shared<MultiSubGraphOp::BodyOutputDescription>(0, 0)};

    ov::TensorVector args{make_trip_count_tensor(1), make_bool_tensor(true), make_tensor(element::f32, Shape{1}, 1.0f)};
    ov::TensorVector out{ov::Tensor(element::f32, Shape{1})};

    OV_EXPECT_THROW_HAS_SUBSTRING(
        ov::reference::loop(body, output_descs, input_descs, op::v5::Loop::SpecialBodyPorts{-1, 0}, out, args, {}),
        ov::Exception,
        "TensorIterator/Loop body has not parameter");
}

TEST(LoopReferenceSecurityTest, slice_input_zero_length_axis_throws) {
    auto param = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto body = std::make_shared<Model>(OutputVector{param}, ParameterVector{param});

    // args[0] is trip_count and args[1] is execution_condition (hardcoded by
    // reference::loop, not described by any InputDescription); the
    // SliceInputDescription below points its m_input_index at args[2], which has
    // zero length (Shape{0}) along the axis (0) it is sliced on.
    op::util::InputDescriptionVector input_descs{
        std::make_shared<MultiSubGraphOp::SliceInputDescription>(2, 0, 0, 1, 1, -1, 0)};
    op::util::OutputDescriptionVector output_descs{std::make_shared<MultiSubGraphOp::BodyOutputDescription>(0, 0)};

    ov::TensorVector args{make_trip_count_tensor(1), make_bool_tensor(true), ov::Tensor(element::f32, Shape{0})};
    ov::TensorVector out{ov::Tensor(element::f32, Shape{1})};

    OV_EXPECT_THROW_HAS_SUBSTRING(
        ov::reference::loop(body, output_descs, input_descs, op::v5::Loop::SpecialBodyPorts{-1, 0}, out, args, {}),
        ov::Exception,
        "zero length along the slice axis");
}
