// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_test_utils.hpp"

#include "openvino/core/model.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/add.hpp"
#include "intel_gpu/op/fully_connected.hpp"
#include "intel_gpu/op/fully_connected_compressed.hpp"
#include "intel_gpu/op/placeholder.hpp"

#include "plugin/transformations/convert_fc_to_compressed.hpp"

#include <memory>

using namespace testing;
using namespace ov::intel_gpu;

namespace ov {
namespace test {
namespace intel_gpu {

TEST_F(TransformationTestsF, ConvertFCToCompressed1) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 32, 16 }, { 1 });
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f32);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 32, 1 }, { 1 });
        auto scale = std::make_shared<ov::op::v1::Multiply>(convert, scale_const);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, scale, no_bias);

        model = std::make_shared<ov::Model>(ov::NodeVector{ fc }, ov::ParameterVector{ input1 });
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 32, 16 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 32, 1 }, { 1 });
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input1, weights_const, no_bias, scale_const);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{ fc_compressed }, ov::ParameterVector{ input1 });
    }
}

TEST_F(TransformationTestsF, ConvertFCToCompressed2) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 32, 16 }, { 1 });
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f32);
        auto zp_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 32, 1 }, { 1 });
        auto sub = std::make_shared<ov::op::v1::Subtract>(convert, zp_const);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 32, 1 }, { 1 });
        auto scale = std::make_shared<ov::op::v1::Multiply>(sub, scale_const);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, scale, no_bias);

        model = std::make_shared<ov::Model>(ov::NodeVector{ fc }, ov::ParameterVector{ input1 });
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 32, 16 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 32, 1 }, { 1 });
        auto zp_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 32, 1 }, { 1 });
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input1, weights_const, no_bias, scale_const, zp_const);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{ fc_compressed }, ov::ParameterVector{ input1 });
    }
}

TEST_F(TransformationTestsF, ConvertFCToCompressed3) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 32, 4, 4 }, { 1 });
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f32);
        auto zp_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 32, 4, 1 }, { 1 });
        auto sub = std::make_shared<ov::op::v1::Subtract>(convert, zp_const);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 32, 4, 1 }, { 1 });
        auto scale = std::make_shared<ov::op::v1::Multiply>(sub, scale_const);
        auto reshape_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 2 }, { -1, 16 });
        auto reshape = std::make_shared<ov::op::v1::Reshape>(scale, reshape_const, false);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, reshape, no_bias);

        model = std::make_shared<ov::Model>(ov::NodeVector{ fc }, ov::ParameterVector{ input1 });
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 32, 16 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 32, 4 }, { 1 });
        auto zp_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 32, 4 }, { 1 });
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input1, weights_const, no_bias, scale_const, zp_const);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{ fc_compressed }, ov::ParameterVector{ input1 });
    }
}

TEST_F(TransformationTestsF, ConvertFCToCompressed4) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{ 32, 4, 4 }, { 1 });
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f32);
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 1, 1, 1 }, { 1 });
        auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_const, ov::element::f32);
        auto sub = std::make_shared<ov::op::v1::Subtract>(convert, zp_convert);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 32, 4, 1 }, { 1 });
        auto scale = std::make_shared<ov::op::v1::Multiply>(sub, scale_const);
        auto reshape_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 2 }, { -1, 16 });
        auto reshape = std::make_shared<ov::op::v1::Reshape>(scale, reshape_const, false);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, reshape, no_bias);

        model = std::make_shared<ov::Model>(ov::NodeVector{ fc }, ov::ParameterVector{ input1 });
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{ 32, 16 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 32, 4 }, { 1 });
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 1, 1 }, { 1 });
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input1, weights_const, no_bias, scale_const, zp_const);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{ fc_compressed }, ov::ParameterVector{ input1 });
    }
}

TEST_F(TransformationTestsF, ConvertFCToCompressed5) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{ 4, 4, 32 }, { 1 });
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f32);
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 1, 1, 1 }, { 1 });
        auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_const, ov::element::f32);
        auto sub = std::make_shared<ov::op::v1::Subtract>(convert, zp_convert);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 4, 1, 32 }, { 1 });
        auto scale = std::make_shared<ov::op::v1::Multiply>(sub, scale_const);
        auto reshape_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 2 }, { 16, -1 });
        auto reshape = std::make_shared<ov::op::v1::Reshape>(scale, reshape_const, false);
        auto transpose_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 2 }, { 1, 0 });
        auto transpose = std::make_shared<ov::op::v1::Transpose>(reshape, transpose_const);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, transpose, no_bias);

        model = std::make_shared<ov::Model>(ov::NodeVector{ fc }, ov::ParameterVector{ input1 });
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{ 16, 32 }, { 1 });
        auto transpose_weights_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 2 }, { 1, 0 });
        auto transpose_weights = std::make_shared<ov::op::v1::Transpose>(weights_const, transpose_weights_const);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 4, 32 }, { 1 });
        auto transpose_scale_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 2 }, { 1, 0 });
        auto transpose_scale = std::make_shared<ov::op::v1::Transpose>(scale_const, transpose_scale_const);
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 1, 1 }, { 1 });
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input1, transpose_weights, no_bias, transpose_scale, zp_const);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{ fc_compressed }, ov::ParameterVector{ input1 });
    }
}

TEST_F(TransformationTestsF, ConvertFCToCompressed6) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{ 4, 4, 32 }, { 1 });
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f32);
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 4, 1, 32 }, { 1 });
        auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_const, ov::element::f32);
        auto sub = std::make_shared<ov::op::v1::Subtract>(convert, zp_convert);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 4, 1, 32 }, { 1 });
        auto scale = std::make_shared<ov::op::v1::Multiply>(sub, scale_const);
        auto reshape_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 2 }, { 16, -1 });
        auto reshape = std::make_shared<ov::op::v1::Reshape>(scale, reshape_const, false);
        auto transpose_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 2 }, { 1, 0 });
        auto transpose = std::make_shared<ov::op::v1::Transpose>(reshape, transpose_const);
	auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, transpose, no_bias);

        model = std::make_shared<ov::Model>(ov::NodeVector{ fc }, ov::ParameterVector{ input1 });
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{ 16, 32 }, { 1 });
        auto transpose_weights_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 2 }, { 1, 0 });
        auto transpose_weights = std::make_shared<ov::op::v1::Transpose>(weights_const, transpose_weights_const);
	auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 4, 32 }, { 1 });
        auto transpose_scale_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 2 }, { 1, 0 });
        auto transpose_scale = std::make_shared<ov::op::v1::Transpose>(scale_const, transpose_scale_const);
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 4, 32 }, { 1 });
        auto transpose_zp_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 2 }, { 1, 0 });
        auto transpose_zp = std::make_shared<ov::op::v1::Transpose>(zp_const, transpose_zp_const);
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input1, transpose_weights, no_bias, transpose_scale, transpose_zp);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{ fc_compressed }, ov::ParameterVector{ input1 });
    }
}

TEST_F(TransformationTestsF, ConvertFCToCompressed7) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{ 4, 4, 32 }, { 1 });
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f16);
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 4, 1, 32 }, { 1 });
        auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_const, ov::element::f16);
        auto sub = std::make_shared<ov::op::v1::Subtract>(convert, zp_convert);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 4, 1, 32 }, { 1 });
        auto scale = std::make_shared<ov::op::v1::Multiply>(sub, scale_const);
        auto reshape_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 2 }, { 16, -1 });
        auto reshape = std::make_shared<ov::op::v1::Reshape>(scale, reshape_const, false);
        auto transpose_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 2 }, { 1, 0 });
        auto transpose = std::make_shared<ov::op::v1::Transpose>(reshape, transpose_const);
	auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, transpose, no_bias);

        model = std::make_shared<ov::Model>(ov::NodeVector{ fc }, ov::ParameterVector{ input1 });
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{ 16, 32 }, { 1 });
        auto transpose_weights_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 2 }, { 1, 0 });
        auto transpose_weights = std::make_shared<ov::op::v1::Transpose>(weights_const, transpose_weights_const);
	auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 4, 32 }, { 1 });
        auto transpose_scale_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 2 }, { 1, 0 });
        auto transpose_scale = std::make_shared<ov::op::v1::Transpose>(scale_const, transpose_scale_const);
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 4, 32 }, { 1 });
        auto transpose_zp_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 2 }, { 1, 0 });
        auto transpose_zp = std::make_shared<ov::op::v1::Transpose>(zp_const, transpose_zp_const);
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input1, transpose_weights, no_bias, transpose_scale, transpose_zp);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{ fc_compressed }, ov::ParameterVector{ input1 });
    }
}

class TestSubgraph : public ov::op::util::SubGraphOp {
public:
    OPENVINO_OP("TestSubgraph", "Test", ov::op::util::SubGraphOp);

    TestSubgraph() = default;

    TestSubgraph(const OutputVector& args, const std::shared_ptr<ov::Model>& body);

    TestSubgraph(const NodeVector& args, const std::shared_ptr<ov::Model>& body);

    bool visit_attributes(AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override;

private:
    const ov::Model& body() const {
        return *m_bodies[0];
    }
    ov::Model& body() {
        return *m_bodies[0];
    }
    const std::shared_ptr<ov::Model>& body_ptr() const {
        return m_bodies[0];
    }
    std::shared_ptr<ov::Model>& body_ptr() {
        return m_bodies[0];
    }

};

TestSubgraph::TestSubgraph(const ov::OutputVector& args, const std::shared_ptr<ov::Model>& body)
    : SubGraphOp(args) {
    SubGraphOp::set_function(body);
    constructor_validate_and_infer_types();
    for (size_t i = 0; i < body->get_parameters().size(); ++i)
        m_input_descriptions[0].push_back(std::make_shared<InvariantInputDescription>(i, i));
    for (size_t i = 0; i < body->get_output_size(); ++i)
        m_output_descriptions[0].push_back(std::make_shared<BodyOutputDescription>(i, i));
}

TestSubgraph::TestSubgraph(const ov::NodeVector& args, const std::shared_ptr<ov::Model>& body)
    : TestSubgraph(as_output_vector(args), body) {}

std::shared_ptr<ov::Node> TestSubgraph::clone_with_new_inputs(const ov::OutputVector& inputs) const {
    return std::make_shared<TestSubgraph>(inputs, body().clone());
}

void TestSubgraph::validate_and_infer_types() {
    ov::ParameterVector old_parameters;
    for (auto op : body_ptr()->get_parameters()) {
        old_parameters.push_back(op);
    }

    for (size_t i = 0; i < get_input_size(); ++i) {
        body_ptr()->replace_parameter(
            i,
            std::make_shared<ov::op::v0::Parameter>(get_input_element_type(i), get_input_partial_shape(i)));
    }

    body_ptr()->validate_nodes_and_infer_types();

    for (size_t i = 0; i < body_ptr()->get_parameters().size(); i++) {
        body_ptr()->get_parameters()[i]->set_friendly_name(old_parameters[i]->get_friendly_name());
    }

    set_output_size(body_ptr()->get_output_size());
    for (size_t i = 0; i < get_output_size(); ++i) {
        set_output_type(i, body_ptr()->get_output_element_type(i), body_ptr()->get_output_partial_shape(i));
    }
}

bool TestSubgraph::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("body", body_ptr());
    visitor.on_attribute("input_descriptions", m_input_descriptions[0]);
    visitor.on_attribute("output_descriptions", m_output_descriptions[0]);
    return true;
}

TEST_F(TransformationTestsF, ConvertFCToCompressed8) {
    {
        auto weights_const = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{ 4, 4, 32 }, { 1 });
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f16);
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 4, 1, 32 }, { 1 });
        auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_const, ov::element::f16);
        auto sub = std::make_shared<ov::op::v1::Subtract>(convert, zp_convert);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 4, 1, 32 }, { 1 });
        auto scale = std::make_shared<ov::op::v1::Multiply>(sub, scale_const);
        auto reshape_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 2 }, { 16, -1 });
        auto reshape = std::make_shared<ov::op::v1::Reshape>(scale, reshape_const, false);
        auto transpose_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 2 }, { 1, 0 });
        auto transpose = std::make_shared<ov::op::v1::Transpose>(reshape, transpose_const);
	auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();

        auto param1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 15});
        auto const_value1 = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1, 1}, {1});
        auto param2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 16});
        auto const_value2 = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1, 1}, {1});
        auto add1 = std::make_shared<ov::op::v1::Add>(param1, const_value1);
        auto add2 = std::make_shared<ov::op::v1::Add>(param2, const_value2);
        auto result1 = std::make_shared<ov::op::v0::Result>(add1);
        auto result2 = std::make_shared<ov::op::v0::Result>(add2);
        auto submodel = std::make_shared<ov::Model>(ov::ResultVector{result1, result2}, ov::ParameterVector{param1, param2});
        ParameterVector subgraph_parameters{submodel->inputs().size()};
        OutputVector args{submodel->inputs().size()};
        for (size_t i = 0; i < submodel->inputs().size(); i++) {
            auto const& input = submodel->input(i);
            subgraph_parameters[i] =
                std::make_shared<ov::op::v0::Parameter>(input.get_element_type(), input.get_partial_shape());
            args[i] = subgraph_parameters[i]->output(0);
        }
        auto subgraph_op = std::make_shared<TestSubgraph>(args, submodel);
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(subgraph_op->output(1), transpose, no_bias);

        model = std::make_shared<ov::Model>(ov::NodeVector{std::make_shared<ov::op::v0::Result>(subgraph_op->output(0)), fc}, subgraph_parameters);
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto weights_const = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{ 16, 32 }, { 1 });
        auto transpose_weights_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 2 }, { 1, 0 });
        auto transpose_weights = std::make_shared<ov::op::v1::Transpose>(weights_const, transpose_weights_const);
	auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 4, 32 }, { 1 });
        auto transpose_scale_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 2 }, { 1, 0 });
        auto transpose_scale = std::make_shared<ov::op::v1::Transpose>(scale_const, transpose_scale_const);
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 4, 32 }, { 1 });
        auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_const, ov::element::f16);
        auto transpose_zp_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 2 }, { 1, 0 });
        auto transpose_zp = std::make_shared<ov::op::v1::Transpose>(zp_const, transpose_zp_const);

        auto param1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 15});
        auto const_value1 = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1, 1}, {1});
        auto param2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 16});
        auto const_value2 = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1, 1}, {1});
        auto add1 = std::make_shared<ov::op::v1::Add>(param1, const_value1);
        auto add2 = std::make_shared<ov::op::v1::Add>(param2, const_value2);
        auto result1 = std::make_shared<ov::op::v0::Result>(add1);
        auto result2 = std::make_shared<ov::op::v0::Result>(add2);
        auto submodel = std::make_shared<ov::Model>(ov::ResultVector{result1, result2}, ov::ParameterVector{param1, param2});
        ParameterVector subgraph_parameters{submodel->inputs().size()};
        OutputVector args{submodel->inputs().size()};
        for (size_t i = 0; i < submodel->inputs().size(); i++) {
            auto const& input = submodel->input(i);
            subgraph_parameters[i] =
                std::make_shared<ov::op::v0::Parameter>(input.get_element_type(), input.get_partial_shape());
            args[i] = subgraph_parameters[i]->output(0);
        }
        auto subgraph_op = std::make_shared<TestSubgraph>(args, submodel);
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(subgraph_op->output(1), transpose_weights, no_bias, transpose_scale, transpose_zp);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{ std::make_shared<ov::op::v0::Result>(subgraph_op->output(0)), fc_compressed }, subgraph_parameters);
    }
}

TEST_F(TransformationTestsF, ConvertFCToCompressed9) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 32, 16 }, { 1 });
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f16);
        auto zp_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 1 }, { 1 });
        auto sub = std::make_shared<ov::op::v1::Subtract>(convert, zp_const);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 1 }, { 1 });
        auto scale = std::make_shared<ov::op::v1::Multiply>(sub, scale_const);
	    auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, scale, no_bias);

        model = std::make_shared<ov::Model>(ov::NodeVector{ fc }, ov::ParameterVector{ input1 });
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 32, 16 }, { 1 });
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 1 }, { 1 });
        auto zp_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 1 }, { 1 });
	    auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input1, weights_const, no_bias, scale_const, zp_const);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{ fc_compressed }, ov::ParameterVector{ input1 });
    }
}

TEST_F(TransformationTestsF, ConvertFCToCompressed10) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 32, 16 }, { 1 });
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f16);
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 32, 1 }, { 1 });
        auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_const, ov::element::f16);
        auto sub = std::make_shared<ov::op::v1::Subtract>(convert, zp_convert);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 1 }, { 1 });
        auto scale = std::make_shared<ov::op::v1::Multiply>(sub, scale_const);
	    auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, scale, no_bias);

        model = std::make_shared<ov::Model>(ov::NodeVector{ fc }, ov::ParameterVector{ input1 });
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 32, 16 }, { 1 });
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 1 }, { 1 });
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 32, 1 }, { 1 });
	    auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input1, weights_const, no_bias, scale_const, zp_const);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{ fc_compressed }, ov::ParameterVector{ input1 });
    }
}

}  // namespace intel_gpu
}  // namespace test
}  // namespace ov
