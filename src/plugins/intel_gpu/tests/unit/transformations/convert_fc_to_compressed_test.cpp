// Copyright (C) 2018-2026 Intel Corporation
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
#include <sstream>
#include <vector>

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

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 32, 16 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 32, 1 }, { 1 });
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input1, weights_const, no_bias, scale_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1});
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

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 32, 16 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 32, 1 }, { 1 });
        auto zp_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 32, 1 }, { 1 });
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input1, weights_const, no_bias, scale_const, zp_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1});
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

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 32, 16 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 32, 4 }, { 1 });
        auto zp_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 32, 4 }, { 1 });
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input1, weights_const, no_bias, scale_const, zp_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertFCToCompressed4) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 1, 4, 4 }, { 1 });
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f32);
        auto zp_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 1, 4, 1 }, { 1 });
        auto sub = std::make_shared<ov::op::v1::Subtract>(convert, zp_const);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 1, 4, 1 }, { 1 });
        auto scale = std::make_shared<ov::op::v1::Multiply>(sub, scale_const);
        auto reshape_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 2 }, { -1, 16 });
        auto reshape = std::make_shared<ov::op::v1::Reshape>(scale, reshape_const, false);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, reshape, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 1, 16 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 1, 4 }, { 1 });
        auto zp_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 1, 4 }, { 1 });
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input1, weights_const, no_bias, scale_const, zp_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertFCToCompressed5) {
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

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{ 32, 16 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 32, 4 }, { 1 });
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 1, 1 }, { 1 });
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input1, weights_const, no_bias, scale_const, zp_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertFCToCompressed6) {
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
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

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{ 16, 32 }, { 1 });
        auto transpose_weights_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 2 }, { 1, 0 });
        auto transpose_weights = std::make_shared<ov::op::v1::Transpose>(weights_const, transpose_weights_const);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 4, 32 }, { 1 });
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 1, 1 }, { 1 });
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(
            input1, transpose_weights, no_bias, scale_const, zp_const, ov::element::dynamic, true, false, false);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertFCToCompressed7) {
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
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

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{ 16, 32 }, { 1 });
        auto transpose_weights_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 2 }, { 1, 0 });
        auto transpose_weights = std::make_shared<ov::op::v1::Transpose>(weights_const, transpose_weights_const);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 4, 32 }, { 1 });
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 4, 32 }, { 1 });
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(
            input1, transpose_weights, no_bias, scale_const, zp_const, ov::element::dynamic, true, false, false);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertFCToCompressed8) {
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
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

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{ 16, 32 }, { 1 });
        auto transpose_weights_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 2 }, { 1, 0 });
        auto transpose_weights = std::make_shared<ov::op::v1::Transpose>(weights_const, transpose_weights_const);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 4, 32 }, { 1 });
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 4, 32 }, { 1 });
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(
            input1, transpose_weights, no_bias, scale_const, zp_const, ov::element::dynamic, true, false, false);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1});
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

TEST_F(TransformationTestsF, ConvertFCToCompressed9) {
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
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

        model = std::make_shared<ov::Model>(ov::OutputVector{std::make_shared<ov::op::v0::Result>(subgraph_op->output(0)), fc}, subgraph_parameters);
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto weights_const = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{ 16, 32 }, { 1 });
        auto transpose_weights_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 2 }, { 1, 0 });
        auto transpose_weights = std::make_shared<ov::op::v1::Transpose>(weights_const, transpose_weights_const);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 4, 32 }, { 1 });
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 4, 32 }, { 1 });
        auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_const, ov::element::f16);

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
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(
            subgraph_op->output(1), transpose_weights, no_bias, scale_const, zp_const, ov::element::dynamic, true, false, false);

        model_ref =
            std::make_shared<ov::Model>(ov::OutputVector{std::make_shared<ov::op::v0::Result>(subgraph_op->output(0)), fc_compressed}, subgraph_parameters);
    }
}

TEST_F(TransformationTestsF, ConvertFCToCompressed10) {
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

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 32, 16 }, { 1 });
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 1 }, { 1 });
        auto zp_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 1 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input1, weights_const, no_bias, scale_const, zp_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertFCToCompressed11) {
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

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 32, 16 }, { 1 });
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 1 }, { 1 });
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 32, 1 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input1, weights_const, no_bias, scale_const, zp_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertFCToCompressed12) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 16});
        auto weights_param = std::make_shared<ov::op::v0::Parameter>(ov::element::u4, ov::Shape{32, 16});
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_param, ov::element::f16);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{32, 1}, {1});
        auto scale = std::make_shared<ov::op::v1::Multiply>(convert, scale_const);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();

        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, scale, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1, weights_param});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 16});
        auto weights_param = std::make_shared<ov::op::v0::Parameter>(ov::element::u4, ov::Shape{32, 16});
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{32, 1}, {1});

        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input1, weights_param, no_bias, scale_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1, weights_param});
    }
}

TEST_F(TransformationTestsF, ConvertFCToCompressed13) {
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 16});
        auto weights_param = std::make_shared<ov::op::v0::Parameter>(ov::element::u4, ov::Shape{4, 4, 32});
        auto reshape_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {16, 32});
        auto reshape = std::make_shared<ov::op::v1::Reshape>(weights_param, reshape_const, false);
        auto convert = std::make_shared<ov::op::v0::Convert>(reshape, ov::element::f16);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1, 32}, {1});
        auto scale = std::make_shared<ov::op::v1::Multiply>(convert, scale_const);
        auto transpose_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {1, 0});
        auto transpose = std::make_shared<ov::op::v1::Transpose>(scale, transpose_const);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();

        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, transpose, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1, weights_param});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 16});
        auto weights_param = std::make_shared<ov::op::v0::Parameter>(ov::element::u4, ov::Shape{4, 4, 32});
        auto reshape_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {16, 32});
        auto reshape = std::make_shared<ov::op::v1::Reshape>(weights_param, reshape_const, false);
        auto transpose_weights_param = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {1, 0});
        auto transpose_weights = std::make_shared<ov::op::v1::Transpose>(reshape, transpose_weights_param);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1, 32}, {1});
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(
            input1, transpose_weights, no_bias, scale_const, ov::element::dynamic, true, false, false);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1, weights_param});
    }
}

TEST_F(TransformationTestsF, ConvertFCToCompressed14) {
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 16});
        auto weights_param = std::make_shared<ov::op::v0::Parameter>(ov::element::u4, ov::Shape{4, 4, 32});
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_param, ov::element::f16);
        auto reshape_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {16, 32});
        auto reshape = std::make_shared<ov::op::v1::Reshape>(convert, reshape_const, false);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1, 32}, {1});
        auto scale = std::make_shared<ov::op::v1::Multiply>(reshape, scale_const);
        auto transpose_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {1, 0});
        auto transpose = std::make_shared<ov::op::v1::Transpose>(scale, transpose_const);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();

        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, transpose, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1, weights_param});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 16});
        auto weights_param = std::make_shared<ov::op::v0::Parameter>(ov::element::u4, ov::Shape{4, 4, 32});
        auto reshape_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {16, 32});
        auto reshape = std::make_shared<ov::op::v1::Reshape>(weights_param, reshape_const, false);
        auto transpose_weights_param = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {1, 0});
        auto transpose_weights = std::make_shared<ov::op::v1::Transpose>(reshape, transpose_weights_param);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1, 32}, {1});
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(
            input1, transpose_weights, no_bias, scale_const, ov::element::dynamic, true, false, false);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1, weights_param});
    }
}

TEST_F(TransformationTestsF, ConvertFCToCompressed15) {
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 16});
        auto weights_param = std::make_shared<ov::op::v0::Parameter>(ov::element::u4, ov::Shape{4, 4, 32});
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_param, ov::element::f16);
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{4, 1, 32}, {1});
        auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_const, ov::element::f16);
        auto sub = std::make_shared<ov::op::v1::Subtract>(convert, zp_convert);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{4, 1, 32}, {1});
        auto scale = std::make_shared<ov::op::v1::Multiply>(sub, scale_const);
        auto reshape_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {16, 32});
        auto reshape = std::make_shared<ov::op::v1::Reshape>(scale, reshape_const, false);
        auto transpose_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {1, 0});
        auto transpose = std::make_shared<ov::op::v1::Transpose>(reshape, transpose_const);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();

        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, transpose, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1, weights_param});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 16});
        auto weights_param = std::make_shared<ov::op::v0::Parameter>(ov::element::u4, ov::Shape{4, 4, 32});
        auto reshape_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {16, 32});
        auto reshape = std::make_shared<ov::op::v1::Reshape>(weights_param, reshape_const, false);
        auto transpose_weights_param = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {1, 0});
        auto transpose_weights = std::make_shared<ov::op::v1::Transpose>(reshape, transpose_weights_param);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{4, 32}, {1});
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{4, 32}, {1});
        auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_const, ov::element::f16);

        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(
            input1, transpose_weights, no_bias, scale_const, zp_const, ov::element::dynamic, true, false, false);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1, weights_param});
    }
}

TEST_F(TransformationTestsF, ConvertFCToCompressed16_i8ZpNotConvertedToU8) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::i8, ov::Shape{ 32, 16 }, { 1 });
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f16);
        auto zp_const = ov::op::v0::Constant::create(ov::element::i8, ov::Shape{ 32, 1 }, { -128 });
        auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_const, ov::element::f16);
        auto sub = std::make_shared<ov::op::v1::Subtract>(convert, zp_convert);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 1 }, { 1 });
        auto scale = std::make_shared<ov::op::v1::Multiply>(sub, scale_const);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, scale, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        // The i8 ZP constant must be preserved as i8, NOT recast to u8.
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::i8, ov::Shape{ 32, 16 }, { 1 });
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 1 }, { 1 });
        auto zp_const = ov::op::v0::Constant::create(ov::element::i8, ov::Shape{ 32, 1 }, { -128 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input1, weights_const, no_bias, scale_const, zp_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertFCToCompressed16_i4ScalarZpConvertedToI8) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::i4, ov::Shape{ 32, 16 }, { 1 });
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f16);
        auto zp_const = ov::op::v0::Constant::create(ov::element::i4, ov::Shape{ 1 }, { -3 });
        auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_const, ov::element::f16);
        auto sub = std::make_shared<ov::op::v1::Subtract>(convert, zp_convert);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 1 }, { 1 });
        auto scale = std::make_shared<ov::op::v1::Multiply>(sub, scale_const);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, scale, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        // oneDNN rejects sub-byte per-tensor zero points, so the scalar i4 ZP must be widened to i8.
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::i4, ov::Shape{ 32, 16 }, { 1 });
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 1 }, { 1 });
        auto zp_const = ov::op::v0::Constant::create(ov::element::i4, ov::Shape{ 1 }, { -3 });
        auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_const, ov::element::i8);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input1, weights_const, no_bias, scale_const, zp_convert);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1});
    }
}

// 4D non-grouped weight constant: leading dims are unit, only last two are
// real (K, N). is_weight_3d is false, so the reshape branch must collapse
// dims [d0, d1, d2, d3] -> [d0*d1*d2, d3] for weight/scale/zp.
TEST_F(TransformationTestsF, ConvertFCToCompressed17_4dNonGrouped) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 1, 1, 32, 16 }, { 1 });
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f16);
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 1, 1, 32, 1 }, { 1 });
        auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_const, ov::element::f16);
        auto sub = std::make_shared<ov::op::v1::Subtract>(convert, zp_convert);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 1, 1, 32, 1 }, { 1 });
        auto scale = std::make_shared<ov::op::v1::Multiply>(sub, scale_const);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, scale, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 32, 16 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 1 }, { 1 });
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 32, 1 }, { 1 });
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input1, weights_const, no_bias, scale_const, zp_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1});
    }
}

// 4D grouped weight/scale/zp constants with is_weight_3d=true: the FC sees
// a 3D weight (after 4D->3D reshape), so the reshape_const lambda must
// collapse 4D constants [d0, d1, d2, d3] -> [d0, d1, d2*d3] (grouped path).
TEST_F(TransformationTestsF, ConvertFCToCompressed18_4dGroupedWith3dWeight) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 8 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{ 2, 4, 2, 4 }, { 1 });
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f16);
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 2, 4, 2, 1 }, { 1 });
        auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_const, ov::element::f16);
        auto sub = std::make_shared<ov::op::v1::Subtract>(convert, zp_convert);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 2, 4, 2, 1 }, { 1 });
        auto scale = std::make_shared<ov::op::v1::Multiply>(sub, scale_const);
        auto reshape_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 3 }, { 2, 4, -1 });
        auto reshape = std::make_shared<ov::op::v1::Reshape>(scale, reshape_const, false);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, reshape, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 8 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{ 2, 4, 8 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 2, 4, 2 }, { 1 });
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 2, 4, 2 }, { 1 });
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input1, weights_const, no_bias, scale_const, zp_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1});
    }
}
  
TEST_F(TransformationTestsF, ConvertFCToCompressed19_mxfp8e4m3) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 32 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::f8e4m3, ov::Shape{ 32, 32 }, { 1 });
        auto weights_convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f16);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f8e8m0, ov::Shape{ 32, 1 }, { 1 });
        auto scale_convert = std::make_shared<ov::op::v0::Convert>(scale_const, ov::element::f16);
        auto deq_mul = std::make_shared<ov::op::v1::Multiply>(weights_convert, scale_convert);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, deq_mul, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 32 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::f8e4m3, ov::Shape{ 32, 32 }, { 1 });
        auto scale_const = ov::op::v0::Constant::create(ov::element::f8e8m0, ov::Shape{ 32, 1 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input1, weights_const, no_bias, scale_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertFCToCompressed20_mxfp8_e5m2) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 32 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::f8e5m2, ov::Shape{ 32, 32 }, { 1 });
        auto weights_convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f16);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f8e8m0, ov::Shape{ 32, 1 }, { 1 });
        auto scale_convert = std::make_shared<ov::op::v0::Convert>(scale_const, ov::element::f16);
        auto deq_mul = std::make_shared<ov::op::v1::Multiply>(weights_convert, scale_convert);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, deq_mul, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 32 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::f8e5m2, ov::Shape{ 32, 32 }, { 1 });
        auto scale_const = ov::op::v0::Constant::create(ov::element::f8e8m0, ov::Shape{ 32, 1 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input1, weights_const, no_bias, scale_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertFCToCompressed21_fp8e4m3) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 32 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::f8e4m3, ov::Shape{ 32, 32 }, { 1 });
        auto weights_convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f16);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 1 }, { 1 });
        auto deq_mul = std::make_shared<ov::op::v1::Multiply>(weights_convert, scale_const);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, deq_mul, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 32 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::f8e4m3, ov::Shape{ 32, 32 }, { 1 });
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 1 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input1, weights_const, no_bias, scale_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertFCToCompressed22_fp8_e5m2) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 32 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::f8e5m2, ov::Shape{ 32, 32 }, { 1 });
        auto weights_convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f16);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 1 }, { 1 });
        auto deq_mul = std::make_shared<ov::op::v1::Multiply>(weights_convert, scale_const);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, deq_mul, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 32 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::f8e5m2, ov::Shape{ 32, 32 }, { 1 });
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 1 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input1, weights_const, no_bias, scale_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1});
    }
}
TEST_F(TransformationTestsF, ConvertFCToCompressed23_fp4e2m1) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 32 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::f4e2m1, ov::Shape{ 32, 32 }, { 1 });
        auto weights_convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f16);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 1 }, { 1 });
        auto deq_mul = std::make_shared<ov::op::v1::Multiply>(weights_convert, scale_const);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, deq_mul, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 32 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::f4e2m1, ov::Shape{ 32, 32 }, { 1 });
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 1 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input1, weights_const, no_bias, scale_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertFCToCompressed24_mxfp4e2m1) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::f4e2m1, ov::Shape{ 32, 16 }, { 1 });
        auto weights_convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f16);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f8e8m0, ov::Shape{ 32, 1 }, { 1 });
        auto scale_convert = std::make_shared<ov::op::v0::Convert>(scale_const, ov::element::f16);
        auto scale = std::make_shared<ov::op::v1::Multiply>(weights_convert, scale_convert);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, scale, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::f4e2m1, ov::Shape{ 32, 16 }, { 1 });
        auto scale_const = ov::op::v0::Constant::create(ov::element::f8e8m0, ov::Shape{ 32, 1 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input1, weights_const, no_bias, scale_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1});
    }
}

enum class TransposeBeforeReshapePattern {
    ScaleOnly,
    ScaleAndZeroPoint,
    Rank4ScaleAndZeroPoint,
};

struct TransposeBeforeReshapeParams {
    ov::element::Type weights_type;
    bool parameter_weights;
    size_t group_size;
    TransposeBeforeReshapePattern pattern;
};

class ConvertFCToCompressedTransposeBeforeReshapeTest
    : public TransformationTestsF,
      public testing::WithParamInterface<TransposeBeforeReshapeParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<TransposeBeforeReshapeParams>& info) {
        const auto& [weights_type, parameter_weights, group_size, pattern] = info.param;
        std::ostringstream name;
        name << weights_type << (parameter_weights ? "_Parameter" : "_Constant") << "_GroupSize" << group_size;
        switch (pattern) {
        case TransposeBeforeReshapePattern::ScaleOnly:
            name << "_ScaleOnly";
            break;
        case TransposeBeforeReshapePattern::ScaleAndZeroPoint:
            name << "_ScaleAndZeroPoint";
            break;
        case TransposeBeforeReshapePattern::Rank4ScaleAndZeroPoint:
            name << "_Rank4ScaleAndZeroPoint";
            break;
        }
        return name.str();
    }

protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        const auto& params = GetParam();
        model = getModel(params);
        if (isConverted(params)) {
            model_ref = getCompressedModel(params);
        }
        comparator.enable(FunctionsComparator::CONST_VALUES);
        comparator.enable(FunctionsComparator::ATTRIBUTES);
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }

    static bool isRank4(const TransposeBeforeReshapeParams& params) {
        return params.pattern == TransposeBeforeReshapePattern::Rank4ScaleAndZeroPoint;
    }

    static bool hasZeroPoint(const TransposeBeforeReshapeParams& params) {
        return params.pattern != TransposeBeforeReshapePattern::ScaleOnly;
    }

    static bool isConverted(const TransposeBeforeReshapeParams& params) {
        return params.weights_type == ov::element::u2 &&
               (!params.parameter_weights || params.group_size % 64 == 0);
    }

    static ov::Shape getWeightsShape(const TransposeBeforeReshapeParams& params) {
        if (isRank4(params)) {
            return {2, 3, 2, params.group_size};
        }
        return {2, 3, params.group_size};
    }

    static ov::Shape getScaleShape(const TransposeBeforeReshapeParams& params) {
        if (isRank4(params)) {
            return {2, 3, 1, 1};
        }
        return {2, 3, 1};
    }

    static ov::Shape getCompressedWeightShape(const TransposeBeforeReshapeParams& params) {
        if (isRank4(params)) {
            return {3, 2, 2 * params.group_size};
        }
        return {3, 2 * params.group_size};
    }

    static ov::Shape getCompressedMetadataShape(const TransposeBeforeReshapeParams& params) {
        if (isRank4(params)) {
            return {2, 3, 1};
        }
        return {2, 3};
    }

    static std::shared_ptr<ov::op::v0::Parameter> makeInput(const TransposeBeforeReshapeParams& params) {
        return std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                       ov::PartialShape{-1,
                                                                        2 * static_cast<ov::Dimension::value_type>(
                                                                                params.group_size)});
    }

    static std::shared_ptr<ov::Node> makeWeights(const TransposeBeforeReshapeParams& params,
                                                 ov::ParameterVector& parameters) {
        const auto shape = getWeightsShape(params);
        if (params.parameter_weights) {
            auto weights = std::make_shared<ov::op::v0::Parameter>(params.weights_type, shape);
            parameters.push_back(weights);
            return weights;
        }
        return ov::op::v0::Constant::create(params.weights_type, shape, {1});
    }

    static std::shared_ptr<ov::Model> getModel(const TransposeBeforeReshapeParams& params) {
        auto input = makeInput(params);
        ov::ParameterVector parameters{input};
        auto weights = makeWeights(params, parameters);
        auto weights_convert = std::make_shared<ov::op::v0::Convert>(weights, ov::element::f16);
        ov::Output<ov::Node> dequantized_weights = weights_convert;
        const auto scale_shape = getScaleShape(params);
        if (hasZeroPoint(params)) {
            auto zero_point = ov::op::v0::Constant::create(ov::element::u2, scale_shape, {1});
            auto zero_point_convert = std::make_shared<ov::op::v0::Convert>(zero_point, ov::element::f16);
            dequantized_weights = std::make_shared<ov::op::v1::Subtract>(weights_convert, zero_point_convert);
        }
        auto scale = ov::op::v0::Constant::create(ov::element::f16, scale_shape, {1});
        auto multiply = std::make_shared<ov::op::v1::Multiply>(dequantized_weights, scale);
        const auto transpose_order = isRank4(params) ? ov::Shape{1, 0, 2, 3} : ov::Shape{1, 0, 2};
        auto transpose_order_const = ov::op::v0::Constant::create(ov::element::i32,
                                                                  ov::Shape{transpose_order.size()},
                                                                  transpose_order);
        auto transpose = std::make_shared<ov::op::v1::Transpose>(multiply, transpose_order_const);
        const auto compressed_weight_shape = getCompressedWeightShape(params);
        auto reshape_shape = ov::op::v0::Constant::create(ov::element::i32,
                                                          ov::Shape{compressed_weight_shape.size()},
                                                          compressed_weight_shape);
        auto reshape = std::make_shared<ov::op::v1::Reshape>(transpose, reshape_shape, false);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input, reshape, no_bias);
        return std::make_shared<ov::Model>(ov::OutputVector{fc}, parameters);
    }

    static std::shared_ptr<ov::Model> getCompressedModel(const TransposeBeforeReshapeParams& params) {
        auto input = makeInput(params);
        ov::ParameterVector parameters{input};
        auto weights = makeWeights(params, parameters);
        const auto transpose_order = isRank4(params) ? ov::Shape{1, 0, 2, 3} : ov::Shape{1, 0, 2};
        auto transpose_order_const = ov::op::v0::Constant::create(ov::element::i32,
                                                                  ov::Shape{transpose_order.size()},
                                                                  transpose_order);
        auto transpose_weights = std::make_shared<ov::op::v1::Transpose>(weights, transpose_order_const);
        const auto compressed_weight_shape = getCompressedWeightShape(params);
        auto reshape_shape = ov::op::v0::Constant::create(ov::element::i32,
                                                          ov::Shape{compressed_weight_shape.size()},
                                                          compressed_weight_shape);
        auto reshape_weights = std::make_shared<ov::op::v1::Reshape>(transpose_weights, reshape_shape, false);

        const auto scale_shape = getScaleShape(params);
        auto scale = ov::op::v0::Constant::create(ov::element::f16, scale_shape, {1});
        const auto compressed_metadata_shape = getCompressedMetadataShape(params);
        auto scale_shape_const = ov::op::v0::Constant::create(ov::element::i32,
                                                              ov::Shape{compressed_metadata_shape.size()},
                                                              compressed_metadata_shape);
        auto reshape_scale = std::make_shared<ov::op::v1::Reshape>(scale, scale_shape_const, false);

        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        std::shared_ptr<ov::intel_gpu::op::FullyConnectedCompressed> fc_compressed;
        if (hasZeroPoint(params)) {
            auto zero_point = ov::op::v0::Constant::create(ov::element::u2, scale_shape, {1});
            auto zero_point_convert = std::make_shared<ov::op::v0::Convert>(zero_point, ov::element::u8);
            auto zero_point_shape_const = ov::op::v0::Constant::create(ov::element::i32,
                                                                       ov::Shape{compressed_metadata_shape.size()},
                                                                       compressed_metadata_shape);
            auto reshape_zero_point = std::make_shared<ov::op::v1::Reshape>(zero_point_convert,
                                                                             zero_point_shape_const,
                                                                             false);
            fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input,
                                                                                          reshape_weights,
                                                                                          no_bias,
                                                                                          reshape_scale,
                                                                                          reshape_zero_point,
                                                                                          ov::element::dynamic,
                                                                                          true,
                                                                                          false,
                                                                                          false);
        } else {
            fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input,
                                                                                          reshape_weights,
                                                                                          no_bias,
                                                                                          reshape_scale,
                                                                                          ov::element::dynamic,
                                                                                          true,
                                                                                          false,
                                                                                          false);
        }
        return std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, parameters);
    }
};

TEST_P(ConvertFCToCompressedTransposeBeforeReshapeTest, ConvertsOnlySupportedWeightPatterns) {}

static std::vector<TransposeBeforeReshapeParams> getTransposeBeforeReshapeTestParams() {
    const std::vector<ov::element::Type> weights_types{ov::element::u2,
                                                       ov::element::u4,
                                                       ov::element::i4,
                                                       ov::element::u8,
                                                       ov::element::i8,
                                                       ov::element::f8e4m3,
                                                       ov::element::f8e5m2,
                                                       ov::element::f4e2m1};
    const std::vector<size_t> group_sizes{4, 63, 64, 128};
    std::vector<TransposeBeforeReshapeParams> params;
    for (const auto& weights_type : weights_types) {
        for (const auto parameter_weights : {false, true}) {
            for (const auto group_size : group_sizes) {
                params.push_back({weights_type,
                                  parameter_weights,
                                  group_size,
                                  TransposeBeforeReshapePattern::ScaleOnly});
            }
        }
    }
    params.push_back({ov::element::u2, false, 4, TransposeBeforeReshapePattern::ScaleAndZeroPoint});
    params.push_back({ov::element::u2, false, 2, TransposeBeforeReshapePattern::Rank4ScaleAndZeroPoint});
    return params;
}

INSTANTIATE_TEST_SUITE_P(
    WeightsTypeAndGroupAlignment,
    ConvertFCToCompressedTransposeBeforeReshapeTest,
    testing::ValuesIn(getTransposeBeforeReshapeTestParams()),
    ConvertFCToCompressedTransposeBeforeReshapeTest::getTestCaseName);

}  // namespace intel_gpu
}  // namespace test
}  // namespace ov
