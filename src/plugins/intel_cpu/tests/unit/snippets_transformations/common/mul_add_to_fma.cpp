// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <subgraph_simple.hpp>
#include <transformations/snippets/common/pass/mul_add_to_fma.hpp>
#include <transformations/snippets/x64/shape_inference.hpp>
#include <transformations/snippets/common/op/fused_mul_add.hpp>
#include <transformations/snippets/x64/shape_inference.hpp>
#include "snippets/op/scalar.hpp"
#include "lowering_utils.hpp"
#include "common_test_utils/common_utils.hpp"
#include "snippets/pass/manager.hpp"

namespace ov {
namespace test {
namespace snippets {

/// Simple Eltwise graph fully convertible to Subgraph.
/// Tokenized simply by attaching eltwises.
// in1   in2                   in1     in2
//  Multiply   in3 or    in3    Multiply
//          Add             Add
//        Result           Result
class EltwiseWithMulAddFunction : public SnippetsFunctionBase {
public:
    explicit EltwiseWithMulAddFunction(const std::vector<PartialShape>& inputShapes,
                                       const size_t add_input_idx = 0,
                                       const bool scalar_input = false)
        : SnippetsFunctionBase(inputShapes),
          add_input_idx(add_input_idx),
          scalar_input(scalar_input) {
        OPENVINO_ASSERT(input_shapes.size() == 3, "Got invalid number of input shapes");
        OPENVINO_ASSERT(add_input_idx < 2, "Got invalid input idx for add operation");
    }

protected:
    std::shared_ptr<ov::Model> initOriginal() const override {
        auto data0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
        auto data1 = std::make_shared<op::v0::Parameter>(precision, input_shapes[1]);
        ParameterVector parameters{data0, data1};

        std::shared_ptr<Node> data2;
        if (scalar_input) {
            data2 = op::v0::Constant::create(precision, {}, {2.f});
        } else {
            auto parameter = std::make_shared<op::v0::Parameter>(precision, input_shapes[2]);
            parameters.push_back(parameter);
            data2 = parameter;
        }

        auto mul = std::make_shared<op::v1::Multiply>(data0, data1);
        const auto& fst_input = add_input_idx == 0 ? mul->output(0) : data2->output(0);
        const auto& sec_input = add_input_idx == 0 ? data2->output(0) : mul->output(0);
        auto add = std::make_shared<op::v1::Add>(fst_input, sec_input);

        return std::make_shared<Model>(NodeVector{add}, parameters);
    }

    std::shared_ptr<ov::Model> initLowered() const override {
        auto data0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
        auto data1 = std::make_shared<op::v0::Parameter>(precision, input_shapes[1]);
        ParameterVector parameters{data0, data1};
        std::shared_ptr<Node> data2;
        if (scalar_input) {
            data2 = std::make_shared<ov::snippets::op::Scalar>(precision, Shape{1}, 2.f);
        } else {
            auto parameter = std::make_shared<op::v0::Parameter>(precision, input_shapes[2]);
            parameters.push_back(parameter);
            data2 = parameter;
        }


        auto a = scalar_input || add_input_idx == 0 ? data0 : data1;
        auto b = scalar_input || add_input_idx == 0 ? data1 : data2;
        auto c = scalar_input || add_input_idx == 0 ? data2 : data0;

        auto fma = std::make_shared<ov::intel_cpu::FusedMulAdd>(a, b, c);
        return std::make_shared<ov::Model>(NodeVector{fma}, parameters);
    }

    void validate_function(const std::shared_ptr<Model> &m) const override {
        OPENVINO_ASSERT(m != nullptr, "The test requires Model to be defined");
        const auto &params = m->get_parameters();
        OPENVINO_ASSERT(params.size() == (scalar_input ? input_shapes.size() - 1 : input_shapes.size()),
                        "Passed input shapes and produced function are inconsistent.");
        for (size_t i = 0; i < params.size(); i++)
            OPENVINO_ASSERT(std::equal(input_shapes[i].begin(), input_shapes[i].end(), params[i]->get_shape().begin()),
                            "Passed input shapes and produced function are inconsistent.");
    }

private:
    size_t add_input_idx;
    bool scalar_input;
};

typedef std::tuple<
        PartialShape,  // Input shape 0
        PartialShape,  // Input shape 1
        PartialShape,  // Input shape 2
        size_t         // Add input index
> MulAddToFMAParams;

class MulAddToFMATests : public LoweringTests, public testing::WithParamInterface<MulAddToFMAParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<MulAddToFMAParams> obj) {
        std::vector<PartialShape> inputShapes(3);
        size_t add_input_idx;
        std::tie(inputShapes[0], inputShapes[1], inputShapes[2], add_input_idx) = obj.param;

        std::ostringstream result;
        for (size_t i = 0; i < inputShapes.size(); i++)
            result << "IS[" << i << "]=" <<  ov::test::utils::partialShape2str({inputShapes[i]}) << "_";
        result << "add_input_idx=" << add_input_idx;
        return result.str();
    }

protected:
    void SetUp() override {
        using PassPosition = ov::snippets::pass::PassPosition;
        LoweringTests::SetUp();
        std::vector<PartialShape> inputShapes(3);
        size_t add_input_idx;
        std::tie(inputShapes[0], inputShapes[1], inputShapes[2], add_input_idx) = this->GetParam();
        const bool scalar_input = ov::shape_size(inputShapes[2].to_shape()) == 1;
        snippets_model = std::make_shared<EltwiseWithMulAddFunction>(inputShapes, add_input_idx, scalar_input);

        // Note: this inserts MulAddToFMA at the end of the pipeline
        backend_passes.emplace_back(PassPosition(PassPosition::Place::PipelineEnd),
                                    std::make_shared<ov::intel_cpu::pass::MulAddToFMA>());

        std::vector<ov::Node::type_info_t> custom_opset{ov::intel_cpu::FusedMulAdd::get_type_info_static()};
        auto target_machine = std::make_shared<DummyTargetMachine>(custom_opset);
        generator = std::make_shared<DummyGenerator>(target_machine);
    }

    std::shared_ptr<SnippetsFunctionBase> snippets_model;
    std::shared_ptr<ov::snippets::Generator> generator;
    std::vector<ov::snippets::pass::Manager::PositionedPassBase> backend_passes;
};

TEST_P(MulAddToFMATests, MulAddToFMATests) {
    auto subgraph = getLoweredSubgraph(snippets_model->getOriginal(),
                                       backend_passes,
                                       std::make_shared<ov::snippets::lowered::pass::PassConfig>(),
                                       {},
                                       generator,
                                       8, 256,
                                       std::make_shared<ov::snippets::CPUShapeInferSnippetsFactory>());
    model = subgraph->body_ptr();
    model_ref = snippets_model->getLowered();
}

namespace MulAddToFMATestsInstantiation {
std::vector<PartialShape> in_shapes_0 = {{1, 3, 16, 16}};
std::vector<PartialShape> in_shapes_1 = {{1, 3, 16, 16}};
std::vector<PartialShape> in_shapes_2 = {{1, 3, 16, 16}, {}};
std::vector<size_t> in_idxes_for_add = {0, 1};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets, MulAddToFMATests,
                        ::testing::Combine(
                                ::testing::ValuesIn(in_shapes_0),
                                ::testing::ValuesIn(in_shapes_1),
                                ::testing::ValuesIn(in_shapes_2),
                                ::testing::ValuesIn(in_idxes_for_add)),
                        MulAddToFMATests::getTestCaseName);
} // namespace MulAddToFMATestsInstantiation

TEST_F(TransformationTestsF, smoke_Snippets_MulAddToFMATestsNegative) {
    auto data0 = std::make_shared<op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 3, 16, 16});
    auto data1 = std::make_shared<op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 3, 16, 16});
    auto data2 = std::make_shared<op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 3, 16, 16});

    auto mul = std::make_shared<op::v1::Multiply>(data0, data1);
    auto additional_consumer = std::make_shared<op::v0::Relu>(mul);
    auto add = std::make_shared<op::v1::Add>(mul, data2);

    model = std::make_shared<Model>(ov::NodeVector{add, additional_consumer}, ov::ParameterVector{data0, data1, data2});
    manager.register_pass<ov::intel_cpu::pass::MulAddToFMA>();
}
}  // namespace snippets
}  // namespace test
}  // namespace ov
