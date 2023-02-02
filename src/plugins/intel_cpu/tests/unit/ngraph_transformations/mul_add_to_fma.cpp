// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <subgraph_simple.hpp>
#include <snippets_transformations/mul_add_to_fma.hpp>
#include <snippets_transformations/op/fused_mul_add.hpp>

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
                                       const bool constant_input = false)
        : SnippetsFunctionBase(inputShapes),
          add_input_idx(add_input_idx),
          constant_input(constant_input) {
        NGRAPH_CHECK(input_shapes.size() == 3, "Got invalid number of input shapes");
        NGRAPH_CHECK(add_input_idx < 2, "Got invalid input idx for add operation");
    }

protected:
    std::shared_ptr<ov::Model> initOriginal() const override {
        auto data0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
        auto data1 = std::make_shared<op::v0::Parameter>(precision, input_shapes[1]);
        ParameterVector parameters{data0, data1};

        std::shared_ptr<Node> data2;
        if (constant_input) {
            data2 = op::v0::Constant::create(precision, input_shapes[2].to_shape(), {2.f});
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

    std::shared_ptr<ov::Model> initReference() const override {
        auto data0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
        auto data1 = std::make_shared<op::v0::Parameter>(precision, input_shapes[1]);
        ParameterVector parameters{data0, data1};

        std::shared_ptr<Node> data2;
        if (constant_input) {
            data2 = op::v0::Constant::create(precision, input_shapes[2].to_shape(), {2.f});
        } else {
            auto parameter = std::make_shared<op::v0::Parameter>(precision, input_shapes[2]);
            parameters.push_back(parameter);
            data2 = parameter;
        }

        auto fma = std::make_shared<ov::intel_cpu::FusedMulAdd>(data0, data1, data2);
        return std::make_shared<Model>(NodeVector{fma}, parameters);
    }

    void validate_function(const std::shared_ptr<Model> &m) const override {
        NGRAPH_CHECK(m != nullptr, "The test requires Model to be defined");
        const auto &params = m->get_parameters();
        NGRAPH_CHECK(params.size() == (constant_input ? input_shapes.size() - 1 : input_shapes.size()),
                    "Passed input shapes and produced function are inconsistent.");
        for (size_t i = 0; i < params.size(); i++)
            NGRAPH_CHECK(std::equal(input_shapes[i].begin(), input_shapes[i].end(), params[i]->get_shape().begin()),
                        "Passed input shapes and produced function are inconsistent.");
    }

private:
    size_t add_input_idx;
    bool constant_input;
};

typedef std::tuple<
        PartialShape,  // Input shape 0
        PartialShape,  // Input shape 1
        PartialShape,  // Input shape 2
        size_t,        // Add input index
        bool           // Constant input
> MulAddToFMAParams;

class MulAddToFMATests : public TransformationTestsF, public testing::WithParamInterface<MulAddToFMAParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<MulAddToFMAParams> obj) {
        std::vector<PartialShape> inputShapes(3);
        size_t add_input_idx;
        bool constant_input;
        std::tie(inputShapes[0], inputShapes[1], inputShapes[2], add_input_idx, constant_input) = obj.param;

        std::ostringstream result;
        for (size_t i = 0; i < inputShapes.size(); i++)
            result << "IS[" << i << "]=" << inputShapes[i] << "_";
        result << "add_input_idx=" << add_input_idx << (constant_input ? "_constant_input" : "");
        return result.str();
    }

protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        std::vector<PartialShape> inputShapes(3);
        size_t add_input_idx;
        bool constant_input;
        std::tie(inputShapes[0], inputShapes[1], inputShapes[2], add_input_idx, constant_input) = this->GetParam();
        snippets_function = std::make_shared<EltwiseWithMulAddFunction>(inputShapes, add_input_idx, constant_input);

        manager.register_pass<ov::intel_cpu::pass::MulAddToFMA>();
    }

    std::shared_ptr<SnippetsFunctionBase> snippets_function;
};

TEST_P(MulAddToFMATests, MulAddToFMATests) {
    model = snippets_function->getOriginal();
    model_ref = snippets_function->getReference();
}

namespace MulAddToFMATestsInstantiation {
std::vector<PartialShape> in_shapes_0 = {{1, 3, 2, 2}};
std::vector<PartialShape> in_shapes_1 = {{1, 3, 2, 2}};
std::vector<PartialShape> in_shapes_2 = {{1, 3, 2, 2}, {}};
std::vector<size_t> in_idxes_for_add = {0, 1};
std::vector<bool> constant_input = {false, true};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets, MulAddToFMATests,
                        ::testing::Combine(
                                ::testing::ValuesIn(in_shapes_0),
                                ::testing::ValuesIn(in_shapes_1),
                                ::testing::ValuesIn(in_shapes_2),
                                ::testing::ValuesIn(in_idxes_for_add),
                                ::testing::ValuesIn(constant_input)),
                        MulAddToFMATests::getTestCaseName);

} // namespace MulAddToFMATestsInstantiation

TEST_F(TransformationTestsF, smoke_Snippets_MulAddToFMATestsNegative) {
    auto data0 = std::make_shared<op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 3, 2, 2});
    auto data1 = std::make_shared<op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 3, 2, 2});
    auto data2 = std::make_shared<op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 3, 2, 2});

    auto mul = std::make_shared<op::v1::Multiply>(data0, data1);
    auto additional_consumer = std::make_shared<op::v0::Relu>(mul);
    auto add = std::make_shared<op::v1::Add>(mul, data2);

    model = std::make_shared<Model>(ov::NodeVector{add, additional_consumer}, ov::ParameterVector{data0, data1, data2});
}
}  // namespace snippets
}  // namespace test
}  // namespace ov
