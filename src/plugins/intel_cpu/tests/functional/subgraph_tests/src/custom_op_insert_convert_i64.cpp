// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/ov_tensor_utils.hpp>
#include <openvino/op/op.hpp>
#include <shared_test_classes/base/ov_subgraph.hpp>
#include <ov_models/builders.hpp>
#include "test_utils/cpu_test_utils.hpp"

using namespace ov::test;
using namespace CPUTestUtils;

namespace ov {
namespace test {
using CustomOpI64CPUTestParams = std::tuple<ElementType, InputShape>;

class CustomOpI64 : public ov::op::Op {
public:
    OPENVINO_OP("CustomOpI64");

    CustomOpI64() = default;
    CustomOpI64(const ov::OutputVector& args) : Op(args) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        const auto& inputs_count = input_values().size();
        OPENVINO_ASSERT(inputs_count == 1,
                        "Input count must be 1, Got: ",
                        inputs_count);
        OPENVINO_ASSERT(get_input_element_type(0) == ov::element::Type_t::i32,
                        "The input must be i32.");
        set_output_size(2);

        auto inShape = get_input_partial_shape(0);

        set_output_type(0, ov::element::Type_t::i64, inShape);
        set_output_type(1, ov::element::Type_t::i32, inShape);
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override {
        OPENVINO_ASSERT(new_args.size() == 1, "Incorrect number of new arguments");

        return std::make_shared<CustomOpI64>(new_args);
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override {
        const auto& in = inputs[0];
        auto& out0 = outputs[0];
        auto& out1 = outputs[1];

        auto inData = in.data<int32_t>();

        auto outData0 = out0.data<int64_t>();
        for (size_t i = 0lu; i < in.get_size(); i++) {
            outData0[i] = static_cast<int64_t>(inData[i]);
        }

        memcpy(out1.data(), inData, out1.get_byte_size());

        return true;
    }

    bool evaluate(ov::TensorVector& output_values,
                  const ov::TensorVector& input_values,
                  const ov::EvaluationContext& evaluationContext) const override {
        return evaluate(output_values, input_values);
    }

    bool has_evaluate() const override {
        return true;
    }
};

class CustomOpConvertI64CPUTest : public testing::WithParamInterface<CustomOpI64CPUTestParams>,
                                  virtual public SubgraphBaseTest,
                                  public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<CustomOpI64CPUTestParams>& obj) {
        ElementType inType;
        InputShape inputShape;
        std::tie(inType, inputShape) = obj.param;

        std::ostringstream result;
        result << "IS=" << inputShape << "_";
        result << "Prc=" << inType;
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;

        ElementType inType;
        InputShape inputShape;
        std::tie(inType, inputShape) = this->GetParam();

        init_input_shapes({inputShape});
        ov::ParameterVector inputParams;
        ov::OutputVector paramsOuts;
        for (auto&& shape : inputDynamicShapes) {
            auto param = std::make_shared<ov::op::v0::Parameter>(inType, shape);
            inputParams.push_back(param);
            paramsOuts.push_back(param);
        }
        auto customOp = std::make_shared<CustomOpI64>(paramsOuts);

        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(customOp)};
        function = std::make_shared<ov::Model>(results, inputParams, "customOpTest");
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            auto tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i]);
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }

    void compare(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual) override {
        ASSERT_EQ(expected.size(), actual.size());
        ASSERT_EQ(expected.size(), function->get_results().size());
        const auto& results = function->get_results();
        for (size_t j = 0; j < results.size(); j++) {
            const auto result = results[j];
            for (size_t i = 0; i < result->get_input_size(); ++i) {
                ov::test::utils::compare(expected[j], actual[j], abs_threshold, rel_threshold);
            }
        }
    }
};

TEST_P(CustomOpConvertI64CPUTest, CompareWithRefs) {
    run();
    // TODO: Graph could not be dumped with int64 for now. Swith on this in scope of int64 enabling.
    // CPUTestUtils::CheckNumberOfNodesWithType(compiledModel, "Convert", 1);
}

const InputShape inputShapes = {
    {}, {{2, 3, 64}}
};

INSTANTIATE_TEST_SUITE_P(smoke_CustomOp,
                         CustomOpConvertI64CPUTest,
                         ::testing::Combine(::testing::Values(ElementType::i32), ::testing::Values(inputShapes)),
                         CustomOpConvertI64CPUTest::getTestCaseName);

}  // namespace test
}  // namespace ov
