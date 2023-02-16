// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/op/op.hpp>
#include <shared_test_classes/base/ov_subgraph.hpp>
#include <ngraph_functions/builders.hpp>
#include <common_test_utils/ov_tensor_utils.hpp>

using namespace ov::test;

namespace CPULayerTestsDefinitions {

/* This is a synthetic op that mimics the general behaviour of operations with internal dynamics, i.e. nodes where the
   the output shapes may only be defined after the actual computations. */

class CustomOp : public ov::op::Op {
public:
    OPENVINO_OP("CustomOp");

    CustomOp() = default;
    CustomOp(const ov::OutputVector& args) : Op(args) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        const auto& inputs_count = input_values().size();
        OPENVINO_ASSERT(inputs_count == 1,
                        "Input count must be 1, Got: ",
                        inputs_count);
        set_output_size(2);

        auto shape0 = get_input_partial_shape(0);
        auto rank0  = shape0.rank();

        OPENVINO_ASSERT(rank0.compatible(3),
                        "The input must be 3D.");

        //here we set undefined shapes since they can only be determined after the actual calculations
        set_output_type(0, get_input_element_type(0), ov::PartialShape({ov::Dimension()}));
        set_output_type(1, get_input_element_type(0), ov::PartialShape({ov::Dimension(), ov::Dimension(), ov::Dimension()}));
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override {
        OPENVINO_ASSERT(new_args.size() == 1, "Incorrect number of new arguments");

        return std::make_shared<CustomOp>(new_args);
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override {
        auto in = inputs[0];
        auto out0 = outputs[0];
        auto out1 = outputs[1];
        std::vector<float> out0_data(100, 1.5);
        auto out0_shape = ov::Shape({100});
        out0.set_shape(out0_shape);
        memcpy(out0.data(), out0_data.data(), sizeof(out0_data[0]) * out0_data.size());

        out1.set_shape(in.get_shape());
        memcpy(out1.data(), in.data(), in.get_byte_size());
        return true;
    }

    OPENVINO_SUPPRESS_DEPRECATED_START
    // old fashion evaluate method, just to make the whole test subsystem work
    bool evaluate(const ov::HostTensorVector& outputs, const ov::HostTensorVector& inputs) const override {
        ov::TensorVector tmp_inputs;
        for (auto& input : inputs) {
            tmp_inputs.emplace_back(input->get_element_type(), input->get_shape(), input->get_data_ptr());
        }
        ov::TensorVector tmp_outputs;
        for (auto& output : outputs) {
            tmp_outputs.emplace_back(output->get_element_type(), ov::Shape(output->get_partial_shape().rank().get_length()));
        }
        evaluate(tmp_outputs, tmp_inputs);
        OPENVINO_ASSERT(tmp_outputs.size() == outputs.size());
        for (size_t i = 0; i < tmp_outputs.size(); ++i) {
            outputs[i]->set_shape(tmp_outputs[i].get_shape());
            outputs[i]->write(tmp_outputs[i].data(), tmp_outputs[i].get_byte_size());
        }
        return true;
    }
    OPENVINO_SUPPRESS_DEPRECATED_END

    bool has_evaluate() const override {
        return true;
    }
};

class CustomOpCPUTest : public SubgraphBaseTest {
protected:
    void SetUp() override {
        targetDevice = CommonTestUtils::DEVICE_CPU;

        InputShape inputShapes{{-1, -1, -1}, {{10, 5, 3}, {16, 24, 16}, {4, 8, 12}}};

        init_input_shapes({inputShapes});
        auto ngPrc = ngraph::element::f32;
        auto inputParams = ngraph::builder::makeDynamicParams(ngPrc, inputDynamicShapes);
        auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(inputParams));
        auto customOp = std::make_shared<CustomOp>(paramOuts);

        ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(customOp)};
        function = std::make_shared<ngraph::Function>(results, inputParams, "customOpTest");
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (int i = 0; i < funcInputs.size(); ++i) {
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

TEST_F(CustomOpCPUTest, smoke_CustomOpInternalDynamismCPUTest) {
    run();
}
} // namespace CPULayerTestsDefinitions