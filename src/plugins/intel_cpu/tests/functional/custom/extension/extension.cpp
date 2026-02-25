// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/extension.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/util/file_util.hpp"

using testing::ElementsAreArray;

class CustomAbs : public ov::op::Op {
public:
    OPENVINO_OP("CustomAbs", "custom_opset")

    CustomAbs() = default;
    CustomAbs(const ov::Output<ov::Node>& arg) : ov::op::Op({arg}) {
        constructor_validate_and_infer_types();
    }
    void validate_and_infer_types() override {
        set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
    }
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override {
        return std::make_shared<CustomAbs>(new_args.at(0));
    }
    bool visit_attributes(ov::AttributeVisitor&) override {
        return true;
    }

    bool has_evaluate() const override {
        return get_input_element_type(0) == ov::element::f32;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override {
        if (inputs[0].get_element_type() == ov::element::f32) {
            outputs[0].set_shape(inputs[0].get_shape());

            auto first = inputs[0].data<const float>();

            std::transform(first, first + inputs[0].get_size(), outputs[0].data<float>(), [](float v) {
                return v < 0 ? -v * 2.0f : v;
            });
            return true;
        } else {
            return false;
        }
    }
};

static void infer_model(ov::Core& core,
                        ov::CompiledModel& model,
                        std::vector<float>& input_values,
                        const std::vector<float>& expected) {
    auto input_tensor = ov::Tensor(ov::element::f32, model.input(0).get_shape(), input_values.data());

    auto infer_req = model.create_infer_request();
    infer_req.set_input_tensor(input_tensor);
    infer_req.infer();

    auto computed = infer_req.get_output_tensor(0);
    EXPECT_THAT(expected, ElementsAreArray(computed.data<const float>(), computed.get_size()));
}

static std::string model_full_path(const char* path) {
    return ov::util::make_path<char>(ov::util::make_path<char>(ov::test::utils::getExecutableDirectory(), TEST_MODELS),
                                     path);
}

TEST(DISABLED_Extension, XmlModelWithCustomAbs) {
    // Issue: 163252
    std::string model = R"V0G0N(
<net name="Network" version="10">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset1">
            <data element_type="f32" shape="10"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>10</dim>
                </port>
            </output>
        </layer>
        <layer name="activation" id="1" type="CustomAbs" version="custom_opset">
            <input>
                <port id="1" precision="FP32">
                    <dim>10</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>10</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>10</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";

    std::vector<float> input_values{1, -2, 3, -4, 5, -6, 7, -8, 9, -10};
    std::vector<float> expected{1, 4, 3, 8, 5, 12, 7, 16, 9, 20};

    ov::Core core;
    core.add_extension(std::make_shared<ov::OpExtension<CustomAbs>>());
    auto weights = ov::Tensor();
    auto ov_model = core.read_model(model, weights);
    auto compiled_model = core.compile_model(ov_model);

    infer_model(core, compiled_model, input_values, expected);
}


static std::string get_extension_path() {
    return ov::util::make_plugin_library_name<char>(ov::test::utils::getExecutableDirectory(),
                                                    std::string("openvino_template_extension") + OV_BUILD_POSTFIX);
}

TEST(Extension, smoke_XmlModelWithExtensionFromDSO) {
    std::string model = R"V0G0N(
<net name="Network" version="10">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset1">
            <data element_type="f32" shape="2,2,2,1"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>2</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer name="operation" id="1" type="Identity" version="extension">
            <data  add="11"/>
            <input>
                <port id="1" precision="FP32">
                    <dim>2</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>2</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>2</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                    <dim>1</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";

    std::vector<float> input_values{1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> expected{1, 2, 3, 4, 5, 6, 7, 8};

    ov::Core core;
    core.set_property("CPU", {{ov::hint::inference_precision.name(), ov::element::f32.get_type_name()}});
    core.add_extension(get_extension_path());
    auto weights = ov::Tensor();
    auto ov_model = core.read_model(model, weights);
    auto compiled_model = core.compile_model(ov_model);

    infer_model(core, compiled_model, input_values, expected);
}

TEST(DISABLED_Extension, OnnxModelWithExtensionFromDSO) {
    // Issue: 163252
    std::vector<float> input_values{1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> expected{1, 2, 3, 4, 5, 6, 7, 8};

    ov::Core core;
    core.add_extension(get_extension_path());
    auto ov_model = core.read_model(model_full_path("func_tests/models/custom_template_op.onnx"));
    auto compiled_model = core.compile_model(ov_model);

    infer_model(core, compiled_model, input_values, expected);
}
