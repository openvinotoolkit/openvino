// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "conversion_with_reference.hpp"
#include "openvino/frontend/decoder.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/frontend/extension.hpp"
#include "openvino/frontend/manager.hpp"
#include "openvino/frontend/node_context.hpp"
#include "openvino/frontend/tensorflow/exception.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/log.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "tf_framework_node.hpp"
#include "tf_utils.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::element;
using namespace ov::op;
using namespace ov::frontend;
using namespace ov::frontend::tensorflow::tests;

namespace {
class TestDecoder : public ov::frontend::DecoderBase {
public:
    explicit TestDecoder(const std::string& op_type) : m_op_type(op_type) {}

    ov::Any get_attribute(const std::string& name) const override {
        throw "Not implemented";
    }

    size_t get_input_size() const override {
        throw "Not implemented";
    }

    void get_input_node(size_t input_port_idx,
                        std::string& producer_name,
                        std::string& producer_output_port_name,
                        size_t& producer_output_port_index) const override {
        throw "Not implemented";
    }

    const std::string& get_op_type() const override {
        return m_op_type;
    }

    const std::string& get_op_name() const override {
        throw "Not implemented";
    }

private:
    const std::string m_op_type;
};

shared_ptr<Model> convert_model_partially(const std::string& model_path) {
    FrontEndManager fem;
    auto front_end = fem.load_by_framework(TF_FE);
    if (!front_end) {
        throw "TensorFlow Frontend is not initialized";
    }
    auto model_filename = FrontEndTestUtils::make_model_path(std::string(TEST_TENSORFLOW_MODELS_DIRNAME) + model_path);
    auto input_model = front_end->load(model_filename);
    if (!input_model) {
        throw "Input model is not read";
    }
    auto model = front_end->convert_partially(input_model);
    if (!model) {
        throw "Model is not converted partially";
    }

    return model;
}

ov::OutputVector incorrect_less_translator(const ov::frontend::NodeContext& node) {
    // NOTE: pay attention that this is a fake translator for Less operation
    // only serves for testing purposes
    TENSORFLOW_OP_VALIDATION(node, false, "Less expects ten inputs.");
    return {};
}

ov::OutputVector add_translator_with_unknown_exception(const ov::frontend::NodeContext& node) {
    // NOTE: pay attention that this is a fake translator for Add operation
    // only serves for testing purposes
    throw 0;
    return {};
}
}  // namespace

TEST(FrontEndConvertModelTest, test_unsupported_op) {
    FrontEndManager fem;
    FrontEnd::Ptr frontEnd;
    InputModel::Ptr inputModel;
    OV_ASSERT_NO_THROW(frontEnd = fem.load_by_framework(TF_FE));
    ASSERT_NE(frontEnd, nullptr);
    auto model_filename = FrontEndTestUtils::make_model_path(std::string(TEST_TENSORFLOW_MODELS_DIRNAME) +
                                                             std::string("relu_unsupported/relu_unsupported.pb"));
    OV_ASSERT_NO_THROW(inputModel = frontEnd->load(model_filename));
    ASSERT_NE(inputModel, nullptr);
    shared_ptr<ov::Model> model;
    ASSERT_THROW(model = frontEnd->convert(inputModel), OpConversionFailure);
    ASSERT_EQ(model, nullptr);
    OV_ASSERT_NO_THROW(model = frontEnd->decode(inputModel));
    ASSERT_THROW(frontEnd->convert(model), OpConversionFailure);
    OV_ASSERT_NO_THROW(model = frontEnd->convert_partially(inputModel));
    ASSERT_THROW(frontEnd->convert(model), OpConversionFailure);

    for (auto& node : model->get_ordered_ops()) {
        if (node->get_friendly_name() == "relu_0" && dynamic_pointer_cast<ov::op::util::FrameworkNode>(node)) {
            model->replace_node(node, make_shared<v0::Relu>(node->input(0).get_source_output()));
        }
    }
    OV_ASSERT_NO_THROW(frontEnd->convert(model));
}

TEST_F(FrontEndConversionWithReferenceTestsF, ModelWithDynamicType) {
    { model = convert_model_partially("dynamic_type_model/dynamic_type_model.pb"); }
    {
        auto x = make_shared<v0::Parameter>(f32, Shape{2, 3});
        auto unsupported_op = make_shared<ov::frontend::tensorflow::FrameworkNode>(make_shared<TestDecoder>("Rrrr"),
                                                                                   ov::OutputVector{x},
                                                                                   1);
        ASSERT_EQ(unsupported_op->get_output_element_type(0), ov::element::dynamic);
        ov::Output<ov::Node> const_one = make_shared<v0::Constant>(ov::element::f32, ov::Shape{}, 1);
        const_one = make_shared<v1::ConvertLike>(const_one, unsupported_op);
        auto input_plus_one = make_shared<v1::Add>(unsupported_op, const_one);
        auto log1p_node = make_shared<v0::Log>(input_plus_one);
        ASSERT_EQ(log1p_node->get_output_element_type(0), ov::element::dynamic);
        model_ref = make_shared<Model>(OutputVector{log1p_node}, ParameterVector{x});
    }
}

TEST(FrontEndConvertModelTest, test_unsupported_tf1_while_and_incorrect_less_translator) {
    shared_ptr<Model> model = nullptr;
    try {
        auto conv_ext = std::make_shared<ov::frontend::ConversionExtension>("Less", incorrect_less_translator);
        model = convert_model("model_tf1_while/model_tf1_while.pbtxt", conv_ext);
        FAIL() << "TensorFlow 1 While is not supported and the fake translator registered in TF FE but conversion "
                  "passed without errors. "
                  "OpConversionFailure is expected.";
    } catch (const OpConversionFailure& error) {
        std::string error_message = error.what();
        std::string ref_message = "Less expects ten inputs.\n";
        std::string not_found_message =
            "[TensorFlow Frontend] Internal error, no translator found for operation(s): Enter, Exit, "
            "LoopCond, Merge, NextIteration, Switch";
        ASSERT_TRUE(error_message.find(ref_message) != std::string::npos);
        ASSERT_TRUE(error_message.find(not_found_message) == std::string::npos);
        ASSERT_EQ(model, nullptr);
    } catch (...) {
        FAIL() << "Conversion of TensorFlow 1 While failed by wrong reason.";
    }
}

TEST(FrontEndConvertModelTest, conversion_with_unknown_exception) {
    shared_ptr<Model> model = nullptr;
    try {
        auto conv_ext =
            std::make_shared<ov::frontend::ConversionExtension>("Add", add_translator_with_unknown_exception);
        model = convert_model("model_tf1_while/model_tf1_while.pbtxt", conv_ext);
        FAIL() << "TensorFlow 1 While is not supported and the fake translator registered in TF FE but conversion "
                  "passed without errors. "
                  "OpConversionFailure is expected.";
    } catch (const OpConversionFailure& error) {
        std::string error_message = error.what();
        std::string ref_message = "Unknown exception type\n";
        std::string doc_message =
            "To facilitate the conversion of unsupported operations, refer to Frontend Extension documentation: "
            "https://docs.openvino.ai/latest/openvino_docs_Extensibility_UG_Frontend_Extensions.html";
        ASSERT_TRUE(error_message.find(ref_message) != std::string::npos);
        ASSERT_TRUE(error_message.find(doc_message) == std::string::npos);
        ASSERT_EQ(model, nullptr);
    } catch (...) {
        FAIL() << "Conversion of TensorFlow 1 While failed by wrong reason.";
    }
}

TEST(FrontEndConvertModelTest, test_unsupported_resource_gather_translator) {
    shared_ptr<Model> model = nullptr;
    try {
        auto conv_ext =
            std::make_shared<ov::frontend::ConversionExtension>("ResourceGather", incorrect_less_translator);
        model = convert_model("resource_gather_model/resource_gather_model.pbtxt", conv_ext);
        FAIL() << "The model with ResourceGather node must not be converted due to incorrect "
                  "ResourceGather translator. "
                  "OpConversionFailure is expected.";
    } catch (const OpConversionFailure& error) {
        std::string error_message = error.what();
        std::string ref_message = "Less expects ten inputs.\n";
        std::string no_ref_message = "[TensorFlow Frontend] Internal error: No translator found for";
        ASSERT_TRUE(error_message.find(ref_message) != std::string::npos);
        ASSERT_TRUE(error_message.find(no_ref_message) == std::string::npos);
        std::string ref_message2 = "Install OpenVINO Tokenizers, refer to the documentation";
        ASSERT_TRUE(error_message.find(ref_message2) == std::string::npos);
        ASSERT_EQ(model, nullptr);
    } catch (...) {
        FAIL() << "Conversion of the model with ResourceGather failed by wrong reason.";
    }
}

TEST(FrontEndConvertModelTest, test_uninitialized_variablev2) {
    shared_ptr<Model> model = nullptr;
    try {
        model = convert_model("unitialized_variablev2/unitialized_variablev2.pb",
                              nullptr,
                              {"x"},
                              {element::f32},
                              {Shape{3, 2}},
                              {},
                              {},
                              false,
                              {"mul"});
        FAIL() << "The model with VariableV2 node must not be converted due to incorrect uninitialized VariableV2.";
    } catch (const OpConversionFailure& error) {
        std::string error_message = error.what();
        std::string ref_message = "Variable or resource `variable_yy2` is not initialized, model is inconsistent";
        ASSERT_TRUE(error_message.find(ref_message) != string::npos);
        ASSERT_EQ(model, nullptr);
    } catch (...) {
        FAIL() << "Conversion of the model with VariableV2 failed by wrong reason.";
    }
}

TEST(FrontEndConvertModelTest, conversion_with_tokenizer_operation) {
    shared_ptr<Model> model = nullptr;
    try {
        model = convert_model("string_lower/string_lower.pb");
        FAIL() << "TensorFlow 1 model with StringLower operation is not supported without openvino-tokenizers.";
    } catch (const OpConversionFailure& error) {
        std::string error_message = error.what();
        std::string ref_message = "Install OpenVINO Tokenizers, refer to the documentation";
        ASSERT_TRUE(error_message.find(ref_message) != std::string::npos);
        ASSERT_EQ(model, nullptr);
    } catch (...) {
        FAIL() << "Conversion of TensorFlow 1 model with StringLower operation failed by wrong reason.";
    }
}
