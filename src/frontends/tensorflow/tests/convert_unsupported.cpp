// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/frontend/decoder.hpp>
#include <openvino/frontend/exception.hpp>
#include <openvino/frontend/manager.hpp>
#include <openvino/op/util/framework_node.hpp>
#include <openvino/opsets/opset10.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "tf_framework_node.hpp"
#include "tf_utils.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::element;
using namespace ov::opset10;
using namespace ov::frontend;

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
                        size_t& producer_output_port_index) const override {
        throw "Not implemented";
    }

    void get_input_node(size_t input_port_idx,
                        std::string& producer_name,
                        size_t& producer_output_port_index,
                        const OpTypeByName& op_type_by_name) const override {
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

shared_ptr<Model> convert_model_partially(const string& model_path) {
    FrontEndManager fem;
    auto front_end = fem.load_by_framework(TF_FE);
    if (!front_end) {
        throw "TensorFlow Frontend is not initialized";
    }
    auto model_filename = FrontEndTestUtils::make_model_path(string(TEST_TENSORFLOW_MODELS_DIRNAME) + model_path);
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
}  // namespace

TEST(FrontEndConvertModelTest, test_unsupported_op) {
    FrontEndManager fem;
    FrontEnd::Ptr frontEnd;
    InputModel::Ptr inputModel;
    ASSERT_NO_THROW(frontEnd = fem.load_by_framework(TF_FE));
    ASSERT_NE(frontEnd, nullptr);
    auto model_filename = FrontEndTestUtils::make_model_path(string(TEST_TENSORFLOW_MODELS_DIRNAME) +
                                                             string("relu_unsupported/relu_unsupported.pb"));
    ASSERT_NO_THROW(inputModel = frontEnd->load(model_filename));
    ASSERT_NE(inputModel, nullptr);
    shared_ptr<ov::Model> model;
    ASSERT_THROW(model = frontEnd->convert(inputModel), OpConversionFailure);
    ASSERT_EQ(model, nullptr);
    ASSERT_NO_THROW(model = frontEnd->decode(inputModel));
    ASSERT_THROW(frontEnd->convert(model), OpConversionFailure);
    ASSERT_NO_THROW(model = frontEnd->convert_partially(inputModel));
    ASSERT_THROW(frontEnd->convert(model), OpConversionFailure);

    for (auto& node : model->get_ordered_ops()) {
        if (node->get_friendly_name() == "relu_0" && dynamic_pointer_cast<ov::op::util::FrameworkNode>(node)) {
            model->replace_node(node, make_shared<opset10::Relu>(node->input(0).get_source_output()));
        }
    }
    ASSERT_NO_THROW(frontEnd->convert(model));
}

TEST(FrontEndConvertModelTest, test_unsupported_tf1_while) {
    FrontEndManager fem;
    FrontEnd::Ptr frontEnd;
    InputModel::Ptr inputModel;
    ASSERT_NO_THROW(frontEnd = fem.load_by_framework(TF_FE));
    ASSERT_NE(frontEnd, nullptr);
    auto model_filename = FrontEndTestUtils::make_model_path(string(TEST_TENSORFLOW_MODELS_DIRNAME) +
                                                             string("model_tf1_while/model_tf1_while.pb"));
    ASSERT_NO_THROW(inputModel = frontEnd->load(model_filename));
    ASSERT_NE(inputModel, nullptr);
    shared_ptr<ov::Model> model;

    try {
        model = frontEnd->convert(inputModel);
        FAIL() << "TensorFlow 1 While is not supported in TF FE but conversion passed without errors. "
                  "OpConversionFailure is expected.";
    } catch (const OpConversionFailure& error) {
        string error_message = error.what();
        string ref_message = "No translator found for Enter node.";
        ASSERT_TRUE(error_message.find(ref_message) != string::npos);
        ASSERT_EQ(model, nullptr);
    } catch (...) {
        FAIL() << "Conversion of TensorFlow 1 While failed by wrong reason.";
    }
}

TEST_F(TransformationTestsF, ModelWithDynamicType) {
    { model = convert_model_partially("dynamic_type_model/dynamic_type_model.pb"); }
    {
        auto x = make_shared<Parameter>(f32, Shape{2, 3});
        auto unsupported_op = make_shared<ov::frontend::tensorflow::FrameworkNode>(make_shared<TestDecoder>("Rrrr"),
                                                                                   ov::OutputVector{x},
                                                                                   1);
        ASSERT_EQ(unsupported_op->get_output_element_type(0), ov::element::dynamic);
        ov::Output<ov::Node> const_one = make_shared<Constant>(ov::element::f32, ov::Shape{}, 1);
        const_one = make_shared<ConvertLike>(const_one, unsupported_op);
        auto input_plus_one = make_shared<Add>(unsupported_op, const_one);
        auto log1p_node = make_shared<Log>(input_plus_one);
        ASSERT_EQ(log1p_node->get_output_element_type(0), ov::element::dynamic);
        model_ref = make_shared<Model>(OutputVector{log1p_node}, ParameterVector{x});
    }
}
