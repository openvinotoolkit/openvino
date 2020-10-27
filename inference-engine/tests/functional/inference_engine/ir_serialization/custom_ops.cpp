// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ie_api.h>
#include <ie_iextension.h>
#include "common_test_utils/ngraph_test_utils.hpp"
#include "ie_core.hpp"
#include "ngraph/ngraph.hpp"
#include "transformations/serialize.hpp"

#ifndef IR_SERIALIZATION_MODELS_PATH  // should be already defined by cmake
#define IR_SERIALIZATION_MODELS_PATH ""
#endif

class SerializationTestOp : public ngraph::op::Op {
public:
    static constexpr ngraph::NodeTypeInfo type_info{"SerializationTestOp", 0};
    const ngraph::NodeTypeInfo& get_type_info() const override {
        return type_info;
    }

    SerializationTestOp() = default;
    SerializationTestOp(const ngraph::Output<ngraph::Node>& arg, bool test1,
                        int64_t test2)
        : Op({arg}), test1(test1), test2(test2) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        auto input_shape = get_input_partial_shape(0).to_shape();

        ngraph::Shape output_shape(input_shape);
        for (int i = 0; i < input_shape.size(); ++i) {
            output_shape[i] = input_shape[i] * test2 + (test1 ? 0 : 1);
        }

        set_output_type(0, get_input_element_type(0),
                        ngraph::PartialShape(output_shape));
    }

    std::shared_ptr<ngraph::Node> clone_with_new_inputs(
        const ngraph::OutputVector& new_args) const override {
        if (new_args.size() != 1) {
            throw ngraph::ngraph_error("Incorrect number of new arguments");
        }

        return std::make_shared<SerializationTestOp>(new_args.at(0), test1,
                                                     test2);
    }

    bool visit_attributes(ngraph::AttributeVisitor& visitor) override {
        visitor.on_attribute("test1", test1);
        visitor.on_attribute("test2", test2);
        return true;
    }

private:
    bool test1;
    int64_t test2;
};

constexpr ngraph::NodeTypeInfo SerializationTestOp::type_info;

class SerializationTestInPlaceExtension : public InferenceEngine::IExtension {
public:
    void GetVersion(const InferenceEngine::Version*& versionInfo) const
        noexcept override {}

    void Unload() noexcept override {}

    void Release() noexcept override {}

    std::map<std::string, ngraph::OpSet> getOpSets() override {
        static std::map<std::string, ngraph::OpSet> opsets;
        if (opsets.empty()) {
            ngraph::OpSet opset;
            opset.insert<SerializationTestOp>();
            opsets["serialization_extension"] = opset;
        }
        return opsets;
    }

private:
};
class CustomOpsSerializationTest : public ::testing::Test {
protected:
    std::string test_name =
        ::testing::UnitTest::GetInstance()->current_test_info()->name();
    std::string m_out_xml_path = test_name + ".xml";
    std::string m_out_bin_path = test_name + ".bin";

    void TearDown() override {
        std::remove(m_out_xml_path.c_str());
        std::remove(m_out_bin_path.c_str());
    }
};

// TODO: need to pass extension opsets to transformation in
// CNNetwork->Serialize()
TEST_F(CustomOpsSerializationTest, DISABLED_CustomOpUser) {
    const std::string model = IR_SERIALIZATION_MODELS_PATH "custom_op.xml";

    InferenceEngine::Core ie;
    ie.AddExtension(std::make_shared<SerializationTestInPlaceExtension>());

    auto expected = ie.ReadNetwork(model);
    expected.serialize(m_out_xml_path, m_out_bin_path);
    auto result = ie.ReadNetwork(m_out_xml_path, m_out_bin_path);

    bool success;
    std::string message;
    std::tie(success, message) =
        compare_functions(result.getFunction(), expected.getFunction());

    ASSERT_TRUE(success) << message;
}

TEST_F(CustomOpsSerializationTest, CustomOpTransformation) {
    const std::string model = IR_SERIALIZATION_MODELS_PATH "custom_op.xml";

    InferenceEngine::Core ie;
    auto extension = std::make_shared<SerializationTestInPlaceExtension>();
    ie.AddExtension(extension);
    auto expected = ie.ReadNetwork(model);
    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::Serialize>(
        m_out_xml_path, m_out_bin_path,
        ngraph::pass::Serialize::Version::IR_V10, extension->getOpSets());
    manager.run_passes(expected.getFunction());
    auto result = ie.ReadNetwork(m_out_xml_path, m_out_bin_path);

    bool success;
    std::string message;
    std::tie(success, message) =
        compare_functions(result.getFunction(), expected.getFunction());

    ASSERT_TRUE(success) << message;
}
