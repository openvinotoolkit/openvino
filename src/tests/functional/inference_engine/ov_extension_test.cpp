// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "common_test_utils/test_common.hpp"
#include "file_utils.h"
#include "ie_iextension.h"
#include "ngraph/op/op.hpp"
#include "openvino/core/op_extension.hpp"
#include "openvino/runtime/core.hpp"

using namespace testing;
using namespace InferenceEngine;
using namespace CommonTestUtils;

class OVExtensionTests : public TestsCommon {
public:
    ov::Core core;

    void test() {
        std::string model = R"V0G0N(
<net name="Activation" version="10">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset1">
            <data shape="1,3,22,22" element_type="f32"/>
            <output>
                <port id="0" precision="FP32" names="in_data">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="activation" id="1" type="Identity" version="extension">
            <input>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32" names="out_data">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
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
        ov::Tensor weights;
        ov::PartialShape refBeforeReshape{1, 3, 22, 22};
        ov::PartialShape refAfterReshape{8, 9, 33, 66};

        auto network = core.read_model(model, weights);
        std::map<std::string, ov::PartialShape> newShapes;
        newShapes["in_data"] = refAfterReshape;

        EXPECT_EQ(refBeforeReshape, network->output().get_partial_shape());
        EXPECT_NO_THROW(network->reshape(newShapes));
        EXPECT_EQ(refAfterReshape, network->output().get_partial_shape());
    }

    void test_two_op() {
        std::string model = R"V0G0N(
<net name="Activation" version="10">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset1">
            <data shape="1,3,22,22" element_type="f32"/>
            <output>
                <port id="0" precision="FP32" names="in_data">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="activation" id="1" type="Identity" version="extension">
            <input>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32" names="out_data">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="activation2" id="2" type="CustomReLU" version="extension">
            <input>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32" names="out_relu_data">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="3" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="1"/>
        <edge from-layer="2" from-port="2" to-layer="3" to-port="0"/>
    </edges>
</net>
)V0G0N";
        ov::Tensor weights;
        ov::PartialShape refBeforeReshape{1, 3, 22, 22};
        ov::PartialShape refAfterReshape{8, 9, 33, 66};

        auto network = core.read_model(model, weights);
        std::map<std::string, ov::PartialShape> newShapes;
        newShapes["in_data"] = refAfterReshape;

        EXPECT_EQ(refBeforeReshape, network->output().get_partial_shape());
        EXPECT_NO_THROW(network->reshape(newShapes));
        EXPECT_EQ(refAfterReshape, network->output().get_partial_shape());
    }
};

namespace {

std::string getOVExtensionPath() {
    return FileUtils::makePluginLibraryName<char>({}, std::string("openvino_template_extension") + IE_BUILD_POSTFIX);
}

}  // namespace

class CustomOldIdentity : public ngraph::op::Op {
public:
    static constexpr ngraph::NodeTypeInfo type_info{"Identity", static_cast<uint64_t>(0)};
    const ngraph::NodeTypeInfo& get_type_info() const override {
        return type_info;
    }

    CustomOldIdentity() = default;
    CustomOldIdentity(const ngraph::Output<ngraph::Node>& arg) : Op({arg}) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
    }

    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override {
        if (new_args.size() != 1) {
            throw ngraph::ngraph_error("Incorrect number of new arguments");
        }

        return std::make_shared<CustomOldIdentity>(new_args.at(0));
    }

    bool visit_attributes(ngraph::AttributeVisitor& visitor) override {
        return true;
    }
};

constexpr ngraph::NodeTypeInfo CustomOldIdentity::type_info;

class TestTileOldExtension : public InferenceEngine::IExtension {
public:
    void GetVersion(const InferenceEngine::Version*& versionInfo) const noexcept override {}

    void Unload() noexcept override {}

    std::map<std::string, ngraph::OpSet> getOpSets() override {
        static std::map<std::string, ngraph::OpSet> opsets;
        if (opsets.empty()) {
            ngraph::OpSet opset;
            opset.insert<CustomOldIdentity>();
            opsets["extension"] = opset;
        }
        return opsets;
    }
};

class CustomNewIdentity : public ov::op::Op {
public:
    OPENVINO_OP("Identity")

    CustomNewIdentity() = default;
    CustomNewIdentity(const ov::Output<ov::Node>& arg) : Op({arg}) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override {
        OPENVINO_ASSERT(new_args.size() != 1, "Incorrect number of new arguments");

        return std::make_shared<CustomNewIdentity>(new_args.at(0));
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        return true;
    }
};

class CustomReLU : public ov::op::Op {
public:
    OPENVINO_OP("CustomReLU")

    CustomReLU() = default;
    CustomReLU(const ov::Output<ov::Node>& arg) : Op({arg}) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override {
        OPENVINO_ASSERT(new_args.size() != 1, "Incorrect number of new arguments");

        return std::make_shared<CustomReLU>(new_args.at(0));
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        return true;
    }
};

TEST_F(OVExtensionTests, ReshapeIRWithOldExtension) {
    OPENVINO_SUPPRESS_DEPRECATED_START
    core.add_extension(std::make_shared<TestTileOldExtension>());
    OPENVINO_SUPPRESS_DEPRECATED_END
    test();
}

TEST_F(OVExtensionTests, ReshapeIRWithNewExtensionsLib) {
    core.add_extension(getOVExtensionPath());
    test();
}

TEST_F(OVExtensionTests, ReshapeIRWithNewExtensionPtr) {
    core.add_extension(std::make_shared<ov::OpExtension<CustomNewIdentity>>());
    test();
}

TEST_F(OVExtensionTests, ReshapeIRWithNewExtension) {
    core.add_extension(ov::OpExtension<CustomNewIdentity>());
    test();
}

TEST_F(OVExtensionTests, ReshapeIRWithNewOp) {
    core.add_extension<CustomNewIdentity>();
    test();
}

TEST_F(OVExtensionTests, IncorrectReshapeIRWithNewExtensionPtr) {
    core.add_extension(std::make_shared<ov::OpExtension<CustomNewIdentity>>());
    EXPECT_ANY_THROW(test_two_op());
}

TEST_F(OVExtensionTests, IncorrectReshapeIRWithNewExtension) {
    core.add_extension(ov::OpExtension<CustomNewIdentity>());
    EXPECT_ANY_THROW(test_two_op());
}

TEST_F(OVExtensionTests, IncorrectReshapeIRWithNewOp) {
    core.add_extension<CustomNewIdentity>();
    EXPECT_ANY_THROW(test_two_op());
}

TEST_F(OVExtensionTests, ReshapeIRWithSeveralNewExtensionPtrs) {
    core.add_extension(
        {std::make_shared<ov::OpExtension<CustomNewIdentity>>(), std::make_shared<ov::OpExtension<CustomReLU>>()});
    test_two_op();
}

TEST_F(OVExtensionTests, ReshapeIRWithSeveralNewExtensions) {
    core.add_extension(ov::OpExtension<CustomNewIdentity>(), ov::OpExtension<CustomReLU>());
    test_two_op();
}

TEST_F(OVExtensionTests, ReshapeIRWithSeveralNewOps) {
    core.add_extension<CustomNewIdentity, CustomReLU>();
    test_two_op();
}
