// Copyright (C) 2018-2021 Intel Corporation
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
#include "openvino/runtime/core.hpp"

using namespace testing;
using namespace InferenceEngine;
using namespace CommonTestUtils;

using OVExtensionTests = TestsCommon;

namespace {

std::string getOVExtensionPath() {
    return FileUtils::makePluginLibraryName<char>({}, std::string("template_ov_extension") + IE_BUILD_POSTFIX);
}

}  // namespace

class CustomOldTile : public ngraph::op::Op {
public:
    static constexpr ngraph::NodeTypeInfo type_info{"Tile", 0};
    const ngraph::NodeTypeInfo& get_type_info() const override {
        return type_info;
    }

    CustomOldTile() = default;
    CustomOldTile(const ngraph::Output<ngraph::Node>& arg, std::vector<int64_t> repeats)
        : Op({arg}),
          repeats(std::move(repeats)) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        auto input_pshape = get_input_partial_shape(0);
        if (input_pshape.is_static()) {
            auto input_shape = input_pshape.to_shape();
            OPENVINO_ASSERT(input_shape.size() == repeats.size());
            ngraph::Shape output_shape(input_shape);
            for (int i = 0; i < input_shape.size(); ++i) {
                output_shape[i] = input_shape[i] * repeats[i];
            }
            set_output_type(0, get_input_element_type(0), ngraph::PartialShape(output_shape));
        } else {
            set_output_type(0, get_input_element_type(0), ngraph::PartialShape::dynamic());
        }
    }

    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override {
        if (new_args.size() != 1) {
            throw ngraph::ngraph_error("Incorrect number of new arguments");
        }

        return std::make_shared<CustomOldTile>(new_args.at(0), repeats);
    }

    bool visit_attributes(ngraph::AttributeVisitor& visitor) override {
        visitor.on_attribute("repeats", repeats);
        return true;
    }

private:
    std::vector<int64_t> repeats;
};

constexpr ngraph::NodeTypeInfo CustomOldTile::type_info;

class TestTileOldExtension : public InferenceEngine::IExtension {
public:
    void GetVersion(const InferenceEngine::Version*& versionInfo) const noexcept override {}

    void Unload() noexcept override {}

    std::map<std::string, ngraph::OpSet> getOpSets() override {
        static std::map<std::string, ngraph::OpSet> opsets;
        if (opsets.empty()) {
            ngraph::OpSet opset;
            opset.insert<CustomOldTile>();
            opsets["extension"] = opset;
        }
        return opsets;
    }

private:
};

TEST_F(OVExtensionTests, ReshapeIRWithOldExtension) {
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
        <layer name="activation" id="1" type="Tile" version="extension">
            <data repeats="4,3,1,2" test2="2"/>
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
                    <dim>4</dim>
                    <dim>9</dim>
                    <dim>22</dim>
                    <dim>44</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>4</dim>
                    <dim>9</dim>
                    <dim>22</dim>
                    <dim>44</dim>
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
    ov::runtime::Core core;
    OPENVINO_SUPPRESS_DEPRECATED_START
    core.add_extension(std::make_shared<TestTileOldExtension>());
    OPENVINO_SUPPRESS_DEPRECATED_END
    ov::runtime::Tensor weights;
    ov::PartialShape refBeforeReshape{4, 9, 22, 44};
    ov::PartialShape refAfterReshape{8, 9, 33, 66};

    auto network = core.read_model(model, weights);
    std::map<std::string, ov::PartialShape> newShapes;
    newShapes["in_data"] = ov::PartialShape{2, 3, 33, 33};

    EXPECT_EQ(refBeforeReshape, network->output().get_partial_shape());
    EXPECT_NO_THROW(network->reshape(newShapes));
    EXPECT_EQ(refAfterReshape, network->output().get_partial_shape());
}

TEST_F(OVExtensionTests, ReshapeIRWithNewExtension) {
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
        <layer name="activation" id="1" type="Tile" version="extension">
            <data repeats="4,3,1,2" test2="2"/>
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
                    <dim>4</dim>
                    <dim>9</dim>
                    <dim>22</dim>
                    <dim>44</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>4</dim>
                    <dim>9</dim>
                    <dim>22</dim>
                    <dim>44</dim>
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
    ov::runtime::Core core;
    OPENVINO_SUPPRESS_DEPRECATED_START
    core.add_extension(getOVExtensionPath());
    OPENVINO_SUPPRESS_DEPRECATED_END
    ov::runtime::Tensor weights;
    ov::PartialShape refBeforeReshape{4, 9, 22, 44};
    ov::PartialShape refAfterReshape{8, 9, 33, 66};

    auto network = core.read_model(model, weights);
    std::map<std::string, ov::PartialShape> newShapes;
    newShapes["in_data"] = ov::PartialShape{2, 3, 33, 33};

    EXPECT_EQ(refBeforeReshape, network->output().get_partial_shape());
    EXPECT_NO_THROW(network->reshape(newShapes));
    EXPECT_EQ(refAfterReshape, network->output().get_partial_shape());
}
