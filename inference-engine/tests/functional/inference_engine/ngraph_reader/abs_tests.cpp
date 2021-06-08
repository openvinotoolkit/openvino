// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include "ngraph_reader_tests.hpp"
#include <ngraph/opsets/opset.hpp>
#include <ngraph/ngraph.hpp>
#include <ie_iextension.h>

class FakeAbs : public ngraph::op::Op {
public:
    static constexpr ngraph::NodeTypeInfo type_info{"Abs", 100500};
    const ngraph::NodeTypeInfo& get_type_info() const override { return type_info;  }

    FakeAbs() = default;
    FakeAbs(const ngraph::Output<ngraph::Node>& arg): ngraph::op::Op({arg}) {
        constructor_validate_and_infer_types();
    }
    void validate_and_infer_types() override {
        set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
    }
    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override {
        return std::make_shared<FakeAbs>(new_args.at(0));
    }
    bool visit_attributes(ngraph::AttributeVisitor& visitor) override {
        (void) visitor;
        return true;
    }
};
constexpr ngraph::NodeTypeInfo FakeAbs::type_info;

class AbsFakeExtension: public InferenceEngine::IExtension {
public:
    void GetVersion(const InferenceEngine::Version*& versionInfo) const noexcept override {}
    void Unload() noexcept override {}

    std::map<std::string, ngraph::OpSet> getOpSets() override{
        std::map<std::string, ngraph::OpSet> opsets;
        ngraph::OpSet opset;
        opset.insert<FakeAbs>();
        opsets["experimental"] = opset;
        return opsets;
    }
};

TEST_F(NGraphReaderTests, ReadAbsFromCustomOpsetNetwork) {
    std::string model = R"V0G0N(
<net name="Network" version="10">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset1">
            <data element_type="f32" shape="1,3,22,22"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="Abs" id="1" type="Abs" version="experimental">
            <input>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
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

    Core ie;
    ie.AddExtension(std::make_shared<AbsFakeExtension>());
    Blob::Ptr weights;

    auto network = ie.ReadNetwork(model, weights);
    auto nGraph = network.getFunction();
    bool genericNodeExists = false;
    const std::string type = "Abs";
    for (auto op : nGraph->get_ops()) {
        if (type == op->get_type_info().name && 100500 == op->get_type_info().version)
            genericNodeExists = true;
    }
    ASSERT_TRUE(genericNodeExists);
}

TEST_F(NGraphReaderTests, ReadAbsNetwork) {
    std::string model = R"V0G0N(
<net name="Network" version="10">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset1">
            <data element_type="f32" shape="1,3,22,22"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="Abs" id="1" type="Abs" version="opset1">
            <input>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
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
    std::string modelV5 = R"V0G0N(
<net name="Network" version="5" precision="FP32" batch="1">
    <layers>
        <layer name="in1" type="Input" precision="FP32" id="0">
            <data originalLayersNames="in1"/>
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="Abs" id="1" type="Abs" precision="FP32">
            <data originalLayersNames="Abs"/>
            <input>
                <port id="1">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
    </edges>
</net>
)V0G0N";

    compareIRs(model, modelV5, 0);
}
