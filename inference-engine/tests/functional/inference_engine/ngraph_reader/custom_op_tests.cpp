// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <ngraph/ngraph.hpp>
#include <common_test_utils/xml_net_builder/xml_filler.hpp>
#include "ngraph_reader_tests.hpp"

class CustomAddConst : public ngraph::op::Op {
public:
    static constexpr ngraph::NodeTypeInfo type_info{"CustomAddConst", 100600};
    const ngraph::NodeTypeInfo& get_type_info() const override { return type_info;  }
    CustomAddConst() = default;
    CustomAddConst(const ngraph::Output<ngraph::Node>& arg, const ngraph::element::Type element_type,
        const ngraph::Shape shape, const std::shared_ptr<ngraph::runtime::AlignedBuffer> data):
        ngraph::op::Op({arg}),
        m_element_type(element_type),
        m_shape(shape),
        m_data(data) {
        constructor_validate_and_infer_types();
    }
    void validate_and_infer_types() override {
        set_output_type(0, m_element_type, m_shape);
    }
    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override {
        return std::make_shared<CustomAddConst>(new_args.at(0), m_element_type, m_shape, m_data);
    }
    bool visit_attributes(ngraph::AttributeVisitor& visitor) override {
        visitor.on_attribute("element_type", m_element_type);
        visitor.on_attribute("shape", m_shape);
        if (!m_data) {
            m_data = std::make_shared<ngraph::runtime::AlignedBuffer>(shape_size(m_shape) * m_element_type.size(), 64);
        }
        visitor.on_attribute("value", m_data);
        return true;
    }

    ngraph::Shape getShapeAttr() const { return m_shape; }
    void* getDataPtr() { return (m_data ? m_data->get_ptr() : nullptr); }

private:
    ngraph::element::Type m_element_type;
    ngraph::Shape m_shape{};
    std::shared_ptr<ngraph::runtime::AlignedBuffer> m_data;
};

constexpr ngraph::NodeTypeInfo CustomAddConst::type_info;

class CustomAddConstExtension : public InferenceEngine::IExtension {
public:
    void GetVersion(const InferenceEngine::Version*& versionInfo) const noexcept override {}

    void Unload() noexcept override {}

    std::map<std::string, ngraph::OpSet> getOpSets() override {
        std::map<std::string, ngraph::OpSet> opsets;
        ngraph::OpSet opset;
        opset.insert<CustomAddConst>();
        opsets["custom_opset"] = opset;
        return opsets;
    }
};

TEST_F(NGraphReaderTests, ReadCustomAddConstNetwork) {
    std::string model = R"V0G0N(
  <net name="Network" version="10">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset1">
        <data element_type="i32" shape="4"/>
            <output>
                <port id="0" precision="I32">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer name="activation" id="1" type="CustomAddConst" version="custom_opset">
        <data element_type="i32" shape="4" value="_VALUE_"/>
            <input>
                <port id="1" precision="I32">
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="I32">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset1">
            <input>
                <port id="0" precision="I32">
                    <dim>4</dim>
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

    const std::string expectedValue = std::string("0?|%.g6/,-{5~P1>");
    REPLACE_WITH_STR(model, "_VALUE_", expectedValue);
    InferenceEngine::Blob::CPtr weights;

    InferenceEngine::Core ie;
    ie.AddExtension(std::make_shared<CustomAddConstExtension>());
    auto network = ie.ReadNetwork(model, weights);

    bool found = false;
    for (const auto & op : network.getFunction()->get_ops()) {
        if (auto casted = std::dynamic_pointer_cast<CustomAddConst>(op)) {
            std::string actualValue(reinterpret_cast<char *>(casted->getDataPtr()),
                expectedValue.length());
            ASSERT_EQ(expectedValue, actualValue);
            found = true;
        }
    }
    ASSERT_TRUE(found);
}
