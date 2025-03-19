// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common_test_utils/graph_comparator.hpp"
#include "openvino/core/preprocess/input_tensor_info.hpp"
#include "openvino/frontend/manager.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/runtime/core.hpp"

class PartialShapeDeserialization : public testing::Test {
protected:
    std::shared_ptr<ov::Model> getWithIRFrontend(const std::string& model) {
        std::istringstream modelStringStream(model);
        std::istream& modelStream = modelStringStream;

        ov::frontend::FrontEnd::Ptr FE;
        ov::frontend::InputModel::Ptr inputModel;

        ov::AnyVector params{&modelStream};

        FE = manager.load_by_model(params);
        if (FE)
            inputModel = FE->load(params);

        if (inputModel)
            return FE->convert(inputModel);

        return nullptr;
    }

private:
    ov::frontend::FrontEndManager manager;
};

TEST_F(PartialShapeDeserialization, shape_with_boundaries_test_case1) {
    std::string model = R"V0G0N(
<net name="Network" version="11">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset8">
            <data element_type="f16" shape="1,3,100..200,120..320"/>
            <output>
                <port id="0" precision="FP16" names="input_tensor">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>-1</dim>
                    <dim>-1</dim>
                </port>
            </output>
        </layer>
        <layer name="Round" id="1" type="Round" version="opset8">
            <data mode="half_to_even"/>
            <input>
                <port id="1" precision="FP16">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>-1</dim>
                    <dim>-1</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP16" names="output_tensor">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>-1</dim>
                    <dim>-1</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset8">
            <input>
                <port id="0" precision="FP16">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>-1</dim>
                    <dim>-1</dim>
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
    auto f = getWithIRFrontend(model);
    ASSERT_NE(nullptr, f);

    ov::PartialShape shape{1, 3, ov::Dimension(100, 200), ov::Dimension(120, 320)};
    auto type = ov::element::f16;
    auto param = std::make_shared<ov::opset8::Parameter>(type, shape);

    param->set_friendly_name("in1");
    param->get_output_tensor(0).set_names({"input_tensor"});

    auto round = std::make_shared<ov::opset8::Round>(param, ov::opset8::Round::RoundMode::HALF_TO_EVEN);

    round->set_friendly_name("Round");
    round->get_output_tensor(0).set_names({"output_tensor"});

    auto result = std::make_shared<ov::opset8::Result>(round);
    result->set_friendly_name("output");

    auto f_11_ref = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
    f_11_ref->set_friendly_name("Network");

    const auto fc = FunctionsComparator::with_default()
                        .enable(FunctionsComparator::ATTRIBUTES)
                        .enable(FunctionsComparator::PRECISIONS)
                        .enable(FunctionsComparator::RUNTIME_KEYS)
                        .enable(FunctionsComparator::NAMES)
                        .enable(FunctionsComparator::CONST_VALUES);
    auto res = fc.compare(f, f_11_ref);
    EXPECT_TRUE(res.valid) << res.message;
}

TEST_F(PartialShapeDeserialization, shape_with_boundaries_testC_case2) {
    std::string model = R"V0G0N(
<net name="Network" version="11">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset8">
            <data element_type="f16" shape="1,?,..200,120.."/>
            <output>
                <port id="0" precision="FP16" names="input_tensor">
                    <dim>1</dim>
                    <dim>-1</dim>
                    <dim>-1</dim>
                    <dim>-1</dim>
                </port>
            </output>
        </layer>
        <layer name="Round" id="1" type="Round" version="opset8">
            <data mode="half_to_even"/>
            <input>
                <port id="1" precision="FP16">
                    <dim>1</dim>
                    <dim>-1</dim>
                    <dim>-1</dim>
                    <dim>-1</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP16" names="output_tensor">
                    <dim>1</dim>
                    <dim>-1</dim>
                    <dim>-1</dim>
                    <dim>-1</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset8">
            <input>
                <port id="0" precision="FP16">
                    <dim>1</dim>
                    <dim>-1</dim>
                    <dim>-1</dim>
                    <dim>-1</dim>
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
    auto f = getWithIRFrontend(model);
    ASSERT_NE(nullptr, f);

    ov::PartialShape shape{1, ov::Dimension(), ov::Dimension(0, 200), ov::Dimension(120, -1)};
    auto type = ov::element::f16;
    auto param = std::make_shared<ov::opset8::Parameter>(type, shape);

    param->set_friendly_name("in1");
    param->get_output_tensor(0).set_names({"input_tensor"});

    auto round = std::make_shared<ov::opset8::Round>(param, ov::opset8::Round::RoundMode::HALF_TO_EVEN);

    round->set_friendly_name("Round");
    round->get_output_tensor(0).set_names({"output_tensor"});

    auto result = std::make_shared<ov::opset8::Result>(round);
    result->set_friendly_name("output");

    auto f_11_ref = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
    f_11_ref->set_friendly_name("Network");

    const auto fc = FunctionsComparator::with_default()
                        .enable(FunctionsComparator::ATTRIBUTES)
                        .enable(FunctionsComparator::PRECISIONS)
                        .enable(FunctionsComparator::RUNTIME_KEYS)
                        .enable(FunctionsComparator::NAMES)
                        .enable(FunctionsComparator::CONST_VALUES);
    auto res = fc.compare(f, f_11_ref);
    EXPECT_TRUE(res.valid) << res.message;
}

TEST_F(PartialShapeDeserialization, shape_with_boundaries_test_dynamic_rank) {
    std::string model = R"V0G0N(
<net name="Network" version="11">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset8">
            <data element_type="f16" shape="..."/>
            <output>
                <port id="0" precision="FP16" names="input_tensor">
                </port>
            </output>
        </layer>
        <layer name="Round" id="1" type="Round" version="opset8">
            <data mode="half_to_even"/>
            <input>
                <port id="1" precision="FP16">
                </port>
            </input>
            <output>
                <port id="2" precision="FP16" names="output_tensor">
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset8">
            <input>
                <port id="0" precision="FP16">
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
    auto f = getWithIRFrontend(model);
    ASSERT_NE(nullptr, f);

    ov::PartialShape shape = ov::PartialShape::dynamic();
    auto type = ov::element::f16;
    auto param = std::make_shared<ov::opset8::Parameter>(type, shape);

    param->set_friendly_name("in1");
    param->get_output_tensor(0).set_names({"input_tensor"});

    auto round = std::make_shared<ov::opset8::Round>(param, ov::opset8::Round::RoundMode::HALF_TO_EVEN);

    round->set_friendly_name("Round");
    round->get_output_tensor(0).set_names({"output_tensor"});

    auto result = std::make_shared<ov::opset8::Result>(round);
    result->set_friendly_name("output");

    auto f_11_ref = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
    f_11_ref->set_friendly_name("Network");

    const auto fc = FunctionsComparator::with_default()
                        .enable(FunctionsComparator::ATTRIBUTES)
                        .enable(FunctionsComparator::PRECISIONS)
                        .enable(FunctionsComparator::RUNTIME_KEYS)
                        .enable(FunctionsComparator::NAMES)
                        .enable(FunctionsComparator::CONST_VALUES);
    auto res = fc.compare(f, f_11_ref);
    EXPECT_TRUE(res.valid) << res.message;
}

TEST_F(PartialShapeDeserialization, shape_with_boundaries_test_dynamic_rank_negative) {
    std::string model = R"V0G0N(
<net name="Network" version="11">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset8">
            <data element_type="f16" shape="...,..."/>
            <output>
                <port id="0" precision="FP16" names="input_tensor">
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="1" version="opset8">
            <input>
                <port id="0" precision="FP16">
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
    </edges>
</net>
)V0G0N";
    // TODO: change to ov::Exception (69781)
    ASSERT_THROW(getWithIRFrontend(model), ov::Exception);
}

TEST_F(PartialShapeDeserialization, shape_with_boundaries_test_dynamic_dim_negative) {
    std::string model = R"V0G0N(
<net name="Network" version="11">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset8">
            <data element_type="f16" shape="1,...,2"/>
            <output>
                <port id="0" precision="FP16" names="input_tensor">
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="1" version="opset8">
            <input>
                <port id="0" precision="FP16">
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
    </edges>
</net>
)V0G0N";
    // TODO: change to ov::Exception (69781)
    ASSERT_THROW(getWithIRFrontend(model), ov::Exception);
}

TEST_F(PartialShapeDeserialization, shape_with_boundaries_test_wrong_dim) {
    std::string model = R"V0G0N(
<net name="Network" version="11">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset8">
            <data element_type="f16" shape="1s,2"/>
            <output>
                <port id="0" precision="FP16" names="input_tensor">
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="1" version="opset8">
            <input>
                <port id="0" precision="FP16">
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
    </edges>
</net>
)V0G0N";
    // TODO: change to ov::Exception (69781)
    ASSERT_THROW(getWithIRFrontend(model), ov::Exception);
}

TEST_F(PartialShapeDeserialization, shape_with_boundaries_test_wrong_boundary) {
    std::string model = R"V0G0N(
<net name="Network" version="11">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset8">
            <data element_type="f16" shape="1..g,2"/>
            <output>
                <port id="0" precision="FP16" names="input_tensor">
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="1" version="opset8">
            <input>
                <port id="0" precision="FP16">
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
    </edges>
</net>
)V0G0N";
    // TODO: change to ov::Exception (69781)
    ASSERT_THROW(getWithIRFrontend(model), ov::Exception);
}
