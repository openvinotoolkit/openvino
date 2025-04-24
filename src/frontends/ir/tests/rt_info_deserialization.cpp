// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <sstream>
#include <string>

#include "common_test_utils/graph_comparator.hpp"
#include "openvino/core/preprocess/input_tensor_info.hpp"
#include "openvino/frontend/manager.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/round.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/runtime/core.hpp"
#include "transformations/rt_info/fused_names_attribute.hpp"
#include "transformations/rt_info/old_api_map_element_type_attribute.hpp"
#include "transformations/rt_info/old_api_map_order_attribute.hpp"

class RTInfoDeserialization : public testing::Test {
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

TEST_F(RTInfoDeserialization, node_v10) {
    std::string model = R"V0G0N(
<net name="Network" version="10">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset8">
            <data element_type="f16" shape="1,3,22,22"/>
            <rt_info>
                <attribute name="fused_names" version="0" value="in1"/>
                <attribute name="old_api_map_order" version="0" value="0,2,3,1" />
                <attribute name="old_api_map_element_type" version="0" value="f16"/>
            </rt_info>
            <output>
                <port id="0" precision="FP16" names="input_tensor">
                    <rt_info>
                        <attribute name="layout" version="0" layout="[N,C,H,W]"/>
                    </rt_info>
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="Round" id="1" type="Round" version="opset8">
            <data mode="half_to_even"/>
            <rt_info>
                <attribute name="fused_names" version="0" value="Round1,Round2"/>
            </rt_info>
            <input>
                <port id="1" precision="FP16">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP16" names="output_tensor">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset8">
            <input>
                <port id="0" precision="FP16">
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
    auto f = getWithIRFrontend(model);
    ASSERT_NE(nullptr, f);

    auto check_rt_info = [](const ov::RTMap& info) {
        EXPECT_FALSE(info.count(ov::FusedNames::get_type_info_static()));

        const std::string& key_old_api_order = ov::OldApiMapOrder::get_type_info_static();
        EXPECT_FALSE(info.count(key_old_api_order));
        const std::string& key_old_api_element_type = ov::OldApiMapElementType::get_type_info_static();
        EXPECT_FALSE(info.count(key_old_api_element_type));
    };

    auto check_version = [](const std::shared_ptr<ov::Model>& f, int version_ref) {
        auto& rt_info = f->get_rt_info();
        ASSERT_TRUE(rt_info.count("version"));
        ASSERT_TRUE(rt_info.at("version").is<int64_t>());
        ASSERT_EQ(rt_info.at("version").as<int64_t>(), version_ref);
    };
    check_version(f, 10);

    auto param = f->get_parameters()[0];
    check_rt_info(param->get_rt_info());

    auto result = f->get_results()[0];
    auto round = result->get_input_node_ptr(0);
    check_rt_info(round->get_rt_info());

    // read IR v10 with new API and check that CNNNetwork precision conversions are applied
    {
        ov::Shape shape{1, 3, 22, 22};
        auto type = ov::element::f32;
        auto param = std::make_shared<ov::op::v0::Parameter>(type, shape);
        param->set_friendly_name("in1");
        param->get_output_tensor(0).set_names({"input_tensor", param->get_friendly_name()});

        // TODO: No guarantee that exactly 'Convert' will be added
        auto convert_param = std::make_shared<ov::op::v0::Convert>(param, ov::element::f16);

        auto round = std::make_shared<ov::op::v5::Round>(convert_param, ov::op::v5::Round::RoundMode::HALF_TO_EVEN);

        auto convert_result = std::make_shared<ov::op::v0::Convert>(round, type);
        convert_result->set_friendly_name("Round");
        convert_result->get_output_tensor(0).set_names({"output_tensor", convert_result->get_friendly_name()});

        auto result = std::make_shared<ov::op::v0::Result>(convert_result);
        result->set_friendly_name("output");

        auto f_10_ref = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
        f_10_ref->set_friendly_name("Network");

        ov::Core core;
        auto f_10_core = core.read_model(model, ov::Tensor());
        ASSERT_NE(nullptr, f_10_core);

        check_version(f_10_core, 10);

        const auto fc = FunctionsComparator::with_default()
                            .enable(FunctionsComparator::ATTRIBUTES)
                            .enable(FunctionsComparator::PRECISIONS)
                            .enable(FunctionsComparator::RUNTIME_KEYS)
                            .enable(FunctionsComparator::NAMES)
                            .enable(FunctionsComparator::CONST_VALUES);
        auto res = fc.compare(f_10_core, f_10_ref);
        EXPECT_TRUE(res.valid) << res.message;
    }
}

TEST_F(RTInfoDeserialization, names_collision_v10) {
    std::string model = R"V0G0N(
<net name="Network" version="10">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset8">
            <data element_type="f16" shape="1,3,22,22"/>
            <rt_info>
                <attribute name="fused_names" version="0" value="in1"/>
                <attribute name="old_api_map_order" version="0" value="0,2,3,1" />
                <attribute name="old_api_map_element_type" version="0" value="f16"/>
            </rt_info>
            <output>
                <port id="0" precision="FP16" names="input_tensor">
                    <rt_info>
                        <attribute name="layout" version="0" layout="[N,C,H,W]"/>
                    </rt_info>
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="input_tensor" id="1" type="Round" version="opset8">
            <data mode="half_to_even"/>
            <rt_info>
                <attribute name="fused_names" version="0" value="Round1,Round2"/>
            </rt_info>
            <input>
                <port id="1" precision="FP16">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP16" names="output_tensor">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset8">
            <input>
                <port id="0" precision="FP16">
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
    auto f = getWithIRFrontend(model);
    ASSERT_NE(nullptr, f);

    auto check_version = [](const std::shared_ptr<ov::Model>& f, int version_ref) {
        auto& rt_info = f->get_rt_info();
        ASSERT_TRUE(rt_info.count("version"));
        ASSERT_TRUE(rt_info.at("version").is<int64_t>());
        ASSERT_EQ(rt_info.at("version").as<int64_t>(), version_ref);
    };
    check_version(f, 10);

    // read IR v10 with new API
    {
        ov::Core core;
        EXPECT_THROW(core.read_model(model, ov::Tensor()), ov::Exception);
    }
}

TEST_F(RTInfoDeserialization, input_and_output_v10) {
    std::string model = R"V0G0N(
<net name="Network" version="10">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset8">
            <data element_type="i64" shape="1,3,22,22"/>
            <output>
                <port id="0" precision="I64" names="input_tensor">
                    <rt_info>
                        <attribute name="fused_names" version="0" value="test1,test2"/>
                        <attribute name="layout" version="0" layout="[N,C,H,W]" />
                    </rt_info>
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="sum" type="Add" version="opset1">
            <input>
                <port id="0">
                    <rt_info>
                        <attribute name="fused_names" version="0" value="test2,test3"/>
                    </rt_info>
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
                <port id="1">
                    <rt_info>
                        <attribute name="fused_names" version="0" value="test3,test4"/>
                    </rt_info>
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="I64" names="output_tensor">
                    <rt_info>
                        <attribute name="fused_names" version="0" value="test4,test5"/>
                    </rt_info>
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset8">
            <input>
                <port id="0" precision="I64">
                    <rt_info>
                        <attribute name="fused_names" version="0" value="test5,test6"/>
                        <attribute name="layout" version="0" layout="[?,C,H,W]" />
                    </rt_info>
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";
    auto f = getWithIRFrontend(model);
    ASSERT_NE(nullptr, f);

    auto check_rt_info = [](const ov::RTMap& info) {
        ASSERT_FALSE(info.count(ov::FusedNames::get_type_info_static()));
    };

    auto check_version = [](const std::shared_ptr<ov::Model>& f, int ref_version) {
        auto& rt_info = f->get_rt_info();
        ASSERT_TRUE(rt_info.count("version"));
        ASSERT_TRUE(rt_info.at("version").is<int64_t>());
        ASSERT_EQ(rt_info.at("version").as<int64_t>(), ref_version);
    };
    check_version(f, 10);

    auto param = f->get_parameters()[0];
    EXPECT_EQ(param->get_layout(), "");
    check_rt_info(param->output(0).get_rt_info());

    auto result = f->get_results()[0];
    EXPECT_EQ(result->get_layout(), "");
    check_rt_info(result->input(0).get_rt_info());

    auto add = result->get_input_node_ptr(0);
    check_rt_info(add->input(0).get_rt_info());
    check_rt_info(add->input(1).get_rt_info());
    check_rt_info(add->output(0).get_rt_info());

    // read IR v10 with new API and check that CNNNetwork precision conversions are applied
    {
        const ov::Shape shape{1, 3, 22, 22};
        const auto type = ov::element::i64;
        auto param = std::make_shared<ov::op::v0::Parameter>(type, shape);
        param->set_friendly_name("in1");
        param->get_output_tensor(0).set_names({"input_tensor", param->get_friendly_name()});

        auto sum = std::make_shared<ov::op::v1::Add>(param, param);

        // TODO: No guarantee that exactly 'convert' will be added by post-processing
        auto convert_result = std::make_shared<ov::op::v0::Convert>(sum, ov::element::i32);
        convert_result->set_friendly_name("sum");
        convert_result->get_output_tensor(0).set_names({"output_tensor", convert_result->get_friendly_name()});

        auto result = std::make_shared<ov::op::v0::Result>(convert_result);
        result->set_friendly_name("output");

        auto f_10_ref = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
        f_10_ref->set_friendly_name("Network");

        ov::Core core;
        auto f_10_core = core.read_model(model, ov::Tensor());
        ASSERT_NE(nullptr, f_10_core);
        check_version(f_10_core, 10);

        const auto fc = FunctionsComparator::with_default()
                            .enable(FunctionsComparator::ATTRIBUTES)
                            .enable(FunctionsComparator::PRECISIONS)
                            .enable(FunctionsComparator::RUNTIME_KEYS)
                            .enable(FunctionsComparator::NAMES)
                            .enable(FunctionsComparator::CONST_VALUES);
        auto res = fc.compare(f_10_core, f_10_ref);
        EXPECT_TRUE(res.valid) << res.message;
    }
}

TEST_F(RTInfoDeserialization, node_v11) {
    std::string model = R"V0G0N(
<net name="Network" version="11">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset8">
            <data element_type="f32" shape="1,22,22,3"/>
            <rt_info>
                <attribute name="old_api_map_element_type" version="0" value="f16"/>
                <attribute name="old_api_map_order" version="0" value="0,2,3,1"/>
                <attribute name="fused_names" version="0" value="in1"/>
                <attribute name="if no version" value="then ignore"/>
                <attribute name="fused_names" version="1" comment="unknown version"/>
            </rt_info>
            <output>
                <port id="0" precision="FP32" names="input_tensor">
                    <rt_info>
                        <attribute name="layout" version="0" layout="[N,H,W,C]"/>
                        <no_name version="0" value="param1"/>
                        <empty_node/>
                    </rt_info>
                    <dim>1</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                    <dim>3</dim>
                </port>
            </output>
        </layer>
        <layer name="Round" id="1" type="Round" version="opset8">
            <data mode="half_to_even"/>
            <rt_info>
                <attribute name="fused_names" version="0" value="    Round 1  ,  Round 2  "/>
            </rt_info>
            <input>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                    <dim>3</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32" names="output_tensor">
                    <rt_info>
                        <attribute name="layout" version="0" layout="[N,H,W,C]"/>
                    </rt_info>
                    <dim>1</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                    <dim>3</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset8">
            <rt_info>
                <attribute name="old_api_map_order" version="0" value="0,3,1,2"/>
            </rt_info>
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                    <dim>3</dim>
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

    auto check_fused_names = [](const ov::RTMap& info, const std::string& names) {
        ASSERT_TRUE(info.count(ov::FusedNames::get_type_info_static()));
        auto fused_names_attr = info.at(ov::FusedNames::get_type_info_static()).as<ov::FusedNames>();
        EXPECT_EQ(fused_names_attr.getNames(), names);
    };
    auto check_old_api_map_order = [](const ov::RTMap& info, const std::vector<uint64_t>& order) {
        const std::string& old_api_map_key = ov::OldApiMapOrder::get_type_info_static();
        ASSERT_TRUE(info.count(old_api_map_key));
        auto old_api_map_attr_val = info.at(old_api_map_key).as<ov::OldApiMapOrder>().value;
        EXPECT_EQ(old_api_map_attr_val, order);
    };
    auto check_old_api_map_type = [](const ov::RTMap& info, const ov::element::Type& type) {
        const std::string& old_api_map_key = ov::OldApiMapElementType::get_type_info_static();
        ASSERT_TRUE(info.count(old_api_map_key));
        auto old_api_map_attr_val = info.at(old_api_map_key).as<ov::OldApiMapElementType>().value;
        EXPECT_EQ(old_api_map_attr_val, type);
    };

    auto check_version = [](const std::shared_ptr<ov::Model>& f, int ref_version) {
        auto& rt_info = f->get_rt_info();
        ASSERT_TRUE(rt_info.count("version"));
        ASSERT_TRUE(rt_info.at("version").is<int64_t>());
        ASSERT_EQ(rt_info.at("version").as<int64_t>(), ref_version);
    };
    check_version(f, 11);

    auto param = f->get_parameters()[0];
    check_fused_names(param->get_rt_info(), "in1");
    check_old_api_map_order(param->get_rt_info(), std::vector<uint64_t>({0, 2, 3, 1}));
    check_old_api_map_type(param->get_rt_info(), ov::element::Type_t::f16);

    auto result = f->get_result();
    check_old_api_map_order(result->get_rt_info(), std::vector<uint64_t>({0, 3, 1, 2}));
    auto round = result->get_input_node_ptr(0);
    check_fused_names(round->get_rt_info(), "Round 1,Round 2");

    // read IR v11 with new API
    {
        ov::Core core;
        auto f_11 = core.read_model(model, ov::Tensor());
        ASSERT_NE(nullptr, f_11);

        check_old_api_map_order(f_11->get_parameters()[0]->get_rt_info(), std::vector<uint64_t>({0, 2, 3, 1}));
        check_old_api_map_type(f_11->get_parameters()[0]->get_rt_info(), ov::element::Type_t::f16);

        check_old_api_map_order(f_11->get_result()->get_rt_info(), std::vector<uint64_t>({0, 3, 1, 2}));

        auto res = compare_functions(f, f_11);
        EXPECT_TRUE(res.first) << res.second;

        check_version(f_11, 11);
    }
}

TEST_F(RTInfoDeserialization, node_v11_multiple_rt_keys) {
    std::string model = R"V0G0N(
<net name="Network" version="11">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset8">
            <data element_type="f32" shape="1,22,22,3"/>
            <rt_info>
                <attribute name="old_api_map_order" version="0" value="0,2,3,1"/>
                <attribute name="old_api_map_order" version="0" value="0,1,2,3"/>
                <attribute name="fused_names" version="0" value="in1"/>
            </rt_info>
            <output>
                <port id="0" precision="FP32" names="input_tensor">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="Round" id="1" type="Round" version="opset8">
            <data mode="half_to_even"/>
            <rt_info>
                <attribute name="fused_names" version="0" value="Round1,Round2"/>
            </rt_info>
            <input>
                <rt_info>
                    <attribute name="fused_names" version="0" value="check"/>
                    <attribute name="fused_names" version="0" value="multiple_keys"/>
                </rt_info>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32" names="output_tensor">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset8">
            <rt_info>
                <attribute name="old_api_map_order" version="0" value="0,3,1,2"/>
            </rt_info>
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
    ASSERT_THROW(getWithIRFrontend(model), ov::Exception);
}

TEST_F(RTInfoDeserialization, input_and_output_v11) {
    std::string model = R"V0G0N(
<net name="Network" version="11">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset8">
            <data element_type="f32" shape="1,3,22,22"/>
            <output>
                <port id="0" precision="FP32">
                    <rt_info>
                        <attribute name="fused_names" version="0" value="test1,test2"/>
                        <attribute name="layout" version="0" layout="[N,C,H,W]" />
                        <attribute name="memory_type" version="0" value="test_memory_type" />
                    </rt_info>
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="sum" type="Add" version="opset1">
            <input>
                <port id="0">
                    <rt_info>
                        <attribute name="fused_names" version="0" value="test2,test3"/>
                    </rt_info>
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
                <port id="1">
                    <rt_info>
                        <attribute name="fused_names" version="0" value="test3,test4"/>
                    </rt_info>
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <rt_info>
                        <attribute name="fused_names" version="0" value="test4,test5"/>
                        <attribute name="layout" version="0" layout="[?,C,H,W]" />
                    </rt_info>
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset8">
            <input>
                <port id="0" precision="FP32">
                    <rt_info>
                        <attribute name="fused_names" version="0" value="test5,test6"/>
                    </rt_info>
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";
    auto f = getWithIRFrontend(model);
    ASSERT_NE(nullptr, f);

    auto check_version = [](const std::shared_ptr<ov::Model>& f, int ref_version) {
        auto& rt_info = f->get_rt_info();
        ASSERT_TRUE(rt_info.count("version"));
        ASSERT_TRUE(rt_info.at("version").is<int64_t>());
        ASSERT_EQ(rt_info.at("version").as<int64_t>(), ref_version);
    };
    check_version(f, 11);

    auto check_fused_names = [](const ov::RTMap& info, const std::string& names) {
        const std::string& key = ov::FusedNames::get_type_info_static();
        ASSERT_TRUE(info.count(key));
        auto fused_names_attr = info.at(key).as<ov::FusedNames>();
        ASSERT_EQ(fused_names_attr.getNames(), names);
    };

    auto param = f->get_parameters()[0];
    check_fused_names(param->output(0).get_rt_info(), "test1,test2");
    EXPECT_EQ(param->get_layout(), "NCHW");
    auto var0 = f->input(0)
                    .get_rt_info()
                    .at(ov::preprocess::TensorInfoMemoryType::get_type_info_static())
                    .as<ov::preprocess::TensorInfoMemoryType>()
                    .value;
    EXPECT_EQ(var0, "test_memory_type");

    auto result = f->get_result();
    check_fused_names(result->input(0).get_rt_info(), "test5,test6");
    EXPECT_EQ(f->get_results()[0]->get_layout(), "?CHW");

    auto add = result->get_input_node_ptr(0);
    check_fused_names(add->input(0).get_rt_info(), "test2,test3");
    check_fused_names(add->input(1).get_rt_info(), "test3,test4");
    check_fused_names(add->output(0).get_rt_info(), "test4,test5");
}

TEST_F(RTInfoDeserialization, indexes_input_and_output_v11) {
    std::string model = R"V0G0N(
<net name="Network" version="11">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset8">
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
        <layer name="in2" type="Parameter" id="1" version="opset8">
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
        <layer id="2" name="sum" type="Add" version="opset1">
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
                <port id="1">
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
        <layer id="4" name="relu" type="Relu" version="opset8">
            <input>
                <port id="0">
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
        <layer name="output2" type="Result" id="5" version="opset8">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
        </layer>
        <layer name="output1" type="Result" id="3" version="opset8">
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
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
        <edge from-layer="2" from-port="2" to-layer="3" to-port="0"/>
        <edge from-layer="2" from-port="2" to-layer="4" to-port="0"/>
        <edge from-layer="4" from-port="2" to-layer="5" to-port="0"/>
    </edges>
</net>
)V0G0N";
    auto f = getWithIRFrontend(model);
    ASSERT_NE(nullptr, f);

    auto check_version = [](const std::shared_ptr<ov::Model>& f, int ref_version) {
        auto& rt_info = f->get_rt_info();
        ASSERT_TRUE(rt_info.count("version"));
        ASSERT_TRUE(rt_info.at("version").is<int64_t>());
        ASSERT_EQ(rt_info.at("version").as<int64_t>(), ref_version);
    };
    check_version(f, 11);

    ASSERT_EQ(2, f->get_parameters().size());
    ASSERT_EQ(f->get_parameters()[0]->get_friendly_name(), "in1");
    ASSERT_EQ(f->get_parameters()[1]->get_friendly_name(), "in2");

    ASSERT_EQ(2, f->get_results().size());
    ASSERT_EQ(f->get_results()[0]->get_friendly_name(), "output2");
    ASSERT_EQ(f->get_results()[1]->get_friendly_name(), "output1");
}

TEST_F(RTInfoDeserialization, node_naming_v11) {
    std::string model = R"V0G0N(
<net name="Network" version="11">
	<layers>
		<layer id="0" name="input" type="Parameter" version="opset1">
			<data shape="1,3,224,224" element_type="f32" />
			<output>
				<port id="0" precision="FP32" names="input">
					<dim>1</dim>
					<dim>3</dim>
					<dim>224</dim>
					<dim>224</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="output" type="Result" version="opset1">
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>3</dim>
					<dim>224</dim>
					<dim>224</dim>
				</port>
			</input>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0" />
	</edges>
	<rt_info>
		<framework>
			<item0 value="0" /> <!-- Old-style notation, may cause an issue if starts with an unexpected character. Compatibility. -->
			<info name="item1" value="1" /> <!-- New-style notation, name can contain any characters -->
		</framework>
		<info name="conversion_parameters"> <!-- New-style whole block -->
			<info name="input_model" value="DIR\model.onnx" />
			<info name="is_python_api_used" value="True" />
		</info>
	</rt_info>
</net>
)V0G0N";
    auto f = getWithIRFrontend(model);
    ASSERT_NE(nullptr, f);

    auto check_version = [](const std::shared_ptr<ov::Model>& f, int ref_version) {
        auto& rt_info = f->get_rt_info();
        ASSERT_TRUE(rt_info.count("version"));
        ASSERT_TRUE(rt_info.at("version").is<int64_t>());
        ASSERT_EQ(rt_info.at("version").as<int64_t>(), ref_version);
    };
    check_version(f, 11);

    auto& rt_info = f->get_rt_info();
    ASSERT_TRUE(rt_info.count("framework"));
    ASSERT_TRUE(rt_info.count("conversion_parameters"));

    auto& item0 = f->get_rt_info<std::string>("framework", "item0");
    ASSERT_EQ(item0, "0");

    auto& item1 = f->get_rt_info<std::string>("framework", "item1");
    ASSERT_EQ(item1, "1");

    auto& is_python_api_used = f->get_rt_info<std::string>("conversion_parameters", "is_python_api_used");
    ASSERT_EQ(is_python_api_used, "True");
}
