// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/xml_util/xml_serialize_util.hpp"

namespace ov {
namespace test {
namespace {

std::string get_prefixed_name(const std::string& custom_name) {
    return std::string{util::rt_map_user_data_prefix} + custom_name;
}
Any& get_user_data(AnyMap& rt_map, const std::string& custom_name) {
    return rt_map.at(get_prefixed_name(custom_name));
}
const Any& get_user_data(const AnyMap& rt_map, const std::string& custom_name) {
    return rt_map.at(get_prefixed_name(custom_name));
}
}  // namespace

TEST(RTInfoCustom, simple_entries) {
    std::string ref_ir_xml = R"V0G0N(
<net name="Network" version="11">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset8">
            <data element_type="f32" shape="27" />
            <rt_info>
                <!-- 'version' tag-attribute presence or value shouldn't matter -->
                <user_data name="infoA" value="A" version="" />
                <user_data name="infoB" value="B" version="0" />
                <user_data name="infoB" value="BB" version="A" />
                <user_data name="fused_names_0" value="a_name" />
                <user_data name="fused_names" value="b_name" />
                <attribute name="fused_names" version="0" value="the_name" />
            </rt_info>
            <output>
                <port id="0" precision="FP32" names="input_tensor">
                    <dim>27</dim>
                </port>
            </output>
        </layer>
        <layer name="Abs" id="1" type="Abs" version="opset8">
            <rt_info>
                <user_data name="infoC" value="C" />
                <attribute name="fused_names" version="0" value="" />
            </rt_info>
            <input>
                <port id="0" precision="FP32">
                    <dim>27</dim>
                </port>
            </input>
            <output>
                <port id="1" precision="FP32" names="output_tensor">
                    <rt_info>
                        <user_data name="infoD" value="D" />
                    </rt_info>
                    <dim>27</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset8">
            <rt_info>
                <attribute name="primitives_priority" version="0" value="the_prior" />
                <user_data name="primitives_priority_0" value="a_prior" />
            </rt_info>
            <input>
                <port id="0" precision="FP32">
                    <dim>27</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0" />
        <edge from-layer="1" from-port="1" to-layer="2" to-port="0" />
    </edges>
</net>
)V0G0N";

    const auto check_model = [](const Model* const model) {
        std::string value;

        const auto& param_rti = model->get_parameters().at(0)->get_rt_info();
        EXPECT_EQ(param_rti.size(), 5);

        OV_ASSERT_NO_THROW(value = param_rti.at("fused_names_0").as<std::string>());
        EXPECT_EQ(value.compare("the_name"), 0);
        OV_ASSERT_NO_THROW(value = get_user_data(param_rti, "fused_names_0").as<std::string>());
        EXPECT_EQ(value.compare("a_name"), 0);
        OV_ASSERT_NO_THROW(value = get_user_data(param_rti, "fused_names").as<std::string>());
        EXPECT_EQ(value.compare("b_name"), 0);

        OV_ASSERT_NO_THROW(value = get_user_data(param_rti, "infoA").as<std::string>());
        EXPECT_EQ(value.compare("A"), 0);
        OV_ASSERT_NO_THROW(value = get_user_data(param_rti, "infoB").as<std::string>());
        EXPECT_EQ(value.compare("B"), 0);

        const auto& result = model->get_results().at(0);
        const auto abs = result->get_input_node_ptr(0);

        const auto& abs_rti = abs->get_rt_info();
        EXPECT_EQ(abs_rti.size(), 2);
        OV_ASSERT_NO_THROW(value = get_user_data(abs_rti, "infoC").as<std::string>());
        EXPECT_EQ(value.compare("C"), 0);

        const auto& abs_output_rti = abs->output(0).get_rt_info();
        EXPECT_EQ(abs_output_rti.size(), 1);
        OV_ASSERT_NO_THROW(value = get_user_data(abs_output_rti, "infoD").as<std::string>());
        EXPECT_EQ(value.compare("D"), 0);

        const auto& result_rti = result->get_rt_info();
        EXPECT_EQ(result_rti.size(), 2);
        OV_ASSERT_NO_THROW(value = result_rti.at("primitives_priority_0").as<std::string>());
        EXPECT_EQ(value.compare("the_prior"), 0);
        OV_ASSERT_NO_THROW(value = get_user_data(result_rti, "primitives_priority_0").as<std::string>());
        EXPECT_EQ(value.compare("a_prior"), 0);
    };

    Core core;
    auto model_0 = core.read_model(ref_ir_xml, Tensor{});
    ASSERT_NE(nullptr, model_0);
    check_model(model_0.get());

    std::stringstream model_s, weights_s;
    pass::Serialize{model_s, weights_s}.run_on_model(model_0);
    const auto model_1 = core.read_model(model_s.str(), Tensor{});
    ASSERT_NE(nullptr, model_1);
    check_model(model_1.get());
}

TEST(RTInfoCustom, nested_entries) {
    std::string ref_ir_xml = R"V0G0N(
<net name="Network" version="11">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset8">
            <data element_type="f32" shape="27" />
            <rt_info>
                <user_data name="infoA" value="A" />
                <user_data name="nested">
                    <user_data name="infoB" value="B" />
                    <user_data name="infoC" value="C" />
                </user_data>
            </rt_info>
            <output>
                <port id="0" precision="FP32" names="input_tensor">
                    <dim>27</dim>
                </port>
            </output>
        </layer>
        <layer name="Abs" id="1" type="Abs" version="opset8">
            <input>
                <port id="0" precision="FP32">
                    <dim>27</dim>
                </port>
            </input>
            <output>
                <port id="1" precision="FP32" names="output_tensor">
                    <rt_info>
                        <user_data name="nested_0">
                            <user_data name="nested_1">
                                <user_data name="infoD" value="D" />
                            </user_data>
                        </user_data>
                    </rt_info>
                    <dim>27</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset8">
            <input>
                <port id="0" precision="FP32">
                    <dim>27</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0" />
        <edge from-layer="1" from-port="1" to-layer="2" to-port="0" />
    </edges>
</net>
)V0G0N";

    auto model = Core{}.read_model(ref_ir_xml, Tensor{});
    ASSERT_NE(nullptr, model);
    std::string value;
    AnyMap any_map;

    const auto& param_rti = model->get_parameters().at(0)->get_rt_info();
    EXPECT_EQ(param_rti.size(), 2);
    OV_ASSERT_NO_THROW(any_map = get_user_data(param_rti, "nested").as<AnyMap>());
    EXPECT_EQ(any_map.size(), 2);
    OV_ASSERT_NO_THROW(value = any_map.at("infoB").as<std::string>());
    EXPECT_EQ(value.compare("B"), 0);
    OV_ASSERT_NO_THROW(value = any_map.at("infoC").as<std::string>());
    EXPECT_EQ(value.compare("C"), 0);

    const auto abs = model->get_results().at(0)->get_input_node_ptr(0);
    const auto& abs_rti = abs->output(0).get_rt_info();
    EXPECT_EQ(abs_rti.size(), 1);
    OV_ASSERT_NO_THROW(any_map = get_user_data(abs_rti, "nested_0").as<AnyMap>());
    EXPECT_EQ(any_map.size(), 1);

    AnyMap nested_map;
    OV_ASSERT_NO_THROW(nested_map = any_map.at("nested_1").as<AnyMap>());
    EXPECT_EQ(nested_map.size(), 1);
    OV_ASSERT_NO_THROW(value = nested_map.at("infoD").as<std::string>());
    EXPECT_EQ(value.compare("D"), 0);
}

TEST(RTInfoCustom, RuntimeAttribute_priority) {
    const auto data = std::make_shared<op::v0::Parameter>(element::Type_t::f64, Shape{111});
    const auto abs = std::make_shared<op::v0::Abs>(data);
    const auto result = std::make_shared<op::v0::Result>(abs);
    const auto model = std::make_shared<Model>(ResultVector{result}, ParameterVector{data});

    auto& info = abs->get_rt_info();
    const auto layout_custom_id = get_prefixed_name("layout");
    const auto layout_custom_value = std::string{"ABCxyz"};
    const auto layout_attribute_id = std::string{LayoutAttribute::get_type_info_static()};
    const auto layout_attribute_value = LayoutAttribute{"NCHW"};
    info[layout_custom_id] = layout_custom_value;
    info[layout_attribute_id] = "CWHN";
    info["L_A_Y_O_U_T"] = layout_attribute_value;

    std::stringstream model_s, weights_s;
    pass::Serialize{model_s, weights_s}.run_on_model(model);
    const auto r_model = Core{}.read_model(model_s.str(), Tensor{});

    const auto& r_abs_rt_info = r_model->get_output_op(0)->input(0).get_source_output().get_node()->get_rt_info();
    EXPECT_EQ(r_abs_rt_info.size(), 2);

    LayoutAttribute la;
    OV_ASSERT_NO_THROW(la = r_abs_rt_info.at(layout_attribute_id).as<LayoutAttribute>());
    EXPECT_EQ(la.to_string(), layout_attribute_value.to_string());

    std::string custom;
    OV_ASSERT_NO_THROW(custom = r_abs_rt_info.at(layout_custom_id).as<std::string>());
    EXPECT_EQ(custom, layout_custom_value);
}
}  // namespace test
}  // namespace ov
