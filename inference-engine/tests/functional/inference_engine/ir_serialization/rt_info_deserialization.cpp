// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <sstream>
#include <string>
#include <memory>

#include <inference_engine.hpp>
#include <memory>
#include <ngraph/opsets/opset8.hpp>
#include <string>
#include <transformations/rt_info/fused_names_attribute.hpp>
#include <transformations/rt_info/old_api_map_attribute.hpp>
#include "frontend_manager/frontend_manager.hpp"
#include "graph_comparator.hpp"
#include "ie_blob.h"
#include "ie_precision.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/pass/pass.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"
#include "ngraph/variant.hpp"
#include "ngraph/pass/manager.hpp"
#include "openvino/runtime/core.hpp"

using namespace ngraph;

class RTInfoDeserialization : public testing::Test {
protected:
    std::shared_ptr<ngraph::Function> getWithIRFrontend(const std::string& model) {
        std::istringstream modelStringStream(model);
        std::istream& modelStream = modelStringStream;

        ngraph::frontend::FrontEnd::Ptr FE;
        ngraph::frontend::InputModel::Ptr inputModel;

        ov::VariantVector params{ov::make_variant(&modelStream)};

        FE = manager.load_by_model(params);
        if (FE)
            inputModel = FE->load(params);

        if (inputModel)
            return FE->convert(inputModel);

        return nullptr;
    }

private:
    ngraph::frontend::FrontEndManager manager;
};

TEST_F(RTInfoDeserialization, NodeV10) {
    std::string model = R"V0G0N(
<net name="Network" version="10">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset8">
            <data element_type="f16" shape="1,3,22,22"/>
            <rt_info>
                <attribute name="fused_names" version="0" value="in1"/>
                <attribute name="old_api_map" version="0" order="0,2,3,1" element_type="f16"/>
            </rt_info>
            <output>
                <port id="0" precision="FP16" names="input_tensor">
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

    auto check_rt_info = [](const RTMap& info) {
        const std::string& key = VariantWrapper<ngraph::FusedNames>::get_type_info_static();
        EXPECT_FALSE(info.count(key));

        const std::string& key_old_api = ov::OldApiMap::get_type_info_static();
        EXPECT_FALSE(info.count(key_old_api));
    };

    auto check_version = [](const std::shared_ptr<ov::Function>& f, int version_ref) {
        auto& rt_info = f->get_rt_info();
        ASSERT_TRUE(rt_info.count("version"));
        auto version = std::dynamic_pointer_cast<VariantWrapper<int64_t>>(rt_info.at("version"));
        ASSERT_NE(version, nullptr);
        ASSERT_EQ(version->get(), version_ref);
    };
    check_version(f, 10);

    auto param = f->get_parameters()[0];
    check_rt_info(param->get_rt_info());

    auto result = f->get_results()[0];
    auto round = result->get_input_node_ptr(0);
    check_rt_info(round->get_rt_info());

    // read IR v10 with old API
    {
        InferenceEngine::Core core;
        auto f_10 = core.ReadNetwork(model, InferenceEngine::Blob::CPtr());
        ASSERT_NE(nullptr, f_10.getFunction());

        auto res = compare_functions(f, f_10.getFunction());
        EXPECT_TRUE(res.first) << res.second;

        EXPECT_EQ(InferenceEngine::Precision::FP32, f_10.getInputsInfo()["in1"]->getPrecision());
        EXPECT_EQ(InferenceEngine::Precision::FP32, f_10.getOutputsInfo()["Round"]->getPrecision());
    }

    // read IR v10 with new API and check that CNNNetwork precision conversions are applied
    {
        ngraph::Shape shape{1, 3, 22, 22};
        auto type = ngraph::element::f32;
        auto param = std::make_shared<ngraph::opset8::Parameter>(type, shape);
        param->set_friendly_name("in1");
        param->get_output_tensor(0).set_names({"input_tensor", param->get_friendly_name()});

        auto convert_param = std::make_shared<opset8::Convert>(param, ngraph::element::f16);

        auto round = std::make_shared<opset8::Round>(convert_param,
            ngraph::opset8::Round::RoundMode::HALF_TO_EVEN);

        auto convert_result = std::make_shared<opset8::Convert>(round, type);
        convert_result->set_friendly_name("Round");
        convert_result->get_output_tensor(0).set_names({"output_tensor",
            convert_result->get_friendly_name()});

        auto result = std::make_shared<opset8::Result>(convert_result);
        result->set_friendly_name("output");

        auto f_10_ref =
            std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{param});
        f_10_ref->set_friendly_name("Network");

        ov::runtime::Core core;
        auto f_10_core = core.read_model(model, ov::runtime::Tensor());
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

TEST_F(RTInfoDeserialization, InputAndOutputV10) {
    std::string model = R"V0G0N(
<net name="Network" version="10">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset8">
            <data element_type="i64" shape="1,3,22,22"/>
            <output>
                <port id="0" precision="I64" names="input_tensor">
                    <rt_info>
                        <attribute name="fused_names" version="0" value="test1,test2"/>
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

    auto check_rt_info = [](const RTMap& info) {
        const std::string& key = VariantWrapper<ngraph::FusedNames>::get_type_info_static();
        ASSERT_FALSE(info.count(key));
    };

    auto check_version = [](const std::shared_ptr<ov::Function>& f, int ref_version) {
        auto& rt_info = f->get_rt_info();
        ASSERT_TRUE(rt_info.count("version"));
        auto version = std::dynamic_pointer_cast<VariantWrapper<int64_t>>(rt_info.at("version"));
        ASSERT_NE(version, nullptr);
        ASSERT_EQ(version->get(), ref_version);
    };
    check_version(f, 10);

    auto param = f->get_parameters()[0];
    check_rt_info(param->output(0).get_rt_info());

    auto result = f->get_results()[0];
    check_rt_info(result->input(0).get_rt_info());

    auto add = result->get_input_node_ptr(0);
    check_rt_info(add->input(0).get_rt_info());
    check_rt_info(add->input(1).get_rt_info());
    check_rt_info(add->output(0).get_rt_info());

    // read IR v10 with old API
    {
        InferenceEngine::Core core;
        auto f_10 = core.ReadNetwork(model, InferenceEngine::Blob::CPtr());
        ASSERT_NE(nullptr, f_10.getFunction());

        auto res = compare_functions(f, f_10.getFunction());
        EXPECT_TRUE(res.first) << res.second;

        EXPECT_EQ(InferenceEngine::Precision::I64, f_10.getInputsInfo()["in1"]->getPrecision());
        EXPECT_EQ(InferenceEngine::Precision::I32, f_10.getOutputsInfo()["sum"]->getPrecision());
    }

    // read IR v10 with new API and check that CNNNetwork precision conversions are applied
    {
        const ngraph::Shape shape{1, 3, 22, 22};
        const auto type = ngraph::element::i64;
        auto param = std::make_shared<ngraph::opset8::Parameter>(type, shape);
        param->set_friendly_name("in1");
        param->get_output_tensor(0).set_names({"input_tensor", param->get_friendly_name()});

        auto sum = std::make_shared<opset8::Add>(param, param);
        sum->set_friendly_name("sum");

        auto convert_result = std::make_shared<opset8::Convert>(sum, ngraph::element::i32);
        convert_result->set_friendly_name("sum");
        convert_result->get_output_tensor(0).set_names({"output_tensor", convert_result->get_friendly_name()});

        auto result = std::make_shared<opset8::Result>(convert_result);
        result->set_friendly_name("output");

        auto f_10_ref =
            std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{param});
        f_10_ref->set_friendly_name("Network");

        ov::runtime::Core core;
        auto f_10_core = core.read_model(model, ov::runtime::Tensor());
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

TEST_F(RTInfoDeserialization, NodeV11) {
    std::string model = R"V0G0N(
<net name="Network" version="11">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset8">
            <data element_type="f32" shape="1,22,22,3"/>
            <rt_info>
                <attribute name="old_api_map" version="0" order="0,2,3,1" element_type="f16"/>
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
                <attribute name="old_api_map" version="0" order="0,3,1,2" element_type="f16"/>
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
    auto f = getWithIRFrontend(model);
    ASSERT_NE(nullptr, f);

    auto check_fused_names = [](const RTMap& info, const std::string& names) {
        const std::string& key = VariantWrapper<ngraph::FusedNames>::get_type_info_static();
        ASSERT_TRUE(info.count(key));
        auto fused_names_attr = std::dynamic_pointer_cast<VariantWrapper<ngraph::FusedNames>>(info.at(key));
        ASSERT_TRUE(fused_names_attr);
        EXPECT_EQ(fused_names_attr->get().getNames(), names);
    };

    auto check_old_api_map = [](const RTMap & info, const std::vector<uint64_t> & order, const ngraph::element::Type& type) {
        const std::string & old_api_map_key = ov::OldApiMap::get_type_info_static();
        ASSERT_TRUE(info.count(old_api_map_key));
        auto old_api_map_attr = std::dynamic_pointer_cast<ov::OldApiMap>(info.at(old_api_map_key));
        ASSERT_TRUE(old_api_map_attr);
        auto old_api_map_attr_val = old_api_map_attr->get();
        EXPECT_EQ(old_api_map_attr_val.get_order(), order);
        EXPECT_EQ(old_api_map_attr_val.get_type(), type);
    };
    auto check_version = [](const std::shared_ptr<ov::Function>& f, int ref_version) {
        auto& rt_info = f->get_rt_info();
        ASSERT_TRUE(rt_info.count("version"));
        auto version = std::dynamic_pointer_cast<VariantWrapper<int64_t>>(rt_info.at("version"));
        ASSERT_NE(version, nullptr);
        EXPECT_EQ(version->get(), ref_version);
    };
    check_version(f, 11);

    auto param = f->get_parameters()[0];
    check_fused_names(param->get_rt_info(), "in1");
    check_old_api_map(param->get_rt_info(), std::vector<uint64_t>({0, 2, 3, 1}), ngraph::element::Type_t::f16);

    auto result = f->get_result();
    check_old_api_map(result->get_rt_info(), std::vector<uint64_t>({0, 3, 1, 2}), ngraph::element::Type_t::f16);
    auto round = result->get_input_node_ptr(0);
    check_fused_names(round->get_rt_info(), "Round1,Round2");

    // read IR v11 with new API
    {
        ov::runtime::Core core;
        auto f_11 = core.read_model(model, ov::runtime::Tensor());
        ASSERT_NE(nullptr, f_11);

        check_old_api_map(f_11->get_parameters()[0]->get_rt_info(),
                          std::vector<uint64_t>({0, 2, 3, 1}),
                          ngraph::element::Type_t::f16);

        check_old_api_map(f_11->get_result()->get_rt_info(),
                          std::vector<uint64_t>({0, 3, 1, 2}),
                          ngraph::element::Type_t::f16);

        auto res = compare_functions(f, f_11);
        EXPECT_TRUE(res.first) << res.second;

        check_version(f_11, 11);
    }

    // read IR v11 with old API and check that old_api_map is applied
    {
        const ngraph::PartialShape shape{1, 3, 22, 22};
        auto type = ngraph::element::f16;
        auto param = std::make_shared<ngraph::opset8::Parameter>(type, shape);
        param->set_friendly_name("in1");
        param->get_output_tensor(0).set_names({"input_tensor"});

        auto convert_param = std::make_shared<opset8::Convert>(param, ngraph::element::f32);

        auto constant_param = std::make_shared<opset8::Constant>(ngraph::element::i64,
                                                                 ngraph::Shape{4},
                                                                 std::vector<int64_t>{0, 2, 3, 1});
        auto transpose_param = std::make_shared<opset8::Transpose>(convert_param, constant_param);

        auto round = std::make_shared<opset8::Round>(transpose_param,
            ngraph::opset8::Round::RoundMode::HALF_TO_EVEN);
        // TODO: runtime information should migrate as well?
        round->get_rt_info()[VariantWrapper<ngraph::FusedNames>::get_type_info_static()] =
            std::make_shared<VariantWrapper<ngraph::FusedNames>>(ngraph::FusedNames("Round1,Round2"));

        auto constant_result = std::make_shared<opset8::Constant>(ngraph::element::i64,
                                                                  ngraph::Shape{4},
                                                                  std::vector<int64_t>{0, 3, 1, 2});
        auto transpose_result = std::make_shared<opset8::Transpose>(round, constant_result);

        auto convert_result = std::make_shared<opset8::Convert>(transpose_result, type);
        convert_result->set_friendly_name("Round");
        convert_result->get_output_tensor(0).set_names({"output_tensor"});

        auto result = std::make_shared<opset8::Result>(convert_result);
        result->set_friendly_name("output");

        auto f_10_ref =
            std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{param});
        f_10_ref->set_friendly_name("Network");

        InferenceEngine::Core core;
        auto cnn_core = core.ReadNetwork(model, InferenceEngine::Blob::CPtr());
        auto f_10_core = cnn_core.getFunction();
        ASSERT_NE(nullptr, f_10_core);

        check_version(f_10_core, 10);

        EXPECT_EQ(InferenceEngine::Precision::FP32, cnn_core.getInputsInfo()["in1"]->getPrecision());
        EXPECT_EQ(InferenceEngine::Precision::FP32, cnn_core.getOutputsInfo()["Round"]->getPrecision());

        const auto fc = FunctionsComparator::with_default()
                            .enable(FunctionsComparator::ATTRIBUTES)
                            .enable(FunctionsComparator::PRECISIONS)
                            .enable(FunctionsComparator::RUNTIME_KEYS)
                            .enable(FunctionsComparator::NAMES)
                            .enable(FunctionsComparator::CONST_VALUES);
        auto res = fc.compare(f_10_core, f_10_ref);
        EXPECT_TRUE(res.valid) << res.message;

        EXPECT_EQ(shape, f_10_ref->input().get_partial_shape());
        EXPECT_EQ(shape, f_10_core->input().get_partial_shape());
        EXPECT_EQ(shape, f_10_ref->get_output_partial_shape(0));
        EXPECT_EQ(shape, f_10_core->get_output_partial_shape(0));

        // check that old api map is removed once applied
        auto check_old_api_rt_info = [](const RTMap & info) {
            const std::string & key = ov::OldApiMap::get_type_info_static();
            EXPECT_EQ(0, info.count(key));
        };

        check_old_api_rt_info(f_10_core->get_parameters()[0]->get_rt_info());
        check_old_api_rt_info(f_10_core->get_result()->get_rt_info());

        // check information about layout
        EXPECT_TRUE(f_10_core->get_parameters()[0]->get_layout().empty())
            << f_10_core->get_parameters()[0]->get_layout().to_string();
        EXPECT_TRUE(f_10_core->get_results()[0]->get_layout().empty())
            << f_10_core->get_results()[0]->get_layout().to_string();
    }
}

TEST_F(RTInfoDeserialization, InputAndOutputV11) {
    std::string model = R"V0G0N(
<net name="Network" version="11">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset8">
            <data element_type="f32" shape="1,3,22,22"/>
            <rt_info>
                <attribute name="old_api_map" version="0" order="" element_type="undefined"/>
            </rt_info>
            <output>
                <port id="0" precision="FP32">
                    <rt_info>
                        <attribute name="fused_names" version="0" value="test1,test2"/>
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
                    </rt_info>
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset8">
            <rt_info>
                <attribute name="old_api_map" version="0" order="" element_type="undefined"/>
            </rt_info>
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

    auto check_version = [](const std::shared_ptr<ov::Function>& f, int ref_version) {
        auto& rt_info = f->get_rt_info();
        ASSERT_TRUE(rt_info.count("version"));
        auto version = std::dynamic_pointer_cast<VariantWrapper<int64_t>>(rt_info.at("version"));
        ASSERT_NE(version, nullptr);
        ASSERT_EQ(version->get(), ref_version);
    };
    check_version(f, 11);

    auto check_fused_names = [](const RTMap& info, const std::string& names) {
        const std::string& key = VariantWrapper<ngraph::FusedNames>::get_type_info_static();
        ASSERT_TRUE(info.count(key));
        auto fused_names_attr = std::dynamic_pointer_cast<VariantWrapper<ngraph::FusedNames>>(info.at(key));
        ASSERT_TRUE(fused_names_attr);
        ASSERT_EQ(fused_names_attr->get().getNames(), names);
    };

    auto check_old_api_map = [](const RTMap& info, const std::vector<uint64_t>& order, ngraph::element::Type type) {
        const std::string& old_api_map_key = ov::OldApiMap::get_type_info_static();
        ASSERT_TRUE(info.count(old_api_map_key));
        auto old_api_map_attr = std::dynamic_pointer_cast<ov::OldApiMap>(info.at(old_api_map_key));
        ASSERT_TRUE(old_api_map_attr);
        auto old_api_map_attr_val = old_api_map_attr->get();
        ASSERT_EQ(old_api_map_attr_val.get_order(), order);
        ASSERT_EQ(old_api_map_attr_val.get_type(), type);
    };

    auto param = f->get_parameters()[0];
    check_fused_names(param->output(0).get_rt_info(), "test1,test2");
    check_old_api_map(param->get_rt_info(), std::vector<uint64_t>({}), ngraph::element::Type_t::undefined);

    auto result = f->get_result();
    check_fused_names(result->input(0).get_rt_info(), "test5,test6");
    check_old_api_map(result->get_rt_info(), std::vector<uint64_t>({}), ngraph::element::Type_t::undefined);

    auto add = result->get_input_node_ptr(0);
    check_fused_names(add->input(0).get_rt_info(), "test2,test3");
    check_fused_names(add->input(1).get_rt_info(), "test3,test4");
    check_fused_names(add->output(0).get_rt_info(), "test4,test5");

    // read IR v11 with old API - the function is the same since no old_api_map is applied
    {
        InferenceEngine::Core core;
        auto cnn = core.ReadNetwork(model, InferenceEngine::Blob::CPtr());
        auto f_10 = cnn.getFunction();
        ASSERT_NE(nullptr, f_10);

        EXPECT_EQ(InferenceEngine::Precision::FP32, cnn.getInputsInfo()["in1"]->getPrecision());
        EXPECT_EQ(InferenceEngine::Precision::FP32, cnn.getOutputsInfo()["sum"]->getPrecision());

        // check that old api map is removed once applied
        auto check_old_api_rt_info = [](const RTMap& info) {
            const std::string& key = ov::OldApiMap::get_type_info_static();
            EXPECT_FALSE(info.count(key));
        };

        check_old_api_rt_info(f_10->get_parameters()[0]->get_rt_info());
        check_old_api_rt_info(f_10->get_result()->get_rt_info());

        auto res = compare_functions(f, f_10);
        EXPECT_TRUE(res.first) << res.second;

        check_version(f_10, 10);
    }
}

TEST_F(RTInfoDeserialization, IndexesInputAndOutputV11) {
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

    auto check_version = [](const std::shared_ptr<ov::Function>& f, int ref_version) {
        auto& rt_info = f->get_rt_info();
        ASSERT_TRUE(rt_info.count("version"));
        auto version = std::dynamic_pointer_cast<VariantWrapper<int64_t>>(rt_info.at("version"));
        ASSERT_NE(version, nullptr);
        ASSERT_EQ(version->get(), ref_version);
    };
    check_version(f, 11);

    ASSERT_EQ(2, f->get_parameters().size());
    ASSERT_EQ(f->get_parameters()[0]->get_friendly_name(), "in1");
    ASSERT_EQ(f->get_parameters()[1]->get_friendly_name(), "in2");

    ASSERT_EQ(2, f->get_results().size());
    ASSERT_EQ(f->get_results()[0]->get_friendly_name(), "output2");
    ASSERT_EQ(f->get_results()[1]->get_friendly_name(), "output1");
}
