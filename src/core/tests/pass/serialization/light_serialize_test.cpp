// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/pass/light_serialize.hpp"

#include <gtest/gtest.h>

#include <fstream>

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/graph_comparator.hpp"
#include "common_test_utils/test_common.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/pass/light_deserialize.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/util/file_util.hpp"
#include "read_ir.hpp"

namespace ov::test {

using op::v0::Parameter, op::v0::Constant, op::v1::Add;
class LightSerializePassTest : public testing::Test {
protected:
    std::stringstream m_out_xml_stream;
    std::map<int64_t, std::reference_wrapper<ov::ValueAccessor<void>>> m_offset_const_map;
    std::shared_ptr<Model> m_model;

    void SetUp() override {
        // Build a subgraph resembling a ResNet block with 50 nodes
        auto input = std::make_shared<Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto x = input;
        for (size_t i = 0; i < 50; ++i) {
            auto weights = op::v0::Constant::create(element::f32, Shape{3, 3, 3, 3}, {0.1f});
            auto conv = std::make_shared<op::v1::Convolution>(x,
                                                              weights,
                                                              Strides{1, 1},
                                                              CoordinateDiff{1, 1},
                                                              CoordinateDiff{1, 1},
                                                              Strides{1, 1});
            auto relu = std::make_shared<op::v0::Relu>(conv);

            x = relu;
        }
        m_model = std::make_shared<Model>(OutputVector{x}, ParameterVector{input}, "resnet50_subgraph");
    }

    void TearDown() override {
        // Nothing to clean up for stringstreams
    }
};

TEST_F(LightSerializePassTest, light_serialize_resnet50_subgraph) {
    ASSERT_NO_THROW(pass::LightSerialize(m_out_xml_stream, m_offset_const_map).run_on_model(m_model));
    // Check that streams are not empty
    ASSERT_FALSE(m_out_xml_stream.str().empty());
}

TEST_F(LightSerializePassTest, light_serialize_and_deserialize_resnet50_subgraph) {
    // 1. 序列化
    ASSERT_NO_THROW(pass::LightSerialize(m_out_xml_stream, m_offset_const_map).run_on_model(m_model));
    ASSERT_FALSE(m_out_xml_stream.str().empty());

    // 2. 反序列化
    std::stringstream xml_in(m_out_xml_stream.str());
    std::map<int64_t, std::reference_wrapper<ov::ValueAccessor<void>>> offset_const_map_for_deserialize =
        m_offset_const_map;
    std::shared_ptr<ov::Model> deserialized_model;
    ASSERT_NO_THROW(
        { deserialized_model = pass::LightDeserialize(xml_in, offset_const_map_for_deserialize).get_model(); });
    ASSERT_NE(deserialized_model, nullptr);

    // 3. 比较模型结构和常量
    auto comparator = FunctionsComparator::with_default()
                          .enable(FunctionsComparator::ATTRIBUTES)
                          .enable(FunctionsComparator::CONST_VALUES);
    const auto& [is_valid, error_msg] = comparator.compare(deserialized_model, m_model);
    EXPECT_TRUE(is_valid) << error_msg;
}
TEST_F(LightSerializePassTest, light_serialize_and_deserialize_with_custom_opsets) {
    // 1. 序列化
    ASSERT_NO_THROW(pass::LightSerialize(m_out_xml_stream, m_offset_const_map).run_on_model(m_model));
    ASSERT_FALSE(m_out_xml_stream.str().empty());

    // 2. 反序列化
    std::stringstream xml_in(m_out_xml_stream.str());
    std::map<int64_t, ov::ValueAccessor<void>&> offset_const_map_for_deserialize = m_offset_const_map;
    std::shared_ptr<ov::Model> deserialized_model;
    ASSERT_NO_THROW(
        { deserialized_model = pass::LightDeserialize(xml_in, offset_const_map_for_deserialize).get_model(); });
    ASSERT_NE(deserialized_model, nullptr);

    // 3. 比较模型结构和常量
    auto comparator = FunctionsComparator::with_default()
                          .enable(FunctionsComparator::ATTRIBUTES)
                          .enable(FunctionsComparator::CONST_VALUES);
    const auto& [is_valid, error_msg] = comparator.compare(deserialized_model, m_model);
    EXPECT_TRUE(is_valid) << error_msg;
}

}  // namespace ov::test
