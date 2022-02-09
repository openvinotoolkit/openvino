// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <fstream>

#include "common_test_utils/graph_comparator.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/util/file_util.hpp"
#include "read_ir.hpp"
#include "util/test_common.hpp"

using ModelBuilder = std::function<std::shared_ptr<ov::Model>()>;
using SerializationFromModelParams = std::tuple<ModelBuilder, std::string>;

class SerializationFromModelTest : public ov::test::TestsCommon,
                                   public testing::WithParamInterface<SerializationFromModelParams> {
public:
    ModelBuilder m_builder;
    std::string m_out_xml_path;
    std::string m_out_bin_path;

    static std::string getTestCaseName(const testing::TestParamInfo<SerializationFromModelParams>& obj) {
        std::string res = std::get<1>(obj.param);
        return res;
    }

    void SetUp() override {
        m_builder = std::get<0>(GetParam());
        std::string test_name = GetTestName() + "_" + GetTimestamp();
        m_out_xml_path = test_name + ".xml";
        m_out_bin_path = test_name + ".bin";
    }

    void TearDown() override {
        std::remove(m_out_xml_path.c_str());
        std::remove(m_out_bin_path.c_str());
    }
};

TEST_P(SerializationFromModelTest, CompareFunctions) {
    auto expected = m_builder();
    ov::pass::Serialize(m_out_xml_path, m_out_bin_path).run_on_model(expected);
    auto result = ov::test::readModel(m_out_xml_path, m_out_bin_path);

    const auto fc = FunctionsComparator::with_default()
                        .enable(FunctionsComparator::ATTRIBUTES)
                        .enable(FunctionsComparator::CONST_VALUES);
    const auto res = fc.compare(result, expected);
    EXPECT_TRUE(res.valid) << res.message;
}

namespace {
std::shared_ptr<ov::Model> create_model_if_mixed_inputs() {
    // Then inputs mapping: 1->0, 0->1
    // Else inputs mapping: 0->0
    // Shapes of all inputs are different to ensure each parameter is connected properly
    using namespace ov;
    auto X = std::make_shared<op::v0::Parameter>(element::f32, Shape{2});
    X->output(0).get_tensor().set_names({"X"});
    auto Y = std::make_shared<op::v0::Parameter>(element::f32, Shape{4});
    Y->output(0).get_tensor().set_names({"Y"});
    auto Z = std::make_shared<op::v0::Parameter>(element::f32, Shape{8});
    Z->output(0).get_tensor().set_names({"Z"});
    auto Xt = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    Xt->output(0).get_tensor().set_names({"X_then"});
    auto Yt = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    Yt->output(0).get_tensor().set_names({"Y_then"});
    auto Ze = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    Ze->output(0).get_tensor().set_names({"Z_else"});
    auto cond = std::make_shared<op::v0::Constant>(element::boolean, Shape{1}, true);
    auto axis_then = std::make_shared<op::v0::Constant>(element::i32, Shape{}, 0);
    auto split_y = std::make_shared<opset8::Split>(Yt, axis_then, 2);
    auto then_op = std::make_shared<opset8::Subtract>(Xt, split_y->output(0));
    auto res0 = std::make_shared<op::v0::Result>(then_op);
    auto axis_else = std::make_shared<op::v0::Constant>(element::i32, Shape{}, 0);
    auto split_z = std::make_shared<opset8::Split>(Ze, axis_else, 4);
    auto else_op = std::make_shared<opset8::Relu>(split_z);
    auto res1 = std::make_shared<op::v0::Result>(else_op);
    auto then_body = std::make_shared<ov::Model>(OutputVector{res0}, ParameterVector{Yt, Xt}, "then_body");
    auto else_body = std::make_shared<ov::Model>(OutputVector{res1}, ParameterVector{Ze}, "else_body");
    auto if_op = std::make_shared<op::v8::If>(cond);
    if_op->set_then_body(then_body);
    if_op->set_else_body(else_body);
    if_op->set_input(X, Xt, nullptr);
    if_op->set_input(Y, Yt, nullptr);
    if_op->set_input(Z, nullptr, Ze);
    auto result = if_op->set_output(res0, res1);
    auto res = std::make_shared<op::v0::Result>(result);
    res->output(0).get_tensor().set_names({"Res"});
    return std::make_shared<Model>(OutputVector{res}, ParameterVector{X, Y, Z});
}

INSTANTIATE_TEST_SUITE_P(IRSerializationFromModel,
                         SerializationFromModelTest,
                         testing::Values(std::make_tuple(create_model_if_mixed_inputs, "Model_with_if_mixed_inputs")),
                         SerializationFromModelTest::getTestCaseName);
}  // namespace
