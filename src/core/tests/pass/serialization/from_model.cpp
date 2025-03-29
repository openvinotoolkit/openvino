// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <fstream>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/graph_comparator.hpp"
#include "common_test_utils/test_common.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/if.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/util/file_util.hpp"
#include "read_ir.hpp"

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
        std::string filePrefix = ov::test::utils::generateTestFilePrefix();
        m_out_xml_path = filePrefix + ".xml";
        m_out_bin_path = filePrefix + ".bin";
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
    auto split_y = std::make_shared<op::v1::Split>(Yt, axis_then, 2);
    auto then_op = std::make_shared<op::v1::Subtract>(Xt, split_y->output(0));
    auto res0 = std::make_shared<op::v0::Result>(then_op);
    auto axis_else = std::make_shared<op::v0::Constant>(element::i32, Shape{}, 0);
    auto split_z = std::make_shared<op::v1::Split>(Ze, axis_else, 4);
    auto else_op = std::make_shared<op::v0::Relu>(split_z);
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

std::vector<SerializationFromModelParams> get_models() {
    auto result = std::vector<SerializationFromModelParams>{};
    result.emplace_back(std::make_tuple(create_model_if_mixed_inputs, "Model_with_if_mixed_inputs"));
    // Zero size
    {
        auto builder = []() {
            using namespace ov;
            auto p1 = std::make_shared<op::v0::Parameter>(element::f32, Shape{2});
            p1->output(0).set_names({"X"});
            auto p2 = std::make_shared<op::v0::Parameter>(element::f32, Shape{2});
            p2->output(0).set_names({"Y"});
            auto op = std::make_shared<op::v1::Add>(p1, p2);
            auto res = std::make_shared<op::v0::Result>(op);
            return std::make_shared<Model>(OutputVector{res}, ParameterVector{p1, p2});
        };
        result.emplace_back(std::make_tuple(builder, "Model_with_no_weights"));
    }
    // Various constant size 2^shift
    std::vector<size_t> shifts = {0, 1, 2, 4, 8, 16, 20};
    for (const auto& shift : shifts) {
        for (size_t offset = 0; offset < 2; offset++) {
            size_t s = (1llu << shift) + offset;
            auto builder = [s]() {
                using namespace ov;
                auto shape = Shape{s};
                auto data = std::vector<uint8_t>(shape_size(shape));
                std::iota(data.begin(), data.end(), 42);
                auto p1 = std::make_shared<op::v0::Parameter>(element::u8, shape);
                p1->output(0).set_names({"X"});
                auto c1 = std::make_shared<op::v0::Constant>(element::u8, shape, data.data());
                c1->output(0).set_names({"C"});
                auto op = std::make_shared<op::v1::Add>(p1, c1);
                auto res = std::make_shared<op::v0::Result>(op);
                return std::make_shared<Model>(OutputVector{res}, ParameterVector{p1});
            };
            result.emplace_back(
                std::make_tuple(builder,
                                std::string("Model_size_") + std::to_string(s) + "_" + std::to_string(offset)));
        }
    }
    return result;
}

INSTANTIATE_TEST_SUITE_P(IRSerializationFromModel,
                         SerializationFromModelTest,
                         testing::ValuesIn(get_models()),
                         SerializationFromModelTest::getTestCaseName);
}  // namespace

class SerializationFromModelTest_large : public ov::test::TestsCommon, public testing::WithParamInterface<size_t> {
public:
    std::string m_out_xml_path;
    std::string m_out_bin_path;

    static std::string getTestCaseName(const testing::TestParamInfo<size_t>& obj) {
        std::string res = std::to_string(obj.param);
        return res;
    }

    void SetUp() override {
        std::string filePrefix = ov::test::utils::generateTestFilePrefix();
        m_out_xml_path = filePrefix + ".xml";
        m_out_bin_path = filePrefix + ".bin";
    }

    void TearDown() override {
        std::remove(m_out_xml_path.c_str());
        std::remove(m_out_bin_path.c_str());
    }
};

// Disabled just because of long execution time. Enable for nightly builds in future
TEST_P(SerializationFromModelTest_large, DISABLED_Model_very_large) {
    using namespace ov;
    std::string test_name = GetTimestamp();
    size_t s = (1llu << GetParam()) + 5;
    {
        auto shape = Shape{s};
        auto data = std::vector<uint8_t>(shape_size(shape), 42);
        std::iota(data.begin(), data.end(), 42);
        auto p1 = std::make_shared<op::v0::Parameter>(element::u8, shape);
        p1->output(0).set_names({"X"});
        auto c1 = std::make_shared<op::v0::Constant>(element::u8, shape, data.data());
        c1->output(0).set_names({"C"});
        auto op = std::make_shared<op::v1::Add>(p1, c1);
        auto res = std::make_shared<op::v0::Result>(op);
        auto model = std::make_shared<Model>(OutputVector{res}, ParameterVector{p1});
        ov::pass::Serialize(m_out_xml_path, m_out_bin_path).run_on_model(model);
    }
    auto actual = ov::test::readModel(m_out_xml_path, m_out_bin_path);
    bool found = false;
    for (const auto& op : actual->get_ordered_ops()) {
        if (auto const1 = ov::as_type_ptr<op::v0::Constant>(op)) {
            auto ptr = const1->get_data_ptr<uint8_t>();
            for (size_t i = 0; i < s; i++) {
                EXPECT_EQ(ptr[i], uint8_t(i + 42)) << "Index " << i << " has value " << static_cast<int>(ptr[i]);
            }
            found = true;
        }
    }
    EXPECT_TRUE(found);
}

namespace {
INSTANTIATE_TEST_SUITE_P(nightly_IRSerializationFromModel_large,
                         SerializationFromModelTest_large,
                         testing::ValuesIn(std::vector<size_t>{32}),
                         SerializationFromModelTest_large::getTestCaseName);
}  // namespace
