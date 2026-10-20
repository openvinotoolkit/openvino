// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <vector>
#include <transformations/cpu_opset/arm/pass/convert_reduce_multi_axis.hpp>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/opsets/opset1_decl.hpp"

using namespace testing;
using namespace ov::intel_cpu;

struct ReduceMultiAxisTestParams {
    std::string name;
    std::function<std::shared_ptr<ov::Node>(const ov::Output<ov::Node>&, const ov::Output<ov::Node>&, bool)> makeReduce;
    std::function<void(ov::pass::Manager&)> registerPass;
};

static auto makeReduceMultiAxisParams() {
    return ::testing::Values(
        ReduceMultiAxisTestParams{
            "ReduceMin",
            [](const ov::Output<ov::Node>& data, const ov::Output<ov::Node>& axes, bool keep_dims) {
                return std::make_shared<ov::opset1::ReduceMin>(data, axes, keep_dims);
            },
            [](ov::pass::Manager& m) { m.register_pass<ConvertReduceMin>(); }},
        ReduceMultiAxisTestParams{
            "ReduceMax",
            [](const ov::Output<ov::Node>& data, const ov::Output<ov::Node>& axes, bool keep_dims) {
                return std::make_shared<ov::opset1::ReduceMax>(data, axes, keep_dims);
            },
            [](ov::pass::Manager& m) { m.register_pass<ConvertReduceMax>(); }},
        ReduceMultiAxisTestParams{
            "ReduceSum",
            [](const ov::Output<ov::Node>& data, const ov::Output<ov::Node>& axes, bool keep_dims) {
                return std::make_shared<ov::opset1::ReduceSum>(data, axes, keep_dims);
            },
            [](ov::pass::Manager& m) { m.register_pass<ConvertReduceSum>(); }},
        ReduceMultiAxisTestParams{
            "ReduceProd",
            [](const ov::Output<ov::Node>& data, const ov::Output<ov::Node>& axes, bool keep_dims) {
                return std::make_shared<ov::opset1::ReduceProd>(data, axes, keep_dims);
            },
            [](ov::pass::Manager& m) { m.register_pass<ConvertReduceProd>(); }});
}

using ConvertReduceMultiAxisParams = std::tuple<ReduceMultiAxisTestParams, ov::PartialShape>;

class ConvertReduceMultiAxisTest : public TransformationTestsF,
                                   public WithParamInterface<ConvertReduceMultiAxisParams> {
public:
    static std::string getTestCaseName(const TestParamInfo<ConvertReduceMultiAxisParams>& info) {
        const auto& [params, shape] = info.param;
        return params.name + (shape.is_dynamic() ? "_Dynamic" : "_Static");
    }
};

TEST_P(ConvertReduceMultiAxisTest, CompareWithRef) {
    const auto& [params, shape] = GetParam();
    auto param = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape);
    auto axes = ov::opset1::Constant::create(ov::element::i64, ov::Shape{2}, {0, 1});
    auto reduce = params.makeReduce(param, axes, true);
    model = std::make_shared<ov::Model>(ov::OutputVector{reduce}, ov::ParameterVector{param});

    if (shape.is_static()) {
        auto ref_param = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape);
        std::shared_ptr<ov::Node> node = ref_param;
        for (auto axis : std::vector<int64_t>{0, 1}) {
            auto reduction_axis = ov::opset1::Constant::create(ov::element::i64, ov::Shape{}, {axis});
            node = params.makeReduce(node, reduction_axis, true);
        }
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{node}, ov::ParameterVector{ref_param});
    }

    params.registerPass(manager);
}

INSTANTIATE_TEST_SUITE_P(TransformationTests, ConvertReduceMultiAxisTest,
    ::testing::Combine(
        makeReduceMultiAxisParams(),
        ::testing::Values(ov::PartialShape{2, 19, 2, 9}, ov::PartialShape{2, -1, 2, 9})),
    ConvertReduceMultiAxisTest::getTestCaseName);
