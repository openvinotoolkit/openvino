// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <vector>
#include <transformations/cpu_opset/arm/pass/convert_reduce_no_keep_dims.hpp>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reduce_logical_and.hpp"
#include "openvino/op/reduce_logical_or.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/opsets/opset1_decl.hpp"

using namespace testing;
using namespace ov::intel_cpu;

struct ReduceNoKeepDimsTestParams {
    std::string name;
    ov::element::Type dataType;
    std::function<std::shared_ptr<ov::Node>(const ov::Output<ov::Node>&, const ov::Output<ov::Node>&, bool)> makeReduce;
    std::function<void(ov::pass::Manager&)> registerPass;
};

static auto makeReduceNoKeepDimsParams() {
    using namespace ov::op::util;
    return ::testing::Values(
        ReduceNoKeepDimsTestParams{
            "ReduceMin", ov::element::f32,
            [](const ov::Output<ov::Node>& d, const ov::Output<ov::Node>& a, bool k) {
                return std::make_shared<ov::opset1::ReduceMin>(d, a, k);
            },
            [](ov::pass::Manager& m) { m.register_pass<ConvertReduction<ArithmeticReductionKeepDims>>(); }},
        ReduceNoKeepDimsTestParams{
            "ReduceMax", ov::element::f32,
            [](const ov::Output<ov::Node>& d, const ov::Output<ov::Node>& a, bool k) {
                return std::make_shared<ov::opset1::ReduceMax>(d, a, k);
            },
            [](ov::pass::Manager& m) { m.register_pass<ConvertReduction<ArithmeticReductionKeepDims>>(); }},
        ReduceNoKeepDimsTestParams{
            "ReduceSum", ov::element::f32,
            [](const ov::Output<ov::Node>& d, const ov::Output<ov::Node>& a, bool k) {
                return std::make_shared<ov::opset1::ReduceSum>(d, a, k);
            },
            [](ov::pass::Manager& m) { m.register_pass<ConvertReduction<ArithmeticReductionKeepDims>>(); }},
        ReduceNoKeepDimsTestParams{
            "ReduceProd", ov::element::f32,
            [](const ov::Output<ov::Node>& d, const ov::Output<ov::Node>& a, bool k) {
                return std::make_shared<ov::opset1::ReduceProd>(d, a, k);
            },
            [](ov::pass::Manager& m) { m.register_pass<ConvertReduction<ArithmeticReductionKeepDims>>(); }},
        ReduceNoKeepDimsTestParams{
            "ReduceMean", ov::element::f32,
            [](const ov::Output<ov::Node>& d, const ov::Output<ov::Node>& a, bool k) {
                return std::make_shared<ov::opset1::ReduceMean>(d, a, k);
            },
            [](ov::pass::Manager& m) { m.register_pass<ConvertReduction<ArithmeticReductionKeepDims>>(); }},
        ReduceNoKeepDimsTestParams{
            "ReduceLogicalAnd", ov::element::boolean,
            [](const ov::Output<ov::Node>& d, const ov::Output<ov::Node>& a, bool k) {
                return std::make_shared<ov::opset1::ReduceLogicalAnd>(d, a, k);
            },
            [](ov::pass::Manager& m) { m.register_pass<ConvertReduction<LogicalReductionKeepDims>>(); }},
        ReduceNoKeepDimsTestParams{
            "ReduceLogicalOr", ov::element::boolean,
            [](const ov::Output<ov::Node>& d, const ov::Output<ov::Node>& a, bool k) {
                return std::make_shared<ov::opset1::ReduceLogicalOr>(d, a, k);
            },
            [](ov::pass::Manager& m) { m.register_pass<ConvertReduction<LogicalReductionKeepDims>>(); }});
}

using ConvertReduceNoKeepDimsParams = std::tuple<ReduceNoKeepDimsTestParams, ov::PartialShape>;

class ConvertReduceNoKeepDimsTest : public TransformationTestsF,
                                    public WithParamInterface<ConvertReduceNoKeepDimsParams> {
public:
    static std::string getTestCaseName(const TestParamInfo<ConvertReduceNoKeepDimsParams>& info) {
        const auto& [params, shape] = info.param;
        return params.name + (shape.is_dynamic() ? "_Dynamic" : "_Static");
    }
};

TEST_P(ConvertReduceNoKeepDimsTest, CompareWithRef) {
    const auto& [params, shape] = GetParam();
    auto param = std::make_shared<ov::opset1::Parameter>(params.dataType, shape);
    auto axes = ov::opset1::Constant::create(ov::element::i64, ov::Shape{2}, {0, 1});
    auto reduce = params.makeReduce(param, axes, false);
    model = std::make_shared<ov::Model>(ov::OutputVector{reduce}, ov::ParameterVector{param});

    {
        auto ref_param = std::make_shared<ov::opset1::Parameter>(params.dataType, shape);
        auto ref_axes = ov::opset1::Constant::create(ov::element::i64, ov::Shape{2}, {0, 1});
        auto ref_reduce = params.makeReduce(ref_param, ref_axes, true);
        auto squeeze = std::make_shared<ov::opset1::Squeeze>(ref_reduce, ref_axes);
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{squeeze}, ov::ParameterVector{ref_param});
    }

    params.registerPass(manager);
}

INSTANTIATE_TEST_SUITE_P(TransformationTests, ConvertReduceNoKeepDimsTest,
    ::testing::Combine(
        makeReduceNoKeepDimsParams(),
        ::testing::Values(ov::PartialShape{2, 19, 2, 9}, ov::PartialShape{2, -1, 2, 9})),
    ConvertReduceNoKeepDimsTest::getTestCaseName);
