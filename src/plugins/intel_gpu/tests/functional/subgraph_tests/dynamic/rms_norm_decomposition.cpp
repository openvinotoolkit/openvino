// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/file_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_mean.hpp"

namespace {
using ov::test::InputShape;

/*
 *           Input(F32) Const(F32)
 *              |  \     /
 *              |  Power(F32) Const(I64)
 *              |      \       /
 *              |   ReduceMean(F32)
 *              |       |  Const(F32)
 *              |       |  /
 *              |      Add(F32)
 *              |       |
 *              |     Sqrt(F32) Const(F32)
 *              |       |      /
 *              |    Divide(F32)
 *              |      /
 *  Const(F32) Multiply(F32)
 *         \    |
 *         Multiply(F32)
 *              |
 *          Convert(F16)
 */
using RMSNormDecompositionParams = std::tuple<std::vector<InputShape>,             // input shapes
                                              ov::element::Type>;                  // input precision

class RMSNormDecomposition : public testing::WithParamInterface<RMSNormDecompositionParams>,
                             virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<RMSNormDecompositionParams>& obj) {
        const auto& [input_shapes, input_precision] = obj.param;

        std::ostringstream result;
        result << "IS=(";
        for (const auto& shape : input_shapes) {
            result << ov::test::utils::partialShape2str({shape.first}) << "_";
        }
        result << ")_TS=";
        for (const auto& shape : input_shapes) {
            result << "(";
            if (!shape.second.empty()) {
                auto itr = shape.second.begin();
                do {
                    result << ov::test::utils::vec2str(*itr);
                } while (++itr != shape.second.end() && result << "_");
            }
            result << ")_";
        }
        result << "input_precision=" << input_precision;
        return result.str();
    }

protected:
    std::shared_ptr<ov::Model> init_subgraph(std::vector<ov::PartialShape>& input_shapes,
                                             const ov::Shape& target_shape,
                                             const ov::element::Type input_precision) {
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(input_precision, input_shapes[0])};

        // x^2
        auto power_const = ov::op::v0::Constant::create(input_precision, {}, {2.f});
        auto power = std::make_shared<ov::op::v1::Power>(params[0], power_const);

        // ReduceMean(x^2,axes)
        auto mean_axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
        auto mean = std::make_shared<ov::op::v1::ReduceMean>(power, mean_axes, true);

        // ReduceMean(x^2,axes)+eps
        auto eps = ov::op::v0::Constant::create(input_precision, {}, {1e-5f});
        auto add_eps = std::make_shared<ov::op::v1::Add>(mean, eps);

        // Sqrt(ReduceMean(x^2,axes)+eps)
        auto sqrt = std::make_shared<ov::op::v0::Sqrt>(add_eps);

        // 1/Sqrt(ReduceMean(x^2,axes)+eps)
        auto div_const = ov::op::v0::Constant::create(input_precision, {}, {1});
        auto div = std::make_shared<ov::op::v1::Divide>(div_const, sqrt);

        // x * 1/Sqrt(ReduceMean(x^2,axes)+eps)
        auto mul1 = std::make_shared<ov::op::v1::Multiply>(params[0], div);

        // x * 1/Sqrt(ReduceMean(x^2,axes)+eps) * gamma
        auto dim = *target_shape.rbegin();

        auto tensor = ov::test::utils::create_and_fill_tensor(input_precision, ov::Shape{dim});
        auto gamma = std::make_shared<ov::op::v0::Constant>(tensor);
        auto mul2 = std::make_shared<ov::op::v1::Multiply>(gamma, mul1);

        auto comp = std::make_shared<ov::op::v0::Convert>(mul2, ov::element::f16);

        return std::make_shared<ov::Model>(ov::OutputVector{comp}, params, "RMSNormDecomposition");
    }

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;

        const auto& [input_shapes, input_precision] = GetParam();

        init_input_shapes(input_shapes);

        inType = outType = input_precision;

        function = init_subgraph(inputDynamicShapes, targetStaticShapes.front().front(), input_precision);
    }
};

TEST_P(RMSNormDecomposition, Inference) {
    run();
}

TEST_P(RMSNormDecomposition, Inference_cached) {
    std::stringstream ss;
    ss << "gpu_model_cache_" << std::hash<std::string>{}(
          std::string(::testing::UnitTest::GetInstance()->current_test_info()->test_suite_name()) +
          std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()));
    std::string cacheDirName = ss.str();
    {
        ov::test::utils::removeFilesWithExt(cacheDirName, "blob");
        ov::test::utils::removeFilesWithExt(cacheDirName, "cl_cache");
        ov::test::utils::removeDir(cacheDirName);
        core->set_property(ov::cache_dir(cacheDirName));
        compile_model();
    }
    {
        run();
        ov::test::utils::removeFilesWithExt(cacheDirName, "blob");
        ov::test::utils::removeFilesWithExt(cacheDirName, "cl_cache");
        ov::test::utils::removeDir(cacheDirName);
    }
}

const std::vector<ov::element::Type> input_precisions = {ov::element::f32, ov::element::f16};

const std::vector<std::vector<InputShape>> input_shapes_basic = {
    {{{-1, -1, 96}, {{1, 4, 96}}}},
    {{{-1, -1, -1}, {{1, 2, 16}}}},
    {{{}, {{1, 2, 6}}}},
    {{{}, {{1, 2, 18}}}},
};

INSTANTIATE_TEST_SUITE_P(smoke_RMSNormDecomposition_basic,
                         RMSNormDecomposition,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_basic),
                                            ::testing::ValuesIn(input_precisions)),
                         RMSNormDecomposition::getTestCaseName);
} // namespace
