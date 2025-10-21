// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/file_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/variadic_split.hpp"

namespace {
using ov::test::InputShape;

using SwiGLUFusionParams = std::tuple<std::vector<InputShape>,   // input shapes
                                             ov::element::Type>;        // input precision

class SwiGLUFusion : public testing::WithParamInterface<SwiGLUFusionParams>,
                            virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<SwiGLUFusionParams>& obj) {
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

        // VariadicSplit(X, axis, split_lengths) = Xw, Xv
        auto axis_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {-1});
        auto split_lengths_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {48, -1});
        auto variadic_split = std::make_shared<ov::op::v1::VariadicSplit>(params[0], axis_const, split_lengths_const);

        // Swish(Xw) = Xw * (1.0 + exp(-beta * Xw))
        auto swish = std::make_shared<ov::op::v4::Swish>(variadic_split->output(0));

        // Mul(Xw, Xv) = Swish(Xw) * Xv
        auto mul = std::make_shared<ov::op::v1::Multiply>(swish, variadic_split->output(1));

        return std::make_shared<ov::Model>(ov::OutputVector{mul}, params, "SwiGLUFusion");
    }

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;

        const auto& [input_shapes, input_precision] = GetParam();

        init_input_shapes(input_shapes);

        inType = outType = input_precision;

        function = init_subgraph(inputDynamicShapes, targetStaticShapes.front().front(), input_precision);
    }
};

TEST_P(SwiGLUFusion, Inference) {
    run();
}

TEST_P(SwiGLUFusion, Inference_cached) {
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

const std::vector<std::vector<InputShape>> input_shapes_dyn = {
    {{{-1, -1, 96}, {{20, 1, 96}}}},
    {{{-1, -1, -1}, {{1, 1, 96}}}},
};

INSTANTIATE_TEST_SUITE_P(smoke_SwiGLUFusion_basic,
                         SwiGLUFusion,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_dyn),
                                            ::testing::ValuesIn(input_precisions)),
                         SwiGLUFusion::getTestCaseName);
} // namespace
