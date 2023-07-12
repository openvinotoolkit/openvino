// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <utility>
#include <vector>
#include <memory>

#include "openvino/runtime/core.hpp"

#include <common_test_utils/test_common.hpp>
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "base/ov_behavior_test_utils.hpp"
#include "functional_test_utils/ov_plugin_cache.hpp"

using namespace ::testing;

using params = std::tuple<ov::element::Type, ov::element::Type>;

class InferencePrecisionTests : public testing::WithParamInterface<params>,
                                virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<params> &obj) {
        ov::element::Type model_precision;
        ov::element::Type inference_precision;
        std::tie(model_precision, inference_precision) = obj.param;
        std::stringstream s;
        s << "model_precision=" << model_precision << "_inference_precison=" << inference_precision;
        return s.str();
    }
};

TEST_P(InferencePrecisionTests, smoke_canSetInferencePrecisionAndInfer) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    auto core =  ov::test::utils::PluginCache::get().core();
    ov::element::Type model_precision;
    ov::element::Type inference_precision;
    std::tie(model_precision, inference_precision) = GetParam();
    auto function = ov::test::behavior::getDefaultNGraphFunctionForTheDevice({1, 2, 32, 32}, model_precision);
    ov::CompiledModel compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = core->compile_model(function, ov::test::utils::DEVICE_GPU, ov::hint::inference_precision(inference_precision)));
    auto req = compiled_model.create_infer_request();
    OV_ASSERT_NO_THROW(req.infer());
}

static const std::vector<params> test_params = {
    {ov::element::f16, ov::element::f32},
    {ov::element::f16, ov::element::f16},
    {ov::element::f32, ov::element::f32},
    {ov::element::f32, ov::element::f16},
};

INSTANTIATE_TEST_SUITE_P(smoke_GPU_BehaviorTests, InferencePrecisionTests, ::testing::ValuesIn(test_params), InferencePrecisionTests::getTestCaseName);

TEST(InferencePrecisionTests, CantSetInvalidInferencePrecision) {
    ov::Core core;

    ASSERT_NO_THROW(core.get_property(ov::test::utils::DEVICE_GPU, ov::hint::inference_precision));
    ASSERT_ANY_THROW(core.set_property(ov::test::utils::DEVICE_GPU, ov::hint::inference_precision(ov::element::bf16)));
    ASSERT_ANY_THROW(core.set_property(ov::test::utils::DEVICE_GPU, ov::hint::inference_precision(ov::element::undefined)));
}

TEST(ExecutionModeTest, SetCompileGetInferPrecisionAndExecMode) {
    ov::Core core;

    core.set_property(ov::test::utils::DEVICE_GPU, ov::hint::execution_mode(ov::hint::ExecutionMode::PERFORMANCE));
    auto model = ngraph::builder::subgraph::makeConvPoolRelu();
    {
        auto compiled_model = core.compile_model(model, ov::test::utils::DEVICE_GPU, ov::hint::inference_precision(ov::element::f32));
        ASSERT_EQ(ov::hint::ExecutionMode::PERFORMANCE, compiled_model.get_property(ov::hint::execution_mode));
        ASSERT_EQ(ov::element::f32, compiled_model.get_property(ov::hint::inference_precision));
    }

    {
        auto compiled_model = core.compile_model(model, ov::test::utils::DEVICE_GPU, ov::hint::execution_mode(ov::hint::ExecutionMode::ACCURACY));
        ASSERT_EQ(ov::hint::ExecutionMode::ACCURACY, compiled_model.get_property(ov::hint::execution_mode));
        ASSERT_EQ(ov::element::f32, compiled_model.get_property(ov::hint::inference_precision));
    }

    {
        auto compiled_model = core.compile_model(model, ov::test::utils::DEVICE_GPU);
        ASSERT_EQ(ov::hint::ExecutionMode::PERFORMANCE, compiled_model.get_property(ov::hint::execution_mode));
        ASSERT_EQ(ov::element::f16, compiled_model.get_property(ov::hint::inference_precision));
    }
}
