// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/test_common.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include <ie_core.hpp>
#include <ie_plugin_config.hpp>

inline std::shared_ptr<ngraph::Function> makeTestModel(std::vector<size_t> inputShape = {1, 1, 32, 32}) {
    ngraph::Shape in_shape(inputShape);
    auto et = ngraph::element::Type_t::f16;
    auto in = std::make_shared<ngraph::opset1::Parameter>(et, in_shape);
    auto gelu = std::make_shared<ngraph::opset7::Gelu>(in);
    auto swish_const = ngraph::op::Constant::create(et, ngraph::Shape{}, {2.5f});
    auto swish = std::make_shared<ngraph::opset4::Swish>(gelu, swish_const);
    ngraph::Shape reluShape = swish->outputs()[0].get_tensor().get_shape();
    std::vector<size_t> constShape2 = {1, ngraph::shape_size(reluShape)};
    auto const2 = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{2}, constShape2);
    auto reshape2 = std::make_shared<ngraph::opset1::Reshape>(swish, const2, false);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(reshape2)};
    std::shared_ptr<ngraph::Function> fnPtr = std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{in});
    return fnPtr;
}

class CustomLocaleTest : public CommonTestUtils::TestsCommon {
protected:
    std::string test_name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
    std::shared_ptr<ngraph::Function> function;

    void SetUp() override {
        function = makeTestModel();
    }
};
TEST_F(CustomLocaleTest, CanLoadNetworkWithCustomLocale) {
    auto prev = std::locale();
    try {
        std::locale::global(std::locale("ru_RU.UTF-8"));
    } catch (...) {
        GTEST_SKIP();
    }

    std::shared_ptr<InferenceEngine::Core> ie = PluginCache::get().ie();
    InferenceEngine::CNNNetwork cnnNet(function);
    ASSERT_NO_THROW(ie->LoadNetwork(cnnNet, "GPU"));

    std::locale::global(prev);
}
