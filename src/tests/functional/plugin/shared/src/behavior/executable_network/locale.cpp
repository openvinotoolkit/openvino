// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/executable_network/locale.hpp"
#include "functional_test_utils/summary/api_summary.hpp"

namespace BehaviorTestsDefinitions {

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

std::string CustomLocaleTest::getTestCaseName(const testing::TestParamInfo<LocaleParams> &obj) {
    std::ostringstream results;
    std::string target_device, localeName;
    std::tie(localeName, target_device) = obj.param;
    results << "locale=" << localeName << "_"
            << "targetDevice=" << target_device;
    return results.str();
}

void CustomLocaleTest::SetUp() {
    APIBaseTest::SetUp();
    std::tie(localeName, target_device) = GetParam();
    testName = ::testing::UnitTest::GetInstance()->current_test_info()->name();
    function = makeTestModel();
}

TEST_P(CustomLocaleTest, CanLoadNetworkWithCustomLocale) {
    auto prev = std::locale();
    try {
        std::locale::global(std::locale(localeName.c_str()));
    } catch (...) {
        GTEST_SKIP();
    }

    std::shared_ptr<InferenceEngine::Core> ie = PluginCache::get().ie(target_device);
    InferenceEngine::CNNNetwork cnnNet(function);
    ASSERT_NO_THROW(ie->LoadNetwork(cnnNet, target_device));

    std::locale::global(prev);
}

} // namespace BehaviorTestsDefinitions
