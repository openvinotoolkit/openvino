// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/executable_network/locale.hpp"

#include <locale.h>

#include "functional_test_utils/summary/api_summary.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gelu.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/swish.hpp"

namespace BehaviorTestsDefinitions {

inline std::shared_ptr<ov::Model> makeTestModel(std::vector<size_t> inputShape = {1, 1, 32, 32}) {
    ov::Shape in_shape(inputShape);
    auto et = ov::element::Type_t::f16;
    auto in = std::make_shared<ov::op::v0::Parameter>(et, in_shape);
    auto gelu = std::make_shared<ov::op::v7::Gelu>(in);
    auto swish_const = ov::op::v0::Constant::create(et, ov::Shape{}, {2.5f});
    auto swish = std::make_shared<ov::op::v4::Swish>(gelu, swish_const);
    ov::Shape reluShape = swish->outputs()[0].get_tensor().get_shape();
    std::vector<size_t> constShape2 = {1, ov::shape_size(reluShape)};
    auto const2 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, constShape2);
    auto reshape2 = std::make_shared<ov::op::v1::Reshape>(swish, const2, false);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(reshape2)};
    std::shared_ptr<ov::Model> fnPtr = std::make_shared<ov::Model>(results, ov::ParameterVector{in});
    return fnPtr;
}

std::string CustomLocaleTest::getTestCaseName(const testing::TestParamInfo<LocaleParams>& obj) {
    std::ostringstream results;
    std::string targetDevice, localeName;
    std::tie(localeName, targetDevice) = obj.param;
    std::replace(localeName.begin(), localeName.end(), '-', '.');
    std::replace(targetDevice.begin(), targetDevice.end(), ':', '.');
    results << "locale=" << localeName << "_"
            << "targetDevice=" << targetDevice;
    return results.str();
}

void CustomLocaleTest::SetUp() {
    std::tie(localeName, target_device) = GetParam();
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    APIBaseTest::SetUp();
    testName = ::testing::UnitTest::GetInstance()->current_test_info()->name();
    function = makeTestModel();
}

TEST_P(CustomLocaleTest, CanLoadNetworkWithCustomLocale) {
    auto prev = std::locale().name();
    setlocale(LC_ALL, localeName.c_str());
    setlocale(LC_NUMERIC, localeName.c_str());
    setlocale(LC_TIME, localeName.c_str());

    std::shared_ptr<InferenceEngine::Core> ie = PluginCache::get().ie(target_device);
    InferenceEngine::CNNNetwork cnnNet(function);
    ASSERT_NO_THROW(ie->LoadNetwork(cnnNet, target_device));

    setlocale(LC_ALL, prev.c_str());
    setlocale(LC_NUMERIC, prev.c_str());
    setlocale(LC_TIME, prev.c_str());
}

}  // namespace BehaviorTestsDefinitions
