// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common_test_utils/test_common.hpp"
#include "common_test_utils/file_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "ov_models/subgraph_builders.hpp"

#include "base/behavior_test_utils.hpp"

namespace BehaviorTestsDefinitions {

typedef std::tuple<
        std::string,    // Locale name
        std::string>    // Target device name
        LocaleParams;

class CustomLocaleTest : public BehaviorTestsUtils::IEExecutableNetworkTestBase,
                         public ::testing::WithParamInterface<LocaleParams> {
protected:
    std::shared_ptr<ngraph::Function> function;
    std::string localeName;
    std::string testName;

    void SetUp() override;
public:
    static std::string getTestCaseName(const testing::TestParamInfo<LocaleParams> &obj);
};

} // namespace BehaviorTestsDefinitions
