// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common_test_utils/test_common.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include "common_test_utils/file_utils.hpp"

namespace BehaviorTestsDefinitions {

typedef std::tuple<
        std::string,    // Locale name
        std::string>    // Target device name
        LocaleParams;

class CustomLocaleTest : public CommonTestUtils::TestsCommon,
                         public ::testing::WithParamInterface<LocaleParams> {
protected:
    std::shared_ptr<ngraph::Function> function;
    std::string localeName;
    std::string testName;
    std::string deviceName;

    void SetUp() override;
public:
    static std::string getTestCaseName(const testing::TestParamInfo<LocaleParams> &obj);
};

} // namespace BehaviorTestsDefinitions
