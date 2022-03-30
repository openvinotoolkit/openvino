// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common_test_utils/test_common.hpp"
#include "common_test_utils/file_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "ngraph_functions/subgraph_builders.hpp"

#include "base/behavior_test_utils.hpp"

namespace BehaviorTestsDefinitions {

typedef std::tuple<
        std::string,    // Locale name
        std::string>    // Target device name
        LocaleParams;

class CustomLocaleTest : public ov::test::behavior::APIBaseTest,
                         public ::testing::WithParamInterface<LocaleParams> {
protected:
    std::shared_ptr<ngraph::Function> function;
    std::string localeName;
    std::string testName;

    void set_api_entity() override { api_entity = ov::test::utils::ov_entity::ie_executable_network; }

    void SetUp() override;
public:
    static std::string getTestCaseName(const testing::TestParamInfo<LocaleParams> &obj);
};

} // namespace BehaviorTestsDefinitions
