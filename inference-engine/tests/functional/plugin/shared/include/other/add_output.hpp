// Copyright (C) 2020 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <map>

#include "common_test_utils/common_layers_params.hpp"
#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/test_common.hpp"
#include "common_test_utils/test_constants.hpp"
#include "common_test_utils/xml_net_builder/ir_net.hpp"
#include "common_test_utils/xml_net_builder/xml_filler.hpp"
#include "ie_core.hpp"

class AddOutputTestsCommonClass : public CommonTestUtils::TestsCommon,
                                  public testing::WithParamInterface<std::tuple<std::string, std::string>> {
private:
    static std::string generate_model();

public:
    static std::string getTestCaseName(testing::TestParamInfo<std::tuple<std::string, std::string>> obj);
    void run_test();
};
