// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once


#include "common_test_utils/test_common.hpp"

typedef std::tuple<
        std::shared_ptr<ov::Model>,  // Model to work with
        std::vector<std::string>,    // Target layers to add as outputs
        std::string>                 // Target device name
        addOutputsParams;

class AddOutputsTest : public ov::test::TestsCommon,
                       public testing::WithParamInterface<addOutputsParams> {
protected:
    void SetUp() override;
public:
    static std::string getTestCaseName(const testing::TestParamInfo<addOutputsParams> &obj);
};
