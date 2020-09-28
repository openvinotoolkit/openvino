// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>
#include <memory>

#include "functional_test_utils/layer_test_utils.hpp"

namespace LayerTestsDefinitions {

typedef std::tuple<
    LayerTestsUtils::TargetDevice, // Device name
    unsigned int                   // Allocations count
> MultipleAllocationsParams;

class MultipleAllocations : public testing::WithParamInterface<MultipleAllocationsParams>,
                        virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MultipleAllocationsParams>& obj);

protected:
    void SetUp() override;

protected:
    unsigned int  m_allocationsCount = 0;
};

}  // namespace LayerTestsDefinitions
