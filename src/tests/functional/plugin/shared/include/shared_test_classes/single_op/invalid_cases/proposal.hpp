// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_op/proposal.hpp"

namespace ov {
namespace test {
typedef std::tuple<proposalSpecificParams, std::vector<float>, std::string> proposalBehTestParamsSet;

class ProposalBehTest : public testing::WithParamInterface<proposalBehTestParamsSet>,
                        virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<proposalBehTestParamsSet> obj);

protected:
    void SetUp() override;
    void run() override;
};

}  // namespace test
}  // namespace ov
