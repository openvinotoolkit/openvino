// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/snippets_test_utils.hpp"

namespace ov {
namespace test {
namespace snippets {

class CheckBroadcastTestCaseParams {
public:
    std::pair<InputShape, InputShape> input_shapes;
    ov::op::AutoBroadcastSpec broadcast;
    size_t num_nodes;
    size_t num_subgraphs;
};

typedef std::tuple <
    ov::element::Type,            // input types
    CheckBroadcastTestCaseParams, // test case details
    std::string                   // target device
> CheckBroadcastParams;

class CheckBroadcast : public testing::WithParamInterface<CheckBroadcastParams>, virtual public SnippetsTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<CheckBroadcastParams> obj);

protected:
    void SetUp() override;
};

} // namespace snippets
} // namespace test
} // namespace ov
