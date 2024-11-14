// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/snippets_test_utils.hpp"

namespace ov {
namespace test {
namespace snippets {

typedef std::tuple<
        std::vector<InputShape>,        // Input shapes
        std::vector<int64_t>,           // Transpose Order
        int64_t,                        // Softmax Axis
        size_t,                         // Expected num nodes
        size_t,                         // Expected num subgraphs
        std::string                     // Target Device
> TransposeSoftmaxParams;


class TransposeSoftmax : public testing::WithParamInterface<ov::test::snippets::TransposeSoftmaxParams>,
                         virtual public SnippetsTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ov::test::snippets::TransposeSoftmaxParams> obj);

protected:
    void SetUp() override;
};

class TransposeSoftmaxEltwise : public TransposeSoftmax {
protected:
    void SetUp() override;
};


} // namespace snippets
} // namespace test
} // namespace ov