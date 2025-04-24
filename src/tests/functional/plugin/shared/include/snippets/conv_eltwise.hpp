// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/snippets_test_utils.hpp"

namespace ov {
namespace test {
namespace snippets {

typedef std::tuple<
        ov::Shape,                   // Input Shape #0
        ov::Shape,                   // Input Shape #1
        std::shared_ptr<ov::Node>,   // The first binary eltwise op after the Convolution
        size_t,                      // Expected num nodes
        size_t,                      // Expected num subgraphs
        std::string                  // Target Device
> ConvEltwiseParams;

class ConvEltwise : public testing::WithParamInterface<ov::test::snippets::ConvEltwiseParams>,
                    virtual public SnippetsTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ov::test::snippets::ConvEltwiseParams> obj);

protected:
    void SetUp() override;
};


} // namespace snippets
} // namespace test
} // namespace ov
