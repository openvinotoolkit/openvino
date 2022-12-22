// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/snippets_test_utils.hpp"

namespace ov {
namespace test {
namespace snippets {

typedef std::tuple<
        std::vector<ov::Shape>,          // Input shapes
        bool,                            // With Multiply
        size_t,                          // Expected num nodes
        size_t,                          // Expected num subgraphs
        std::string                      // Target Device
> MHAParams;


class MHA : public testing::WithParamInterface<ov::test::snippets::MHAParams>,
        virtual public ov::test::SnippetsTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ov::test::snippets::MHAParams> obj);

protected:
    void SetUp() override;

    void generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) override;
};

class MHASelect : public MHA {
protected:
    void SetUp() override;

    void generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) override;
};

class MHAWOTransposeOnInputs : public MHA {
protected:
    void SetUp() override;
};

} // namespace snippets
} // namespace test
} // namespace ov
