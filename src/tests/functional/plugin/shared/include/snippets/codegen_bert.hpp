// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>
#include <memory>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ov_models/utils/ov_helpers.hpp"
#include "ov_models/builders.hpp"
//  todo: Rewrite this test using Snippets test infrastructure. See add_convert or conv_eltwise for example

namespace ov {
namespace test {
namespace snippets {

typedef std::tuple<
        ov::element::Type_t,  // Network Precision
        ov::Shape,            // Input 0 Shape
        ov::Shape,            // Input 1 Shape
        std::string           // Target Device
> CodegenBertParams;

class CodegenBert : public testing::WithParamInterface<ov::test::snippets::CodegenBertParams>,
virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ov::test::snippets::CodegenBertParams> obj);

protected:
    void SetUp() override;
};


} // namespace snippets
} // namespace test
} // namespace ov
