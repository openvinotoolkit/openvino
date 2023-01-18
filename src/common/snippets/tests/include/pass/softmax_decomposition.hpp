// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "lowering_utils.hpp"
#include "snippets_helpers.hpp"

namespace ov {
namespace test {
namespace snippets {

typedef std::tuple<
        Shape, // Input shape 0
        int  // Axis
> SoftmaxParams;

typedef std::tuple<
        Shape, // Input shape 0
        Shape, // Input shape 1
        int  // Axis
> AddSoftmaxParams;

class SoftmaxTests : public LoweringTests, public testing::WithParamInterface<SoftmaxParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<SoftmaxParams> obj);
protected:
    void SetUp() override;
    std::shared_ptr<SnippetsFunctionBase> snippets_function;
};

class AddSoftmaxTests : public LoweringTests, public testing::WithParamInterface<AddSoftmaxParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<AddSoftmaxParams> obj);
protected:
    void SetUp() override;
    std::shared_ptr<SnippetsFunctionBase> snippets_function;
};

}  // namespace snippets
}  // namespace test
}  // namespace ov
