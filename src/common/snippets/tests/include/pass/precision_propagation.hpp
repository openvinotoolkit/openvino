// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "lowering_utils.hpp"
#include "snippets_helpers.hpp"

namespace ov {
namespace test {
namespace snippets {

class PrecisionPropagationParamsValues {
public:
    class Actual {
    public:
        std::pair<element::Type, element::Type> convertion_before_op1;
        element::Type convertion_before_op2_1;
        std::pair<element::Type, element::Type> convertion_before_op2_2;
        std::set<std::vector<element::Type>> op1_supported_precisions;
        std::set<std::vector<element::Type>> op2_supported_precisions;
    };

    class Expected {
    public:
        std::pair<element::Type, element::Type> convertion_before_op1;
        element::Type convertion_before_op2_1;
        std::pair<element::Type, element::Type> convertion_before_op2_2;
        element::Type convertion_after_op2;
    };

    std::vector<element::Type> input_types;
    Actual actual;
    Expected expected;
};

typedef std::tuple<
    std::pair<PartialShape, PartialShape>, // input shapes
    PrecisionPropagationParamsValues
> PrecisionPropagationParams;

class PrecisionPropagationTest : public TransformationTestsF,
                                 public testing::WithParamInterface<PrecisionPropagationParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<PrecisionPropagationParams> obj);

protected:
    std::shared_ptr<SnippetsFunctionBase> snippets_model;
};

}  // namespace snippets
}  // namespace test
}  // namespace ov
