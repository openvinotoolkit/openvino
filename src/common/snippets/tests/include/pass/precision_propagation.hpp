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
        Actual() = default;

        Actual(const std::pair<element::Type, element::Type>& convertion_before_op1,
               const std::pair<element::Type, element::Type>& convertion_before_op2,
               const std::set<std::vector<InferenceEngine::Precision>>& op1_supported_precisions,
               const std::set<std::vector<InferenceEngine::Precision>>& op2_supported_precisions)
            : convertion_before_op1(convertion_before_op1),
              convertion_before_op2(convertion_before_op2),
              op1_supported_precisions(op1_supported_precisions),
              op2_supported_precisions(op2_supported_precisions) {}

        std::pair<element::Type, element::Type> convertion_before_op1;
        std::pair<element::Type, element::Type> convertion_before_op2;
        std::set<std::vector<InferenceEngine::Precision>> op1_supported_precisions;
        std::set<std::vector<InferenceEngine::Precision>> op2_supported_precisions;
    };

    class Expected {
    public:
        Expected() = default;

        Expected(
            const std::pair<element::Type, element::Type>& convertion_before_op1,
            const std::pair<element::Type, element::Type>& convertion_before_op2,
            const element::Type convertion_after_op2)
            : convertion_before_op1(convertion_before_op1),
              convertion_before_op2(convertion_before_op2),
              convertion_after_op2(convertion_after_op2) {}

        std::pair<element::Type, element::Type> convertion_before_op1;
        std::pair<element::Type, element::Type> convertion_before_op2;
        element::Type convertion_after_op2;
    };

    PrecisionPropagationParamsValues() = default;
    PrecisionPropagationParamsValues(const std::vector<element::Type>& input_types,
                                     const Actual& actual,
                                     const Expected& expected)
        : input_types(input_types),
          actual(actual),
          expected(expected) {}

    std::vector<element::Type> input_types;
    Actual actual;
    Expected expected;
};

typedef std::tuple<
    std::pair<Shape, Shape>, // input shapes
    PrecisionPropagationParamsValues
> PrecisionPropagationParams;

class PrecisionPropagationTest : public TransformationTestsF,
                                 public testing::WithParamInterface<PrecisionPropagationParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<PrecisionPropagationParams> obj);

protected:
    std::shared_ptr<SnippetsFunctionBase> snippets_function;
};

}  // namespace snippets
}  // namespace test
}  // namespace ov
