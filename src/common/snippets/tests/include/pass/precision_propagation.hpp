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
        Actual(const std::pair<element::Type, element::Type>& input_types,
               const std::vector<std::pair<element::Type, element::Type>>& supported_precisions,
               const std::vector<std::pair<element::Type, element::Type>>& expected_convertion_types)
            : input_types(input_types),
              supported_precisions(supported_precisions),
              expected_convertion_types(expected_convertion_types) {}

        std::pair<element::Type, element::Type> input_types;
        std::vector<std::pair<element::Type, element::Type>> supported_precisions;
        std::vector<std::pair<element::Type, element::Type>> expected_convertion_types;
    };

    class Expected {
    public:
        Expected() = default;
        Expected(const std::vector<std::pair<element::Type, element::Type>>& convertion_output_types)
            : convertion_output_types(convertion_output_types) {}
        std::vector<std::pair<element::Type, element::Type>> convertion_output_types;
    };

    PrecisionPropagationParamsValues() = default;
    PrecisionPropagationParamsValues(const Actual& actual, const Expected& expected)
        : actual(actual),
          expected(expected) {}

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
    void SetUp() override;
    std::shared_ptr<SnippetsFunctionBase> snippets_function;
};

}  // namespace snippets
}  // namespace test
}  // namespace ov
