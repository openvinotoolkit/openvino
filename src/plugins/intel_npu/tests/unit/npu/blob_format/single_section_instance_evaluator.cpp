// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/single_section_instance_evaluator.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"

namespace {

constexpr std::string_view STRING_THAT_EVALUATES_TO_SUPPORTED = "0";
constexpr std::string_view STRING_THAT_EVALUATES_TO_UNSUPPORTED = "1";
constexpr std::string_view STRING_THAT_EVALUATES_TO_NA = "";

}  // namespace

using namespace intel_npu;

class DumbSectionInstanceEvaluator : public ISectionInstanceEvaluator {
public:
    DumbSectionInstanceEvaluator() = default;

    ov::CompatibilityCheck evaluate(std::string_view runtime_requirements) const override {
        if (runtime_requirements.empty()) {
            return ov::CompatibilityCheck::NOT_APPLICABLE;
        }
        if (runtime_requirements == STRING_THAT_EVALUATES_TO_SUPPORTED) {
            return ov::CompatibilityCheck::SUPPORTED;
        }
        return ov::CompatibilityCheck::UNSUPPORTED;
    }
};

class MockSectionInstanceEvaluator : public ISectionInstanceEvaluator {
public:
    MockSectionInstanceEvaluator() = default;
    MOCK_METHOD(ov::CompatibilityCheck, evaluate, (std::string_view), (const, override));
};

using SingleSectionInstanceEvaluatorTest = ::testing::Test;
using testing::_;

TEST(SingleSectionInstanceEvaluatorTest, NullEvaluator) {
    OV_EXPECT_THROW(SingleSectionInstanceEvaluator(nullptr, ""), ov::Exception, _);
}

TEST(SingleSectionInstanceEvaluatorTest, ReturnsTheCorrectResult) {
    SingleSectionInstanceEvaluator evaluator(std::make_shared<DumbSectionInstanceEvaluator>(),
                                             STRING_THAT_EVALUATES_TO_SUPPORTED);
    ASSERT_EQ(evaluator.get_result(), ov::CompatibilityCheck::SUPPORTED);
    evaluator = SingleSectionInstanceEvaluator(std::make_shared<DumbSectionInstanceEvaluator>(),
                                               STRING_THAT_EVALUATES_TO_UNSUPPORTED);
    ASSERT_EQ(evaluator.get_result(), ov::CompatibilityCheck::UNSUPPORTED);
    evaluator =
        SingleSectionInstanceEvaluator(std::make_shared<DumbSectionInstanceEvaluator>(), STRING_THAT_EVALUATES_TO_NA);
    ASSERT_EQ(evaluator.get_result(), ov::CompatibilityCheck::NOT_APPLICABLE);
}

TEST(SingleSectionInstanceEvaluatorTest, Evaluated) {
    const SingleSectionInstanceEvaluator evaluator(std::make_shared<DumbSectionInstanceEvaluator>(),
                                                   STRING_THAT_EVALUATES_TO_SUPPORTED);
    ASSERT_FALSE(evaluator.evaluated());
    evaluator.get_result();
    ASSERT_TRUE(evaluator.evaluated());
    evaluator.get_result();
    ASSERT_TRUE(evaluator.evaluated());
}

TEST(SingleSectionInstanceEvaluatorTest, EvaluatesOnlyOnceWhenRequested) {
    const auto mock = std::make_shared<MockSectionInstanceEvaluator>();
    const auto evaluator = SingleSectionInstanceEvaluator(mock, STRING_THAT_EVALUATES_TO_SUPPORTED);
    EXPECT_CALL(*mock, evaluate(STRING_THAT_EVALUATES_TO_SUPPORTED)).Times(1);
    evaluator.get_result();
    evaluator.get_result();
}

TEST(SingleSectionInstanceEvaluatorTest, NoEvaluation) {
    const auto mock = std::make_shared<MockSectionInstanceEvaluator>();
    const auto evaluator = SingleSectionInstanceEvaluator(mock, STRING_THAT_EVALUATES_TO_SUPPORTED);
    EXPECT_CALL(*mock, evaluate(STRING_THAT_EVALUATES_TO_SUPPORTED)).Times(0);
    evaluator.evaluated();
}
