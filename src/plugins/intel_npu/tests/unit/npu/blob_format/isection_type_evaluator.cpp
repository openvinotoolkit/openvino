// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/isection_type_evaluator.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace intel_npu;

class DumbSectionTypeEvaluator : public ISectionTypeEvaluator {
public:
    explicit DumbSectionTypeEvaluator(const bool evaluation_result)
        : ISectionTypeEvaluator(),
          m_result(evaluation_result) {}

private:
    bool evaluate() const override {
        return m_result;
    }

    bool m_result;
};

class MockSectionTypeEvaluator : public ISectionTypeEvaluator {
public:
    MockSectionTypeEvaluator() = default;
    MOCK_METHOD(bool, evaluate, (), (const, override));
};

using ISectionTypeEvaluatorUnitTests = ::testing::Test;

TEST(ISectionTypeEvaluatorUnitTests, ReturnsTheCorrectResult) {
    ASSERT_TRUE(DumbSectionTypeEvaluator(true).get_result());
    ASSERT_FALSE(DumbSectionTypeEvaluator(false).get_result());
}

TEST(ISectionTypeEvaluatorUnitTests, Evaluated) {
    DumbSectionTypeEvaluator evaluator(false);
    ASSERT_FALSE(evaluator.evaluated());
    evaluator.get_result();
    ASSERT_TRUE(evaluator.evaluated());
    evaluator.get_result();
    ASSERT_TRUE(evaluator.evaluated());
}

TEST(ISectionTypeEvaluatorUnitTests, EvaluatesOnlyOnceWhenRequested) {
    MockSectionTypeEvaluator evaluator;
    EXPECT_CALL(evaluator, evaluate()).Times(1);
    evaluator.get_result();
    evaluator.get_result();
}

TEST(ISectionTypeEvaluatorUnitTests, NoEvaluation) {
    MockSectionTypeEvaluator evaluator;
    EXPECT_CALL(evaluator, evaluate()).Times(0);
    evaluator.evaluated();
}
