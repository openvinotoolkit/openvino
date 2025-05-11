// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <string>

#include "gtest/gtest.h"

// Copied from gtest

namespace ov {
std::string prepend_disabled(const std::string& backend_name,
                             const std::string& test_name,
                             const std::string& manifest);

std::string combine_test_backend_and_case(const std::string& backend_name, const std::string& test_casename);
}  // namespace ov

#define OPENVINO_GTEST_TEST_CLASS_NAME_(backend_name, test_case_name, test_name) \
    backend_name##_##test_case_name##_##test_name##_Test

#define OPENVINO_GTEST_TEST_(backend_name, test_case_name, test_name, parent_class, parent_id)                        \
    class OPENVINO_GTEST_TEST_CLASS_NAME_(backend_name, test_case_name, test_name) : public parent_class {            \
    public:                                                                                                           \
        OPENVINO_GTEST_TEST_CLASS_NAME_(backend_name, test_case_name, test_name)() {}                                 \
                                                                                                                      \
    private:                                                                                                          \
        void TestBody() override;                                                                                     \
        static ::testing::TestInfo* const test_info_ GTEST_ATTRIBUTE_UNUSED_;                                         \
        GTEST_DISALLOW_COPY_AND_ASSIGN_(OPENVINO_GTEST_TEST_CLASS_NAME_(backend_name, test_case_name, test_name));    \
    };                                                                                                                \
                                                                                                                      \
    ::testing::TestInfo* const OPENVINO_GTEST_TEST_CLASS_NAME_(backend_name, test_case_name, test_name)::test_info_ = \
        ::testing::internal::MakeAndRegisterTestInfo(                                                                 \
            ::ov::combine_test_backend_and_case(#backend_name, #test_case_name).c_str(),                              \
            ::ov::prepend_disabled(#backend_name, #test_name, s_manifest).c_str(),                                    \
            nullptr,                                                                                                  \
            nullptr,                                                                                                  \
            ::testing::internal::CodeLocation(__FILE__, __LINE__),                                                    \
            (parent_id),                                                                                              \
            parent_class::SetUpTestCase,                                                                              \
            parent_class::TearDownTestCase,                                                                           \
            new ::testing::internal::TestFactoryImpl<                                                                 \
                OPENVINO_GTEST_TEST_CLASS_NAME_(backend_name, test_case_name, test_name)>);                           \
    void OPENVINO_GTEST_TEST_CLASS_NAME_(backend_name, test_case_name, test_name)::TestBody()

#define OPENVINO_TEST(test_case_name, test_name) \
    OPENVINO_GTEST_TEST_(test_case_name,         \
                         test_case_name,         \
                         test_name,              \
                         ::testing::Test,        \
                         ::testing::internal::GetTestTypeId())

// OPENVINO_TEST_F facilitates the use of the same configuration parameters for multiple
// unit tests similar to the original TEST_F, but with the introduction of a new 0th
// parameter for the backend name, which allows openvino's manifest controlled unit testing.
//
// Start by defining a class derived from testing::Test, which you'll pass for the
// text_fixture parameter.
// Then use this class to define multiple related unit tests (which share some common
// configuration information and/or setup code).
//
// Generated test names take the form:
// BACKENDNAME/FixtureClassName.test_name
// where the test case name is "BACKENDNAME/FixtureClassName"
// and the test name is "test_name"
//
// With the use of OPENVINO_TEST_F the filter to run all the tests for a given backend
// should be:
// --gtest_filter=BACKENDNAME*.*
// (rather than the BACKENDNAME.* that worked before the use of OPENVINO_TEST_F)
#define OPENVINO_TEST_F(backend_name, test_fixture, test_name) \
    OPENVINO_GTEST_TEST_(backend_name,                         \
                         test_fixture,                         \
                         test_name,                            \
                         test_fixture,                         \
                         ::testing::internal::GetTypeId<test_fixture>())

// OPENVINO_TEST_P combined with OPENVINO_INSTANTIATE_TEST_SUITE_P facilate the generation
// of value parameterized tests (similar to the original TEST_P and INSTANTIATE_TEST_SUITE_P
// with the addition of a new 0th parameter for the backend name, which allows openvino's
// manifest controlled unit testing).
//
// Start by defining a class derived from ::testing::TestWithParam<T>, which you'll pass
// for the test_case_name parameter.
// Then use OPENVINO_INSTANTIATE_TEST_SUITE_P to define each generation of test cases (see below).
#define OPENVINO_TEST_P(backend_name, test_case_name, test_name)                                                   \
    class OPENVINO_GTEST_TEST_CLASS_NAME_(backend_name, test_case_name, test_name) : public test_case_name {       \
    public:                                                                                                        \
        OPENVINO_GTEST_TEST_CLASS_NAME_(backend_name, test_case_name, test_name)() {}                              \
        void TestBody() override;                                                                                  \
                                                                                                                   \
    private:                                                                                                       \
        static int AddToRegistry() {                                                                               \
            ::testing::UnitTest::GetInstance()                                                                     \
                ->parameterized_test_registry()                                                                    \
                .GetTestCasePatternHolder<test_case_name>(#backend_name "/" #test_case_name,                       \
                                                          ::testing::internal::CodeLocation(__FILE__, __LINE__))   \
                ->AddTestPattern(                                                                                  \
                    #backend_name "/" #test_case_name,                                                             \
                    ::ov::prepend_disabled(#backend_name "/" #test_case_name, #test_name, s_manifest).c_str(),     \
                    new ::testing::internal::TestMetaFactory<                                                      \
                        OPENVINO_GTEST_TEST_CLASS_NAME_(backend_name, test_case_name, test_name)>());              \
            return 0;                                                                                              \
        }                                                                                                          \
        static int gtest_registering_dummy_ GTEST_ATTRIBUTE_UNUSED_;                                               \
        GTEST_DISALLOW_COPY_AND_ASSIGN_(OPENVINO_GTEST_TEST_CLASS_NAME_(backend_name, test_case_name, test_name)); \
    };                                                                                                             \
    int OPENVINO_GTEST_TEST_CLASS_NAME_(backend_name, test_case_name, test_name)::gtest_registering_dummy_ =       \
        OPENVINO_GTEST_TEST_CLASS_NAME_(backend_name, test_case_name, test_name)::AddToRegistry();                 \
    void OPENVINO_GTEST_TEST_CLASS_NAME_(backend_name, test_case_name, test_name)::TestBody()

// Use OPENVINO_INSTANTIATE_TEST_SUITE_P to create a generated set of test case variations.
// The prefix parameter is a label that you optionally provide (no quotes) for a unique
// test name (helpful for labelling a set of inputs and for filtering).
// The prefix parameter can be skipped by simply using a bare comma (see example below).
//
// Unlike INSTANTIATE_TEST_SUITE_P we don't currently support passing a custom param
// name generator. Supporting this with a single macro name requires the use of
// ... and __VA_ARGS__ which in turn generates a warning using INSTANTIATE_TEST_SUITE_P
// without a trailing , parameter.
//
// Examples:
// OPENVINO_INSTANTIATE_TEST_SUITE_P(BACKENDNAME,                  // backend_name
//                                ,                             // empty/skipped prefix
//                                TestWithParamSubClass,        // test_suite_name
//                                ::testing::Range(0, 3) )      // test generator
// would generate:
// BACKENDNAME/TestWithParamSubClass.test_name/0
// BACKENDNAME/TestWithParamSubClass.test_name/1
// BACKENDNAME/TestWithParamSubClass.test_name/2
//
// OPENVINO_INSTANTIATE_TEST_SUITE_P(BACKENDNAME,                  // backend_name
//                                NumericRangeTests,            // prefix
//                                TestWithParamSubClass,        // test_suite_name
//                                ::testing::Range(0, 3) )      // test generator
// would generate:
// BACKENDNAME/NumericRangeTests/BACKENDNAME/TestWithParamSubClass.test_name/0
// BACKENDNAME/NumericRangeTests/BACKENDNAME/TestWithParamSubClass.test_name/1
// BACKENDNAME/NumericRangeTests/BACKENDNAME/TestWithParamSubClass.test_name/2
//
// With the use of OPENVINO_TEST_P and OPENVINO_INSTANTIATE_TEST_SUITE_P
// the filter to run all the tests for a given backend should be:
// --gtest_filter=BACKENDNAME*.*
// (rather than the BACKENDNAME.* that worked before the use of OPENVINO_TEST_P)
#define OPENVINO_INSTANTIATE_TEST_SUITE_P(backend_name, prefix, test_suite_name, generator)                   \
    static ::testing::internal::ParamGenerator<test_suite_name::ParamType>                                    \
        gtest_##prefix##backend_name##test_suite_name##_EvalGenerator_() {                                    \
        return generator;                                                                                     \
    }                                                                                                         \
    static ::std::string gtest_##prefix##backend_name##test_suite_name##_EvalGenerateName_(                   \
        const ::testing::TestParamInfo<test_suite_name::ParamType>& info) {                                   \
        return ::testing::internal::DefaultParamName<test_suite_name::ParamType>(info);                       \
    }                                                                                                         \
    static int gtest_##prefix##backend_name##test_suite_name##_dummy_ GTEST_ATTRIBUTE_UNUSED_ =               \
        ::testing::UnitTest::GetInstance()                                                                    \
            ->parameterized_test_registry()                                                                   \
            .GetTestCasePatternHolder<test_suite_name>(#backend_name "/" #test_suite_name,                    \
                                                       ::testing::internal::CodeLocation(__FILE__, __LINE__)) \
            ->AddTestSuiteInstantiation(#prefix[0] != '\0' ? #backend_name "/" #prefix : "",                  \
                                        &gtest_##prefix##backend_name##test_suite_name##_EvalGenerator_,      \
                                        &gtest_##prefix##backend_name##test_suite_name##_EvalGenerateName_,   \
                                        __FILE__,                                                             \
                                        __LINE__)
