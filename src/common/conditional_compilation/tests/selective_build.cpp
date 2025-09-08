// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#ifdef SELECTIVE_BUILD_ANALYZER
#    define SELECTIVE_BUILD_ANALYZER_ON
#    undef SELECTIVE_BUILD_ANALYZER
#elif defined(SELECTIVE_BUILD)
#    define SELECTIVE_BUILD_ON
#    undef SELECTIVE_BUILD
#endif

#define SELECTIVE_BUILD

#include <openvino/cc/factory.h>

namespace {
OV_CC_DOMAINS(CCTests);

template <typename T>
struct TestTemplateClass;

template <>
struct TestTemplateClass<int> {
    void operator()(int& v) {
        v = 42;
    }
};

template <>
struct TestTemplateClass<bool> {
    void operator()(int& v) {
        v = 43;
    }
};

template <>
struct TestTemplateClass<float> {
    void operator()(int& v) {
        v = 44;
    }
};

struct TestNodeBase {
    TestNodeBase(int k, int v) : key(k), value(v) {}
    virtual ~TestNodeBase() = default;
    int key;
    int value;
};

template <int N>
struct TestNode : public TestNodeBase {
    TestNode(int value) : TestNodeBase(N, value) {}
};

}  // namespace

TEST(ConditionalCompilationTests, SimpleScope) {
#define CCTests_Scope0 1
    int n = 0;

    // Simple scope is enabled
    OV_SCOPE(CCTests, Scope0) {
        n = 42;
    }
    EXPECT_EQ(n, 42);

    // Simple scope is disabled
    OV_SCOPE(CCTests, Scope1) n = 43;
    EXPECT_EQ(n, 42);

#undef CCTests_Scope0
}

TEST(ConditionalCompilationTests, SwitchCase) {
    // Cases 0 and 2 are enabled
#define CCTests_TestTemplateClass       1
#define CCTests_TestTemplateClass_cases OV_CASE(0, int), OV_CASE(2, float)

    int n = 0;

    OV_SWITCH(CCTests, TestTemplateClass, n, 0, OV_CASE(0, int), OV_CASE(1, bool), OV_CASE(2, float));
    EXPECT_EQ(n, 42);

    OV_SWITCH(CCTests, TestTemplateClass, n, 1, OV_CASE(0, int), OV_CASE(1, bool), OV_CASE(2, float));
    EXPECT_EQ(n, 42);

    OV_SWITCH(CCTests, TestTemplateClass, n, 2, OV_CASE(0, int), OV_CASE(1, bool), OV_CASE(2, float));
    EXPECT_EQ(n, 44);

#undef CCTests_TestTemplateClass
#undef CCTests_TestTemplateClass_cases
}

TEST(ConditionalCompilationTests, Factory) {
#define CCTests_TestNode0 1
#define CCTests_TestNode1 0
#define CCTests_TestNode2 1

    openvino::cc::Factory<int, TestNodeBase*(int)> testFactory("TestFactory");
    testFactory.registerNodeIfRequired(CCTests, TestNode0, 0, TestNode<0>);
    testFactory.registerNodeIfRequired(CCTests, TestNode1, 1, TestNode<1>);
    testFactory.registerNodeIfRequired(CCTests, TestNode2, 2, TestNode<2>);

    TestNodeBase* node0 = testFactory.createNodeIfRegistered(CCTests, 0, 42);
    TestNodeBase* node1 = testFactory.createNodeIfRegistered(CCTests, 1, 43);
    TestNodeBase* node2 = testFactory.createNodeIfRegistered(CCTests, 2, 44);

    EXPECT_TRUE(node0 && node0->key == 0 && node0->value == 42);
    EXPECT_TRUE(!node1);
    EXPECT_TRUE(node2 && node2->key == 2 && node2->value == 44);

    delete node0;
    delete node1;
    delete node2;

#undef CCTests_TestNode0
#undef CCTests_TestNode1
#undef CCTests_TestNode2
}

#undef SELECTIVE_BUILD

#ifdef SELECTIVE_BUILD_ANALYZER_ON
#    define SELECTIVE_BUILD_ANALYZER
#elif defined(SELECTIVE_BUILD_ON)
#    define SELECTIVE_BUILD
#endif
