// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ie_parameter.hpp>
#include <ie_layouts.h>

using namespace InferenceEngine;

class DestructorTest {
public:
    DestructorTest() {
        constructorCount++;
    }

    DestructorTest(const DestructorTest& c) {
        constructorCount++;
    }

    DestructorTest(const DestructorTest&& c) {
        constructorCount++;
    }

    ~DestructorTest() {
        destructorCount++;
    }

    static size_t destructorCount;
    static size_t constructorCount;
};
size_t DestructorTest::destructorCount = 0;
size_t DestructorTest::constructorCount = 0;

class ParameterTests : public ::testing::Test {
public:
    void SetUp() override {
        DestructorTest::destructorCount = 0;
        DestructorTest::constructorCount = 0;
    }
};

TEST_F(ParameterTests, ParameterAsInt) {
    Parameter p = 4;
    ASSERT_TRUE(p.is<int>());
    int test = p;
    ASSERT_EQ(4, test);
}

TEST_F(ParameterTests, ParameterAsUInt) {
    Parameter p = 4u;
    ASSERT_TRUE(p.is<unsigned int>());
#ifdef __i386__
    ASSERT_TRUE(p.is<size_t>());
#else
    ASSERT_FALSE(p.is<size_t>());
#endif
    unsigned int test = p;
    ASSERT_EQ(4, test);
}

TEST_F(ParameterTests, ParameterAsSize_t) {
    size_t ref = 4;
    Parameter p = ref;
    ASSERT_TRUE(p.is<size_t>());
    size_t test = p;
    ASSERT_EQ(ref, test);
}

TEST_F(ParameterTests, ParameterAsFloat) {
    Parameter p = 4.f;
    ASSERT_TRUE(p.is<float>());
    float test = p;
    ASSERT_EQ(4.f, test);
}

TEST_F(ParameterTests, ParameterAsString) {
    std::string ref = "test";
    Parameter p = ref;
    std::string test = p;
    ASSERT_TRUE(p.is<std::string>());
    ASSERT_EQ(ref, test);
}

TEST_F(ParameterTests, ParameterAsStringInLine) {
    Parameter p = "test";
    std::string test = p;
    ASSERT_TRUE(p.is<std::string>());
    ASSERT_EQ("test", test);
}

TEST_F(ParameterTests, IntParameterAsString) {
    Parameter p = 4;
    ASSERT_TRUE(p.is<int>());
    ASSERT_FALSE(p.is<std::string>());
    ASSERT_THROW(std::string test = p, std::bad_cast);
    ASSERT_THROW(std::string test = p.as<std::string>(), std::bad_cast);
}

TEST_F(ParameterTests, StringParameterAsInt) {
    Parameter p = "4";
    ASSERT_FALSE(p.is<int>());
    ASSERT_TRUE(p.is<std::string>());
    ASSERT_THROW((void)static_cast<int>(p), std::bad_cast);
    ASSERT_THROW((void)p.as<int>(), std::bad_cast);
}

TEST_F(ParameterTests, ParameterAsTensorDesc) {
    TensorDesc ref(Precision::FP32, {1, 3, 2, 2}, Layout::NCHW);
    Parameter p = ref;
    ASSERT_TRUE(p.is<TensorDesc>());
    TensorDesc test = p;
    ASSERT_EQ(ref, test);
}

TEST_F(ParameterTests, ParameterAsInts) {
    std::vector<int> ref = {1, 2, 3, 4, 5};
    Parameter p = ref;
    ASSERT_TRUE(p.is<std::vector<int>>());
    std::vector<int> test = p;
    ASSERT_EQ(ref.size(), test.size());
    for (size_t i = 0; i < ref.size(); i++) {
        ASSERT_EQ(ref[i], test[i]);
    }
}

TEST_F(ParameterTests, ParameterAsUInts) {
    std::vector<unsigned int> ref = {1, 2, 3, 4, 5};
    Parameter p = ref;
    ASSERT_TRUE(p.is<std::vector<unsigned int>>());
    std::vector<unsigned int> test = p;
    ASSERT_EQ(ref.size(), test.size());
    for (size_t i = 0; i < ref.size(); i++) {
        ASSERT_EQ(ref[i], test[i]);
    }
}

TEST_F(ParameterTests, ParameterAsSize_ts) {
    std::vector<size_t> ref = {1, 2, 3, 4, 5};
    Parameter p = ref;
    ASSERT_TRUE(p.is<std::vector<size_t>>());
    std::vector<size_t> test = p;
    ASSERT_EQ(ref.size(), test.size());
    for (size_t i = 0; i < ref.size(); i++) {
        ASSERT_EQ(ref[i], test[i]);
    }
}

TEST_F(ParameterTests, ParameterAsFloats) {
    std::vector<float> ref = {1, 2, 3, 4, 5};
    Parameter p = ref;
    ASSERT_TRUE(p.is<std::vector<float>>());
    std::vector<float> test = p;
    ASSERT_EQ(ref.size(), test.size());
    for (size_t i = 0; i < ref.size(); i++) {
        ASSERT_EQ(ref[i], test[i]);
    }
}

TEST_F(ParameterTests, ParameterAsStrings) {
    std::vector<std::string> ref = {"test1", "test2", "test3", "test4", "test1"};
    Parameter p = ref;
    ASSERT_TRUE(p.is<std::vector<std::string>>());
    std::vector<std::string> test = p;
    ASSERT_EQ(ref.size(), test.size());
    for (size_t i = 0; i < ref.size(); i++) {
        ASSERT_EQ(ref[i], test[i]);
    }
}

TEST_F(ParameterTests, ParameterAsMapOfParameters) {
    std::map<std::string, Parameter> refMap;
    refMap["testParamInt"] = 4;
    refMap["testParamString"] = "test";
    Parameter p = refMap;
    bool isMap = p.is<std::map<std::string, Parameter>>();
    ASSERT_TRUE(isMap);
    std::map<std::string, Parameter> testMap = p;

    ASSERT_NE(testMap.find("testParamInt"), testMap.end());
    ASSERT_NE(testMap.find("testParamString"), testMap.end());

    int testInt = testMap["testParamInt"];
    std::string testString = testMap["testParamString"];

    ASSERT_EQ(refMap["testParamInt"].as<int>(), testInt);
    ASSERT_EQ(refMap["testParamString"].as<std::string>(), testString);
}

TEST_F(ParameterTests, ParameterNotEmpty) {
    Parameter p = 4;
    ASSERT_FALSE(p.empty());
}

TEST_F(ParameterTests, ParameterEmpty) {
    Parameter p;
    ASSERT_TRUE(p.empty());
}

TEST_F(ParameterTests, ParameterClear) {
    Parameter p = 4;
    ASSERT_FALSE(p.empty());
    p.clear();
    ASSERT_TRUE(p.empty());
}

TEST_F(ParameterTests, ParametersNotEqualByType) {
    Parameter p1 = 4;
    Parameter p2 = "string";
    ASSERT_TRUE(p1 != p2);
    ASSERT_FALSE(p1 == p2);
}

TEST_F(ParameterTests, ParametersNotEqualByValue) {
    Parameter p1 = 4;
    Parameter p2 = 5;
    ASSERT_TRUE(p1 != p2);
    ASSERT_FALSE(p1 == p2);
}

TEST_F(ParameterTests, ParametersEqual) {
    Parameter p1 = 4;
    Parameter p2 = 4;
    ASSERT_TRUE(p1 == p2);
    ASSERT_FALSE(p1 != p2);
}

TEST_F(ParameterTests, ParametersStringEqual) {
    std::string s1 = "abc";
    std::string s2 = std::string("a") + "bc";
    Parameter p1 = s1;
    Parameter p2 = s2;
    ASSERT_TRUE(s1 == s2);
    ASSERT_TRUE(p1 == p2);
    ASSERT_FALSE(p1 != p2);
}

TEST_F(ParameterTests, ParametersCStringEqual) {
    const char s1[] = "abc";
    const char s2[] = "abc";
    Parameter p1 = s1;
    Parameter p2 = s2;
    ASSERT_TRUE(s1 != s2);
    ASSERT_TRUE(p1 == p2);
    ASSERT_FALSE(p1 != p2);
}

TEST_F(ParameterTests, CompareParametersWithoutEqualOperator) {
    class TestClass {
    public:
        TestClass(int test, int* testPtr): test(test), testPtr(testPtr) {}

    private:
        int test;
        int* testPtr;
    };

    TestClass a(2, reinterpret_cast<int*>(0x234));
    TestClass b(2, reinterpret_cast<int*>(0x234));
    TestClass c(3, reinterpret_cast<int*>(0x234));
    Parameter parA = a;
    Parameter parB = b;
    Parameter parC = c;

    ASSERT_THROW((void)(parA == parB), details::InferenceEngineException);
    ASSERT_THROW((void)(parA != parB), details::InferenceEngineException);
    ASSERT_THROW((void)(parA == parC), details::InferenceEngineException);
    ASSERT_THROW((void)(parA != parC), details::InferenceEngineException);
}

TEST_F(ParameterTests, ParameterRemovedRealObject) {
    ASSERT_EQ(0, DestructorTest::constructorCount);
    ASSERT_EQ(0, DestructorTest::destructorCount);
    {
        DestructorTest t;
        Parameter p1 = t;
    }
    ASSERT_EQ(2, DestructorTest::constructorCount);
    ASSERT_EQ(2, DestructorTest::destructorCount);
}

TEST_F(ParameterTests, ParameterRemovedRealObjectWithDuplication) {
    ASSERT_EQ(0, DestructorTest::constructorCount);
    ASSERT_EQ(0, DestructorTest::destructorCount);
    {
        DestructorTest t;
        Parameter p = t;
        ASSERT_EQ(0, DestructorTest::destructorCount);
        p = t;
        ASSERT_EQ(2, DestructorTest::destructorCount);
    }
    ASSERT_EQ(4, DestructorTest::constructorCount);
    ASSERT_EQ(4, DestructorTest::destructorCount);
}

TEST_F(ParameterTests, ParameterRemovedRealObjectPointerWithDuplication) {
    ASSERT_EQ(0, DestructorTest::constructorCount);
    ASSERT_EQ(0, DestructorTest::destructorCount);
    {
        auto * t = new DestructorTest();
        Parameter p = t;
        ASSERT_EQ(1, DestructorTest::constructorCount);
        ASSERT_EQ(0, DestructorTest::destructorCount);
        p = t;
        ASSERT_TRUE(p.is<DestructorTest *>());
        DestructorTest* t2 = p;
        ASSERT_EQ(0, DestructorTest::destructorCount);
        delete t;
        auto * t3 = p.as<DestructorTest *>();
        ASSERT_EQ(t2, t3);
    }
    ASSERT_EQ(1, DestructorTest::constructorCount);
    ASSERT_EQ(1, DestructorTest::destructorCount);
}