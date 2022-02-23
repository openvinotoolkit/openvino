// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/deprecated.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
#include "openvino/core/any.hpp"
OPENVINO_SUPPRESS_DEPRECATED_END

#include <ngraph/variant.hpp>
#include <string>

#include "gtest/gtest.h"
#include "openvino/core/runtime_attribute.hpp"

using namespace ov;

class DestructorTest {
public:
    // Base member type defined. ov:Any can access all derived class
    using Base = std::tuple<DestructorTest>;
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

    virtual const std::type_info& type_info() const {
        return typeid(DestructorTest);
    }

    static size_t destructorCount;
    static size_t constructorCount;
};

class Derived : public DestructorTest {
public:
    using DestructorTest::DestructorTest;
    const std::type_info& type_info() const override {
        return typeid(Derived);
    }
};

size_t DestructorTest::destructorCount = 0;
size_t DestructorTest::constructorCount = 0;

class AnyTests : public ::testing::Test {
public:
    void SetUp() override {
        DestructorTest::destructorCount = 0;
        DestructorTest::constructorCount = 0;
    }
};

TEST_F(AnyTests, parameter_std_string) {
    auto parameter = Any{"My string"};
    ASSERT_TRUE(parameter.is<std::string>());
    EXPECT_EQ(parameter.as<std::string>(), "My string");
}

TEST_F(AnyTests, parameter_int64_t) {
    auto parameter = Any{int64_t(27)};
    ASSERT_TRUE(parameter.is<int64_t>());
    EXPECT_FALSE(parameter.is<std::string>());
    EXPECT_EQ(parameter.as<int64_t>(), 27);
}

struct Ship {
    Ship(const std::string& name_, const int16_t x_, const int16_t y_) : name{name_}, x{x_}, y{y_} {}
    std::string name;
    int16_t x;
    int16_t y;
};

TEST_F(AnyTests, parameter_ship) {
    {
        auto parameter = Any{Ship{"Lollipop", 3, 4}};
        ASSERT_TRUE(parameter.is<Ship>());
        Ship& ship = parameter.as<Ship>();
        EXPECT_EQ(ship.name, "Lollipop");
        EXPECT_EQ(ship.x, 3);
        EXPECT_EQ(ship.y, 4);
    }
    {
        auto parameter = Any::make<Ship>("Lollipop", int16_t(3), int16_t(4));
        ASSERT_TRUE(parameter.is<Ship>());
        Ship& ship = parameter.as<Ship>();
        EXPECT_EQ(ship.name, "Lollipop");
        EXPECT_EQ(ship.x, 3);
        EXPECT_EQ(ship.y, 4);
    }
}

TEST_F(AnyTests, AnyAsInt) {
    Any p = 4;
    ASSERT_TRUE(p.is<int>());
    int test = p.as<int>();
    ASSERT_EQ(4, test);
}

TEST_F(AnyTests, AnyAsUInt) {
    Any p = uint32_t(4);
    ASSERT_TRUE(p.is<uint32_t>());
    ASSERT_TRUE(p.is<uint32_t>());
    uint32_t test = p.as<uint32_t>();
    ASSERT_EQ(4, test);
}

TEST_F(AnyTests, AnyAsString) {
    std::string ref = "test";
    Any p = ref;
    ASSERT_TRUE(p.is<std::string>());
    std::string test = p.as<std::string>();
    ASSERT_EQ(ref, test);
}

TEST_F(AnyTests, AnyAsStringInLine) {
    Any p = "test";
    ASSERT_TRUE(p.is<std::string>());
    std::string test = p.as<std::string>();
    ASSERT_EQ("test", test);
}

TEST_F(AnyTests, AnyAsInts) {
    std::vector<int> ref = {1, 2, 3, 4, 5};
    Any p = ref;
    ASSERT_TRUE(p.is<std::vector<int>>());
    std::vector<int> test = p.as<std::vector<int>>();
    ASSERT_EQ(ref.size(), test.size());
    for (size_t i = 0; i < ref.size(); i++) {
        ASSERT_EQ(ref[i], test[i]);
    }
}

TEST_F(AnyTests, AnyAsMapOfAnys) {
    std::map<std::string, Any> refMap;
    refMap["testParamInt"] = 4;
    refMap["testParamString"] = "test";
    Any p = refMap;
    bool isMap = p.is<std::map<std::string, Any>>();
    ASSERT_TRUE(isMap);
    auto testMap = p.as<std::map<std::string, Any>>();

    ASSERT_NE(testMap.find("testParamInt"), testMap.end());
    ASSERT_NE(testMap.find("testParamString"), testMap.end());

    int testInt = testMap["testParamInt"].as<int>();
    std::string testString = testMap["testParamString"].as<std::string>();

    ASSERT_EQ(refMap["testParamInt"].as<int>(), testInt);
    ASSERT_EQ(refMap["testParamString"].as<std::string>(), testString);
}

TEST_F(AnyTests, AnyNotEmpty) {
    Any p = 4;
    ASSERT_FALSE(p.empty());
}

TEST_F(AnyTests, AnyEmpty) {
    Any p;
    ASSERT_TRUE(p.empty());
}

TEST_F(AnyTests, AnyClear) {
    Any p = 4;
    ASSERT_FALSE(p.empty());
    p = {};
    ASSERT_TRUE(p.empty());
}

TEST_F(AnyTests, AnysNotEqualByType) {
    Any p1 = 4;
    Any p2 = "string";
    ASSERT_TRUE(p1 != p2);
    ASSERT_FALSE(p1 == p2);
}

TEST_F(AnyTests, AnysNotEqualByValue) {
    Any p1 = 4;
    Any p2 = 5;
    ASSERT_TRUE(p1 != p2);
    ASSERT_FALSE(p1 == p2);
}

TEST_F(AnyTests, AnysEqual) {
    Any p1 = 4;
    Any p2 = 4;
    ASSERT_TRUE(p1 == p2);
    ASSERT_FALSE(p1 != p2);
}

TEST_F(AnyTests, AnysStringEqual) {
    std::string s1 = "abc";
    std::string s2 = std::string("a") + "bc";
    Any p1 = s1;
    Any p2 = s2;
    ASSERT_TRUE(s1 == s2);
    ASSERT_TRUE(p1 == p2);
    ASSERT_FALSE(p1 != p2);
}

TEST_F(AnyTests, MapOfAnysEqual) {
    std::map<std::string, Any> map0;
    map0["testParamInt"] = 4;
    map0["testParamString"] = "test";
    const auto map1 = map0;

    Any p0 = map0;
    Any p1 = map1;
    ASSERT_TRUE(p0 == p1);
    ASSERT_FALSE(p0 != p1);
}

TEST_F(AnyTests, CompareAnysWithoutEqualOperator) {
    class TestClass {
    public:
        TestClass(int test, int* testPtr) : test(test), testPtr(testPtr) {}

    private:
        int test;
        int* testPtr;
    };

    TestClass a(2, reinterpret_cast<int*>(0x234));
    TestClass b(2, reinterpret_cast<int*>(0x234));
    TestClass c(3, reinterpret_cast<int*>(0x234));
    Any parA = a;
    Any parB = b;
    Any parC = c;

    ASSERT_THROW((void)(parA == parB), ov::Exception);
    ASSERT_THROW((void)(parA != parB), ov::Exception);
    ASSERT_THROW((void)(parA == parC), ov::Exception);
    ASSERT_THROW((void)(parA != parC), ov::Exception);
}

TEST_F(AnyTests, AnyRemovedRealObject) {
    ASSERT_EQ(0, DestructorTest::constructorCount);
    ASSERT_EQ(0, DestructorTest::destructorCount);
    {
        DestructorTest t;
        Any p1 = t;
    }
    ASSERT_EQ(2, DestructorTest::constructorCount);
    ASSERT_EQ(2, DestructorTest::destructorCount);
}

TEST_F(AnyTests, AnyRemovedRealObjectWithDuplication) {
    ASSERT_EQ(0, DestructorTest::constructorCount);
    ASSERT_EQ(0, DestructorTest::destructorCount);
    {
        DestructorTest t;
        Any p = t;
        ASSERT_EQ(0, DestructorTest::destructorCount);
        p = t;
        ASSERT_EQ(1, DestructorTest::destructorCount);
    }
    ASSERT_EQ(3, DestructorTest::constructorCount);
    ASSERT_EQ(3, DestructorTest::destructorCount);
}

TEST_F(AnyTests, AnyRemovedRealObjectPointerWithDuplication) {
    ASSERT_EQ(0, DestructorTest::constructorCount);
    ASSERT_EQ(0, DestructorTest::destructorCount);
    {
        auto* t = new DestructorTest();
        Any p = t;
        ASSERT_EQ(1, DestructorTest::constructorCount);
        ASSERT_EQ(0, DestructorTest::destructorCount);
        p = t;
        ASSERT_TRUE(p.is<DestructorTest*>());
        DestructorTest* t2 = p.as<DestructorTest*>();
        ASSERT_EQ(0, DestructorTest::destructorCount);
        delete t;
        auto* t3 = p.as<DestructorTest*>();
        ASSERT_EQ(t2, t3);
    }
    ASSERT_EQ(1, DestructorTest::constructorCount);
    ASSERT_EQ(1, DestructorTest::destructorCount);
}

void PrintTo(const Any& object, std::ostream* stream) {
    if (object.empty() || !stream) {
        return;
    }
    object.print(*stream);
}

TEST_F(AnyTests, PrintToEmpty) {
    Any p;
    std::stringstream stream;
    ASSERT_NO_THROW(p.print(stream));
    ASSERT_EQ(stream.str(), std::string{});
}

TEST_F(AnyTests, PrintToIntAny) {
    int value = -5;
    Any p = value;
    std::stringstream stream;
    ASSERT_NO_THROW(p.print(stream));
    ASSERT_EQ(stream.str(), std::to_string(value));
}

TEST_F(AnyTests, ReadToIntAny) {
    int value = -5;
    std::stringstream strm;
    strm << value;
    Any p = int{};
    ASSERT_NO_THROW(p.read(strm));
    ASSERT_FALSE(strm.fail());
    ASSERT_EQ(value, p.as<int>());
}

TEST_F(AnyTests, PrintToUIntAny) {
    unsigned int value = 5;
    Any p = value;
    std::stringstream stream;
    ASSERT_NO_THROW(p.print(stream));
    ASSERT_EQ(stream.str(), std::to_string(value));
}

TEST_F(AnyTests, PrintToSize_tAny) {
    std::size_t value = 5;
    Any p = value;
    std::stringstream stream;
    ASSERT_NO_THROW(p.print(stream));
    ASSERT_EQ(stream.str(), std::to_string(value));
}

TEST_F(AnyTests, PrintToFloatAny) {
    Any p = 5.5f;
    std::stringstream stream;
    ASSERT_NO_THROW(p.print(stream));
    ASSERT_EQ(stream.str(), std::string{"5.5"});
}

TEST_F(AnyTests, PrintToStringAny) {
    std::string value = "some text";
    Any p = value;
    std::stringstream stream;
    ASSERT_NO_THROW(p.print(stream));
    ASSERT_EQ(stream.str(), value);
}

TEST_F(AnyTests, PrintToVectorOfInts) {
    Any p = std::vector<int>{-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5};
    std::stringstream stream;
    ASSERT_NO_THROW(p.print(stream));
    ASSERT_EQ(stream.str(), std::string{"-5 -4 -3 -2 -1 0 1 2 3 4 5"});
}

TEST_F(AnyTests, PrintToVectorOfUInts) {
    Any p = std::vector<unsigned int>{0, 1, 2, 3, 4, 5};
    std::stringstream stream;
    ASSERT_NO_THROW(p.print(stream));
    ASSERT_EQ(stream.str(), std::string{"0 1 2 3 4 5"});
}

TEST_F(AnyTests, PrintToVectorOfFloats) {
    auto ref_vec = std::vector<float>{0.0f, 1.1f, 2.2f, 3.3f, 4.4f, 5.5f};
    {
        Any p = std::vector<float>{0.0f, 1.1f, 2.2f, 3.3f, 4.4f, 5.5f};
        ASSERT_EQ(p.as<std::string>(), std::string{"0 1.1 2.2 3.3 4.4 5.5"});
    }

    {
        Any p = "0 1.1 2.2 3.3 4.4 5.5";
        ASSERT_EQ((p.as<std::vector<float>>()), ref_vec);
    }
}

TEST_F(AnyTests, PrintToVectorOfStrings) {
    Any p = std::vector<std::string>{"zero", "one", "two", "three", "four", "five"};
    std::stringstream stream;
    ASSERT_NO_THROW(p.print(stream));
    ASSERT_EQ(stream.str(), std::string{"zero one two three four five"});
}

TEST_F(AnyTests, PrintToMapOfAnys) {
    std::map<std::string, Any> refMap;
    refMap["testParamInt"] = 4;
    refMap["testParamString"] = "test";
    std::stringstream stream;
    {
        Any p = refMap;
        ASSERT_NO_THROW(p.print(stream));
        ASSERT_EQ(stream.str(), std::string{"testParamInt 4 testParamString test"});
    }
}

TEST_F(AnyTests, constructFromVariantImpl) {
    OPENVINO_SUPPRESS_DEPRECATED_START
    auto parameter = Any{4};
    auto get_impl = [&] {
        return std::make_shared<ngraph::VariantImpl<int>>();
    };
    auto other_parameter = Any{get_impl()};
    OPENVINO_SUPPRESS_DEPRECATED_END
}

TEST_F(AnyTests, dynamicPointerCastToVariantWrapper) {
    OPENVINO_SUPPRESS_DEPRECATED_START
    Any p = std::make_shared<ngraph::VariantWrapper<std::string>>("42");
    auto str_variant = std::dynamic_pointer_cast<ngraph::VariantWrapper<std::string>>(p);
    ASSERT_EQ("42", str_variant->get());
    OPENVINO_SUPPRESS_DEPRECATED_END
}

TEST_F(AnyTests, asTypePtrToVariantWrapper) {
    OPENVINO_SUPPRESS_DEPRECATED_START
    Any p = std::make_shared<ngraph::VariantWrapper<std::string>>("42");
    auto str_variant = ov::as_type_ptr<ngraph::VariantWrapper<std::string>>(p);
    ASSERT_EQ("42", str_variant->get());
    OPENVINO_SUPPRESS_DEPRECATED_END
}

TEST_F(AnyTests, castToVariantWrapper) {
    {
        OPENVINO_SUPPRESS_DEPRECATED_START
        Any p = std::make_shared<ngraph::VariantWrapper<std::string>>("42");
        std::shared_ptr<ngraph::VariantWrapper<std::string>> str_variant = p;
        ASSERT_EQ("42", str_variant->get());
        OPENVINO_SUPPRESS_DEPRECATED_END
    }
    {
        OPENVINO_SUPPRESS_DEPRECATED_START
        Any p = std::make_shared<ngraph::VariantWrapper<std::string>>("42");
        auto f = [](const std::shared_ptr<ngraph::VariantWrapper<std::string>>& str_variant) {
            ASSERT_EQ("42", str_variant->get());
        };
        f(p);
        OPENVINO_SUPPRESS_DEPRECATED_END
    }
    {
        OPENVINO_SUPPRESS_DEPRECATED_START
        Any p = std::make_shared<ngraph::VariantWrapper<std::string>>("42");
        auto f = [](std::shared_ptr<ngraph::VariantWrapper<std::string>>& str_variant) {
            ASSERT_EQ("42", str_variant->get());
        };
        f(p);
        OPENVINO_SUPPRESS_DEPRECATED_END
    }
    {
        OPENVINO_SUPPRESS_DEPRECATED_START
        std::shared_ptr<RuntimeAttribute> v = std::make_shared<ngraph::VariantWrapper<std::string>>("42");
        Any p = v;
        auto f = [](std::shared_ptr<ngraph::VariantWrapper<std::string>>& str_variant) {
            ASSERT_NE(nullptr, str_variant);
            ASSERT_EQ("42", str_variant->get());
        };
        f(p);
        OPENVINO_SUPPRESS_DEPRECATED_END
    }
}

TEST_F(AnyTests, accessUsingBaseReference) {
    ASSERT_EQ(0, DestructorTest::constructorCount);
    ASSERT_EQ(0, DestructorTest::destructorCount);
    {
        using Base = DestructorTest;
        auto p = Any::make<Derived>();
        ASSERT_TRUE(p.is<Derived>());
        ASSERT_TRUE(p.is<Base>());
        ASSERT_NO_THROW(p.as<Derived>());
        ASSERT_NO_THROW(p.as<Base>());
        ASSERT_EQ(typeid(Derived), p.as<Base>().type_info());
    }
    ASSERT_EQ(1, DestructorTest::constructorCount);
    ASSERT_EQ(1, DestructorTest::destructorCount);
}

struct WrongBase {
    // No Base member type defined
    // Should be: using Base = std::tuple<WrongBase>;
};

struct WrongDerived : public WrongBase {
    // No Base member type defined
    // Should be: using Base = std::tuple<WrongBase>;
};

TEST_F(AnyTests, accessUsingWrongBaseReference) {
    Any p = WrongDerived{};
    ASSERT_TRUE(p.is<WrongDerived>());
    ASSERT_FALSE(p.is<WrongBase>());
    ASSERT_NO_THROW(p.as<WrongDerived>());
    ASSERT_THROW(p.as<WrongBase>(), ov::Exception);
}

TEST_F(AnyTests, ToString) {
    Any p = 42;
    ASSERT_TRUE(p.is<int>());
    ASSERT_EQ("42", p.as<std::string>());
}

TEST_F(AnyTests, FromString) {
    Any p = "42";
    ASSERT_TRUE(p.is<std::string>());
    ASSERT_EQ(42, p.as<int>());
}

TEST_F(AnyTests, BoolFromString) {
    {
        Any p = "YES";
        ASSERT_TRUE(p.is<std::string>());
        ASSERT_EQ(true, p.as<bool>());
    }
    {
        Any p = "NO";
        ASSERT_TRUE(p.is<std::string>());
        ASSERT_EQ(false, p.as<bool>());
    }
    {
        Any p = "Some";
        ASSERT_TRUE(p.is<std::string>());
        ASSERT_THROW(p.as<bool>(), ov::Exception);
    }
}

TEST_F(AnyTests, NotIntFromStringThrow) {
    Any p = "not42";
    ASSERT_TRUE(p.is<std::string>());
    ASSERT_THROW(p.as<int>(), ov::Exception);
}

TEST_F(AnyTests, AddressofNoThrow) {
    Any p;
    ASSERT_EQ(nullptr, p.addressof());
    p = 42;
    ASSERT_NE(nullptr, p.addressof());
}