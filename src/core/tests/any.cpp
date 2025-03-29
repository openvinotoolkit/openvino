// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/any.hpp"

#include <gtest/gtest.h>

#include <string>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/core/runtime_attribute.hpp"

namespace ov {
namespace test {

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

    virtual ~DestructorTest() {
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

TEST_F(AnyTests, AnyAsSetOfAnys) {
    std::set<std::string> refSet0;
    std::set<int> refSet1;
    refSet0.insert("test");
    refSet1.insert(4);
    Any s0 = refSet0;
    Any s1 = refSet1;
    bool isSet0 = s0.is<std::set<std::string>>();
    bool isSet1 = s1.is<std::set<int>>();
    ASSERT_TRUE(isSet0);
    ASSERT_TRUE(isSet1);
    auto testSet0 = s0.as<std::set<std::string>>();
    auto testSet1 = s1.as<std::set<int>>();
    ASSERT_NE(testSet0.count("test"), 0);
    ASSERT_NE(testSet1.count(4), 0);
}

TEST_F(AnyTests, AnyAsMapOfMapOfAnys) {
    std::map<std::string, Any> refMap1;
    refMap1["testParamInt"] = 4;
    refMap1["testParamString"] = "test";

    std::map<std::string, Any> refMap2;
    refMap2["testParamInt"] = 5;
    refMap2["testParamString"] = "test2";

    std::map<std::string, Any> refMap;
    refMap["refMap1"] = refMap1;
    refMap["refMap2"] = refMap2;

    Any p = refMap;
    bool isMap = p.is<std::map<std::string, Any>>();
    ASSERT_TRUE(isMap);
    auto testMap = p.as<std::map<std::string, Any>>();

    ASSERT_NE(testMap.find("refMap1"), testMap.end());
    auto testMap1 = testMap.at("refMap1").as<std::map<std::string, Any>>();
    ASSERT_NE(testMap1.find("testParamInt"), testMap1.end());
    ASSERT_NE(testMap1.find("testParamString"), testMap1.end());

    int testInt1 = testMap1["testParamInt"].as<int>();
    std::string testString1 = testMap1["testParamString"].as<std::string>();

    ASSERT_EQ(refMap1["testParamInt"].as<int>(), testInt1);
    ASSERT_EQ(refMap1["testParamString"].as<std::string>(), testString1);

    ASSERT_NE(testMap.find("refMap2"), testMap.end());
    auto testMap2 = testMap.at("refMap2").as<std::map<std::string, Any>>();
    ASSERT_NE(testMap2.find("testParamInt"), testMap2.end());
    ASSERT_NE(testMap2.find("testParamString"), testMap2.end());

    int testInt2 = testMap2["testParamInt"].as<int>();
    std::string testString2 = testMap2["testParamString"].as<std::string>();

    ASSERT_EQ(refMap2["testParamInt"].as<int>(), testInt2);
    ASSERT_EQ(refMap2["testParamString"].as<std::string>(), testString2);
}

TEST_F(AnyTests, AnyAsMapOfMapOfAnysFromString) {
    const std::string string_props = "{map1:{prop1:1,prop2:2.0},map2:{prop1:value}}";
    ov::Any any(string_props);

    ov::AnyMap map;
    ASSERT_TRUE(any.is<std::string>());
    ASSERT_FALSE(any.is<ov::AnyMap>());
    OV_ASSERT_NO_THROW(map = any.as<ov::AnyMap>());
    ASSERT_EQ(string_props, ov::Any(map).as<std::string>());

    // check map1
    using MapStrDouble = std::map<std::string, double>;
    MapStrDouble map1;
    ASSERT_TRUE(map["map1"].is<std::string>());
    ASSERT_FALSE(map["map1"].is<ov::AnyMap>());
    ASSERT_FALSE(map["map1"].is<MapStrDouble>());
    OV_ASSERT_NO_THROW(map1 = map["map1"].as<MapStrDouble>());
    ASSERT_EQ(2, map1.size());

    // check map1:prop1
    ASSERT_EQ(1.0, map1["prop1"]);
    // check map1:prop2
    ASSERT_EQ(2.0, map1["prop2"]);

    // check map2
    ov::AnyMap map2;
    ASSERT_TRUE(map["map2"].is<std::string>());
    ASSERT_FALSE(map["map2"].is<ov::AnyMap>());
    OV_ASSERT_NO_THROW(map2 = map["map2"].as<ov::AnyMap>());
    ASSERT_EQ(1, map2.size());

    // check map1:prop1
    ASSERT_TRUE(map2["prop1"].is<std::string>());
    ASSERT_FALSE(map2["prop1"].is<int>());
    ASSERT_EQ("value", map2["prop1"].as<std::string>());
}

TEST_F(AnyTests, AnyAsMapOfMapOfMapOfAnysFromString) {
    const std::string string_props = "{map1:{subprop_map:{prop:value}},prop1:1,prop2:2.0}";
    ov::Any any(string_props);

    ov::AnyMap map;
    ASSERT_TRUE(any.is<std::string>());
    ASSERT_FALSE(any.is<ov::AnyMap>());
    OV_ASSERT_NO_THROW(map = any.as<ov::AnyMap>());
    ASSERT_EQ(3, map.size());
    ASSERT_EQ(string_props, ov::Any(map).as<std::string>());

    // check prop1
    ASSERT_TRUE(map["prop1"].is<std::string>());
    ASSERT_FALSE(map["prop1"].is<int>());
    ASSERT_EQ("1", map["prop1"].as<std::string>());
    ASSERT_EQ(1, map["prop1"].as<int>());

    // check prop2
    ASSERT_TRUE(map["prop2"].is<std::string>());
    ASSERT_FALSE(map["prop2"].is<int>());
    ASSERT_FALSE(map["prop2"].is<double>());
    ASSERT_EQ("2.0", map["prop2"].as<std::string>());
    ASSERT_EQ(2, map["prop2"].as<int>());
    ASSERT_EQ(2.0, map["prop2"].as<double>());

    // check map1
    ov::AnyMap map1;
    ASSERT_TRUE(map["map1"].is<std::string>());
    ASSERT_FALSE(map["map1"].is<ov::AnyMap>());
    OV_ASSERT_NO_THROW(map1 = map["map1"].as<ov::AnyMap>());

    // check subprop
    ov::AnyMap subprop_map;
    ASSERT_TRUE(map1["subprop_map"].is<std::string>());
    ASSERT_FALSE(map1["subprop_map"].is<ov::AnyMap>());
    OV_ASSERT_NO_THROW(subprop_map = map1["subprop_map"].as<ov::AnyMap>());

    // check prop
    ASSERT_TRUE(subprop_map["prop"].is<std::string>());
    ASSERT_FALSE(subprop_map["prop"].is<ov::AnyMap>());
    ASSERT_EQ("value", subprop_map["prop"].as<std::string>());
}

TEST_F(AnyTests, AnyDoesNotShareValues) {
    // simple types
    {
        Any a = 1;
        Any b = a;
        a = 2;
        ASSERT_EQ(1, b.as<int>());
        ASSERT_EQ(2, a.as<int>());
        b = 3;
        ASSERT_EQ(2, a.as<int>());
        ASSERT_EQ(3, b.as<int>());
    }

    // AnyMap's
    {
        AnyMap map{
            {"1", ov::Any(1)},
            {"2", ov::Any(2)},
        };

        Any a = map;

        // check initial state
        ASSERT_EQ(1, a.as<AnyMap>()["1"].as<int>());
        ASSERT_EQ(2, a.as<AnyMap>()["2"].as<int>());

        map["1"] = 3;                                 // change map
        ASSERT_EQ(1, a.as<AnyMap>()["1"].as<int>());  // Any is not changed

        a.as<AnyMap>()["2"] = 4;           // change Any
        ASSERT_EQ(2, map["2"].as<int>());  // map is not changed

        // erase from Any's map
        AnyMap from_any_map = a.as<AnyMap>();
        from_any_map.erase(from_any_map.begin());
        ASSERT_EQ(2, map.size());

        // erase from map
        map.erase(map.find("2"));
        ASSERT_NE(from_any_map.end(), from_any_map.find("2"));
        ASSERT_EQ(4, a.as<AnyMap>()["2"].as<int>());
    }
}

TEST_F(AnyTests, AnyMapSharesValues) {
    AnyMap map{
        {"1", 1},
        {"2", 2},
    };

    AnyMap copy_map = map;

    // check initial state
    ASSERT_EQ(1, copy_map["1"].as<int>());
    ASSERT_EQ(2, copy_map["2"].as<int>());

    // change map
    map["1"].as<int>() = 110;

    // check copied state
    EXPECT_EQ(110, map["1"].as<int>());
    EXPECT_EQ(1, copy_map["1"].as<int>());
}

TEST_F(AnyTests, AnyMapSharesComplexValues) {
    const std::string string_props = "{map1:{subprop_map:{prop:value}},prop1:1,prop2:2.0}";
    ov::Any any(string_props);
    ov::AnyMap map;
    OV_ASSERT_NO_THROW(map = any.as<ov::AnyMap>());

    AnyMap copy_map = map;

    // check initial state
    ASSERT_EQ(1, copy_map["prop1"].as<int>());

    // change map
    map["prop1"].as<std::string>() = "110";

    // check original and copied state
    EXPECT_EQ("110", map["prop1"].as<std::string>());
    EXPECT_EQ("1", copy_map["prop1"].as<std::string>());
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

        int get_test() {
            return test;
        }
        int* get_test_ptr() {
            return testPtr;
        }

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

void PrintTo(const Any& object, std::ostream* stream);
void PrintTo(const Any& object, std::ostream* stream) {
    if (object.empty() || !stream) {
        return;
    }
    object.print(*stream);
}

TEST_F(AnyTests, PrintToEmpty) {
    Any p;
    std::stringstream stream;
    OV_ASSERT_NO_THROW(p.print(stream));
    ASSERT_EQ(stream.str(), std::string{});
}

TEST_F(AnyTests, PrintToIntAny) {
    int value = -5;
    Any p = value;
    std::stringstream stream;
    OV_ASSERT_NO_THROW(p.print(stream));
    ASSERT_EQ(stream.str(), std::to_string(value));
}

TEST_F(AnyTests, ReadToIntAny) {
    int value = -5;
    std::stringstream strm;
    strm << value;
    Any p = int{};
    OV_ASSERT_NO_THROW(p.read(strm));
    ASSERT_FALSE(strm.fail());
    ASSERT_EQ(value, p.as<int>());
}

TEST_F(AnyTests, PrintToUIntAny) {
    unsigned int value = 5;
    Any p = value;
    std::stringstream stream;
    OV_ASSERT_NO_THROW(p.print(stream));
    ASSERT_EQ(stream.str(), std::to_string(value));
}

TEST_F(AnyTests, PrintToSize_tAny) {
    std::size_t value = 5;
    Any p = value;
    std::stringstream stream;
    OV_ASSERT_NO_THROW(p.print(stream));
    ASSERT_EQ(stream.str(), std::to_string(value));
}

TEST_F(AnyTests, PrintToFloatAny) {
    Any p = 5.5f;
    std::stringstream stream;
    OV_ASSERT_NO_THROW(p.print(stream));
    ASSERT_EQ(stream.str(), std::string{"5.5"});
}

TEST_F(AnyTests, PrintToStringAny) {
    std::string value = "some text";
    Any p = value;
    std::stringstream stream;
    OV_ASSERT_NO_THROW(p.print(stream));
    ASSERT_EQ(stream.str(), value);
}

TEST_F(AnyTests, PrintToVectorOfInts) {
    Any p = std::vector<int>{-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5};
    std::stringstream stream;
    OV_ASSERT_NO_THROW(p.print(stream));
    ASSERT_EQ(stream.str(), std::string{"-5 -4 -3 -2 -1 0 1 2 3 4 5"});
}

TEST_F(AnyTests, PrintToVectorOfUInts) {
    Any p = std::vector<unsigned int>{0, 1, 2, 3, 4, 5};
    std::stringstream stream;
    OV_ASSERT_NO_THROW(p.print(stream));
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
    OV_ASSERT_NO_THROW(p.print(stream));
    ASSERT_EQ(stream.str(), std::string{"zero one two three four five"});
}

TEST_F(AnyTests, PrintToMapOfAnys) {
    std::map<std::string, Any> refMap;
    refMap["testParamInt"] = 4;
    refMap["testParamString"] = "test";
    std::stringstream stream;
    {
        Any p = refMap;
        OV_ASSERT_NO_THROW(p.print(stream));
        ASSERT_EQ(stream.str(), std::string{"{testParamInt:4,testParamString:test}"});
    }
}

TEST_F(AnyTests, PrintToMapOfMapsOfAnys) {
    std::map<std::string, Any> refMap1;
    refMap1["testParamInt"] = 4;
    refMap1["testParamString"] = "test";

    std::map<std::string, Any> refMap2;
    refMap2["testParamInt"] = 5;
    refMap2["testParamString"] = "test2";

    std::map<std::string, Any> refMap;
    refMap["refMap1"] = refMap1;
    refMap["refMap2"] = refMap2;

    std::stringstream stream;
    {
        Any p = refMap;
        OV_ASSERT_NO_THROW(p.print(stream));
        ASSERT_EQ(
            stream.str(),
            std::string{
                "{refMap1:{testParamInt:4,testParamString:test},refMap2:{testParamInt:5,testParamString:test2}}"});
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
        OV_ASSERT_NO_THROW(p.as<Derived>());
        OV_ASSERT_NO_THROW(p.as<Base>());
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
    OV_ASSERT_NO_THROW(p.as<WrongDerived>());
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

TEST_F(AnyTests, EmptyStringAsAny) {
    Any p = "";
    std::vector<float> ref_f;
    std::vector<int> ref_i;
    ASSERT_TRUE(p.is<std::string>());
    ASSERT_EQ(p.as<int>(), 0);
    ASSERT_EQ(p.as<std::vector<float>>(), ref_f);
    ASSERT_EQ(p.as<std::vector<int>>(), ref_i);
}

template <class T>
class AnyConversionTest : public AnyTests {};

TYPED_TEST_SUITE_P(AnyConversionTest);

using AnyArithmeticTypes = ::testing::Types<char,
                                            signed char,
                                            short,
                                            int,
                                            long,
                                            long long,
                                            unsigned char,
                                            unsigned short,
                                            unsigned int,
                                            unsigned long,
                                            unsigned long long,
                                            float,
                                            double>;

TYPED_TEST_P(AnyConversionTest, AnyToOtherValue) {
    const TypeParam test_value{static_cast<TypeParam>(23.15f)};
    const auto a = Any{test_value};

    EXPECT_EQ(a.as<int8_t>(), static_cast<int8_t>(test_value));
    EXPECT_EQ(a.as<int16_t>(), static_cast<int16_t>(test_value));
    EXPECT_EQ(a.as<int32_t>(), static_cast<int32_t>(test_value));
    EXPECT_EQ(a.as<int64_t>(), static_cast<int64_t>(test_value));

    EXPECT_EQ(a.as<uint8_t>(), static_cast<uint8_t>(test_value));
    EXPECT_EQ(a.as<uint16_t>(), static_cast<uint16_t>(test_value));
    EXPECT_EQ(a.as<uint32_t>(), static_cast<uint32_t>(test_value));
    EXPECT_EQ(a.as<uint64_t>(), static_cast<uint64_t>(test_value));
    EXPECT_EQ(a.as<size_t>(), static_cast<size_t>(test_value));

    EXPECT_EQ(a.as<float>(), static_cast<float>(test_value));
    EXPECT_EQ(a.as<double>(), static_cast<double>(test_value));
}

REGISTER_TYPED_TEST_SUITE_P(AnyConversionTest, AnyToOtherValue);
INSTANTIATE_TYPED_TEST_SUITE_P(InstantiationName, AnyConversionTest, AnyArithmeticTypes);

TEST_F(AnyTests, AnyAsOtherTypeIsIncosisoinet) {
    // To show member `as` current behaviour.
    // Maybe there should be two members `as` which return value
    // and `cast` returns reference if casted type is same as Any underlying type
    auto a = Any{10};

    auto& a_int = a.as<int>();
    auto& a_str = a.as<std::string>();

    EXPECT_EQ(a_int, 10);
    EXPECT_EQ(a_str, "10");

    a_int = 15;
    EXPECT_EQ(a_int, 15);
    // as string ref still has old value
    EXPECT_EQ(a_str, "10");

    a_str = "30";
    EXPECT_EQ(a_int, 15);
    // as string ref has new value but is not in sync what any contains.
    EXPECT_EQ(a_str, "30");
}

}  // namespace test
}  // namespace ov
