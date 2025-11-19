// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/config/config.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"

namespace ov {

namespace unit_test {

namespace intel_npu {

enum class DummyTestOption {
    TESTVALUE_0 = 0,
    TESTVALUE_1 = 1,
    TESTVALUE_2 = 2,
    TESTVALUE_3 = 3  // not parsed
};

std::string_view stringifyEnum(DummyTestOption dummyTestOption);

std::string_view stringifyEnum(DummyTestOption dummyTestOption) {
    switch (dummyTestOption) {
    case DummyTestOption::TESTVALUE_0:
        return "TESTVALUE_0";
    case DummyTestOption::TESTVALUE_1:
        return "TESTVALUE_1";
    case DummyTestOption::TESTVALUE_2:
        return "TESTVALUE_2";
    default:
        OPENVINO_THROW("Cannot stringify DummyTestOption!");
    }
}

static constexpr ov::Property<DummyTestOption, PropertyMutability::RO> dummy_test_option{"DUMMY_TEST_OPTION"};

struct DUMMY_TEST_OPTION final : ::intel_npu::OptionBase<DUMMY_TEST_OPTION, DummyTestOption> {
    static std::string_view key() {
        return dummy_test_option.name();
    }

    static constexpr std::string_view getTypeName() {
        return "ov::unit_test_intel_npu::DummyTestOption";
    }

    static DummyTestOption defaultValue() {
        return DummyTestOption::TESTVALUE_0;
    }
};

static constexpr ov::Property<DummyTestOption, PropertyMutability::RO> dummy_test_option_custom_class_checker{
    "DUMMY_TEST_OPTION_CUSTOM_CLASS_CHECKER"};

struct DUMMY_TEST_OPTION_CUSTOM_CLASS_CHECKER final
    : ::intel_npu::OptionBase<DUMMY_TEST_OPTION_CUSTOM_CLASS_CHECKER, DummyTestOption> {
    static std::string_view key() {
        return dummy_test_option_custom_class_checker.name();
    }

    static constexpr std::string_view getTypeName() {
        return "ov::unit_test_intel_npu::DummyTestOption";
    }

    static DummyTestOption defaultValue() {
        return DummyTestOption::TESTVALUE_0;
    }

    static bool isValueSupported(std::string_view /* unusedVal */) {  // custom class checker is here!
        return true;
    }
};

}  // namespace intel_npu

}  // namespace unit_test

}  // namespace ov

template <>
struct intel_npu::OptionParser<ov::unit_test::intel_npu::DummyTestOption> final {
    static ov::unit_test::intel_npu::DummyTestOption parse(std::string_view val);
};

ov::unit_test::intel_npu::DummyTestOption intel_npu::OptionParser<ov::unit_test::intel_npu::DummyTestOption>::parse(
    std::string_view val) {
    if (val == "TESTVALUE_0") {
        return ov::unit_test::intel_npu::DummyTestOption::TESTVALUE_0;
    } else if (val == "TESTVALUE_1") {
        return ov::unit_test::intel_npu::DummyTestOption::TESTVALUE_1;
    } else if (val == "TESTVALUE_2") {
        return ov::unit_test::intel_npu::DummyTestOption::TESTVALUE_2;
    }
    OPENVINO_THROW("Cannot parse DummyTestOption", val);
}

using ConfigUnitTests = ::testing::Test;

using namespace ov::unit_test::intel_npu;

TEST_F(ConfigUnitTests, DefaultOptionValueCheckerWorks) {
    intel_npu::OptionsDesc options;
    options.add<DUMMY_TEST_OPTION>();
    options.add<DUMMY_TEST_OPTION_CUSTOM_CLASS_CHECKER>();

    ASSERT_EQ(true,
              std::invoke(options.get(dummy_test_option.name()).isValueSupported,
                          options.get(dummy_test_option.name()),
                          DUMMY_TEST_OPTION::toString(
                              DummyTestOption::TESTVALUE_1)));  // parsable values will be returned as supported
    ASSERT_EQ(
        true,
        (options.get(dummy_test_option.name()).*options.get(dummy_test_option.name()).isValueSupported)(
            DUMMY_TEST_OPTION::toString(DummyTestOption::TESTVALUE_2)));  // same check as above, but using other syntax
                                                                          // for calling class member function pointer
}

TEST_F(ConfigUnitTests, OptionImplementationValueCheckerWorks) {
    intel_npu::OptionsDesc options;
    options.add<DUMMY_TEST_OPTION>();
    options.add<DUMMY_TEST_OPTION_CUSTOM_CLASS_CHECKER>();

    static constexpr std::string_view hardcodedTestValue3 = "TESTVALUE_3";

    ASSERT_EQ(false,
              std::invoke(options.get(dummy_test_option.name()).isValueSupported,
                          options.get(dummy_test_option.name()),
                          hardcodedTestValue3));  // unparsable values will not be returned as supported
    ASSERT_EQ(false,
              (options.get(dummy_test_option.name()).*options.get(dummy_test_option.name()).isValueSupported)(
                  hardcodedTestValue3));  // same check as above, but using other syntax for calling class member
                                          // function pointer

    // logic changes for a class that implements its own `isValueSupported` method
    ASSERT_EQ(true,
              std::invoke(options.get(dummy_test_option_custom_class_checker.name()).isValueSupported,
                          options.get(dummy_test_option_custom_class_checker.name()),
                          hardcodedTestValue3));
    ASSERT_EQ(true,
              (options.get(dummy_test_option_custom_class_checker.name()).*
               options.get(dummy_test_option_custom_class_checker.name()).isValueSupported)(
                  hardcodedTestValue3));  // same check as above, but using other syntax for calling class member
                                          // function pointer
}

TEST_F(ConfigUnitTests, CustomValueCheckerWorks) {
    intel_npu::OptionsDesc options;
    options.add<DUMMY_TEST_OPTION>([](std::string_view val) {  // custom value checker is given on option registration
        return false;
    });
    options.add<DUMMY_TEST_OPTION_CUSTOM_CLASS_CHECKER>([](std::string_view val) {
        return false;
    });

    ASSERT_EQ(false,
              std::invoke(
                  options.get(dummy_test_option.name()).isValueSupported,
                  options.get(dummy_test_option.name()),
                  DUMMY_TEST_OPTION::toString(DummyTestOption::TESTVALUE_0)));  // even if TESTVALUE_0 can be parsed,
                                                                                // custom checker should be prioritized
    ASSERT_EQ(
        false,
        (options.get(dummy_test_option.name()).*options.get(dummy_test_option.name()).isValueSupported)(
            DUMMY_TEST_OPTION::toString(DummyTestOption::TESTVALUE_1)));  // same check as above, but using other syntax
                                                                          // for calling class member function pointer

    // same expectations even if class implements self `isValueSupported` method
    ASSERT_EQ(false,
              std::invoke(options.get(dummy_test_option_custom_class_checker.name()).isValueSupported,
                          options.get(dummy_test_option_custom_class_checker.name()),
                          DUMMY_TEST_OPTION_CUSTOM_CLASS_CHECKER::toString(DummyTestOption::TESTVALUE_2)));
    ASSERT_EQ(false,
              (options.get(dummy_test_option_custom_class_checker.name()).*
               options.get(dummy_test_option_custom_class_checker.name()).isValueSupported)(
                  DUMMY_TEST_OPTION_CUSTOM_CLASS_CHECKER::toString(DummyTestOption::TESTVALUE_2)));
}
