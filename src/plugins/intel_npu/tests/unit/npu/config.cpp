// Copyright (C) 2018-2026 Intel Corporation
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
    OPENVINO_THROW("Cannot parse DummyTestOption: ", val);
}

namespace {

using ConfigUnitTests = ::testing::Test;

using namespace ov::unit_test::intel_npu;

static constexpr std::string_view hardcodedTestValue3 = "TESTVALUE_3";
static constexpr std::string_view expectedParseErrorMessage = "Cannot parse DummyTestOption: TESTVALUE_3";

TEST_F(ConfigUnitTests, DefaultOptionValueCheckerWorks) {
    intel_npu::OptionsDesc options;
    options.add<DUMMY_TEST_OPTION>();
    options.add<DUMMY_TEST_OPTION_CUSTOM_CLASS_CHECKER>();

    ASSERT_EQ(true,
              options.get(dummy_test_option.name())
                  .isValueSupported(DUMMY_TEST_OPTION::toString(
                      DummyTestOption::TESTVALUE_1)));  // parsable values will be returned as supported
}

TEST_F(ConfigUnitTests, OptionImplementationValueCheckerWorks) {
    intel_npu::OptionsDesc options;
    options.add<DUMMY_TEST_OPTION>();
    options.add<DUMMY_TEST_OPTION_CUSTOM_CLASS_CHECKER>();

    // make sure TESTVALUE_3 cannot be parsed by DUMMY_TEST_OPTION
    OV_EXPECT_THROW_HAS_SUBSTRING(DUMMY_TEST_OPTION::parse(hardcodedTestValue3),
                                  ov::Exception,
                                  expectedParseErrorMessage.data());

    ASSERT_EQ(false,
              options.get(dummy_test_option.name())
                  .isValueSupported(hardcodedTestValue3));  // unparsable values will not be returned as supported

    // logic changes for a class that implements its own `isValueSupported` method
    ASSERT_EQ(true, options.get(dummy_test_option_custom_class_checker.name()).isValueSupported(hardcodedTestValue3));
}

TEST_F(ConfigUnitTests, CustomValueCheckerWorks) {
    intel_npu::OptionsDesc options;
    options.add<DUMMY_TEST_OPTION>(
        [](std::string_view /* unusedVal */) {  // custom value checker is given on option registration
            return true;
        });
    options.add<DUMMY_TEST_OPTION_CUSTOM_CLASS_CHECKER>([](std::string_view /* unusedVal */) {
        return false;
    });

    // make sure TESTVALUE_3 cannot be parsed by DUMMY_TEST_OPTION
    OV_EXPECT_THROW_HAS_SUBSTRING(DUMMY_TEST_OPTION::parse(hardcodedTestValue3),
                                  ov::Exception,
                                  expectedParseErrorMessage.data());

    ASSERT_EQ(true,
              options.get(dummy_test_option.name())
                  .isValueSupported(hardcodedTestValue3));  // even if TESTVALUE_3 cannot be parsed,
                                                            // custom checker should be prioritized

    // same prioritization expectation for custom value checker even if class implements self `isValueSupported` method
    // which returns `true`
    ASSERT_EQ(false,
              options.get(dummy_test_option_custom_class_checker.name())
                  .isValueSupported(DUMMY_TEST_OPTION_CUSTOM_CLASS_CHECKER::toString(DummyTestOption::TESTVALUE_2)));
}

}  // namespace
