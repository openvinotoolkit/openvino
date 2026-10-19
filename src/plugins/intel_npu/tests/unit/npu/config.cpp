// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/config/config.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <sstream>

#include "common_test_utils/test_assertions.hpp"
#include "intel_npu/config/options.hpp"

namespace ov {

namespace unit_test {

namespace intel_npu {

enum class DummyTestOption {
    TESTVALUE_0 = 0,
    TESTVALUE_1 = 1,
    TESTVALUE_2 = 2  // not parsed
};

static std::string_view stringifyEnum(DummyTestOption dummyTestOption) {
    switch (dummyTestOption) {
    case DummyTestOption::TESTVALUE_0:
        return "TESTVALUE_0";
    case DummyTestOption::TESTVALUE_1:
        return "TESTVALUE_1";
    default:
        OPENVINO_THROW("Cannot stringify DummyTestOption!");
    }
}

// Unscoped enums promote to their underlying type, so they are streamable even with no `operator<<` of their own
enum DummyUnscopedTestOption : uint32_t { UNSCOPED_TESTVALUE_0 = 0, UNSCOPED_TESTVALUE_1 = 1 };

static std::string_view stringifyEnum(DummyUnscopedTestOption dummyTestOption) {
    switch (dummyTestOption) {
    case UNSCOPED_TESTVALUE_0:
        return "UNSCOPED_TESTVALUE_0";
    case UNSCOPED_TESTVALUE_1:
        return "UNSCOPED_TESTVALUE_1";
    default:
        OPENVINO_THROW("Cannot stringify DummyUnscopedTestOption!");
    }
}

// An enum may end up with both a `stringifyEnum` overload and an `operator<<`, each of them printing the value
// differently. The two spellings below only differ so that the printer's choice between them can be observed.
enum class DummyStreamableTestOption { TESTVALUE_0 };

static std::string_view stringifyEnum(DummyStreamableTestOption) {
    return "FROM_STRINGIFY_ENUM";
}

inline std::ostream& operator<<(std::ostream& os, const DummyStreamableTestOption&) {
    return os << "FROM_STREAM_OPERATOR";
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
    }
    OPENVINO_THROW("Cannot parse DummyTestOption: ", val);
}

namespace {

using namespace ov::unit_test::intel_npu;

using intel_npu::Config;
using intel_npu::OptionMode;
using intel_npu::OptionParser;
using intel_npu::OptionPrinter;
using intel_npu::OptionsDesc;

// Aliases for types whose commas would otherwise break the gtest macros
using Milliseconds = std::chrono::duration<int64_t, std::milli>;
using StringToInt64Map = std::map<std::string, int64_t>;

static constexpr std::string_view hardcodedTestValue1 = "TESTVALUE_1";
static constexpr std::string_view hardcodedTestValue2 = "TESTVALUE_2";
static constexpr std::string_view expectedParseErrorMessage = "Cannot parse DummyTestOption: TESTVALUE_2";

//
// Dummy options used to exercise the `OptionsDesc` / `Config` plumbing
//

struct DUMMY_BOTH_OPTION final : intel_npu::OptionBase<DUMMY_BOTH_OPTION, std::string> {
    static std::string_view key() {
        return "DUMMY_BOTH_OPTION";
    }

    static std::string defaultValue() {
        return "both_default";
    }

    static OptionMode mode() {
        return OptionMode::Both;
    }
};

struct DUMMY_COMPILE_TIME_OPTION final : intel_npu::OptionBase<DUMMY_COMPILE_TIME_OPTION, std::string> {
    static std::string_view key() {
        return "DUMMY_COMPILE_TIME_OPTION";
    }

    static std::string defaultValue() {
        return "compile_time_default";
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }
};

struct DUMMY_RUN_TIME_OPTION final : intel_npu::OptionBase<DUMMY_RUN_TIME_OPTION, int64_t> {
    static std::string_view key() {
        return "DUMMY_RUN_TIME_OPTION";
    }

    static int64_t defaultValue() {
        return 42;
    }

    static OptionMode mode() {
        return OptionMode::RunTime;
    }
};

// Intentionally left without a `defaultValue()` override, so `OptionBase` returns `std::nullopt`
struct DUMMY_NO_DEFAULT_OPTION final : intel_npu::OptionBase<DUMMY_NO_DEFAULT_OPTION, std::string> {
    static std::string_view key() {
        return "DUMMY_NO_DEFAULT_OPTION";
    }
};

struct DUMMY_VALIDATED_OPTION final : intel_npu::OptionBase<DUMMY_VALIDATED_OPTION, int64_t> {
    static std::string_view key() {
        return "DUMMY_VALIDATED_OPTION";
    }

    static int64_t defaultValue() {
        return 0;
    }

    static void validateValue(const int64_t& val) {
        OPENVINO_ASSERT(val >= 0, "DUMMY_VALIDATED_OPTION expects non-negative values, got ", val);
    }
};

struct DUMMY_UINT32_OPTION final : intel_npu::OptionBase<DUMMY_UINT32_OPTION, uint32_t> {
    static std::string_view key() {
        return "DUMMY_UINT32_OPTION";
    }

    static uint32_t defaultValue() {
        return 1u;
    }
};

struct DUMMY_ENV_OPTION final : intel_npu::OptionBase<DUMMY_ENV_OPTION, int64_t> {
    static std::string_view key() {
        return "DUMMY_ENV_OPTION";
    }

    static std::string_view envVar() {
        return "OV_NPU_DUMMY_ENV_OPTION";
    }

    static int64_t defaultValue() {
        return 0;
    }
};

struct DUMMY_BOOL_OPTION final : intel_npu::OptionBase<DUMMY_BOOL_OPTION, bool> {
    static std::string_view key() {
        return "DUMMY_BOOL_OPTION";
    }

    static bool defaultValue() {
        return false;
    }
};

// `AUTO` is meaningless to `operator>>`, so this option can only be parsed through its own `parse()`
struct DUMMY_CUSTOM_PARSE_OPTION final : intel_npu::OptionBase<DUMMY_CUSTOM_PARSE_OPTION, int64_t> {
    static std::string_view key() {
        return "DUMMY_CUSTOM_PARSE_OPTION";
    }

    static int64_t defaultValue() {
        return 0;
    }

    static int64_t parse(std::string_view val) {
        if (val == "AUTO") {
            return -1;
        }
        return OptionParser<int64_t>::parse(val);
    }
};

struct DUMMY_DEPRECATED_KEYS_OPTION final : intel_npu::OptionBase<DUMMY_DEPRECATED_KEYS_OPTION, std::string> {
    static std::string_view key() {
        return "DUMMY_DEPRECATED_KEYS_OPTION";
    }

    static std::vector<std::string_view> deprecatedKeys() {
        return {"DUMMY_OLD_KEY"};
    }

    static std::string defaultValue() {
        return "";
    }
};

// Declares more than one deprecated spelling, and is neither a string nor free of validation, so it also
// covers parsing and validation being reached through an alias
struct DUMMY_MULTI_DEPRECATED_KEYS_OPTION final
    : intel_npu::OptionBase<DUMMY_MULTI_DEPRECATED_KEYS_OPTION, int64_t> {
    static std::string_view key() {
        return "DUMMY_MULTI_DEPRECATED_KEYS_OPTION";
    }

    static std::vector<std::string_view> deprecatedKeys() {
        return {"DUMMY_FIRST_OLD_KEY", "DUMMY_SECOND_OLD_KEY"};
    }

    static int64_t defaultValue() {
        return 0;
    }

    static void validateValue(const int64_t& val) {
        OPENVINO_ASSERT(val >= 0, "DUMMY_MULTI_DEPRECATED_KEYS_OPTION expects non-negative values, got ", val);
    }
};

// Claims a deprecated key which `DUMMY_DEPRECATED_KEYS_OPTION` already owns, used only to check that the
// clash is rejected at registration time. Never added to the shared descriptor.
struct DUMMY_CLASHING_DEPRECATED_KEY_OPTION final
    : intel_npu::OptionBase<DUMMY_CLASHING_DEPRECATED_KEY_OPTION, std::string> {
    static std::string_view key() {
        return "DUMMY_CLASHING_DEPRECATED_KEY_OPTION";
    }

    static std::vector<std::string_view> deprecatedKeys() {
        return {"DUMMY_OLD_KEY"};
    }
};

// There is neither an `operator<<` nor an `ov::util::Write` specialization for this type, so `ov::Any::print()`
// is a no-op for it and `ov::Any::as<std::string>()` hands back an empty string instead of failing
struct UnprintablePayload final {
    int value = 0;
};

//
// Helpers
//

// NB: the previous value of the variable is not saved and restored, the variable is simply cleared again on
// destruction. That is enough for the `OV_NPU_*` variables of the dummy options used here, which nothing else
// sets, but this helper must not be reused as is for a variable which may already be set in the environment.
class ScopedEnvVar final {
public:
    ScopedEnvVar(std::string name, const std::string& value) : _name(std::move(name)) {
#ifdef _WIN32
        _putenv_s(_name.c_str(), value.c_str());
#else
        setenv(_name.c_str(), value.c_str(), 1);
#endif
    }

    ScopedEnvVar(const ScopedEnvVar&) = delete;
    ScopedEnvVar& operator=(const ScopedEnvVar&) = delete;

    ~ScopedEnvVar() {
#ifdef _WIN32
        _putenv_s(_name.c_str(), "");
#else
        unsetenv(_name.c_str());
#endif
    }

private:
    std::string _name;
};

bool containsEntry(const std::string& serialized, std::string_view key, std::string_view value) {
    return serialized.find(std::string(key) + "=\"" + std::string(value) + "\"") != std::string::npos;
}

std::shared_ptr<OptionsDesc> makeDummyOptionsDesc() {
    auto desc = std::make_shared<OptionsDesc>();
    desc->add<DUMMY_BOTH_OPTION>();
    desc->add<DUMMY_COMPILE_TIME_OPTION>();
    desc->add<DUMMY_RUN_TIME_OPTION>();
    desc->add<DUMMY_NO_DEFAULT_OPTION>();
    desc->add<DUMMY_VALIDATED_OPTION>();
    desc->add<DUMMY_UINT32_OPTION>();
    desc->add<DUMMY_ENV_OPTION>();
    desc->add<DUMMY_BOOL_OPTION>();
    desc->add<DUMMY_CUSTOM_PARSE_OPTION>();
    desc->add<DUMMY_DEPRECATED_KEYS_OPTION>();
    desc->add<DUMMY_MULTI_DEPRECATED_KEYS_OPTION>();
    return desc;
}

//
// OptionParser
//

using OptionParserUnitTests = ::testing::Test;

TEST_F(OptionParserUnitTests, BoolParserAcceptsAllSupportedSpellings) {
    for (const auto& val : {"YES", "yes", "TRUE", "True", "ON", "on", "1"}) {
        EXPECT_TRUE(OptionParser<bool>::parse(val)) << "value: " << val;
    }
    for (const auto& val : {"NO", "no", "FALSE", "False", "OFF", "off", "0"}) {
        EXPECT_FALSE(OptionParser<bool>::parse(val)) << "value: " << val;
    }
}

TEST_F(OptionParserUnitTests, BoolParserRejectsUnknownValue) {
    OV_EXPECT_THROW_HAS_SUBSTRING(OptionParser<bool>::parse("MAYBE"), ov::Exception, "is not a valid BOOL option");
}

TEST_F(OptionParserUnitTests, Int32ParserAcceptsWholeRange) {
    EXPECT_EQ(std::numeric_limits<int32_t>::min(), OptionParser<int32_t>::parse("-2147483648"));
    EXPECT_EQ(std::numeric_limits<int32_t>::max(), OptionParser<int32_t>::parse("2147483647"));
}

TEST_F(OptionParserUnitTests, Int32ParserRejectsOutOfRangeAndGarbage) {
    OV_EXPECT_THROW_HAS_SUBSTRING(OptionParser<int32_t>::parse("2147483648"),
                                  ov::Exception,
                                  "is not a valid INT32 option");
    OV_EXPECT_THROW_HAS_SUBSTRING(OptionParser<int32_t>::parse("-2147483649"),
                                  ov::Exception,
                                  "is not a valid INT32 option");
    OV_EXPECT_THROW_HAS_SUBSTRING(OptionParser<int32_t>::parse("abc"), ov::Exception, "is not a valid INT32 option");
}

TEST_F(OptionParserUnitTests, UInt32ParserAcceptsWholeRange) {
    EXPECT_EQ(0u, OptionParser<uint32_t>::parse("0"));
    EXPECT_EQ(std::numeric_limits<uint32_t>::max(), OptionParser<uint32_t>::parse("4294967295"));
}

// The parser must not silently wrap negative values around, which is what `std::stoul` would do
TEST_F(OptionParserUnitTests, UInt32ParserRejectsNegativeAndOutOfRange) {
    OV_EXPECT_THROW_HAS_SUBSTRING(OptionParser<uint32_t>::parse("-1"), ov::Exception, "is not a valid UINT32 option");
    OV_EXPECT_THROW_HAS_SUBSTRING(OptionParser<uint32_t>::parse("4294967296"),
                                  ov::Exception,
                                  "is not a valid UINT32 option");
}

TEST_F(OptionParserUnitTests, Int64ParserAcceptsWholeRange) {
    EXPECT_EQ(std::numeric_limits<int64_t>::min(), OptionParser<int64_t>::parse("-9223372036854775808"));
    EXPECT_EQ(std::numeric_limits<int64_t>::max(), OptionParser<int64_t>::parse("9223372036854775807"));
}

TEST_F(OptionParserUnitTests, Int64ParserRejectsGarbage) {
    OV_EXPECT_THROW_HAS_SUBSTRING(OptionParser<int64_t>::parse("abc"), ov::Exception, "is not a valid INT64 option");
}

// The parser must not silently wrap negative values around, which is what `std::stoull` would do
TEST_F(OptionParserUnitTests, UInt64ParserRejectsNegative) {
    EXPECT_EQ(std::numeric_limits<uint64_t>::max(), OptionParser<uint64_t>::parse("18446744073709551615"));
    OV_EXPECT_THROW_HAS_SUBSTRING(OptionParser<uint64_t>::parse("-1"), ov::Exception, "is not a valid UINT64 option");
}

TEST_F(OptionParserUnitTests, DoubleParserWorks) {
    EXPECT_DOUBLE_EQ(1.5, OptionParser<double>::parse("1.5"));
    OV_EXPECT_THROW_HAS_SUBSTRING(OptionParser<double>::parse("abc"), ov::Exception, "is not a valid FP64 option");
}

// `std::sto*` stops at the first character which is not part of the number, so a valid prefix alone must
// not be enough to accept the value
TEST_F(OptionParserUnitTests, NumericParsersRejectTrailingGarbage) {
    OV_EXPECT_THROW_HAS_SUBSTRING(OptionParser<int32_t>::parse("12oops"), ov::Exception, "is not a valid INT32 option");
    OV_EXPECT_THROW_HAS_SUBSTRING(OptionParser<int32_t>::parse("12 oops"),
                                  ov::Exception,
                                  "is not a valid INT32 option");
    OV_EXPECT_THROW_HAS_SUBSTRING(OptionParser<uint32_t>::parse("12oops"),
                                  ov::Exception,
                                  "is not a valid UINT32 option");
    OV_EXPECT_THROW_HAS_SUBSTRING(OptionParser<int64_t>::parse("12oops"), ov::Exception, "is not a valid INT64 option");
    OV_EXPECT_THROW_HAS_SUBSTRING(OptionParser<uint64_t>::parse("12oops"),
                                  ov::Exception,
                                  "is not a valid UINT64 option");
    OV_EXPECT_THROW_HAS_SUBSTRING(OptionParser<double>::parse("1.5oops"), ov::Exception, "is not a valid FP64 option");
    OV_EXPECT_THROW_HAS_SUBSTRING(OptionParser<double>::parse("1.5 oops"), ov::Exception, "is not a valid FP64 option");
}

// An explicit plus sign was accepted by the `std::sto*` functions used before, so it still is
TEST_F(OptionParserUnitTests, NumericParsersAcceptAnExplicitPlusSign) {
    EXPECT_EQ(12, OptionParser<int32_t>::parse("+12"));
    EXPECT_EQ(12u, OptionParser<uint32_t>::parse("+12"));
    EXPECT_EQ(12, OptionParser<int64_t>::parse("+12"));
    EXPECT_EQ(12u, OptionParser<uint64_t>::parse("+12"));
    EXPECT_DOUBLE_EQ(1.5, OptionParser<double>::parse(" +1.5 "));
}

// Dropping the plus sign must not turn a malformed value into a well formed one
TEST_F(OptionParserUnitTests, NumericParsersRejectAMalformedSign) {
    OV_EXPECT_THROW_HAS_SUBSTRING(OptionParser<int32_t>::parse("+-1"), ov::Exception, "is not a valid INT32 option");
    OV_EXPECT_THROW_HAS_SUBSTRING(OptionParser<int64_t>::parse("+-1"), ov::Exception, "is not a valid INT64 option");
    OV_EXPECT_THROW_HAS_SUBSTRING(OptionParser<uint32_t>::parse("+-1"), ov::Exception, "is not a valid UINT32 option");
    OV_EXPECT_THROW_HAS_SUBSTRING(OptionParser<int32_t>::parse("++1"), ov::Exception, "is not a valid INT32 option");
    OV_EXPECT_THROW_HAS_SUBSTRING(OptionParser<int32_t>::parse("+"), ov::Exception, "is not a valid INT32 option");
    OV_EXPECT_THROW_HAS_SUBSTRING(OptionParser<int32_t>::parse("+ 1"), ov::Exception, "is not a valid INT32 option");
    OV_EXPECT_THROW_HAS_SUBSTRING(OptionParser<double>::parse("+-1.5"), ov::Exception, "is not a valid FP64 option");
}

// Surrounding whitespace is still tolerated, only actual content may not be left over
TEST_F(OptionParserUnitTests, NumericParsersIgnoreSurroundingWhitespace) {
    EXPECT_EQ(12, OptionParser<int32_t>::parse(" 12 "));
    EXPECT_EQ(12u, OptionParser<uint32_t>::parse(" 12 "));
    EXPECT_EQ(12, OptionParser<int64_t>::parse(" 12 "));
    EXPECT_EQ(12u, OptionParser<uint64_t>::parse(" 12 "));
    EXPECT_DOUBLE_EQ(1.5, OptionParser<double>::parse(" 1.5 "));
}

TEST_F(OptionParserUnitTests, StringParserKeepsTheWholeView) {
    const std::string_view view = "PREFIX_VALUE";
    EXPECT_EQ("PREFIX", OptionParser<std::string>::parse(view.substr(0, 6)));
}

TEST_F(OptionParserUnitTests, VectorParserSplitsOnComma) {
    const auto parsed = OptionParser<std::vector<int64_t>>::parse("1,2,3");
    EXPECT_EQ(std::vector<int64_t>({1, 2, 3}), parsed);
}

TEST_F(OptionParserUnitTests, MapParserSplitsOnCommaAndColon) {
    const StringToInt64Map expected{{"a", 1}, {"b", 2}};
    EXPECT_EQ(expected, OptionParser<StringToInt64Map>::parse("a:1,b:2"));
}

TEST_F(OptionParserUnitTests, MapParserRejectsEntryWithoutValue) {
    EXPECT_ANY_THROW(OptionParser<StringToInt64Map>::parse("a"));
}

// An empty spelling means "nothing to parse" for the string and container parsers. `Config` relies on this
// when it decides which `ov::Any` payloads may be stringified, so the behaviour is pinned in both directions.
TEST_F(OptionParserUnitTests, StringAndContainerParsersAcceptAnEmptyValue) {
    EXPECT_EQ("", OptionParser<std::string>::parse(""));
    EXPECT_TRUE(OptionParser<std::vector<int64_t>>::parse("").empty());
    EXPECT_TRUE(OptionParser<StringToInt64Map>::parse("").empty());
}

TEST_F(OptionParserUnitTests, ScalarParsersRejectAnEmptyValue) {
    OV_EXPECT_THROW_HAS_SUBSTRING(OptionParser<bool>::parse(""), ov::Exception, "is not a valid BOOL option");
    OV_EXPECT_THROW_HAS_SUBSTRING(OptionParser<int32_t>::parse(""), ov::Exception, "is not a valid INT32 option");
    OV_EXPECT_THROW_HAS_SUBSTRING(OptionParser<uint32_t>::parse(""), ov::Exception, "is not a valid UINT32 option");
    OV_EXPECT_THROW_HAS_SUBSTRING(OptionParser<int64_t>::parse(""), ov::Exception, "is not a valid INT64 option");
    OV_EXPECT_THROW_HAS_SUBSTRING(OptionParser<uint64_t>::parse(""), ov::Exception, "is not a valid UINT64 option");
    OV_EXPECT_THROW_HAS_SUBSTRING(OptionParser<double>::parse(""), ov::Exception, "is not a valid FP64 option");
    OV_EXPECT_THROW_HAS_SUBSTRING(OptionParser<Milliseconds>::parse(""), ov::Exception, "as time duration");
    EXPECT_ANY_THROW(OptionParser<ov::log::Level>::parse(""));
}

// A blank value is trimmed down to an empty one, which the numeric parsers must report instead of reading
// through the null pointer `ov::util::trim` hands back for it
TEST_F(OptionParserUnitTests, NumericParsersRejectABlankValue) {
    OV_EXPECT_THROW_HAS_SUBSTRING(OptionParser<int32_t>::parse("   "), ov::Exception, "is not a valid INT32 option");
    OV_EXPECT_THROW_HAS_SUBSTRING(OptionParser<uint32_t>::parse("   "), ov::Exception, "is not a valid UINT32 option");
    OV_EXPECT_THROW_HAS_SUBSTRING(OptionParser<int64_t>::parse("   "), ov::Exception, "is not a valid INT64 option");
    OV_EXPECT_THROW_HAS_SUBSTRING(OptionParser<uint64_t>::parse("   "), ov::Exception, "is not a valid UINT64 option");
    OV_EXPECT_THROW_HAS_SUBSTRING(OptionParser<double>::parse("   "), ov::Exception, "is not a valid FP64 option");
}

TEST_F(OptionParserUnitTests, DurationParserRejectsNegativeAndGarbage) {
    EXPECT_EQ(Milliseconds(10), OptionParser<Milliseconds>::parse("10"));
    OV_EXPECT_THROW_HAS_SUBSTRING(OptionParser<Milliseconds>::parse("-1"),
                                  ov::Exception,
                                  "is not a valid time duration");
    OV_EXPECT_THROW_HAS_SUBSTRING(OptionParser<Milliseconds>::parse("abc"), ov::Exception, "as time duration");
}

TEST_F(OptionParserUnitTests, DurationParserRejectsTrailingGarbage) {
    OV_EXPECT_THROW_HAS_SUBSTRING(OptionParser<Milliseconds>::parse("10ms"), ov::Exception, "as time duration");
    OV_EXPECT_THROW_HAS_SUBSTRING(OptionParser<Milliseconds>::parse("10 oops"), ov::Exception, "as time duration");
}

// Types providing an `operator>>` (all the `ov::` property enums do) need no parser of their own
TEST_F(OptionParserUnitTests, DefaultParserReliesOnStreamOperator) {
    EXPECT_EQ(ov::log::Level::DEBUG, OptionParser<ov::log::Level>::parse("LOG_DEBUG"));
    EXPECT_EQ(ov::hint::ExecutionMode::ACCURACY, OptionParser<ov::hint::ExecutionMode>::parse("ACCURACY"));
}

TEST_F(OptionParserUnitTests, DefaultParserPropagatesStreamOperatorErrors) {
    EXPECT_ANY_THROW(OptionParser<ov::log::Level>::parse("LOG_SOMETHING"));
    EXPECT_ANY_THROW(OptionParser<ov::hint::ExecutionMode>::parse("SOMETHING"));
}

// `operator>>` stops at the first character it can't consume, so a valid prefix alone must not be enough
TEST_F(OptionParserUnitTests, DefaultParserRejectsTrailingGarbage) {
    OV_EXPECT_THROW_HAS_SUBSTRING(OptionParser<ov::log::Level>::parse("LOG_DEBUG garbage"),
                                  ov::Exception,
                                  "is not a valid option value");
    OV_EXPECT_THROW_HAS_SUBSTRING(OptionParser<ov::hint::ExecutionMode>::parse("ACCURACY PERFORMANCE"),
                                  ov::Exception,
                                  "is not a valid option value");
}

// Surrounding whitespace is still tolerated, only actual content may not be left over
TEST_F(OptionParserUnitTests, DefaultParserIgnoresSurroundingWhitespace) {
    EXPECT_EQ(ov::log::Level::DEBUG, OptionParser<ov::log::Level>::parse(" LOG_DEBUG "));
}

//
// OptionPrinter
//

using OptionPrinterUnitTests = ::testing::Test;

TEST_F(OptionPrinterUnitTests, BooleanIsPrintedAsYesNo) {
    EXPECT_EQ("YES", OptionPrinter<bool>::toString(true));
    EXPECT_EQ("NO", OptionPrinter<bool>::toString(false));
}

TEST_F(OptionPrinterUnitTests, FloatingPointIsPrintedWithTwoDecimals) {
    EXPECT_EQ("1.50", OptionPrinter<double>::toString(1.5));
    EXPECT_EQ("1.00", OptionPrinter<double>::toString(1.0));
}

// Enums providing an `operator<<` are printed through it
TEST_F(OptionPrinterUnitTests, EnumWithStreamOperatorUsesIt) {
    EXPECT_EQ("LOG_DEBUG", OptionPrinter<ov::log::Level>::toString(ov::log::Level::DEBUG));
    EXPECT_EQ("ACCURACY", OptionPrinter<ov::hint::ExecutionMode>::toString(ov::hint::ExecutionMode::ACCURACY));
}

// Enums with no `operator<<` fall back onto the `stringifyEnum` overload
TEST_F(OptionPrinterUnitTests, EnumWithoutStreamOperatorUsesStringifyEnum) {
    EXPECT_EQ(std::string(hardcodedTestValue1), OptionPrinter<DummyTestOption>::toString(DummyTestOption::TESTVALUE_1));
    EXPECT_ANY_THROW(OptionPrinter<DummyTestOption>::toString(DummyTestOption::TESTVALUE_2));
}

// Unscoped enums are implicitly streamable through integral promotion, the `stringifyEnum` overload still wins
TEST_F(OptionPrinterUnitTests, UnscopedEnumUsesStringifyEnum) {
    EXPECT_EQ("UNSCOPED_TESTVALUE_1", OptionPrinter<DummyUnscopedTestOption>::toString(UNSCOPED_TESTVALUE_1));
}

// When an enum provides both, the `stringifyEnum` overload is the one which takes precedence
TEST_F(OptionPrinterUnitTests, StringifyEnumTakesPrecedenceOverStreamOperator) {
    EXPECT_EQ("FROM_STRINGIFY_ENUM",
              OptionPrinter<DummyStreamableTestOption>::toString(DummyStreamableTestOption::TESTVALUE_0));

    // Makes sure the expectation above is not met just because the `operator<<` is unusable to begin with
    std::stringstream stream;
    stream << DummyStreamableTestOption::TESTVALUE_0;
    EXPECT_EQ("FROM_STREAM_OPERATOR", stream.str());
}

TEST_F(OptionPrinterUnitTests, MapIsPrintedAsCommaSeparatedPairs) {
    const StringToInt64Map val{{"a", 1}, {"b", 2}};
    EXPECT_EQ("a:1,b:2", OptionPrinter<StringToInt64Map>::toString(val));
}

TEST_F(OptionPrinterUnitTests, DurationIsPrintedAsCount) {
    EXPECT_EQ("10", OptionPrinter<Milliseconds>::toString(Milliseconds(10)));
}

TEST_F(OptionPrinterUnitTests, OptionModeIsStringified) {
    EXPECT_EQ("Both", intel_npu::stringifyEnum(OptionMode::Both));
    EXPECT_EQ("CompileTime", intel_npu::stringifyEnum(OptionMode::CompileTime));
    EXPECT_EQ("RunTime", intel_npu::stringifyEnum(OptionMode::RunTime));
}

//
// OptionBase
//

using OptionBaseUnitTests = ::testing::Test;

TEST_F(OptionBaseUnitTests, DefaultValueIsOptionalUnlessOverridden) {
    EXPECT_FALSE(DUMMY_NO_DEFAULT_OPTION::defaultValue().has_value());
    EXPECT_EQ("both_default", DUMMY_BOTH_OPTION::defaultValue());
}

TEST_F(OptionBaseUnitTests, TypeNameComesFromTypePrinterForStandardTypes) {
    EXPECT_EQ("std::string", DUMMY_BOTH_OPTION::getTypeName());
    EXPECT_EQ("int64_t", DUMMY_RUN_TIME_OPTION::getTypeName());
    EXPECT_EQ("ov::unit_test_intel_npu::DummyTestOption", DUMMY_TEST_OPTION::getTypeName());
}

TEST_F(OptionBaseUnitTests, DefaultsAreBothModeAndNoEnvVar) {
    EXPECT_EQ(OptionMode::Both, DUMMY_NO_DEFAULT_OPTION::mode());
    EXPECT_TRUE(DUMMY_NO_DEFAULT_OPTION::envVar().empty());
    EXPECT_TRUE(DUMMY_NO_DEFAULT_OPTION::deprecatedKeys().empty());
}

TEST_F(OptionBaseUnitTests, DeprecatedKeysAreReportedInDeclarationOrder) {
    EXPECT_EQ(std::vector<std::string_view>({"DUMMY_OLD_KEY"}), DUMMY_DEPRECATED_KEYS_OPTION::deprecatedKeys());
    EXPECT_EQ(std::vector<std::string_view>({"DUMMY_FIRST_OLD_KEY", "DUMMY_SECOND_OLD_KEY"}),
              DUMMY_MULTI_DEPRECATED_KEYS_OPTION::deprecatedKeys());
}

//
// OptionsDesc
//

using OptionsDescUnitTests = ::testing::Test;

TEST_F(OptionsDescUnitTests, RegisteredOptionCanBeFound) {
    const auto desc = makeDummyOptionsDesc();

    EXPECT_TRUE(desc->has(DUMMY_BOTH_OPTION::key()));
    EXPECT_EQ(DUMMY_BOTH_OPTION::key(), desc->get(DUMMY_BOTH_OPTION::key()).key());
    EXPECT_EQ(OptionMode::CompileTime, desc->get(DUMMY_COMPILE_TIME_OPTION::key()).mode());
}

TEST_F(OptionsDescUnitTests, UnknownOptionIsReported) {
    const auto desc = makeDummyOptionsDesc();

    EXPECT_FALSE(desc->has("SOME_UNKNOWN_OPTION"));
    OV_EXPECT_THROW_HAS_SUBSTRING(desc->get("SOME_UNKNOWN_OPTION"), ov::Exception, "[ NOT_FOUND ]");
}

TEST_F(OptionsDescUnitTests, RegisteringTheSameOptionTwiceThrows) {
    OptionsDesc desc;
    desc.add<DUMMY_BOTH_OPTION>();

    OV_EXPECT_THROW_HAS_SUBSTRING(desc.add<DUMMY_BOTH_OPTION>(), ov::Exception, "was already registered");
}

TEST_F(OptionsDescUnitTests, DeprecatedKeyResolvesToTheActualOption) {
    const auto desc = makeDummyOptionsDesc();

    EXPECT_TRUE(desc->has("DUMMY_OLD_KEY"));
    EXPECT_EQ(DUMMY_DEPRECATED_KEYS_OPTION::key(), desc->get("DUMMY_OLD_KEY").key());
}

TEST_F(OptionsDescUnitTests, EveryDeprecatedKeyOfAnOptionResolvesToIt) {
    const auto desc = makeDummyOptionsDesc();

    for (const auto& deprecatedKey : DUMMY_MULTI_DEPRECATED_KEYS_OPTION::deprecatedKeys()) {
        EXPECT_TRUE(desc->has(deprecatedKey)) << "key: " << deprecatedKey;
        // the whole descriptor, not just the key, is the one of the actual option
        const auto opt = desc->get(deprecatedKey);
        EXPECT_EQ(DUMMY_MULTI_DEPRECATED_KEYS_OPTION::key(), opt.key()) << "key: " << deprecatedKey;
        EXPECT_EQ(DUMMY_MULTI_DEPRECATED_KEYS_OPTION::mode(), opt.mode()) << "key: " << deprecatedKey;
    }
}

TEST_F(OptionsDescUnitTests, DeprecatedKeyIsNotAnOptionOfItsOwn) {
    OptionsDesc desc;
    desc.add<DUMMY_DEPRECATED_KEYS_OPTION>();

    std::vector<std::string> visited;
    desc.walk([&](const intel_npu::details::OptionConcept& opt) {
        visited.emplace_back(opt.key());
    });

    // deprecated keys are a lookup alias only, they are never advertised as options
    EXPECT_EQ(std::vector<std::string>({std::string(DUMMY_DEPRECATED_KEYS_OPTION::key())}), visited);
}

TEST_F(OptionsDescUnitTests, RegisteringTheSameDeprecatedKeyTwiceThrows) {
    OptionsDesc desc;
    desc.add<DUMMY_DEPRECATED_KEYS_OPTION>();

    OV_EXPECT_THROW_HAS_SUBSTRING(desc.add<DUMMY_CLASHING_DEPRECATED_KEY_OPTION>(),
                                  ov::Exception,
                                  "Option 'DUMMY_OLD_KEY' was already registered");
}

TEST_F(OptionsDescUnitTests, ResetRemovesDeprecatedKeys) {
    OptionsDesc desc;
    desc.add<DUMMY_DEPRECATED_KEYS_OPTION>();
    ASSERT_TRUE(desc.has("DUMMY_OLD_KEY"));

    desc.reset();

    EXPECT_FALSE(desc.has("DUMMY_OLD_KEY"));
    OV_EXPECT_THROW_HAS_SUBSTRING(desc.get("DUMMY_OLD_KEY"), ov::Exception, "[ NOT_FOUND ]");
    // the alias table is cleared too, so the same option can be registered again afterwards
    EXPECT_NO_THROW(desc.add<DUMMY_DEPRECATED_KEYS_OPTION>());
}

TEST_F(OptionsDescUnitTests, ResetRemovesRegisteredOptions) {
    OptionsDesc desc;
    desc.add<DUMMY_BOTH_OPTION>();
    ASSERT_TRUE(desc.has(DUMMY_BOTH_OPTION::key()));

    desc.reset();

    EXPECT_FALSE(desc.has(DUMMY_BOTH_OPTION::key()));
}

TEST_F(OptionsDescUnitTests, WalkVisitsEveryRegisteredOption) {
    OptionsDesc desc;
    desc.add<DUMMY_BOTH_OPTION>();
    desc.add<DUMMY_RUN_TIME_OPTION>();

    std::vector<std::string> visited;
    desc.walk([&](const intel_npu::details::OptionConcept& opt) {
        visited.emplace_back(opt.key());
    });

    std::sort(visited.begin(), visited.end());
    EXPECT_EQ(
        std::vector<std::string>({std::string(DUMMY_BOTH_OPTION::key()), std::string(DUMMY_RUN_TIME_OPTION::key())}),
        visited);
}

//
// Config
//

class ConfigUnitTests : public ::testing::Test {
protected:
    void SetUp() override {
        _desc = makeDummyOptionsDesc();
    }

    Config makeConfig() const {
        return Config(_desc);
    }

    std::shared_ptr<OptionsDesc> _desc;
};

TEST_F(ConfigUnitTests, NullOptionsDescIsRejected) {
    const std::shared_ptr<const OptionsDesc> nullDesc;

    OV_EXPECT_THROW_HAS_SUBSTRING((void)Config(nullDesc), ov::Exception, "Got NULL OptionsDesc");
}

TEST_F(ConfigUnitTests, UnsetOptionFallsBackOntoItsDefaultValue) {
    const auto config = makeConfig();

    EXPECT_FALSE(config.has<DUMMY_BOTH_OPTION>());
    EXPECT_EQ("both_default", config.get<DUMMY_BOTH_OPTION>());
    EXPECT_EQ(42, config.get<DUMMY_RUN_TIME_OPTION>());
}

TEST_F(ConfigUnitTests, UnsetOptionWithoutDefaultValueThrowsOnAccess) {
    const auto config = makeConfig();

    OV_EXPECT_THROW_HAS_SUBSTRING(config.get<DUMMY_NO_DEFAULT_OPTION>(),
                                  ov::Exception,
                                  "no default value is available");
}

TEST_F(ConfigUnitTests, UpdateFromMapSetsSeveralOptionsAtOnce) {
    auto config = makeConfig();
    config.update(
        {{std::string(DUMMY_BOTH_OPTION::key()), "custom"}, {std::string(DUMMY_RUN_TIME_OPTION::key()), "7"}});

    EXPECT_TRUE(config.has<DUMMY_BOTH_OPTION>());
    EXPECT_EQ("custom", config.get<DUMMY_BOTH_OPTION>());
    EXPECT_EQ(7, config.get<DUMMY_RUN_TIME_OPTION>());
}

TEST_F(ConfigUnitTests, UpdateFromSingleKeyValueWorks) {
    auto config = makeConfig();
    config.update(DUMMY_RUN_TIME_OPTION::key(), "7");

    EXPECT_EQ(7, config.get<DUMMY_RUN_TIME_OPTION>());
    EXPECT_EQ("7", config.getString<DUMMY_RUN_TIME_OPTION>());
}

TEST_F(ConfigUnitTests, UpdateThroughDeprecatedKeySetsTheActualOption) {
    auto config = makeConfig();
    config.update("DUMMY_OLD_KEY", "some_value");

    EXPECT_TRUE(config.has<DUMMY_DEPRECATED_KEYS_OPTION>());
    EXPECT_EQ("some_value", config.get<DUMMY_DEPRECATED_KEYS_OPTION>());
    // the value is stored under the actual key, not the deprecated one
    EXPECT_FALSE(config.has("DUMMY_OLD_KEY"));
}

TEST_F(ConfigUnitTests, UpdateThroughEachDeprecatedKeySetsTheActualOption) {
    for (const auto& deprecatedKey : DUMMY_MULTI_DEPRECATED_KEYS_OPTION::deprecatedKeys()) {
        auto config = makeConfig();
        config.update({{std::string(deprecatedKey), "7"}});

        EXPECT_TRUE(config.has<DUMMY_MULTI_DEPRECATED_KEYS_OPTION>()) << "key: " << deprecatedKey;
        EXPECT_EQ(7, config.get<DUMMY_MULTI_DEPRECATED_KEYS_OPTION>()) << "key: " << deprecatedKey;
    }
}

TEST_F(ConfigUnitTests, UpdateThroughDeprecatedKeyAcceptsAnAnyPayload) {
    auto config = makeConfig();
    config.update("DUMMY_FIRST_OLD_KEY", ov::Any(int64_t{7}));

    EXPECT_EQ(7, config.get<DUMMY_MULTI_DEPRECATED_KEYS_OPTION>());
}

// The alias is resolved before anything is parsed, so the option's own `parse()` and `validateValue()` are the
// ones which run, and errors are reported against the actual key
TEST_F(ConfigUnitTests, UpdateThroughDeprecatedKeyRunsParsingAndValidation) {
    auto config = makeConfig();

    OV_EXPECT_THROW_HAS_SUBSTRING(config.update("DUMMY_FIRST_OLD_KEY", "abc"),
                                  ov::Exception,
                                  "Failed to parse 'DUMMY_MULTI_DEPRECATED_KEYS_OPTION' option");
    OV_EXPECT_THROW_HAS_SUBSTRING(config.update("DUMMY_FIRST_OLD_KEY", "-1"),
                                  ov::Exception,
                                  "expects non-negative values");
    EXPECT_FALSE(config.has<DUMMY_MULTI_DEPRECATED_KEYS_OPTION>());
}

TEST_F(ConfigUnitTests, DeprecatedAndActualKeysWriteToTheSameSlot) {
    auto config = makeConfig();
    config.update("DUMMY_FIRST_OLD_KEY", "1");
    config.update("DUMMY_SECOND_OLD_KEY", "2");
    config.update(DUMMY_MULTI_DEPRECATED_KEYS_OPTION::key(), "3");

    EXPECT_EQ(3, config.get<DUMMY_MULTI_DEPRECATED_KEYS_OPTION>());
    // a single entry, serialized under the actual key
    EXPECT_EQ("DUMMY_MULTI_DEPRECATED_KEYS_OPTION=\"3\"", config.toString());
}

TEST_F(ConfigUnitTests, OptionDescriptorLookupResolvesDeprecatedKeys) {
    const auto config = makeConfig();

    EXPECT_TRUE(config.hasOpt("DUMMY_OLD_KEY"));
    EXPECT_EQ(DUMMY_DEPRECATED_KEYS_OPTION::key(), config.getOpt("DUMMY_OLD_KEY").key());
}

TEST_F(ConfigUnitTests, UpdateWithUnknownKeyThrows) {
    auto config = makeConfig();

    OV_EXPECT_THROW_HAS_SUBSTRING(config.update("SOME_UNKNOWN_OPTION", "value"), ov::Exception, "[ NOT_FOUND ]");
}

TEST_F(ConfigUnitTests, UpdateWithUnparsableValueThrows) {
    auto config = makeConfig();

    OV_EXPECT_THROW_HAS_SUBSTRING(config.update(DUMMY_RUN_TIME_OPTION::key(), "abc"),
                                  ov::Exception,
                                  "Failed to parse 'DUMMY_RUN_TIME_OPTION' option");
}

TEST_F(ConfigUnitTests, UpdateRunsTheOptionValidation) {
    auto config = makeConfig();
    config.update(DUMMY_VALIDATED_OPTION::key(), "1");
    EXPECT_EQ(1, config.get<DUMMY_VALIDATED_OPTION>());

    OV_EXPECT_THROW_HAS_SUBSTRING(config.update(DUMMY_VALIDATED_OPTION::key(), "-1"),
                                  ov::Exception,
                                  "expects non-negative values");

    OV_EXPECT_THROW_HAS_SUBSTRING(config.update(DUMMY_VALIDATED_OPTION::key(), ov::Any(int64_t{-1})),
                                  ov::Exception,
                                  "expects non-negative values");
}

TEST_F(ConfigUnitTests, UpdateTakesTheValueAsIs) {
    auto config = makeConfig();
    config.update(DUMMY_RUN_TIME_OPTION::key(), ov::Any(int64_t{7}));
    config.update(DUMMY_BOTH_OPTION::key(), ov::Any(std::string("custom")));

    EXPECT_EQ(7, config.get<DUMMY_RUN_TIME_OPTION>());
    EXPECT_EQ("custom", config.get<DUMMY_BOTH_OPTION>());
}

// A string payload must reach the option's own parser instead of being re-parsed by `ov::Any::as()`
TEST_F(ConfigUnitTests, UpdateWithStringPayloadUsesTheOptionParser) {
    auto config = makeConfig();
    config.update(DUMMY_CUSTOM_PARSE_OPTION::key(), ov::Any(std::string("AUTO")));

    EXPECT_EQ(-1, config.get<DUMMY_CUSTOM_PARSE_OPTION>());
}

TEST_F(ConfigUnitTests, StringAndAnyPayloadsAcceptTheSameBooleanSpellings) {
    for (const auto& val : {"YES", "yes", "TRUE", "ON", "1"}) {
        auto fromString = makeConfig();
        fromString.update(DUMMY_BOOL_OPTION::key(), val);

        auto fromAny = makeConfig();
        fromAny.update(DUMMY_BOOL_OPTION::key(), ov::Any(std::string(val)));

        EXPECT_TRUE(fromString.get<DUMMY_BOOL_OPTION>()) << "value: " << val;
        EXPECT_TRUE(fromAny.get<DUMMY_BOOL_OPTION>()) << "value: " << val;
    }
}

TEST_F(ConfigUnitTests, UpdateKeepsStringOptionsIntact) {
    auto config = makeConfig();
    config.update(DUMMY_BOTH_OPTION::key(), ov::Any(std::string("some value with spaces")));

    EXPECT_EQ("some value with spaces", config.get<DUMMY_BOTH_OPTION>());
}

TEST_F(ConfigUnitTests, UpdateWithUnparsableStringPayloadReportsTheOption) {
    auto config = makeConfig();

    OV_EXPECT_THROW_HAS_SUBSTRING(config.update(DUMMY_RUN_TIME_OPTION::key(), ov::Any(std::string("abc"))),
                                  ov::Exception,
                                  "Failed to parse 'DUMMY_RUN_TIME_OPTION' option");
}

TEST_F(ConfigUnitTests, UpdateWithStringPayloadRunsTheOptionValidation) {
    auto config = makeConfig();

    OV_EXPECT_THROW_HAS_SUBSTRING(config.update(DUMMY_VALIDATED_OPTION::key(), ov::Any(std::string("-1"))),
                                  ov::Exception,
                                  "expects non-negative values");
}

// Payloads which already have the option's exact type must keep bypassing the parser, as some options
// cannot be built from a string at all
TEST_F(ConfigUnitTests, UpdateWithExactlyTypedPayloadSkipsTheOptionParser) {
    auto config = makeConfig();
    config.update(DUMMY_CUSTOM_PARSE_OPTION::key(), ov::Any(int64_t{-1}));
    config.update(DUMMY_BOOL_OPTION::key(), ov::Any(true));

    EXPECT_EQ(-1, config.get<DUMMY_CUSTOM_PARSE_OPTION>());
    EXPECT_TRUE(config.get<DUMMY_BOOL_OPTION>());
}

// `ov::Any::as<uint32_t>()` would convert -2.0f through an unchecked arithmetic cast, wrapping it around
// into a huge positive value which then passes any range validation
TEST_F(ConfigUnitTests, UpdateWithMismatchedArithmeticPayloadIsRejected) {
    auto config = makeConfig();

    OV_EXPECT_THROW_HAS_SUBSTRING(config.update(DUMMY_UINT32_OPTION::key(), ov::Any(-2.0f)),
                                  ov::Exception,
                                  "Failed to parse 'DUMMY_UINT32_OPTION' option");
    OV_EXPECT_THROW_HAS_SUBSTRING(config.update(DUMMY_UINT32_OPTION::key(), ov::Any(int64_t{-2})),
                                  ov::Exception,
                                  "Failed to parse 'DUMMY_UINT32_OPTION' option");

    EXPECT_FALSE(config.has<DUMMY_UINT32_OPTION>());
    EXPECT_EQ(1u, config.get<DUMMY_UINT32_OPTION>());
}

// A payload of a different type is still usable as long as its value fits the option
TEST_F(ConfigUnitTests, UpdateWithMismatchedArithmeticPayloadGoesThroughTheParser) {
    auto config = makeConfig();
    config.update(DUMMY_UINT32_OPTION::key(), ov::Any(int64_t{7}));

    EXPECT_EQ(7u, config.get<DUMMY_UINT32_OPTION>());
}

// An unprintable payload is stringified into an empty spelling, which the `std::string` parser would otherwise
// accept, silently setting the option to an empty value instead of reporting the unusable payload
TEST_F(ConfigUnitTests, UpdateWithUnprintablePayloadIsRejected) {
    auto config = makeConfig();

    OV_EXPECT_THROW_HAS_SUBSTRING(config.update(DUMMY_BOTH_OPTION::key(), ov::Any(UnprintablePayload{})),
                                  ov::Exception,
                                  "can't be converted to a string");

    EXPECT_FALSE(config.has<DUMMY_BOTH_OPTION>());
    EXPECT_EQ("both_default", config.get<DUMMY_BOTH_OPTION>());
}

// An empty `ov::Any` is stringified into an empty spelling as well
TEST_F(ConfigUnitTests, UpdateWithEmptyPayloadIsRejected) {
    auto config = makeConfig();

    OV_EXPECT_THROW_HAS_SUBSTRING(config.update(DUMMY_BOTH_OPTION::key(), ov::Any()),
                                  ov::Exception,
                                  "No value was provided");

    EXPECT_FALSE(config.has<DUMMY_BOTH_OPTION>());
    EXPECT_EQ("both_default", config.get<DUMMY_BOTH_OPTION>());
}

// An empty spelling is still accepted when it really was given as a string, for an option whose parser takes it
TEST_F(ConfigUnitTests, UpdateWithEmptyStringPayloadIsAccepted) {
    auto config = makeConfig();
    config.update(DUMMY_BOTH_OPTION::key(), ov::Any(std::string()));

    EXPECT_TRUE(config.has<DUMMY_BOTH_OPTION>());
    EXPECT_EQ("", config.get<DUMMY_BOTH_OPTION>());
}

// A string literal reaches the option's parser as an empty spelling as well, which is not the same thing as
// providing no value at all: here it is the option's own parser which rejects it
TEST_F(ConfigUnitTests, UpdateWithEmptyStringPayloadIsRejectedByTheOptionParser) {
    auto config = makeConfig();

    OV_EXPECT_THROW_HAS_SUBSTRING(config.update(DUMMY_BOOL_OPTION::key(), ""),
                                  ov::Exception,
                                  "is not a valid BOOL option");
    OV_EXPECT_THROW_HAS_SUBSTRING(config.update(DUMMY_RUN_TIME_OPTION::key(), ov::Any(std::string())),
                                  ov::Exception,
                                  "is not a valid INT64 option");

    EXPECT_FALSE(config.has<DUMMY_BOOL_OPTION>());
    EXPECT_FALSE(config.has<DUMMY_RUN_TIME_OPTION>());
}

TEST_F(ConfigUnitTests, RemoveDropsThePreviouslySetValue) {
    auto config = makeConfig();
    config.update(DUMMY_RUN_TIME_OPTION::key(), "7");
    ASSERT_TRUE(config.has(std::string(DUMMY_RUN_TIME_OPTION::key())));

    config.remove(std::string(DUMMY_RUN_TIME_OPTION::key()));

    EXPECT_FALSE(config.has<DUMMY_RUN_TIME_OPTION>());
    // back to the default value
    EXPECT_EQ(42, config.get<DUMMY_RUN_TIME_OPTION>());
}

TEST_F(ConfigUnitTests, ToStringSerializesOnlyTheSetOptions) {
    auto config = makeConfig();
    config.update(DUMMY_BOTH_OPTION::key(), "custom");

    EXPECT_EQ("DUMMY_BOTH_OPTION=\"custom\"", config.toString());
}

TEST_F(ConfigUnitTests, ToStringFromStringRoundTrip) {
    auto config = makeConfig();
    config.update(
        {{std::string(DUMMY_BOTH_OPTION::key()), "custom"}, {std::string(DUMMY_RUN_TIME_OPTION::key()), "7"}});

    auto restored = makeConfig();
    restored.fromString(config.toString());

    EXPECT_EQ("custom", restored.get<DUMMY_BOTH_OPTION>());
    EXPECT_EQ(7, restored.get<DUMMY_RUN_TIME_OPTION>());
}

TEST_F(ConfigUnitTests, OptionDescriptorLookupIsForwardedToTheDesc) {
    const auto config = makeConfig();

    EXPECT_TRUE(config.hasOpt(DUMMY_BOTH_OPTION::key()));
    EXPECT_FALSE(config.hasOpt("SOME_UNKNOWN_OPTION"));
    EXPECT_EQ(OptionMode::RunTime, config.getOpt(DUMMY_RUN_TIME_OPTION::key()).mode());
    OV_EXPECT_THROW_HAS_SUBSTRING(config.getOpt("SOME_UNKNOWN_OPTION"), ov::Exception, "[ NOT_FOUND ]");
}

//
// Config - internal compiler options
//

TEST_F(ConfigUnitTests, InternalOptionCanBeAddedAndRead) {
    auto config = makeConfig();
    config.addOrUpdateInternal("INTERNAL_OPTION", "1");

    EXPECT_TRUE(config.hasInternal("INTERNAL_OPTION"));
    EXPECT_EQ("1", config.getInternal("INTERNAL_OPTION"));
}

TEST_F(ConfigUnitTests, InternalOptionIsOverwrittenOnSecondAdd) {
    auto config = makeConfig();
    config.addOrUpdateInternal("INTERNAL_OPTION", "1");
    config.addOrUpdateInternal("INTERNAL_OPTION", "2");

    EXPECT_EQ("2", config.getInternal("INTERNAL_OPTION"));
}

TEST_F(ConfigUnitTests, ReadingMissingInternalOptionThrows) {
    const auto config = makeConfig();

    EXPECT_FALSE(config.hasInternal("INTERNAL_OPTION"));
    OV_EXPECT_THROW_HAS_SUBSTRING(config.getInternal("INTERNAL_OPTION"), ov::Exception, "does not exist");
}

TEST_F(ConfigUnitTests, RemoveCompileTimeConfigsKeepsRunTimeAndBothOptions) {
    auto config = makeConfig();
    config.update({{std::string(DUMMY_BOTH_OPTION::key()), "custom"},
                   {std::string(DUMMY_COMPILE_TIME_OPTION::key()), "custom"},
                   {std::string(DUMMY_RUN_TIME_OPTION::key()), "7"}});
    config.addOrUpdateInternal("INTERNAL_OPTION", "1");

    config.removeCompileTimeConfigs();

    EXPECT_FALSE(config.has<DUMMY_COMPILE_TIME_OPTION>());
    EXPECT_TRUE(config.has<DUMMY_BOTH_OPTION>());
    EXPECT_TRUE(config.has<DUMMY_RUN_TIME_OPTION>());
    EXPECT_FALSE(config.hasInternal("INTERNAL_OPTION"));
}

//
// Config - compiler serialization
//

TEST_F(ConfigUnitTests, ToStringForCompilerRequiresAPredicate) {
    const auto config = makeConfig();

    OV_EXPECT_THROW_HAS_SUBSTRING(config.toStringForCompiler(nullptr),
                                  ov::Exception,
                                  "requires a valid support predicate");
}

TEST_F(ConfigUnitTests, ToStringForCompilerSerializesSupportedCompileTimeAndBothOptions) {
    auto config = makeConfig();
    config.update({{std::string(DUMMY_BOTH_OPTION::key()), "both_value"},
                   {std::string(DUMMY_COMPILE_TIME_OPTION::key()), "compile_value"}});

    const auto serialized = config.toStringForCompiler([](const std::string&) {
        return true;
    });

    EXPECT_TRUE(containsEntry(serialized, DUMMY_BOTH_OPTION::key(), "both_value")) << serialized;
    EXPECT_TRUE(containsEntry(serialized, DUMMY_COMPILE_TIME_OPTION::key(), "compile_value")) << serialized;
}

TEST_F(ConfigUnitTests, ToStringForCompilerSkipsRunTimeOptions) {
    auto config = makeConfig();
    config.update(DUMMY_RUN_TIME_OPTION::key(), "7");

    const auto serialized = config.toStringForCompiler([](const std::string&) {
        return true;
    });

    EXPECT_EQ("", serialized);
}

// A "Both" option the compiler doesn't know about is still usable by the plugin, so it is silently dropped
TEST_F(ConfigUnitTests, ToStringForCompilerSkipsUnsupportedBothOptions) {
    auto config = makeConfig();
    config.update(DUMMY_BOTH_OPTION::key(), "both_value");

    const auto serialized = config.toStringForCompiler([](const std::string&) {
        return false;
    });

    EXPECT_EQ("", serialized);
}

// A compile-time-only option the compiler doesn't know about cannot be honored, hence the hard error
TEST_F(ConfigUnitTests, ToStringForCompilerThrowsOnUnsupportedCompileTimeOption) {
    auto config = makeConfig();
    config.update(DUMMY_COMPILE_TIME_OPTION::key(), "compile_value");

    OV_EXPECT_THROW_HAS_SUBSTRING(config.toStringForCompiler([](const std::string&) {
        return false;
    }),
                                  ov::Exception,
                                  "[ NOT_FOUND ]");
}

TEST_F(ConfigUnitTests, ToStringForCompilerSerializesInternalOptions) {
    auto config = makeConfig();
    config.addOrUpdateInternal("INTERNAL_OPTION", "1");

    const auto serialized = config.toStringForCompiler([](const std::string&) {
        return true;
    });

    EXPECT_TRUE(containsEntry(serialized, "INTERNAL_OPTION", "1")) << serialized;
}

TEST_F(ConfigUnitTests, ToStringForCompilerThrowsOnUnsupportedInternalOption) {
    auto config = makeConfig();
    config.addOrUpdateInternal("INTERNAL_OPTION", "1");

    OV_EXPECT_THROW_HAS_SUBSTRING(config.toStringForCompiler([](const std::string&) {
        return false;
    }),
                                  ov::Exception,
                                  "[ NOT_FOUND ]");
}

TEST_F(ConfigUnitTests, SerializedEntriesAreSpaceSeparated) {
    auto config = makeConfig();
    config.update(DUMMY_COMPILE_TIME_OPTION::key(), "compile_value");
    config.addOrUpdateInternal("INTERNAL_OPTION", "1");

    const auto serialized = config.toStringForCompiler([](const std::string&) {
        return true;
    });

    EXPECT_EQ(1u, std::count(serialized.begin(), serialized.end(), ' ')) << serialized;
}

//
// Config - environment variables
//

TEST_F(ConfigUnitTests, EnvVarIsAppliedToTheMatchingOption) {
    const ScopedEnvVar envVar("OV_NPU_DUMMY_ENV_OPTION", "7");

    auto config = makeConfig();
    config.parseEnvVars();

    EXPECT_TRUE(config.has<DUMMY_ENV_OPTION>());
    EXPECT_EQ(7, config.get<DUMMY_ENV_OPTION>());
}

TEST_F(ConfigUnitTests, InvalidEnvVarValueIsIgnored) {
    const ScopedEnvVar envVar("OV_NPU_DUMMY_ENV_OPTION", "abc");

    auto config = makeConfig();
    EXPECT_NO_THROW(config.parseEnvVars());

    EXPECT_FALSE(config.has<DUMMY_ENV_OPTION>());
    EXPECT_EQ(0, config.get<DUMMY_ENV_OPTION>());
}

TEST_F(ConfigUnitTests, OptionsWithoutEnvVarAreLeftUntouched) {
    auto config = makeConfig();
    config.parseEnvVars();

    EXPECT_FALSE(config.has<DUMMY_BOTH_OPTION>());
}

//
// Real options
//

using RealOptionsUnitTests = ::testing::Test;

// `ov::log::Level` has no dedicated `OptionParser` / `OptionPrinter` specialization, it goes through the
// stream-operator-based defaults
TEST_F(RealOptionsUnitTests, LogLevelIsParsedAndPrintedThroughTheDefaults) {
    auto desc = std::make_shared<OptionsDesc>();
    desc->add<intel_npu::LOG_LEVEL>();
    Config config(desc);

    config.update(intel_npu::LOG_LEVEL::key(), "LOG_DEBUG");

    EXPECT_EQ(ov::log::Level::DEBUG, config.get<intel_npu::LOG_LEVEL>());
    EXPECT_EQ("LOG_DEBUG", config.getString<intel_npu::LOG_LEVEL>());
    OV_EXPECT_THROW_HAS_SUBSTRING(config.update(intel_npu::LOG_LEVEL::key(), "LOG_SOMETHING"),
                                  ov::Exception,
                                  "Failed to parse 'LOG_LEVEL' option");
}

// A malformed public setting must be rejected instead of being silently applied from its valid prefix
TEST_F(RealOptionsUnitTests, MalformedValuesAreRejectedForPublicOptions) {
    auto desc = std::make_shared<OptionsDesc>();
    desc->add<intel_npu::COMPILER_TYPE>();
    desc->add<intel_npu::PERFORMANCE_HINT_NUM_REQUESTS>();
    Config config(desc);

    OV_EXPECT_THROW_HAS_SUBSTRING(config.update(intel_npu::COMPILER_TYPE::key(), "PLUGIN garbage"),
                                  ov::Exception,
                                  "Failed to parse 'NPU_COMPILER_TYPE' option");
    OV_EXPECT_THROW_HAS_SUBSTRING(config.update(intel_npu::PERFORMANCE_HINT_NUM_REQUESTS::key(), "12oops"),
                                  ov::Exception,
                                  "Failed to parse 'PERFORMANCE_HINT_NUM_REQUESTS' option");
    // the value `OVPropertiesIncorrectTests` passes for this property
    OV_EXPECT_THROW_HAS_SUBSTRING(config.update(intel_npu::PERFORMANCE_HINT_NUM_REQUESTS::key(), ov::Any(-2.0f)),
                                  ov::Exception,
                                  "Failed to parse 'PERFORMANCE_HINT_NUM_REQUESTS' option");

    EXPECT_EQ(ov::intel_npu::CompilerType::PREFER_PLUGIN, config.get<intel_npu::COMPILER_TYPE>());
    EXPECT_FALSE(config.has<intel_npu::PERFORMANCE_HINT_NUM_REQUESTS>());

    // the well formed spellings still go through
    config.update(intel_npu::COMPILER_TYPE::key(), "PLUGIN");
    config.update(intel_npu::PERFORMANCE_HINT_NUM_REQUESTS::key(), "12");

    EXPECT_EQ(ov::intel_npu::CompilerType::PLUGIN, config.get<intel_npu::COMPILER_TYPE>());
    EXPECT_EQ(12u, config.get<intel_npu::PERFORMANCE_HINT_NUM_REQUESTS>());
}

TEST_F(RealOptionsUnitTests, ExecutionModeHintIsParsedAndPrintedThroughTheDefaults) {
    auto desc = std::make_shared<OptionsDesc>();
    desc->add<intel_npu::EXECUTION_MODE_HINT>();
    Config config(desc);

    EXPECT_EQ(ov::hint::ExecutionMode::PERFORMANCE, config.get<intel_npu::EXECUTION_MODE_HINT>());

    config.update(intel_npu::EXECUTION_MODE_HINT::key(), "ACCURACY");

    EXPECT_EQ(ov::hint::ExecutionMode::ACCURACY, config.get<intel_npu::EXECUTION_MODE_HINT>());
    EXPECT_EQ("ACCURACY", config.getString<intel_npu::EXECUTION_MODE_HINT>());
}

}  // namespace
