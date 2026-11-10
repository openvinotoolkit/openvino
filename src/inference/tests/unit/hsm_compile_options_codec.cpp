// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/hsm_compile_options_codec.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <ios>
#include <limits>
#include <locale>
#include <map>
#include <memory>
#include <optional>
#include <ostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/hsm_format.hpp"
#include "openvino/runtime/hsm_reader.hpp"
#include "openvino/runtime/hsm_writer.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/util/hash_util.hpp"

namespace ov::test {
namespace hsm = ov::runtime::hsm;
using namespace std::string_literals;
namespace {

static_assert(std::is_convertible_v<decltype(&hsm::decode_compile_options), hsm::SectionDecoder<ov::AnyMap>>,
              "decode_compile_options(const Section&) must be usable as a SectionDecoder<ov::AnyMap>.");
static_assert(std::is_base_of_v<hsm::ISectionEncoder, hsm::CompileOptionsEncoder>,
              "CompileOptionsEncoder must be an ISectionEncoder.");
static_assert(std::is_convertible_v<std::shared_ptr<hsm::CompileOptionsEncoder>, hsm::SectionEncoderPtr>,
              "CompileOptionsEncoder must be passable as a SectionEncoderPtr.");
static_assert(std::is_constructible_v<hsm::CompileOptionsEncoder, const ov::AnyMap&>,
              "CompileOptionsEncoder must borrow an existing map.");
static_assert(!std::is_constructible_v<hsm::CompileOptionsEncoder, ov::AnyMap&&>,
              "CompileOptionsEncoder must not borrow a temporary map.");
static_assert(!std::is_constructible_v<hsm::CompileOptionsEncoder, const ov::AnyMap&&>,
              "CompileOptionsEncoder must not borrow a temporary const map.");
static_assert(std::is_same_v<decltype(std::declval<hsm::IWriter&>().add_section(
                                 hsm::DeviceId{},
                                 hsm::compile_options_tag,
                                 std::declval<std::shared_ptr<hsm::CompileOptionsEncoder>>())),
                             bool>,
              "CompileOptionsEncoder must fit the unsized IWriter::add_section(SectionEncoderPtr).");

constexpr hsm::DeviceId fake_device_id = hsm::private_device_id_range_start + 42;
constexpr size_t chunk_size = 4096;

// Separators, spelled out here on purpose - the codec's own constants are not visible to the tests.
const std::string US{"\x1f"};
const std::string GS{"\x1d"};
const std::string RS{"\x1e"};
const std::string FS{"\x1c"};

// The digest rule from the format description, applied to everything up to and including FS.
std::string digest_of(std::string_view bytes) {
    uint64_t hash = 0;
    for (const auto byte : bytes) {
        hash = ov::util::u64_hash_combine(hash, static_cast<uint8_t>(byte));
    }
    return std::string{reinterpret_cast<const char*>(&hash), sizeof(hash)};
}

// @p body must end with FS.
std::string with_digest(const std::string& body) {
    return body + digest_of(body);
}

/// @p input is what the encoder gets; @p decoded what the decoder must return (if unset, same as @p input).
struct ReferenceCase {
    std::string name;
    ov::AnyMap input;
    std::optional<ov::AnyMap> decoded;
    std::string entries;  ///< Everything up to and including FS.
    std::string digest;   ///< Little-endian, computed offline by an independent implementation of the rule.

    const ov::AnyMap& expected_decoded() const {
        return decoded ? *decoded : input;
    }

    std::string bytes() const {
        return entries + digest;
    }
};

void PrintTo(const ReferenceCase& reference, std::ostream* os) {
    *os << reference.name;
}

// Reference payloads are spelled out byte by byte - never produced by the codec.
const ReferenceCase empty_map_case{"empty_map", {}, std::nullopt, FS, "\x6c\x57\xd4\xaa\xc2\x65\x0d\x33"s};

const ReferenceCase single_string_case{"single_string",
                                       {{"k", "v"}},
                                       std::nullopt,
                                       "k" + US + "v" + US + FS,
                                       "\x53\x96\x43\xf8\x6b\x0d\x60\xde"s};

const ReferenceCase empty_string_value_case{"empty_string_value",
                                            {{"E", ""}},
                                            std::nullopt,
                                            "E" + US + US + FS,
                                            "\xdb\x32\x5b\x70\xaa\x94\xdf\x52"s};

const ReferenceCase typed_case{"typed",
                               {ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT),
                                ov::hint::inference_precision(ov::element::f16),
                                ov::enable_profiling(true),
                                ov::hint::num_requests(4)},
                               ov::AnyMap{{"INFERENCE_PRECISION_HINT", "f16"},
                                          {"PERFORMANCE_HINT", "THROUGHPUT"},
                                          {"PERFORMANCE_HINT_NUM_REQUESTS", "4"},
                                          {"PERF_COUNT", "YES"}},
                               "INFERENCE_PRECISION_HINT" + US + "f16" + US + "PERFORMANCE_HINT" + US + "THROUGHPUT" +
                                   US + "PERFORMANCE_HINT_NUM_REQUESTS" + US + "4" + US + "PERF_COUNT" + US + "YES" +
                                   US + FS,
                               "\xf9\xf3\xe9\xc6\x9a\x73\x37\x68"s};

const ReferenceCase special_chars_case{"special_chars",
                                       {{"a\nb=c", "{x:y,z}"}, {"\xC5\xBC", "\xE2\x9C\x93"}},
                                       std::nullopt,
                                       "a\nb=c" + US + "{x:y,z}" + US + "\xC5\xBC" + US + "\xE2\x9C\x93" + US + FS,
                                       "\x89\xf1\x68\xef\xf4\x04\x2e\xc0"s};

const ReferenceCase nested_empty_map_case{"nested_empty_map",
                                          {{"M", ov::AnyMap{}}},
                                          std::nullopt,
                                          "M" + GS + GS + FS,
                                          "\xd9\x26\x16\x10\x09\xa9\xe8\xec"s};

const ReferenceCase nested_three_levels_case{
    "nested_three_levels",
    {{"A", ov::AnyMap{{"B", ov::AnyMap{{"C", ov::AnyMap{{"k", "v"}}}}}}}, {"T", "1"}},
    std::nullopt,
    "A" + GS + "B" + GS + "C" + GS + "k" + US + "v" + US + GS + GS + GS + "T" + US + "1" + US + FS,
    "\xbb\xe9\xb1\x26\x54\x81\xb6\xe1"s};

const ReferenceCase sorted_keys_case{"sorted_keys",
                                     {{"b", "2"}, {"a", "1"}},
                                     std::nullopt,
                                     "a" + US + "1" + US + "b" + US + "2" + US + FS,
                                     "\x8b\x6b\x4e\x58\xa3\xd9\x55\xc9"s};

const ReferenceCase numeric_case{
    "numeric",
    {{"D", 0.1}, {"F", 0.1f}, {"I", int64_t{-9000000000}}, {"N", uint32_t{1234}}},
    ov::AnyMap{{"D", "0.10000000000000001"}, {"F", "0.100000001"}, {"I", "-9000000000"}, {"N", "1234"}},
    "D" + US + "0.10000000000000001" + US + "F" + US + "0.100000001" + US + "I" + US + "-9000000000" + US + "N" + US +
        "1234" + US + FS,
    "\xf3\xf7\x10\xcb\x1e\xed\x4a\x85"s};

// ov::device::properties holds exactly this type; it must come out as two nested levels.
const ReferenceCase device_properties_case{
    "device_properties_type",
    {{"DEVICE_PROPERTIES",
      std::map<std::string, ov::AnyMap>{{"CPU", {{"CACHE_DIR", "/tmp/c d"}}}, {"GPU", {{"PERF_COUNT", "YES"}}}}}},
    ov::AnyMap{
        {"DEVICE_PROPERTIES",
         ov::AnyMap{{"CPU", ov::AnyMap{{"CACHE_DIR", "/tmp/c d"}}}, {"GPU", ov::AnyMap{{"PERF_COUNT", "YES"}}}}}},
    "DEVICE_PROPERTIES" + GS + "CPU" + GS + "CACHE_DIR" + US + "/tmp/c d" + US + GS + "GPU" + GS + "PERF_COUNT" + US +
        "YES" + US + GS + GS + FS,
    "\xf8\x1b\x09\x8b\x67\xf8\xaf\x0c"s};

std::vector<ReferenceCase> reference_cases() {
    return {empty_map_case,
            single_string_case,
            empty_string_value_case,
            typed_case,
            special_chars_case,
            nested_empty_map_case,
            nested_three_levels_case,
            sorted_keys_case,
            numeric_case,
            device_properties_case};
}

std::string to_string(ov::util::MemoryView bytes) {
    return {reinterpret_cast<const char*>(bytes.data()), bytes.size()};
}

std::string encode(const ov::AnyMap& options) {
    const hsm::CompileOptionsEncoder encoder{options};
    std::string written;
    encoder.encode([&written](ov::util::MemoryView chunk) {
        written += to_string(chunk);
    });
    return written;
}

hsm::ManifestEntry make_entry(size_t size) {
    hsm::ManifestEntry entry{};
    entry.device = fake_device_id;
    entry.tag = hsm::compile_options_tag;
    entry.size = size;
    return entry;
}

std::optional<ov::AnyMap> decode_buffer(std::string_view bytes) {
    return hsm::decode_compile_options(hsm::Section{make_entry(bytes.size()), ov::util::as_memory_view(bytes)});
}

// The payload starts at a non-zero stream position, as it does inside a container.
std::optional<ov::AnyMap> decode_stream(std::string_view bytes) {
    std::stringstream stream{"prefix" + std::string{bytes}};
    return hsm::decode_compile_options(hsm::Section{make_entry(bytes.size()), stream, 6});
}

// Any::operator== is false for two empty Anys, so compare structurally; decoded leaves must be strings.
::testing::AssertionResult same_options(const ov::AnyMap& actual,
                                        const ov::AnyMap& expected,
                                        const std::string& path = {}) {
    if (actual.size() != expected.size()) {
        return ::testing::AssertionFailure()
               << path << "/: " << actual.size() << " entries, expected " << expected.size();
    }
    for (const auto& [key, value] : expected) {
        const auto where = path + "/" + key;
        const auto found = actual.find(key);
        if (found == actual.end()) {
            return ::testing::AssertionFailure() << where << ": missing";
        }
        const auto& got = found->second;
        if (value.is<ov::AnyMap>()) {
            if (!got.is<ov::AnyMap>()) {
                return ::testing::AssertionFailure() << where << ": expected a nested map";
            }
            if (auto nested = same_options(got.as<ov::AnyMap>(), value.as<ov::AnyMap>(), where); !nested) {
                return nested;
            }
        } else if (!got.is<std::string>() || got.as<std::string>() != value.as<std::string>()) {
            return ::testing::AssertionFailure()
                   << where << ": got " << (got.is<std::string>() ? got.as<std::string>() : "<not a string>")
                   << ", expected " << value.as<std::string>();
        }
    }
    return ::testing::AssertionSuccess();
}

// Decodes from a buffer-backed and from a stream-backed section.
void expect_decoded(std::string_view bytes, const ov::AnyMap& expected) {
    const auto from_buffer = decode_buffer(bytes);
    ASSERT_TRUE(from_buffer);
    EXPECT_TRUE(same_options(*from_buffer, expected));

    const auto from_stream = decode_stream(bytes);
    ASSERT_TRUE(from_stream);
    EXPECT_TRUE(same_options(*from_stream, expected));
}

void expect_rejected(std::string_view bytes) {
    EXPECT_FALSE(decode_buffer(bytes));
    EXPECT_FALSE(decode_stream(bytes));
}

template <typename Value>
void expect_value_roundtrip(const Value& input, const std::optional<std::string>& expected_text = std::nullopt) {
    const ov::AnyMap options{{"k", input}};
    const auto bytes = encode(options);
    for (const auto& [source, decoded] :
         {std::pair{"buffer", decode_buffer(bytes)}, std::pair{"stream", decode_stream(bytes)}}) {
        SCOPED_TRACE(source);
        ASSERT_TRUE(decoded);
        ASSERT_EQ(decoded->size(), 1);
        const ov::Any& leaf = decoded->at("k");
        ASSERT_TRUE(leaf.is<std::string>());
        if (expected_text) {
            EXPECT_EQ(leaf.as<std::string>(), *expected_text);
        }
        const auto restored = leaf.as<Value>();
        EXPECT_EQ(restored, input);
        if constexpr (std::is_floating_point_v<Value>) {
            EXPECT_EQ(std::signbit(restored), std::signbit(input));
        }
    }
}

template <class Case>
std::string case_name(const ::testing::TestParamInfo<Case>& info) {
    return info.param.name;
}

}  // namespace

// --- Format: the reference digests follow the documented rule ---

using HsmCompileOptionsFormatReferenceTest = ::testing::TestWithParam<ReferenceCase>;

TEST_P(HsmCompileOptionsFormatReferenceTest, reference_digest_follows_the_documented_rule) {
    EXPECT_EQ(digest_of(GetParam().entries), GetParam().digest);
}

INSTANTIATE_TEST_SUITE_P(HsmCompileOptions,
                         HsmCompileOptionsFormatReferenceTest,
                         ::testing::ValuesIn(reference_cases()),
                         case_name<ReferenceCase>);

// --- Encoding: options -> reference bytes ---

using HsmCompileOptionsEncodeReferenceTest = ::testing::TestWithParam<ReferenceCase>;

TEST_P(HsmCompileOptionsEncodeReferenceTest, encoder_writes_reference_bytes_to_sink) {
    EXPECT_EQ(encode(GetParam().input), GetParam().bytes());
}

TEST_P(HsmCompileOptionsEncodeReferenceTest, encoding_twice_gives_identical_bytes) {
    const hsm::CompileOptionsEncoder encoder{GetParam().input};
    std::string first;
    std::string second;
    encoder.encode([&first](ov::util::MemoryView chunk) {
        first += to_string(chunk);
    });
    encoder.encode([&second](ov::util::MemoryView chunk) {
        second += to_string(chunk);
    });

    EXPECT_EQ(first, second);
}

INSTANTIATE_TEST_SUITE_P(HsmCompileOptions,
                         HsmCompileOptionsEncodeReferenceTest,
                         ::testing::ValuesIn(reference_cases()),
                         case_name<ReferenceCase>);

namespace {
struct NoStringForm {};

struct ThrowingStringForm {};

std::ostream& operator<<(std::ostream&, const ThrowingStringForm&) {
    throw std::runtime_error{"printer failure"};
}

struct SinkFailure : std::runtime_error {
    using std::runtime_error::runtime_error;
};

class NonSeekableStringBuf : public std::stringbuf {
public:
    using std::stringbuf::stringbuf;

protected:
    pos_type seekpos(pos_type, std::ios_base::openmode) override {
        return pos_type{off_type{-1}};
    }
};

struct InvalidOptionsCase {
    std::string name;
    ov::AnyMap options;
};

void PrintTo(const InvalidOptionsCase& invalid, std::ostream* os) {
    *os << invalid.name;
}

std::vector<InvalidOptionsCase> invalid_options_cases() {
    std::vector<InvalidOptionsCase> cases{
        {"empty_key", {{"", "v"}}},
        {"empty_nested_key", {{"M", ov::AnyMap{{"", "v"}}}}},
        {"empty_key_in_device_properties", {{"M", std::map<std::string, ov::AnyMap>{{"", {{"k", "v"}}}}}}},
        {"empty_any", {{"k", ov::Any{}}}},
        {"empty_nested_any", {{"M", ov::AnyMap{{"k", ov::Any{}}}}}},
        {"non_string_without_text", {{"k", NoStringForm{}}}},
        {"empty_vector_without_text", {{"k", std::vector<int>{}}}},
    };
    for (const char reserved : {'\0', '\x1f', '\x1d', '\x1e', '\x1c'}) {
        const auto suffix = std::to_string(static_cast<int>(reserved));
        const auto with_byte = "a"s + reserved + "b"s;
        cases.push_back({"reserved_byte_in_key_" + suffix, {{with_byte, "v"}}});
        cases.push_back({"reserved_byte_in_nested_key_" + suffix, {{"M", ov::AnyMap{{with_byte, "v"}}}}});
        cases.push_back({"reserved_byte_in_string_value_" + suffix, {{"k", with_byte}}});
        cases.push_back({"reserved_byte_in_printed_value_" + suffix, {{"k", std::vector<std::string>{with_byte}}}});
    }
    return cases;
}
}  // namespace

using HsmCompileOptionsEncodeInvalidTest = ::testing::TestWithParam<InvalidOptionsCase>;

TEST_P(HsmCompileOptionsEncodeInvalidTest, encoder_throws) {
    EXPECT_THROW(encode(GetParam().options), ov::Exception);
}

INSTANTIATE_TEST_SUITE_P(HsmCompileOptions,
                         HsmCompileOptionsEncodeInvalidTest,
                         ::testing::ValuesIn(invalid_options_cases()),
                         case_name<InvalidOptionsCase>);

TEST(HsmCompileOptionsEncodeTest, error_for_non_string_value_without_text_names_key_and_type) {
    const ov::AnyMap options{{"THE_KEY", NoStringForm{}}};
    std::string message;
    try {
        encode(options);
    } catch (const ov::Exception& e) {
        message = e.what();
    }

    EXPECT_NE(message.find("THE_KEY"), std::string::npos) << message;
    EXPECT_NE(message.find("NoStringForm"), std::string::npos) << message;
}

TEST(HsmCompileOptionsEncodeTest, printer_exception_is_wrapped_with_key_and_original_message) {
    const ov::AnyMap options{{"THE_KEY", ThrowingStringForm{}}};

    try {
        encode(options);
        FAIL() << "Expected a conversion exception";
    } catch (const ov::Exception& error) {
        const std::string message{error.what()};
        EXPECT_NE(message.find("THE_KEY"), std::string::npos) << message;
        EXPECT_NE(message.find("printer failure"), std::string::npos) << message;
    }
}

using HsmCompileOptionsEncodeSinkExceptionTest = ::testing::TestWithParam<size_t>;

TEST_P(HsmCompileOptionsEncodeSinkExceptionTest, sink_exception_propagates_without_further_output) {
    const hsm::CompileOptionsEncoder encoder{single_string_case.input};
    std::string written;
    size_t failed_calls = 0;

    try {
        encoder.encode([&](ov::util::MemoryView chunk) {
            if (written.size() >= GetParam()) {
                ++failed_calls;
                throw SinkFailure{"sink failure"};
            }
            written += to_string(chunk);
        });
        FAIL() << "Expected a sink exception";
    } catch (const SinkFailure& error) {
        EXPECT_STREQ(error.what(), "sink failure");
    }

    EXPECT_EQ(failed_calls, 1);
    EXPECT_EQ(written, single_string_case.entries.substr(0, GetParam()));
}

INSTANTIATE_TEST_SUITE_P(HsmCompileOptions,
                         HsmCompileOptionsEncodeSinkExceptionTest,
                         ::testing::Values(size_t{0}, size_t{2}, single_string_case.entries.size()),
                         [](const ::testing::TestParamInfo<size_t>& info) {
                             if (info.param == 0) {
                                 return "first_output";
                             }
                             return info.param == single_string_case.entries.size() ? "digest" : "partial_output";
                         });

TEST(HsmCompileOptionsEncodeTest, encoder_encodes_borrowed_map_only_in_encode) {
    ov::AnyMap options{{"", "invalid until fixed"}};
    const hsm::CompileOptionsEncoder encoder{options};

    EXPECT_THROW(encoder.encode([](ov::util::MemoryView) {}), ov::Exception);

    options = single_string_case.input;
    std::string written;
    encoder.encode([&written](ov::util::MemoryView chunk) {
        written += to_string(chunk);
    });
    EXPECT_EQ(written, single_string_case.bytes());
}

namespace {
struct GroupingNumpunct : std::numpunct<char> {
    char do_decimal_point() const override {
        return '/';
    }
    char do_thousands_sep() const override {
        return '_';
    }
    std::string do_grouping() const override {
        return "\2";
    }
};

class GlobalLocaleGuard {
public:
    explicit GlobalLocaleGuard(const std::locale& locale) : m_previous{std::locale::global(locale)} {}
    ~GlobalLocaleGuard() {
        std::locale::global(m_previous);
    }

private:
    std::locale m_previous;
};
}  // namespace

TEST(HsmCompileOptionsEncodeTest, numbers_do_not_depend_on_global_locale) {
    const GlobalLocaleGuard guard{std::locale{std::locale::classic(), new GroupingNumpunct}};
    std::ostringstream global_format;
    global_format << 1234.5;
    ASSERT_EQ(global_format.str(), "12_34/5");

    EXPECT_EQ(encode(numeric_case.input), numeric_case.bytes());
}

// --- Decoding: reference bytes -> options ---

using HsmCompileOptionsDecodeReferenceTest = ::testing::TestWithParam<ReferenceCase>;

TEST_P(HsmCompileOptionsDecodeReferenceTest, decode_produces_reference_options) {
    expect_decoded(GetParam().bytes(), GetParam().expected_decoded());
}

TEST_P(HsmCompileOptionsDecodeReferenceTest, every_truncated_prefix_is_rejected) {
    const auto bytes = GetParam().bytes();

    for (size_t size = 0; size < bytes.size(); ++size) {
        EXPECT_FALSE(decode_buffer(std::string_view{bytes}.substr(0, size))) << "prefix size: " << size;
    }
}

INSTANTIATE_TEST_SUITE_P(HsmCompileOptions,
                         HsmCompileOptionsDecodeReferenceTest,
                         ::testing::ValuesIn(reference_cases()),
                         case_name<ReferenceCase>);

namespace {
struct AcceptCase {
    std::string name;
    std::string body;  ///< Up to and including FS; the digest is added by the test.
    ov::AnyMap expected;
};

void PrintTo(const AcceptCase& accept, std::ostream* os) {
    *os << accept.name;
}

std::vector<AcceptCase> accept_cases() {
    const std::string reserved_content("\0\x1f\x1d", 3);
    return {
        {"unsorted_keys", "b" + US + "2" + US + "a" + US + "1" + US + FS, {{"a", "1"}, {"b", "2"}}},
        {"same_key_on_different_levels", "a" + GS + "a" + US + "1" + US + GS + FS, {{"a", ov::AnyMap{{"a", "1"}}}}},
        {"empty_key_leaf_is_skipped", US + "x" + US + "k" + US + "v" + US + FS, {{"k", "v"}}},
        {"several_empty_key_leaves_are_not_duplicates",
         US + "x" + US + US + "y" + US + "k" + US + "v" + US + FS,
         {{"k", "v"}}},
        {"empty_key_leaf_in_nested_map",
         "M" + GS + US + "x" + US + "k" + US + "v" + US + GS + FS,
         {{"M", ov::AnyMap{{"k", "v"}}}}},
        {"reserved_block_with_empty_content", "k" + RS + RS + FS, {}},
        {"reserved_block_is_skipped",
         "a" + US + "1" + US + "r" + RS + reserved_content + RS + "b" + US + "2" + US + FS,
         {{"a", "1"}, {"b", "2"}}},
        {"reserved_block_in_nested_map",
         "M" + GS + "r" + RS + "zz" + RS + "k" + US + "v" + US + GS + FS,
         {{"M", ov::AnyMap{{"k", "v"}}}}},
        {"reserved_block_key_is_not_a_duplicate", "a" + US + "1" + US + "a" + RS + "zz" + RS + FS, {{"a", "1"}}},
        {"reserved_block_with_empty_key", RS + "zz" + RS + "k" + US + "v" + US + FS, {{"k", "v"}}},
    };
}
}  // namespace

using HsmCompileOptionsDecodeAcceptTest = ::testing::TestWithParam<AcceptCase>;

TEST_P(HsmCompileOptionsDecodeAcceptTest, decode_accepts) {
    expect_decoded(with_digest(GetParam().body), GetParam().expected);
}

INSTANTIATE_TEST_SUITE_P(HsmCompileOptions,
                         HsmCompileOptionsDecodeAcceptTest,
                         ::testing::ValuesIn(accept_cases()),
                         case_name<AcceptCase>);

namespace {
struct MalformedCase {
    std::string name;
    std::string bytes;
};

void PrintTo(const MalformedCase& malformed, std::ostream* os) {
    *os << malformed.name;
}

std::vector<MalformedCase> malformed_cases() {
    const auto empty_map_digest = digest_of(FS);
    auto wrong_digest = with_digest(single_string_case.entries);
    wrong_digest.back() ^= 1;

    std::vector<MalformedCase> cases{
        {"empty_section", ""},
        {"no_fs_and_no_digest", "k" + US + "v" + US},
        {"section_ends_in_key", "k"},
        {"section_ends_in_value", "k" + US + "v"},
        {"section_ends_in_reserved_block", "k" + RS + "zz"},
        {"digest_mismatch", wrong_digest},
        {"trailing_byte_after_digest", with_digest(single_string_case.entries) + '\0'},
        {"gs_at_top_level", with_digest(GS + FS)},
        {"gs_after_leaf_at_top_level", with_digest("k" + US + "v" + US + GS + FS)},
        {"fs_in_nested_map", with_digest("M" + GS + "k" + US + "v" + US + FS)},
        {"fs_in_key", with_digest("k" + FS)},
        {"fs_in_value", with_digest("k" + US + "v" + FS)},
        {"gs_in_value", with_digest("k" + US + "v" + GS + "v" + US + FS)},
        {"rs_in_value", with_digest("k" + US + "v" + RS + "v" + US + FS)},
        {"nul_in_key", with_digest("k"s + '\0' + US + "v" + US + FS)},
        {"nul_in_value", with_digest("k" + US + "v"s + '\0' + US + FS)},
        {"fs_in_reserved_block", with_digest("k" + RS + "zz" + FS)},
        {"duplicate_leaf", with_digest("k" + US + "1" + US + "k" + US + "2" + US + FS)},
        {"duplicate_nested_map", with_digest("M" + GS + GS + "M" + GS + GS + FS)},
        {"leaf_then_map_with_same_key", with_digest("M" + US + "1" + US + "M" + GS + GS + FS)},
        {"map_then_leaf_with_same_key", with_digest("M" + GS + GS + "M" + US + "1" + US + FS)},
        {"duplicate_leaf_in_nested_map", with_digest("M" + GS + "k" + US + "1" + US + "k" + US + "2" + US + GS + FS)},
    };
    for (size_t size = 1; size < empty_map_digest.size(); ++size) {
        cases.push_back({"digest_cut_to_" + std::to_string(size) + "_bytes", FS + empty_map_digest.substr(0, size)});
    }
    return cases;
}

std::vector<MalformedCase> malformed_multi_chunk_cases() {
    std::vector<MalformedCase> cases;
    for (const size_t offset : {chunk_size - 1, chunk_size, chunk_size + 1}) {
        const auto suffix = "_at_" + std::to_string(offset);
        const auto value_prefix = "k" + US + std::string(offset - 2, 'v');
        using pairs = std::initializer_list<std::pair<const char*, std::string>>;
        for (const auto& [name, illegal] : pairs{{"NUL", "\0"s}, {"GS", GS}, {"RS", RS}, {"FS", FS}}) {
            cases.push_back({"illegal_value_byte_"s + name + suffix, with_digest(value_prefix + illegal + US + FS)});
        }

        const auto pad_to_offset = [offset](const std::string& entry) {
            return entry + "r" + RS + std::string(offset - entry.size() - 3, 'x') + RS;
        };
        const auto leaf_prefix = pad_to_offset("k" + US + "1" + US);
        const auto map_prefix = pad_to_offset("k" + GS + GS);
        cases.push_back({"duplicate_leaf" + suffix, with_digest(leaf_prefix + "k" + US + "2" + US + FS)});
        cases.push_back({"duplicate_map" + suffix, with_digest(map_prefix + "k" + GS + GS + FS)});
        cases.push_back({"leaf_then_map" + suffix, with_digest(leaf_prefix + "k" + GS + GS + FS)});
        cases.push_back({"map_then_leaf" + suffix, with_digest(map_prefix + "k" + US + "2" + US + FS)});
        cases.push_back({"extra_top_level_gs" + suffix, with_digest(leaf_prefix + GS + FS)});
        cases.push_back(
            {"fs_in_nested_map" + suffix, with_digest("M" + GS + "k" + US + std::string(offset - 5, 'v') + US + FS)});

        cases.push_back({"truncated_key" + suffix, std::string(offset, 'k')});
        cases.push_back({"truncated_value" + suffix, value_prefix});
        cases.push_back({"truncated_reserved_block" + suffix, "r" + RS + std::string(offset - 2, 'x')});
    }
    return cases;
}
}  // namespace

using HsmCompileOptionsDecodeMalformedTest = ::testing::TestWithParam<MalformedCase>;

TEST_P(HsmCompileOptionsDecodeMalformedTest, decode_rejects) {
    expect_rejected(GetParam().bytes);
}

INSTANTIATE_TEST_SUITE_P(HsmCompileOptions,
                         HsmCompileOptionsDecodeMalformedTest,
                         ::testing::ValuesIn(malformed_cases()),
                         case_name<MalformedCase>);

INSTANTIATE_TEST_SUITE_P(HsmCompileOptionsMultiChunk,
                         HsmCompileOptionsDecodeMalformedTest,
                         ::testing::ValuesIn(malformed_multi_chunk_cases()),
                         case_name<MalformedCase>);

TEST(HsmCompileOptionsDecodeTest, decoded_strings_convert_to_property_types) {
    const auto decoded = decode_buffer(typed_case.bytes());

    ASSERT_TRUE(decoded);
    EXPECT_EQ(decoded->at(ov::hint::performance_mode.name()).as<ov::hint::PerformanceMode>(),
              ov::hint::PerformanceMode::THROUGHPUT);
    EXPECT_EQ(decoded->at(ov::hint::inference_precision.name()).as<ov::element::Type>(), ov::element::f16);
    EXPECT_TRUE(decoded->at(ov::enable_profiling.name()).as<bool>());
    EXPECT_EQ(decoded->at(ov::hint::num_requests.name()).as<uint32_t>(), 4u);
}

TEST(HsmCompileOptionsDecodeTest, decoded_numbers_restore_exact_values) {
    const auto decoded = decode_buffer(numeric_case.bytes());

    ASSERT_TRUE(decoded);
    EXPECT_EQ(decoded->at("D").as<double>(), 0.1);
    EXPECT_EQ(decoded->at("F").as<float>(), 0.1f);
    EXPECT_EQ(decoded->at("I").as<int64_t>(), int64_t{-9000000000});
    EXPECT_EQ(decoded->at("N").as<uint32_t>(), 1234u);
}

TEST(HsmCompileOptionsDecodeTest, unreadable_section_is_rejected) {
    const auto bytes = single_string_case.bytes();
    std::stringstream stream{bytes.substr(0, bytes.size() - 1)};

    EXPECT_FALSE(hsm::decode_compile_options(hsm::Section{make_entry(bytes.size()), stream, 0}));
}

TEST(HsmCompileOptionsDecodeTest, stream_read_exception_returns_nullopt) {
    const auto bytes = single_string_case.bytes();
    std::stringstream stream{bytes.substr(0, bytes.size() - 1)};
    stream.exceptions(std::ios_base::failbit | std::ios_base::badbit);

    EXPECT_FALSE(hsm::decode_compile_options(hsm::Section{make_entry(bytes.size()), stream, 0}));
    EXPECT_TRUE(stream.fail());
    EXPECT_TRUE(stream.eof());
}

TEST(HsmCompileOptionsDecodeTest, stream_seek_exception_returns_nullopt) {
    const auto bytes = single_string_case.bytes();
    NonSeekableStringBuf buffer{bytes};
    std::istream stream{&buffer};
    stream.exceptions(std::ios_base::failbit | std::ios_base::badbit);

    EXPECT_FALSE(hsm::decode_compile_options(hsm::Section{make_entry(bytes.size()), stream, 0}));
    EXPECT_TRUE(stream.fail());
    EXPECT_FALSE(stream.eof());
}

// --- Decoding: sizes around the decoder's read chunk, no limits ---

TEST(HsmCompileOptionsDecodeTest, payload_ending_around_the_chunk_boundary_is_decoded) {
    for (size_t length = chunk_size - 26; length <= chunk_size + 14; ++length) {
        SCOPED_TRACE(length);
        const std::string value(length, 'x');
        expect_decoded(with_digest("k" + US + value + US + FS), {{"k", value}});
    }
}

TEST(HsmCompileOptionsDecodeTest, key_and_value_larger_than_a_chunk_are_decoded) {
    const std::string key(chunk_size + 100, 'k');
    const std::string value(25 * chunk_size, 'v');

    expect_decoded(with_digest(key + US + value + US + FS), {{key, value}});
}

TEST(HsmCompileOptionsDecodeTest, reserved_block_larger_than_a_chunk_is_skipped) {
    const std::string content(2 * chunk_size + 300, 'z');

    expect_decoded(with_digest("r" + RS + content + RS + "k" + US + "v" + US + FS), {{"k", "v"}});
}

TEST(HsmCompileOptionsDecodeTest, stream_truncated_after_first_chunk_is_rejected) {
    const auto bytes = with_digest("k" + US + std::string(chunk_size + 3, 'v') + US + FS);
    std::stringstream stream{bytes.substr(0, chunk_size)};

    EXPECT_FALSE(hsm::decode_compile_options(hsm::Section{make_entry(bytes.size()), stream, 0}));
    EXPECT_TRUE(stream.fail());
    EXPECT_TRUE(stream.eof());
    EXPECT_EQ(stream.gcount(), 0);
}

TEST(HsmCompileOptionsDecodeTest, stream_missing_digest_after_full_chunk_is_rejected) {
    const auto prefix = "k" + US;
    const auto suffix = US + FS;
    const auto entries = prefix + std::string(chunk_size - prefix.size() - suffix.size(), 'v') + suffix;
    ASSERT_EQ(entries.size(), chunk_size);
    const auto bytes = with_digest(entries);
    std::stringstream stream{entries};

    EXPECT_FALSE(hsm::decode_compile_options(hsm::Section{make_entry(bytes.size()), stream, 0}));
    EXPECT_TRUE(stream.fail());
    EXPECT_TRUE(stream.eof());
    EXPECT_EQ(stream.gcount(), 0);
}

// --- Round trip of generated options ---

TEST(HsmCompileOptionsRoundTripTest, integer_limits_are_preserved) {
    for (const auto value : {std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max()}) {
        SCOPED_TRACE(value);
        expect_value_roundtrip(value, std::to_string(value));
    }
    for (const auto value : {uint64_t{0}, std::numeric_limits<uint64_t>::max()}) {
        SCOPED_TRACE(value);
        expect_value_roundtrip(value, std::to_string(value));
    }
}

template <typename Value>
class HsmCompileOptionsFloatingPointTest : public ::testing::Test {};

using CompileOptionsFloatingPointTypes = ::testing::Types<float, double, long double>;
TYPED_TEST_SUITE(HsmCompileOptionsFloatingPointTest, CompileOptionsFloatingPointTypes);

TYPED_TEST(HsmCompileOptionsFloatingPointTest, finite_limits_roundtrip_exactly) {
    for (const auto value : {std::numeric_limits<TypeParam>::lowest(),
                             std::numeric_limits<TypeParam>::max(),
                             std::numeric_limits<TypeParam>::min(),
                             std::nextafter(TypeParam{1}, TypeParam{2})}) {
        SCOPED_TRACE(value);
        expect_value_roundtrip(value);
    }
}

TYPED_TEST(HsmCompileOptionsFloatingPointTest, signed_zero_is_preserved) {
    expect_value_roundtrip(TypeParam{0});
    expect_value_roundtrip(-TypeParam{0});
}

TEST(HsmCompileOptionsRoundTripTest, vector_is_restored_from_a_string_leaf) {
    expect_value_roundtrip(std::vector<int>{-2, 0, 3}, "-2 0 3");
}

TEST(HsmCompileOptionsRoundTripTest, set_is_restored_from_a_string_leaf) {
    expect_value_roundtrip(std::set<std::string>{"a", "b"}, "a b");
}

TEST(HsmCompileOptionsRoundTripTest, printable_map_is_restored_from_a_string_leaf) {
    expect_value_roundtrip(std::map<std::string, int>{{"a", 1}, {"b", -2}}, "{a:1,b:-2}");
}

TEST(HsmCompileOptionsRoundTripTest, boolean_false_is_restored_from_no) {
    expect_value_roundtrip(false, "NO");
}

TEST(HsmCompileOptionsRoundTripTest, deeply_nested_options_are_restored) {
    // Far deeper than real options: exercises encoder recursion and the decoder's heap level stack.
    constexpr size_t depth = 100;
    ov::AnyMap options{{"level", "0"}};
    for (size_t level = 1; level <= depth; ++level) {
        options = ov::AnyMap{{"level", std::to_string(level)}, {"next", std::move(options)}};
    }

    expect_decoded(encode(options), options);
}

}  // namespace ov::test
