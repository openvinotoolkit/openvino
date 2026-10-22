// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler_impl.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "fake_vcl.hpp"
#include "intel_npu/common/option_support_cache.hpp"
#include "intel_npu/config/config.hpp"
#include "intel_npu/config/options.hpp"
#include "intel_npu/utils/utils.hpp"
#include "model_serializer.hpp"
#include "openvino/core/except.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "ze_graph_ext_wrappers.hpp"

using ::fake_vcl::FakeVcl;
using ::intel_npu::Config;
using ::intel_npu::IDevice;
using ::intel_npu::OptionsDesc;
using ::intel_npu::OptionSupportCache;
using ::intel_npu::ScopedOptionSupportCache;
using ::intel_npu::VCLCompilerImpl;

namespace {

/// Two distinct cache keys, standing in for the plugin and driver adapters sharing one cache.
constexpr OptionSupportCache::CacheKey kFirstKey = 1u;
constexpr OptionSupportCache::CacheKey kSecondKey = 2u;

/// Registers just the options the compiler-in-plugin path reads, so `config.get<>` resolves.
std::shared_ptr<OptionsDesc> makeOptionsDesc() {
    auto desc = std::make_shared<OptionsDesc>();
    desc->add<::intel_npu::MODEL_SERIALIZER_VERSION>();
    desc->add<::intel_npu::WS_COMPILE_CALL_NUMBER>();
    return desc;
}

Config makeConfig() {
    // Registration is all that is needed: compileWsIterative writes WS_COMPILE_CALL_NUMBER via
    // update(), and MODEL_SERIALIZER_VERSION is read through config.get<>.
    return Config(makeOptionsDesc());
}

/// A minimal model with one weight, enough for the serializer to produce a real IR.
std::shared_ptr<ov::Model> makeModel() {
    auto weights = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{5}, std::vector<float>{1.0f});
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{5});
    input->set_friendly_name("Parameter_0");
    auto add = std::make_shared<ov::op::v1::Add>(input, weights);
    add->set_friendly_name("Add_1");
    return std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input}, "compiler_impl_test_model");
}

struct VCLCompilerImplTest : public ::testing::Test {
    FakeVcl fake;

    std::shared_ptr<VCLCompilerImpl> makeCompiler(
        const std::optional<IDevice::DeviceProperties>& props = std::nullopt) {
        return std::make_shared<VCLCompilerImpl>(fake.functions(), props);
    }

    /// A compiler that routes its option-support answers through `cache` under `key`. Without a
    /// cache the compiler is queried on every call, which is the makeCompiler() behaviour above.
    std::shared_ptr<VCLCompilerImpl> makeCachingCompiler(const std::shared_ptr<OptionSupportCache>& cache,
                                                         const OptionSupportCache::CacheKey key = kFirstKey) {
        return std::make_shared<VCLCompilerImpl>(fake.functions(), std::nullopt, ScopedOptionSupportCache{cache, key});
    }

    /// Number of times the compiler library was actually asked about an option.
    size_t optionQueryCount() const {
        return fake.callCount("vclGetCompilerIsOptionSupported");
    }
};

//
// --- construction ---
//

TEST_F(VCLCompilerImplTest, ConstructionQueriesVersionCreatesCompilerAndReadsProperties) {
    auto compiler = makeCompiler();
    ASSERT_NE(compiler, nullptr);

    EXPECT_TRUE(fake.called("vclGetVersion"));
    EXPECT_EQ(fake.callCount("vclCompilerCreate"), 1u);
    EXPECT_EQ(fake.callCount("vclCompilerGetProperties"), 1u);
    // vclGetVersion must precede compiler creation: the negotiated version goes into the desc.
    EXPECT_LT(fake.indexOf("vclGetVersion"), fake.indexOf("vclCompilerCreate"));
}

TEST_F(VCLCompilerImplTest, ConstructionForwardsTheNegotiatedVersionInTheCompilerDesc) {
    fake.reportedCompilerVersion = {VCL_COMPILER_VERSION_MAJOR, VCL_COMPILER_VERSION_MINOR};
    auto compiler = makeCompiler();

    ASSERT_EQ(fake.compilerDescs.size(), 1u);
    EXPECT_EQ(fake.compilerDescs[0].version.major, VCL_COMPILER_VERSION_MAJOR);
    EXPECT_EQ(fake.compilerDescs[0].version.minor, VCL_COMPILER_VERSION_MINOR);
}

TEST_F(VCLCompilerImplTest, DevicePropertiesAreForwardedExactly) {
    IDevice::DeviceProperties props{0x1234u, 7u, 4u};
    auto compiler = makeCompiler(props);

    ASSERT_EQ(fake.deviceDescs.size(), 1u);
    const auto& desc = fake.deviceDescs[0];
    EXPECT_EQ(desc.size, sizeof(vcl_device_desc_t));
    EXPECT_EQ(desc.deviceID, 0x1234u);
    EXPECT_EQ(desc.revision, 7u);
    EXPECT_EQ(desc.tileCount, 4u);
}

TEST_F(VCLCompilerImplTest, OversizedSubdeviceIdClampsToTheInvalidRevisionSentinel) {
    constexpr auto sentinel = std::numeric_limits<uint16_t>::max();
    // Anything at or above the sentinel cannot fit the 16-bit revision field.
    IDevice::DeviceProperties props{0x1u, static_cast<uint32_t>(sentinel) + 5u, 1u};
    auto compiler = makeCompiler(props);

    ASSERT_EQ(fake.deviceDescs.size(), 1u);
    EXPECT_EQ(fake.deviceDescs[0].revision, sentinel);
}

TEST_F(VCLCompilerImplTest, SubdeviceIdExactlyAtTheSentinelAlsoClamps) {
    constexpr auto sentinel = std::numeric_limits<uint16_t>::max();
    IDevice::DeviceProperties props{0x1u, static_cast<uint32_t>(sentinel), 1u};
    auto compiler = makeCompiler(props);

    ASSERT_EQ(fake.deviceDescs.size(), 1u);
    EXPECT_EQ(fake.deviceDescs[0].revision, sentinel);
}

TEST_F(VCLCompilerImplTest, AbsentDevicePropertiesUseDefaultSentinels) {
    auto compiler = makeCompiler(std::nullopt);

    ASSERT_EQ(fake.deviceDescs.size(), 1u);
    const auto& desc = fake.deviceDescs[0];
    EXPECT_EQ(desc.size, sizeof(vcl_device_desc_t));
    EXPECT_EQ(desc.deviceID, 0x00u);
    EXPECT_EQ(desc.revision, std::numeric_limits<uint16_t>::max());
    EXPECT_EQ(desc.tileCount, std::numeric_limits<uint32_t>::max());
}

TEST_F(VCLCompilerImplTest, NullApiTableIsRejected) {
    EXPECT_THROW(
        {
            auto compiler = std::make_shared<VCLCompilerImpl>(nullptr);
            (void)compiler;
        },
        ov::Exception);
}

TEST_F(VCLCompilerImplTest, UnwiredFunctionTableIsRejected) {
    // A default-constructed table: every entry point is null. Without the hasAllRequiredSymbols
    // guard this would dispatch through a null vclGetVersion instead of throwing.
    auto unwired = std::make_shared<const intel_npu::VCLFunctionTable>();
    EXPECT_THROW(
        {
            auto compiler = std::make_shared<VCLCompilerImpl>(unwired);
            (void)compiler;
        },
        ov::Exception);
}

TEST_F(VCLCompilerImplTest, MissingWeakSymbolsDoNotBlockConstruction) {
    // Weak symbols are legitimately null against an older compiler library, so the required-symbol
    // guard must ignore them. FakeVcl already leaves the whole weak list null.
    ASSERT_TRUE(fake.mutableFunctions()->vclAllocatedExecutableCreate2 == nullptr);

    EXPECT_TRUE(fake.functions()->hasAllRequiredSymbols());
    EXPECT_NO_THROW(makeCompiler());
}

TEST_F(VCLCompilerImplTest, MissingRequiredSymbolIsRejected) {
    // vclAllocatedExecutableCreateWSOneShot2 is required, not weak: compileWsOneShot calls it with
    // no null guard, so a library lacking it must be refused at construction rather than crashing
    // on the first weights-separation compile.
    fake.mutableFunctions()->vclAllocatedExecutableCreateWSOneShot2 = nullptr;

    EXPECT_FALSE(fake.functions()->hasAllRequiredSymbols());
    EXPECT_THROW(makeCompiler(), ov::Exception);
}

TEST_F(VCLCompilerImplTest, GetVersionFailureThrows) {
    fake.failWith("vclGetVersion", VCL_RESULT_ERROR_UNKNOWN);
    EXPECT_THROW(makeCompiler(), ov::Exception);
}

TEST_F(VCLCompilerImplTest, CompilerCreateFailureThrowsWithTheVclLogAppended) {
    fake.logString = "compiler-create-exploded";
    fake.failWith("vclCompilerCreate", VCL_RESULT_ERROR_OUT_OF_MEMORY);

    try {
        makeCompiler();
        FAIL() << "Expected construction to throw";
    } catch (const ov::Exception& error) {
        const std::string what = error.what();
        EXPECT_NE(what.find("vclCompilerCreate"), std::string::npos) << what;
        EXPECT_NE(what.find("compiler-create-exploded"), std::string::npos) << what;
    }
}

TEST_F(VCLCompilerImplTest, CompilerGetPropertiesFailureThrows) {
    fake.failWith("vclCompilerGetProperties", VCL_RESULT_ERROR_UNKNOWN);
    EXPECT_THROW(makeCompiler(), ov::Exception);
}

TEST_F(VCLCompilerImplTest, ConstructionAcceptsALibraryBelowTheFloorAndDefersTheCheck) {
    // Documents where the version gate actually lives: construction succeeds even for a too-old
    // library, and the refusal happens on first compile/query (see CompileThrowsWhenTheLibraryIs...).
    fake.reportedCompilerVersion = {VCL_COMPILER_VERSION_MAJOR - 1, 0};
    auto compiler = makeCompiler();

    ASSERT_NE(compiler, nullptr);
    EXPECT_EQ(fake.callCount("vclCompilerCreate"), 1u);
}

//
// --- destruction ---
//

TEST_F(VCLCompilerImplTest, DestructionDestroysTheCompilerExactlyOnce) {
    {
        auto compiler = makeCompiler();
    }
    EXPECT_EQ(fake.compilerDestroyCount, 1);
}

TEST_F(VCLCompilerImplTest, DestructionSwallowsCompilerDestroyFailure) {
    fake.failWith("vclCompilerDestroy", VCL_RESULT_ERROR_UNKNOWN);
    // A throwing destructor would terminate; the implementation only warns.
    EXPECT_NO_THROW({ auto compiler = makeCompiler(); });
    EXPECT_EQ(fake.compilerDestroyCount, 1);
}

TEST_F(VCLCompilerImplTest, FailedConstructionDoesNotDestroyAnUncreatedCompiler) {
    fake.failWith("vclCompilerCreate", VCL_RESULT_ERROR_UNKNOWN);
    EXPECT_THROW(makeCompiler(), ov::Exception);
    EXPECT_EQ(fake.compilerDestroyCount, 0);
}

//
// --- get_version ---
//

TEST_F(VCLCompilerImplTest, GetVersionPacksTheReportedPropertiesVersion) {
    fake.propertiesVersion = {9, 3};
    auto compiler = makeCompiler();

    EXPECT_EQ(compiler->get_version(), ZE_MAKE_VERSION(9, 3));
    EXPECT_EQ(compiler->get_version() >> 16, 9u);
    EXPECT_EQ(compiler->get_version() & 0xFFFFu, 3u);
}

//
// --- is_option_supported ---
//

TEST_F(VCLCompilerImplTest, IsOptionSupportedReturnsTrueOnSuccess) {
    auto compiler = makeCompiler();
    EXPECT_TRUE(compiler->is_option_supported("SOME_OPTION"));
}

TEST_F(VCLCompilerImplTest, IsOptionSupportedPassesNullptrWhenNoValueIsGiven) {
    auto compiler = makeCompiler();
    compiler->is_option_supported("SOME_OPTION");

    ASSERT_EQ(fake.optionSupportQueries.size(), 1u);
    EXPECT_EQ(fake.optionSupportQueries[0].first, "SOME_OPTION");
    EXPECT_FALSE(fake.optionSupportQueries[0].second.has_value());
}

TEST_F(VCLCompilerImplTest, IsOptionSupportedForwardsTheValueWhenGiven) {
    auto compiler = makeCompiler();
    compiler->is_option_supported("SOME_OPTION", std::string("SOME_VALUE"));

    ASSERT_EQ(fake.optionSupportQueries.size(), 1u);
    EXPECT_EQ(fake.optionSupportQueries[0].first, "SOME_OPTION");
    ASSERT_TRUE(fake.optionSupportQueries[0].second.has_value());
    EXPECT_EQ(*fake.optionSupportQueries[0].second, "SOME_VALUE");
}

TEST_F(VCLCompilerImplTest, IsOptionSupportedSwallowsErrorsAndReportsFalse) {
    auto compiler = makeCompiler();
    // The exception is deliberately swallowed: older libraries lack this entry point.
    fake.failWith("vclGetCompilerIsOptionSupported", VCL_RESULT_ERROR_UNSUPPORTED_FEATURE);
    EXPECT_FALSE(compiler->is_option_supported("SOME_OPTION"));

    fake.failWith("vclGetCompilerIsOptionSupported", VCL_RESULT_ERROR_UNKNOWN);
    EXPECT_FALSE(compiler->is_option_supported("SOME_OPTION"));
}

//
// --- get_supported_options ---
//

TEST_F(VCLCompilerImplTest, GetSupportedOptionsUsesTheTwoCallSizeProtocol) {
    auto compiler = makeCompiler();
    const auto options = compiler->get_supported_options();

    EXPECT_EQ(fake.callCount("vclGetCompilerSupportedOptions"), 2u);
    EXPECT_EQ(options, std::vector<std::string>({"OPT_A", "OPT_B", "OPT_C"}));
}

TEST_F(VCLCompilerImplTest, GetSupportedOptionsReturnsEmptyOnZeroSizeWithoutASecondCall) {
    fake.supportedOptionsBuffer.clear();
    auto compiler = makeCompiler();
    const auto options = compiler->get_supported_options();

    EXPECT_TRUE(options.empty());
    // Size 0 short-circuits: only the sizing call happens.
    EXPECT_EQ(fake.callCount("vclGetCompilerSupportedOptions"), 1u);
}

TEST_F(VCLCompilerImplTest, GetSupportedOptionsTrimsTrailingNulsAndTokenises) {
    // VCL pads the buffer with NULs; they must not become part of an option name.
    fake.supportedOptionsBuffer = std::string("OPT_A OPT_B") + std::string(5, '\0');
    auto compiler = makeCompiler();
    const auto options = compiler->get_supported_options();

    ASSERT_EQ(options.size(), 2u);
    EXPECT_EQ(options[0], "OPT_A");
    EXPECT_EQ(options[1], "OPT_B");
}

TEST_F(VCLCompilerImplTest, GetSupportedOptionsCollapsesArbitraryWhitespace) {
    fake.supportedOptionsBuffer = "  OPT_A \t OPT_B \n OPT_C  ";
    auto compiler = makeCompiler();
    EXPECT_EQ(compiler->get_supported_options(), std::vector<std::string>({"OPT_A", "OPT_B", "OPT_C"}));
}

TEST_F(VCLCompilerImplTest, GetSupportedOptionsReturnsEmptyWhenTheBufferIsAllNuls) {
    fake.supportedOptionsBuffer = std::string(8, '\0');
    auto compiler = makeCompiler();
    EXPECT_TRUE(compiler->get_supported_options().empty());
}

TEST_F(VCLCompilerImplTest, GetSupportedOptionsThrowsWhenTheSizingCallFails) {
    auto compiler = makeCompiler();
    fake.failWith("vclGetCompilerSupportedOptions", VCL_RESULT_ERROR_UNKNOWN);
    EXPECT_THROW(compiler->get_supported_options(), ov::Exception);
}

//
// --- option support cache ---
//
// The cache used to live in PluginCompilerAdapter; it now sits behind VCLCompilerImpl, so every
// caller (adapter, serializeConfig, serializeIR) shares one set of answers. The cache is keyed by
// (cache key, option name) only - it carries no value - which is what the tests below pin down.
//

TEST_F(VCLCompilerImplTest, WithoutACacheEveryIsOptionSupportedCallReachesTheCompiler) {
    // The baseline the cache is measured against: a null cache means no memoisation at all.
    auto compiler = makeCompiler();

    EXPECT_TRUE(compiler->is_option_supported("SOME_OPTION"));
    EXPECT_TRUE(compiler->is_option_supported("SOME_OPTION"));

    EXPECT_EQ(optionQueryCount(), 2u);
}

TEST_F(VCLCompilerImplTest, IsOptionSupportedIsAnsweredFromTheCacheOnRepeatedQueries) {
    auto cache = std::make_shared<OptionSupportCache>();
    auto compiler = makeCachingCompiler(cache);

    EXPECT_TRUE(compiler->is_option_supported("SOME_OPTION"));
    EXPECT_TRUE(compiler->is_option_supported("SOME_OPTION"));
    EXPECT_TRUE(compiler->is_option_supported("SOME_OPTION"));

    EXPECT_EQ(optionQueryCount(), 1u);
    EXPECT_EQ(cache->isOptionSupported(kFirstKey, "SOME_OPTION"), std::make_optional(true));
}

TEST_F(VCLCompilerImplTest, IsOptionSupportedCachesNegativeAnswersToo) {
    // "Not supported" is just as expensive to re-derive as "supported", and the compiler's answer
    // cannot change for a given compiler instance.
    auto cache = std::make_shared<OptionSupportCache>();
    auto compiler = makeCachingCompiler(cache);
    fake.unsupportedOptions.insert("MISSING_OPTION");

    EXPECT_FALSE(compiler->is_option_supported("MISSING_OPTION"));
    EXPECT_EQ(optionQueryCount(), 1u);

    // Make the fake start advertising the option: the cached "false" must still win, proving the
    // second call never reached the library.
    fake.unsupportedOptions.clear();
    EXPECT_FALSE(compiler->is_option_supported("MISSING_OPTION"));
    EXPECT_EQ(optionQueryCount(), 1u);
    EXPECT_EQ(cache->isOptionSupported(kFirstKey, "MISSING_OPTION"), std::make_optional(false));
}

TEST_F(VCLCompilerImplTest, QueriesCarryingAValueAlwaysReachTheCompiler) {
    // The cache key is the option name alone, so it cannot tell whether a specific value is
    // accepted. Serving a valued query from it would answer a different question.
    auto cache = std::make_shared<OptionSupportCache>();
    auto compiler = makeCachingCompiler(cache);

    EXPECT_TRUE(compiler->is_option_supported("SOME_OPTION", std::string("VALUE_A")));
    EXPECT_TRUE(compiler->is_option_supported("SOME_OPTION", std::string("VALUE_A")));

    EXPECT_EQ(optionQueryCount(), 2u);
}

TEST_F(VCLCompilerImplTest, AValuedQueryNeitherReadsNorWritesTheCache) {
    auto cache = std::make_shared<OptionSupportCache>();
    auto compiler = makeCachingCompiler(cache);
    fake.unsupportedOptions.insert("SOME_OPTION");

    // A rejected value must not be recorded as "the option is unsupported": the name-only answer
    // is a different fact, and caching the value verdict under the name would corrupt it.
    EXPECT_FALSE(compiler->is_option_supported("SOME_OPTION", std::string("BAD_VALUE")));
    EXPECT_FALSE(cache->isOptionSupported(kFirstKey, "SOME_OPTION").has_value());

    fake.unsupportedOptions.clear();
    EXPECT_TRUE(compiler->is_option_supported("SOME_OPTION"));
    EXPECT_EQ(cache->isOptionSupported(kFirstKey, "SOME_OPTION"), std::make_optional(true));
}

TEST_F(VCLCompilerImplTest, ACachedNameOnlyAnswerDoesNotShortCircuitAValuedQuery) {
    auto cache = std::make_shared<OptionSupportCache>();
    auto compiler = makeCachingCompiler(cache);

    EXPECT_TRUE(compiler->is_option_supported("SOME_OPTION"));
    ASSERT_EQ(optionQueryCount(), 1u);

    EXPECT_TRUE(compiler->is_option_supported("SOME_OPTION", std::string("VALUE_A")));
    EXPECT_EQ(optionQueryCount(), 2u);
    ASSERT_EQ(fake.optionSupportQueries.size(), 2u);
    ASSERT_TRUE(fake.optionSupportQueries[1].second.has_value());
    EXPECT_EQ(*fake.optionSupportQueries[1].second, "VALUE_A");
}

TEST_F(VCLCompilerImplTest, GetSupportedOptionsPopulatesTheCacheForLaterQueries) {
    // The bulk list is the cheap way to fill the cache: one library call, then every listed option
    // is answered locally. This is why get_supported_options writes through to the cache.
    auto cache = std::make_shared<OptionSupportCache>();
    auto compiler = makeCachingCompiler(cache);

    ASSERT_EQ(compiler->get_supported_options(), std::vector<std::string>({"OPT_A", "OPT_B", "OPT_C"}));

    EXPECT_TRUE(compiler->is_option_supported("OPT_A"));
    EXPECT_TRUE(compiler->is_option_supported("OPT_B"));
    EXPECT_TRUE(compiler->is_option_supported("OPT_C"));
    EXPECT_EQ(optionQueryCount(), 0u);
}

TEST_F(VCLCompilerImplTest, OptionsAbsentFromTheBulkListStillReachTheCompiler) {
    // setSupportedOptions only records positives, so an unlisted option is "unknown", not "false".
    // Treating it as false would deny options a newer library accepts but does not enumerate.
    auto cache = std::make_shared<OptionSupportCache>();
    auto compiler = makeCachingCompiler(cache);
    (void)compiler->get_supported_options();

    EXPECT_TRUE(compiler->is_option_supported("OPT_UNLISTED"));
    EXPECT_EQ(optionQueryCount(), 1u);
}

TEST_F(VCLCompilerImplTest, GetSupportedOptionsWithoutACacheDoesNotThrow) {
    auto compiler = makeCompiler();
    EXPECT_NO_THROW((void)compiler->get_supported_options());
}

TEST_F(VCLCompilerImplTest, EntriesAreScopedToTheCacheKey) {
    // One cache is shared by the plugin and driver adapters under different keys. Their compilers
    // accept different options, so an answer must never leak across keys.
    auto cache = std::make_shared<OptionSupportCache>();
    auto first = makeCachingCompiler(cache, kFirstKey);
    auto second = makeCachingCompiler(cache, kSecondKey);

    EXPECT_TRUE(first->is_option_supported("SOME_OPTION"));
    ASSERT_EQ(optionQueryCount(), 1u);

    EXPECT_TRUE(second->is_option_supported("SOME_OPTION"));
    EXPECT_EQ(optionQueryCount(), 2u);
}

TEST_F(VCLCompilerImplTest, CompilersSharingAKeyShareTheirAnswers) {
    auto cache = std::make_shared<OptionSupportCache>();
    auto first = makeCachingCompiler(cache, kFirstKey);
    auto second = makeCachingCompiler(cache, kFirstKey);

    EXPECT_TRUE(first->is_option_supported("SOME_OPTION"));
    ASSERT_EQ(optionQueryCount(), 1u);

    EXPECT_TRUE(second->is_option_supported("SOME_OPTION"));
    EXPECT_EQ(optionQueryCount(), 1u);
}

TEST_F(VCLCompilerImplTest, CachedAnswersOutliveTheCompilerThatProducedThem) {
    // The cache belongs to the plugin, not to a compiler instance: a compiler recreated for a
    // second compile must not have to re-query.
    auto cache = std::make_shared<OptionSupportCache>();
    {
        auto compiler = makeCachingCompiler(cache);
        EXPECT_TRUE(compiler->is_option_supported("SOME_OPTION"));
    }
    ASSERT_EQ(optionQueryCount(), 1u);

    auto recreated = makeCachingCompiler(cache);
    EXPECT_TRUE(recreated->is_option_supported("SOME_OPTION"));
    EXPECT_EQ(optionQueryCount(), 1u);
}

TEST_F(VCLCompilerImplTest, CompileHonoursACachedNegativeWithoutQueryingTheCompiler) {
    // The whole point of moving the cache behind the compiler: the build-flag construction inside
    // compile() goes through the same memoised answers, not just the adapter's public API.
    auto cache = std::make_shared<OptionSupportCache>();
    const std::string serializerVersion{ov::intel_npu::model_serializer_version.name()};
    cache->addSupportedOption(kFirstKey, serializerVersion, false);
    auto compiler = makeCachingCompiler(cache);

    const auto [tensor, compatibility] = compiler->compile(makeModel(), makeConfig());
    (void)tensor;
    (void)compatibility;

    ASSERT_EQ(fake.buildFlags.size(), 1u);
    EXPECT_EQ(fake.buildFlags[0].find(serializerVersion), std::string::npos);
    // The cached "false" is authoritative: the library was never asked whether it knows the option.
    // Valued probes for a concrete serializer version still go through, by design.
    const bool askedByNameOnly = std::any_of(fake.optionSupportQueries.begin(),
                                             fake.optionSupportQueries.end(),
                                             [&serializerVersion](const auto& query) {
                                                 return query.first == serializerVersion && !query.second.has_value();
                                             });
    EXPECT_FALSE(askedByNameOnly);
}

TEST_F(VCLCompilerImplTest, CompileAsksAboutEachOptionNameOnlyOnceWhenCaching) {
    // compile() consults option support from several places - serializeIR, the serializer-version
    // write-back and serializeConfig - so without the cache the same name is queried repeatedly.
    // Valued queries (serializeIR probes concrete serializer versions) are exempt by design.
    auto cache = std::make_shared<OptionSupportCache>();
    auto compiler = makeCachingCompiler(cache);

    const auto result = compiler->compile(makeModel(), makeConfig());
    (void)result;

    ASSERT_FALSE(fake.optionSupportQueries.empty());
    std::vector<std::string> names;
    for (const auto& query : fake.optionSupportQueries) {
        if (!query.second.has_value()) {
            names.push_back(query.first);
        }
    }
    ASSERT_FALSE(names.empty()) << "expected at least one name-only option query during compile()";
    std::sort(names.begin(), names.end());
    EXPECT_EQ(std::adjacent_find(names.begin(), names.end()), names.end())
        << "an option name was queried more than once despite the cache";
}

TEST_F(VCLCompilerImplTest, WithoutACacheCompileRepeatsTheSameOptionQueries) {
    // The counterpart of the test above: it is only the cache that collapses the duplicates, so a
    // regression that stops wiring it would show up here as an unchanged query count.
    auto compiler = makeCompiler();

    const auto result = compiler->compile(makeModel(), makeConfig());
    (void)result;

    std::vector<std::string> names;
    for (const auto& query : fake.optionSupportQueries) {
        if (!query.second.has_value()) {
            names.push_back(query.first);
        }
    }
    std::sort(names.begin(), names.end());
    EXPECT_NE(std::adjacent_find(names.begin(), names.end()), names.end());
}

TEST_F(VCLCompilerImplTest, TheBulkListContradictingACachedNegativeIsRejected) {
    // Documents the cache's conflict guard as seen through the compiler: once an option is recorded
    // as unsupported, a bulk list that advertises it is a genuine inconsistency rather than an
    // update, and setSupportedOptions refuses it instead of silently flipping the answer.
    auto cache = std::make_shared<OptionSupportCache>();
    auto compiler = makeCachingCompiler(cache);
    fake.unsupportedOptions.insert("OPT_A");

    ASSERT_FALSE(compiler->is_option_supported("OPT_A"));

    fake.unsupportedOptions.clear();
    EXPECT_THROW((void)compiler->get_supported_options(), ov::Exception);
}

//
// --- process_profiling_output ---
//

TEST_F(VCLCompilerImplTest, ProcessProfilingOutputFollowsCreateGetDestroyOrdering) {
    fake.profilingPayload.assign(2 * sizeof(ze_profiling_layer_info), 0);
    auto compiler = makeCompiler();

    const std::vector<uint8_t> profData{1, 2, 3};
    const std::vector<uint8_t> network{4, 5, 6, 7};
    const auto info = compiler->process_profiling_output(profData, network);

    EXPECT_EQ(fake.callCount("vclProfilingCreate"), 1u);
    EXPECT_EQ(fake.callCount("vclProfilingGetProperties"), 1u);
    EXPECT_EQ(fake.callCount("vclGetDecodedProfilingBuffer"), 1u);
    EXPECT_EQ(fake.profilingDestroyCount, 1);

    EXPECT_LT(fake.indexOf("vclProfilingCreate"), fake.indexOf("vclProfilingGetProperties"));
    EXPECT_LT(fake.indexOf("vclProfilingGetProperties"), fake.indexOf("vclGetDecodedProfilingBuffer"));
    EXPECT_LT(fake.indexOf("vclGetDecodedProfilingBuffer"), fake.indexOf("vclProfilingDestroy"));

    // One entry per ze_profiling_layer_info in the returned buffer.
    EXPECT_EQ(info.size(), 2u);
}

TEST_F(VCLCompilerImplTest, ProcessProfilingOutputSizesByLayerInfoStride) {
    fake.profilingPayload.assign(5 * sizeof(ze_profiling_layer_info), 0);
    auto compiler = makeCompiler();
    EXPECT_EQ(compiler->process_profiling_output({1}, {2}).size(), 5u);
}

TEST_F(VCLCompilerImplTest, ProcessProfilingOutputThrowsOnNullData) {
    fake.forceNullProfilingData = true;
    auto compiler = makeCompiler();

    try {
        compiler->process_profiling_output({1}, {2});
        FAIL() << "Expected a throw on NULL profiling data";
    } catch (const ov::Exception& error) {
        EXPECT_NE(std::string(error.what()).find("Failed to get VCL profiling output"), std::string::npos);
    }
}

TEST_F(VCLCompilerImplTest, ProcessProfilingOutputThrowsWhenCreateFails) {
    // A decodable payload, so the scripted vclProfilingCreate failure is the only reason to throw.
    fake.profilingPayload.assign(sizeof(ze_profiling_layer_info), 0);
    auto compiler = makeCompiler();
    fake.failWith("vclProfilingCreate", VCL_RESULT_ERROR_UNKNOWN);

    try {
        compiler->process_profiling_output({1}, {2});
        FAIL() << "Expected a throw on vclProfilingCreate failure";
    } catch (const ov::Exception& error) {
        EXPECT_NE(std::string(error.what()).find("vclProfilingCreate"), std::string::npos);
    }
    // Nothing was created, so nothing must be destroyed.
    EXPECT_EQ(fake.profilingDestroyCount, 0);
}

TEST_F(VCLCompilerImplTest, ProcessProfilingOutputThrowsWhenDestroyFails) {
    // The payload must decode successfully, otherwise the null-data guard throws first and the
    // destroy-failure path is never reached.
    fake.profilingPayload.assign(sizeof(ze_profiling_layer_info), 0);
    auto compiler = makeCompiler();
    fake.failWith("vclProfilingDestroy", VCL_RESULT_ERROR_UNKNOWN);

    try {
        compiler->process_profiling_output({1}, {2});
        FAIL() << "Expected a throw on vclProfilingDestroy failure";
    } catch (const ov::Exception& error) {
        EXPECT_NE(std::string(error.what()).find("vclProfilingDestroy"), std::string::npos);
    }
    // The decode ran to completion before destroy was attempted.
    EXPECT_EQ(fake.callCount("vclGetDecodedProfilingBuffer"), 1u);
    EXPECT_EQ(fake.profilingDestroyCount, 1);
}

//
// --- compile ---
//

TEST_F(VCLCompilerImplTest, CompileProducesABlobAndACompatibilityString) {
    auto compiler = makeCompiler();
    auto config = makeConfig();

    const auto [tensor, compatibility] = compiler->compile(makeModel(), config);

    EXPECT_EQ(fake.callCount("vclAllocatedExecutableCreate4"), 1u);
    EXPECT_GT(tensor.get_byte_size(), 0u);
    ASSERT_TRUE(compatibility.has_value());
    EXPECT_EQ(*compatibility, "fake-compat");
}

TEST_F(VCLCompilerImplTest, CompileBuildFlagsAreIoInfoThenSpaceThenSerializedConfig) {
    // This is the plugin's actual contract with the compiler; nothing else pins it down.
    auto compiler = makeCompiler();
    // compile() stamps the resolved serializer version into a local copy of the config, but only
    // when the compiler advertises the option. Model a compiler that does not, so the flags are a
    // pure function of the config the test holds. The stamping branch is covered separately.
    fake.unsupportedOptions.insert(std::string(ov::intel_npu::model_serializer_version.name()));
    auto config = makeConfig();
    const auto model = makeModel();

    const auto [tensor, compatibility] = compiler->compile(model, config);
    (void)tensor;
    (void)compatibility;

    ze_graph_compiler_version_info_t compilerVersion{};
    compilerVersion.major = fake.propertiesVersion.major;
    compilerVersion.minor = fake.propertiesVersion.minor;

    const auto isSupported = [&compiler](const std::string& name) {
        return compiler->is_option_supported(name);
    };
    const std::string expected = ::intel_npu::compiler_utils::serializeIOInfo(model, true) + " " +
                                 ::intel_npu::compiler_utils::serializeConfig(config, compilerVersion, isSupported);

    ASSERT_EQ(fake.buildFlags.size(), 1u);
    EXPECT_EQ(fake.buildFlags[0], expected);
}

TEST_F(VCLCompilerImplTest, CompileStampsTheResolvedSerializerVersionWhenAdvertised) {
    // compile() serializes the IR first, then writes back the serializer version it actually used
    // so the compiler parses the IR the same way. The value is serializeIR's choice, so pin the key.
    auto compiler = makeCompiler();
    const auto [tensor, compatibility] = compiler->compile(makeModel(), makeConfig());
    (void)tensor;
    (void)compatibility;

    const std::string key = std::string(ov::intel_npu::model_serializer_version.name()) + "=\"";
    ASSERT_EQ(fake.buildFlags.size(), 1u);
    EXPECT_NE(fake.buildFlags[0].find(key), std::string::npos);
}

TEST_F(VCLCompilerImplTest, CompileOmitsTheSerializerVersionWhenNotAdvertised) {
    // Sending an option the compiler does not know about is a build-flag parse error, so the
    // write-back must be gated on is_option_supported.
    auto compiler = makeCompiler();
    fake.unsupportedOptions.insert(std::string(ov::intel_npu::model_serializer_version.name()));

    const auto [tensor, compatibility] = compiler->compile(makeModel(), makeConfig());
    (void)tensor;
    (void)compatibility;

    ASSERT_EQ(fake.buildFlags.size(), 1u);
    EXPECT_EQ(fake.buildFlags[0].find(ov::intel_npu::model_serializer_version.name()), std::string::npos);
}

TEST_F(VCLCompilerImplTest, CompileReturnsTheAlignedAllocatorSizeNotTheVclBlobSize) {
    // VCL reports the logical blob size; the tensor must expose the page-aligned allocation, because
    // that is what was actually reserved and what the deleter will free.
    fake.blobPayload.assign(5, 0xAB);
    auto compiler = makeCompiler();
    auto config = makeConfig();

    const auto [tensor, compatibility] = compiler->compile(makeModel(), config);
    (void)compatibility;

    const size_t alignedExpected = ::intel_npu::utils::align_size_to_standard_page_size(5);
    EXPECT_EQ(tensor.get_byte_size(), alignedExpected);
    EXPECT_NE(tensor.get_byte_size(), 5u);
    // The payload still lands at the start of the buffer.
    EXPECT_EQ(static_cast<const uint8_t*>(tensor.data())[0], 0xAB);
}

TEST_F(VCLCompilerImplTest, CompileDestroysTheExecutableOnTheSuccessPath) {
    auto compiler = makeCompiler();
    auto config = makeConfig();

    const auto result = compiler->compile(makeModel(), config);
    (void)result;

    EXPECT_EQ(fake.executableDestroyCount, 1);
}

TEST_F(VCLCompilerImplTest, CompileCleansUpAllocationsWhenExecutableCreationFails) {
    auto compiler = makeCompiler();
    auto config = makeConfig();
    fake.logString = "create4-failed";
    fake.failWith("vclAllocatedExecutableCreate4", VCL_RESULT_ERROR_OUT_OF_MEMORY);

    try {
        compiler->compile(makeModel(), config);
        FAIL() << "Expected compile to throw";
    } catch (const ov::Exception& error) {
        const std::string what = error.what();
        EXPECT_NE(what.find("vclAllocatedExecutableCreate4"), std::string::npos) << what;
        EXPECT_NE(what.find("create4-failed"), std::string::npos) << what;
    }
}

TEST_F(VCLCompilerImplTest, CompileDestroysTheExecutableWhenTheCompatibilityLookupThrows) {
    auto compiler = makeCompiler();
    auto config = makeConfig();
    // A hard failure (not UNSUPPORTED_FEATURE) propagates, but must not leak the executable.
    fake.failWith("vclExecutableGetCompatibilityString", VCL_RESULT_ERROR_UNKNOWN);

    EXPECT_THROW(compiler->compile(makeModel(), config), ov::Exception);
    EXPECT_EQ(fake.executableDestroyCount, 1);
}

TEST_F(VCLCompilerImplTest, CompileTreatsUnsupportedCompatibilityStringAsAbsentNotAnError) {
    auto compiler = makeCompiler();
    auto config = makeConfig();
    fake.compatibilityString.reset();  // makes the fake report UNSUPPORTED_FEATURE

    const auto [tensor, compatibility] = compiler->compile(makeModel(), config);
    (void)tensor;

    EXPECT_FALSE(compatibility.has_value());
    EXPECT_EQ(fake.executableDestroyCount, 1);
}

TEST_F(VCLCompilerImplTest, CompileTrimsTheTrailingNulFromTheCompatibilityString) {
    auto compiler = makeCompiler();
    auto config = makeConfig();
    fake.compatibilityString = std::string("compat-value");

    const auto [tensor, compatibility] = compiler->compile(makeModel(), config);
    (void)tensor;

    ASSERT_TRUE(compatibility.has_value());
    // No embedded NUL should survive into the string.
    EXPECT_EQ(*compatibility, "compat-value");
    EXPECT_EQ(compatibility->size(), std::string("compat-value").size());
}

TEST_F(VCLCompilerImplTest, CompileThrowsOnZeroSizedCompatibilityString) {
    auto compiler = makeCompiler();
    auto config = makeConfig();
    fake.compatibilityString = std::string("ignored");
    fake.compatibilityStringSizeOverride = 0u;

    EXPECT_THROW(compiler->compile(makeModel(), config), ov::Exception);
    EXPECT_EQ(fake.executableDestroyCount, 1);
}

TEST_F(VCLCompilerImplTest, CompileThrowsWhenExecutableDestroyFails) {
    auto compiler = makeCompiler();
    auto config = makeConfig();
    fake.failWith("vclExecutableDestroy", VCL_RESULT_ERROR_UNKNOWN);

    EXPECT_THROW(compiler->compile(makeModel(), config), ov::Exception);
}

TEST_F(VCLCompilerImplTest, CompileThrowsWhenTheLibraryIsBelowTheSupportedFloor) {
    fake.reportedCompilerVersion = {VCL_COMPILER_VERSION_MAJOR - 1, 0};
    auto compiler = makeCompiler();
    auto config = makeConfig();

    EXPECT_THROW(compiler->compile(makeModel(), config), ov::Exception);
    // The version gate fires before any executable is created.
    EXPECT_FALSE(fake.called("vclAllocatedExecutableCreate4"));
}

TEST_F(VCLCompilerImplTest, QueryThrowsWhenTheLibraryIsBelowTheSupportedFloor) {
    // query() negotiates the version like compile() does, so it must apply the same floor: an
    // unsupported library would otherwise be asked to parse an IR it cannot understand.
    fake.reportedCompilerVersion = {VCL_COMPILER_VERSION_MAJOR - 1, 0};
    auto compiler = makeCompiler();
    auto config = makeConfig();

    EXPECT_THROW(compiler->query(makeModel(), config), ov::Exception);
    // The version gate fires before the query handle is created.
    EXPECT_FALSE(fake.called("vclQueryNetworkCreate"));
}

//
// --- compileWsOneShot ---
//

TEST_F(VCLCompilerImplTest, CompileWsOneShotReturnsOneTensorPerAllocation) {
    fake.wsBlobSizes = {8, 16, 32};
    auto compiler = makeCompiler();
    auto config = makeConfig();

    const auto [tensors, compatibility] = compiler->compileWsOneShot(makeModel(), config);

    EXPECT_EQ(fake.callCount("vclAllocatedExecutableCreateWSOneShot2"), 1u);
    ASSERT_EQ(tensors.size(), 3u);
    ASSERT_TRUE(compatibility.has_value());
    EXPECT_EQ(*compatibility, "fake-compat");
}

TEST_F(VCLCompilerImplTest, CompileWsOneShotOrdersInitSchedulesBeforeMain) {
    // The adapter consumes the last tensor as the main schedule, so allocation order is load-bearing.
    fake.wsBlobSizes = {8, 16, 4096 * 3};
    auto compiler = makeCompiler();
    auto config = makeConfig();

    const auto [tensors, compatibility] = compiler->compileWsOneShot(makeModel(), config);
    (void)compatibility;

    ASSERT_EQ(tensors.size(), 3u);
    EXPECT_EQ(tensors.back().get_byte_size(), ::intel_npu::utils::align_size_to_standard_page_size(4096 * 3));
}

TEST_F(VCLCompilerImplTest, CompileWsOneShotThrowsAndDestroysExecutableWhenNothingWasAllocated) {
    fake.wsBlobSizes.clear();  // no allocations -> m_info stays empty
    auto compiler = makeCompiler();
    auto config = makeConfig();

    try {
        compiler->compileWsOneShot(makeModel(), config);
        FAIL() << "Expected compileWsOneShot to throw";
    } catch (const ov::Exception& error) {
        EXPECT_NE(std::string(error.what()).find("blobCount is zero"), std::string::npos) << error.what();
    }
    EXPECT_EQ(fake.executableDestroyCount, 1);
}

TEST_F(VCLCompilerImplTest, CompileWsOneShotThrowsWhenCreationFails) {
    auto compiler = makeCompiler();
    auto config = makeConfig();
    fake.failWith("vclAllocatedExecutableCreateWSOneShot2", VCL_RESULT_ERROR_UNKNOWN);

    EXPECT_THROW(compiler->compileWsOneShot(makeModel(), config), ov::Exception);
}

//
// --- compileWsIterative ---
//

TEST_F(VCLCompilerImplTest, CompileWsIterativeGoesThroughTheSingleBlobPath) {
    auto compiler = makeCompiler();
    auto config = makeConfig();

    const auto [tensor, compatibility] = compiler->compileWsIterative(makeModel(), config, 2);
    (void)compatibility;

    EXPECT_EQ(fake.callCount("vclAllocatedExecutableCreate4"), 1u);
    EXPECT_GT(tensor.get_byte_size(), 0u);
}

//
// --- query ---
//

TEST_F(VCLCompilerImplTest, QueryParsesSupportedLayersAndKeysThemToNPU) {
    const std::string payload = "<Parameter_0><Add_1>";
    fake.queryResultBuffer.assign(payload.begin(), payload.end());
    auto compiler = makeCompiler();
    auto config = makeConfig();

    const auto supported = compiler->query(makeModel(), config);

    ASSERT_EQ(supported.size(), 2u);
    ASSERT_TRUE(supported.count("Parameter_0"));
    ASSERT_TRUE(supported.count("Add_1"));
    EXPECT_EQ(supported.at("Parameter_0"), "NPU");
    EXPECT_EQ(supported.at("Add_1"), "NPU");
}

TEST_F(VCLCompilerImplTest, QueryUsesTheTwoCallSizeProtocolAndDestroysTheHandle) {
    const std::string payload = "<Add_1>";
    fake.queryResultBuffer.assign(payload.begin(), payload.end());
    auto compiler = makeCompiler();
    auto config = makeConfig();

    const auto supported = compiler->query(makeModel(), config);
    (void)supported;

    EXPECT_EQ(fake.callCount("vclQueryNetworkCreate"), 1u);
    EXPECT_EQ(fake.callCount("vclQueryNetwork"), 2u);
    EXPECT_EQ(fake.queryDestroyCount, 1);
}

TEST_F(VCLCompilerImplTest, QueryReturnsAnEmptyMapForAnEmptyResult) {
    fake.queryResultBuffer.clear();
    auto compiler = makeCompiler();
    auto config = makeConfig();

    EXPECT_TRUE(compiler->query(makeModel(), config).empty());
}

TEST_F(VCLCompilerImplTest, QueryThrowsWhenCreationFails) {
    auto compiler = makeCompiler();
    auto config = makeConfig();
    fake.failWith("vclQueryNetworkCreate", VCL_RESULT_ERROR_INVALID_IR);

    EXPECT_THROW(compiler->query(makeModel(), config), ov::Exception);
    EXPECT_EQ(fake.queryDestroyCount, 0);
}

TEST_F(VCLCompilerImplTest, QueryThrowsWhenTheResultFetchFails) {
    auto compiler = makeCompiler();
    auto config = makeConfig();
    fake.failWith("vclQueryNetwork", VCL_RESULT_ERROR_UNKNOWN);

    EXPECT_THROW(compiler->query(makeModel(), config), ov::Exception);
}

TEST_F(VCLCompilerImplTest, QueryThrowsWhenDestroyFails) {
    const std::string payload = "<Add_1>";
    fake.queryResultBuffer.assign(payload.begin(), payload.end());
    auto compiler = makeCompiler();
    auto config = makeConfig();
    fake.failWith("vclQueryNetworkDestroy", VCL_RESULT_ERROR_UNKNOWN);

    EXPECT_THROW(compiler->query(makeModel(), config), ov::Exception);
}

}  // namespace
