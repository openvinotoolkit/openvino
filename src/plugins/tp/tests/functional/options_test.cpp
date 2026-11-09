// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// The TP plugin's options end to end: what it lists, what it forwards to the
// rank contexts, and the state API on a split model.
//
// Needs two rank devices (rank_devices.hpp); the suite skips otherwise.

#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/util/container_util.hpp"
#include "rank_devices.hpp"
#include "tp/internal_properties.hpp"
#include "tp_test_models.hpp"

namespace ov::tp::tests {
namespace {

using ::testing::HasSubstr;

constexpr size_t world_size = 2;

ov::Tensor make_input(const BlockConfig& config, size_t length = 16, int32_t seed = 7) {
    ov::test::utils::InputGenerateData data(-1, 2, 32, seed);
    return ov::test::utils::create_and_fill_tensor(ov::element::f32, ov::Shape{1, length, config.hidden}, data);
}

ov::Tensor to_host(const ov::Tensor& tensor) {
    ov::Tensor host(tensor.get_element_type(), tensor.get_shape());
    tensor.copy_to(host);
    return host;
}

ov::Tensor infer(ov::InferRequest& request, const ov::Tensor& input) {
    request.set_input_tensor(input);
    request.infer();
    return to_host(request.get_output_tensor(0));
}

void expect_same_bits(const ov::Tensor& expected, const ov::Tensor& actual) {
    ASSERT_EQ(expected.get_shape(), actual.get_shape());
    EXPECT_EQ(std::memcmp(expected.data(), actual.data(), expected.get_byte_size()), 0);
}

/// Sets an environment variable for the lifetime of the object.
class ScopedEnv {
public:
    ScopedEnv(const char* name, const std::string& value) : m_name(name) {
#ifdef _WIN32
        _putenv_s(name, value.c_str());
#else
        setenv(name, value.c_str(), 1);
#endif
    }
    ~ScopedEnv() {
#ifdef _WIN32
        _putenv_s(m_name, "");
#else
        unsetenv(m_name);
#endif
    }

private:
    const char* m_name;
};

// Listed by the plugin: the release options only; tuning knobs and debug options are not.
TEST(TPOptionsTest, ListsItsReleaseOptions) {
    ov::Core core;
    const auto supported = core.get_property("TP", ov::supported_properties);
    for (const auto* name : {ov::device::priorities.name(),
                             ov::tp::tp_strategy.name(),
                             ov::tp::collective_backend.name(),
                             ov::tp::internal::communication_timeout_ms.name(),
                             ov::tp::internal::comm_buffer_size.name()}) {
        EXPECT_TRUE(ov::util::contains(supported, name)) << name;
    }
    for (const auto* name : {ov::tp::input_stage_max_bytes.name(),
                             ov::tp::verbose.name(),
                             ov::tp::shard_only.name(),
                             ov::tp::dump_sharding_states.name()}) {
        EXPECT_FALSE(ov::util::contains(supported, name)) << name;
    }
}

TEST(TPOptionsTest, RejectsANegativeTimeout) {
    ov::Core core;
    OV_EXPECT_THROW(core.compile_model(make_transformer_block(BlockConfig{}),
                                       "TP:" + rank_device_type() + ".0," + rank_device_type() + ".1",
                                       {{ov::tp::internal::communication_timeout_ms.name(), -1}}),
                    ov::Exception,
                    HasSubstr("COMMUNICATION_TIMEOUT_MS"));
}

class TPOptionsOnDevicesTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (const auto missing = missing_rank_devices(world_size); !missing.empty())
            GTEST_SKIP() << missing;
    }

    ov::Tensor run(const std::shared_ptr<ov::Model>& model, const ov::Tensor& input, const ov::AnyMap& config = {}) {
        auto request = core.compile_model(model, tp_device(world_size), config).create_infer_request();
        return infer(request, input);
    }

    ov::Core core;
};

// A staging smaller than a prefill's messages splits them into chunks; on two
// ranks the sums are the same, so the result is bit for bit the same.
TEST_F(TPOptionsOnDevicesTest, ACommBufferSizeReachesTheRankContexts) {
    const BlockConfig config;
    const auto model = make_transformer_block(config);
    const auto input = make_input(config);
    const auto whole = run(model, input);
    const auto chunked = run(model, input, {{ov::tp::internal::comm_buffer_size.name(), size_t{4096}}});
    expect_same_bits(whole, chunked);
}

// A staging no device can allocate is refused by name before any context exists.
TEST_F(TPOptionsOnDevicesTest, RejectsAStagingLargerThanADeviceAllocates) {
    OV_EXPECT_THROW(core.compile_model(make_transformer_block(BlockConfig{}),
                                       tp_device(world_size),
                                       {{ov::tp::internal::comm_buffer_size.name(), size_t{1} << 60}}),
                    ov::Exception,
                    HasSubstr("ov_internal_tp_comm_buffer_size of 1152921504606846976 bytes is more than"));
}

// Inputs copied through plugin-owned host memory or handed over as they are: the same result.
TEST_F(TPOptionsOnDevicesTest, InputStagingDoesNotChangeTheResult) {
    const BlockConfig config;
    const auto model = make_transformer_block(config);
    const auto input = make_input(config);
    const auto staged = run(model, input);
    ScopedEnv never_staged("OV_TP_INPUT_STAGE_MAX_BYTES", "0");
    expect_same_bits(staged, run(model, input));
}

// After a reset every rank's slice of the state starts over: a prompt gives what it gave first.
TEST_F(TPOptionsOnDevicesTest, ResetStateStartsEveryRankOver) {
    BlockConfig config;
    config.stateful = true;
    auto compiled = core.compile_model(make_transformer_block(config), tp_device(world_size));
    auto request = compiled.create_infer_request();
    const auto prompt = make_input(config, 16, 1);
    const auto other = make_input(config, 16, 2);

    const auto first = infer(request, prompt);
    const auto after_prompt = infer(request, other);
    request.reset_state();
    expect_same_bits(first, infer(request, prompt));

    // The run in between did read the state, or the reset proved nothing.
    auto fresh = compiled.create_infer_request();
    const auto alone = infer(fresh, other);
    EXPECT_NE(std::memcmp(alone.data(), after_prompt.data(), alone.get_byte_size()), 0);
}

TEST_F(TPOptionsOnDevicesTest, AWholeVocabularyProjectionGivesTheSameLogits) {
    const BlockConfig config;
    const auto model = make_block_with_lm_head(config, 768);
    const auto input = make_input(config);
    const ov::AnyMap f32{ov::hint::inference_precision(ov::element::f32)};
    const auto sharded = run(model, input, f32);
    ScopedEnv whole("OV_TP_DISABLE_LM_HEAD_SHARDING", "YES");
    ov::test::utils::compare(sharded, run(model, input, f32), 1e-5, 1e-5);
}

// The option changes the rank models, so an entry cached with it is not loaded without it.
TEST_F(TPOptionsOnDevicesTest, AWholeVocabularyProjectionGetsItsOwnCacheEntry) {
    const auto dir = std::filesystem::temp_directory_path() / "ov_tp_lm_head_cache_test";
    std::filesystem::remove_all(dir);
    core.set_property(ov::cache_dir(dir.string()));
    const auto model = make_block_with_lm_head(BlockConfig{}, 768);
    const auto from_cache = [&] {
        return core.compile_model(model, tp_device(world_size)).get_property(ov::loaded_from_cache);
    };
    EXPECT_FALSE(from_cache());
    {
        ScopedEnv whole("OV_TP_DISABLE_LM_HEAD_SHARDING", "YES");
        EXPECT_FALSE(from_cache());
        EXPECT_TRUE(from_cache());
    }
    EXPECT_TRUE(from_cache());
    core.set_property(ov::cache_dir(""));
    std::filesystem::remove_all(dir);
}

#ifdef ENABLE_TP_DEBUG_CAPS

TEST_F(TPOptionsOnDevicesTest, DumpsTheShardingStates) {
    const auto dir = std::filesystem::temp_directory_path() / "ov_tp_sharding_states_test";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    {
        ScopedEnv dump("OV_TP_DUMP_SHARDING_STATES", dir.string());
        core.compile_model(make_transformer_block(BlockConfig{}), tp_device(world_size));
    }
    std::vector<std::filesystem::path> files;
    for (const auto& entry : std::filesystem::directory_iterator(dir))
        files.push_back(entry.path());
    std::filesystem::remove_all(dir);
    ASSERT_EQ(files.size(), 1u);
    EXPECT_THAT(files.front().filename().string(), HasSubstr("_2ranks.txt"));
}

#endif  // ENABLE_TP_DEBUG_CAPS

}  // namespace
}  // namespace ov::tp::tests
