// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Export / import of a TP_GPU compiled model.
//
// The blob is an internal cache artifact: it carries one intel_gpu blob per
// rank and only loads back onto the topology it was produced on.  What these
// cases pin down is that a restored model computes the same thing as the one it
// was exported from -- the collectives are rebuilt from scratch on import, so a
// mistake there shows up as wrong numbers rather than as a load failure -- and
// that a blob which does not belong is refused instead of half-loaded.
//
// Needs two or more GPUs; the suite skips otherwise.

#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <sstream>
#include <string>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/runtime/core.hpp"
#include "tp_gpu/properties.hpp"
#include "tp_test_models.hpp"

namespace ov::tp_gpu::tests {
namespace {

constexpr size_t sequence_length = 16;

/// Two ranks everywhere: the blob layout does not vary with the world size,
/// and a third rank only makes the test slower.
constexpr uint32_t world_size = 2;

class TPGpuSerializationTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        try {
            gpu_count = ov::Core{}.get_property("GPU", ov::available_devices).size();
        } catch (const std::exception& e) {
            gpu_error = e.what();
        }
    }

    void SetUp() override {
        if (gpu_count < world_size)
            GTEST_SKIP() << "need >= " << world_size << " GPUs, found " << gpu_count << gpu_error;
    }

    void TearDown() override {
        if (!m_cache_dir.empty()) {
            std::error_code ec;
            std::filesystem::remove_all(m_cache_dir, ec);
        }
    }

    /// A cache directory unique to the running test, removed in TearDown.
    std::string cache_dir() {
        if (m_cache_dir.empty()) {
            const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
            m_cache_dir =
                (std::filesystem::temp_directory_path() / ("ov_tp_gpu_cache_" + std::string(info->name()))).string();
            std::error_code ec;
            std::filesystem::remove_all(m_cache_dir, ec);
        }
        return m_cache_dir;
    }

    static ov::AnyMap tp_config() {
        return {ov::hint::inference_precision(ov::element::f32), ov::tp_gpu::tp_size(world_size)};
    }

    static ov::Tensor make_input(const BlockConfig& config) {
        ov::test::utils::InputGenerateData data(-1, 2, 32, 7);
        return ov::test::utils::create_and_fill_tensor(ov::element::f32,
                                                       ov::Shape{1, sequence_length, config.hidden},
                                                       data);
    }

    static ov::Tensor infer_to_host(ov::CompiledModel& compiled, const ov::Tensor& input) {
        auto request = compiled.create_infer_request();
        request.set_input_tensor(input);
        request.infer();

        // The output tensor belongs to the request; copy it out before the
        // request goes away at the end of this call.
        const auto& device = request.get_output_tensor(0);
        ov::Tensor host(device.get_element_type(), device.get_shape());
        device.copy_to(host);
        return host;
    }

    /// The import path rebuilds the collectives but reuses the very same rank
    /// blobs, so both sides run bit-identical kernels over bit-identical
    /// weights.  Anything but an exact match means the restored model is not
    /// the model that was exported.
    static void expect_identical(const ov::Tensor& exported, const ov::Tensor& imported) {
        ov::test::utils::compare(exported, imported, 0.0, 0.0);
    }

    ov::Core core;

    static size_t gpu_count;
    static std::string gpu_error;

private:
    std::string m_cache_dir;
};

size_t TPGpuSerializationTest::gpu_count = 0;
std::string TPGpuSerializationTest::gpu_error;

TEST_F(TPGpuSerializationTest, ImportedModelMatchesExportedOne) {
    BlockConfig config;
    auto model = make_transformer_block(config);
    const auto input = make_input(config);

    std::stringstream blob;
    ov::Tensor exported;
    {
        auto compiled = core.compile_model(model, "TP_GPU", tp_config());
        exported = infer_to_host(compiled, input);
        compiled.export_model(blob);
    }
    ASSERT_GT(blob.str().size(), 0u);

    auto imported = core.import_model(blob, "TP_GPU", tp_config());
    EXPECT_TRUE(imported.get_property(ov::loaded_from_cache));
    expect_identical(exported, infer_to_host(imported, input));
}

// The imported model has no IR behind it, so its ports come from rank 0.  They
// still have to describe the user model: a caller that binds tensors by name or
// by index against the exported model must keep working against the imported
// one.
TEST_F(TPGpuSerializationTest, ImportedModelKeepsPorts) {
    BlockConfig config;
    auto model = make_transformer_block(config);

    std::stringstream blob;
    auto compiled = core.compile_model(model, "TP_GPU", tp_config());
    compiled.export_model(blob);

    auto imported = core.import_model(blob, "TP_GPU", tp_config());

    ASSERT_EQ(imported.inputs().size(), compiled.inputs().size());
    ASSERT_EQ(imported.outputs().size(), compiled.outputs().size());
    for (size_t i = 0; i < compiled.inputs().size(); ++i) {
        EXPECT_EQ(imported.input(i).get_element_type(), compiled.input(i).get_element_type());
        EXPECT_EQ(imported.input(i).get_partial_shape(), compiled.input(i).get_partial_shape());
        EXPECT_EQ(imported.input(i).get_names(), compiled.input(i).get_names());
    }
    for (size_t i = 0; i < compiled.outputs().size(); ++i) {
        EXPECT_EQ(imported.output(i).get_element_type(), compiled.output(i).get_element_type());
        EXPECT_EQ(imported.output(i).get_partial_shape(), compiled.output(i).get_partial_shape());
    }
}

// The end users never call export_model themselves; they set ov::cache_dir and
// expect the second compile to be a load.  That path goes through Core, which
// only offers it to a plugin that advertises EXPORT_IMPORT and a non-empty set
// of caching properties -- neither of which the plugin can be asked about here
// directly, so the observable effect stands in for both.
TEST_F(TPGpuSerializationTest, CacheDirTurnsSecondCompileIntoImport) {
    BlockConfig config;
    auto model = make_transformer_block(config);
    const auto input = make_input(config);

    core.set_property(ov::cache_dir(cache_dir()));

    auto first = core.compile_model(model, "TP_GPU", tp_config());
    EXPECT_FALSE(first.get_property(ov::loaded_from_cache));
    const auto reference = infer_to_host(first, input);

    auto second = core.compile_model(model, "TP_GPU", tp_config());
    EXPECT_TRUE(second.get_property(ov::loaded_from_cache));
    expect_identical(reference, infer_to_host(second, input));

    core.set_property(ov::cache_dir(""));
}

// A cache entry is keyed on the rank topology as well as on the model: the same
// IR split two ways and three ways produces different blobs, and handing one to
// the other would run a graph whose collectives expect ranks that are not there.
TEST_F(TPGpuSerializationTest, DifferentWorldSizeGetsItsOwnCacheEntry) {
    if (gpu_count < 3)
        GTEST_SKIP() << "need >= 3 GPUs to compare two world sizes, found " << gpu_count;

    BlockConfig config;
    auto model = make_transformer_block(config);

    core.set_property(ov::cache_dir(cache_dir()));

    auto two_ranks = core.compile_model(model, "TP_GPU", {ov::tp_gpu::tp_size(2u)});
    EXPECT_FALSE(two_ranks.get_property(ov::loaded_from_cache));

    auto three_ranks = core.compile_model(model, "TP_GPU", {ov::tp_gpu::tp_size(3u)});
    EXPECT_FALSE(three_ranks.get_property(ov::loaded_from_cache));

    core.set_property(ov::cache_dir(""));
}

// Explicit import bypasses the cache key, so the mismatch has to be caught by
// the blob itself.
TEST_F(TPGpuSerializationTest, ImportRejectsMismatchedTopology) {
    if (gpu_count < 3)
        GTEST_SKIP() << "need >= 3 GPUs to request a topology the blob was not built for, found " << gpu_count;

    BlockConfig config;
    auto model = make_transformer_block(config);

    std::stringstream blob;
    core.compile_model(model, "TP_GPU", tp_config()).export_model(blob);

    EXPECT_THROW(core.import_model(blob, "TP_GPU", {ov::tp_gpu::tp_size(3u)}), ov::Exception);
}

TEST_F(TPGpuSerializationTest, ImportRejectsForeignBlob) {
    std::stringstream blob;
    blob << "this is not a TP_GPU blob, not even close";
    EXPECT_THROW(core.import_model(blob, "TP_GPU", tp_config()), ov::Exception);
}

// A blob that ends inside the TP header must be refused before any of it is
// acted on -- the world size and the per-rank lengths read from it drive both
// an allocation and a seek.
TEST_F(TPGpuSerializationTest, ImportRejectsTruncatedHeader) {
    BlockConfig config;
    auto model = make_transformer_block(config);

    std::stringstream blob;
    core.compile_model(model, "TP_GPU", tp_config()).export_model(blob);

    const auto full = blob.str();
    // Past the magic and the version, short of the device names.
    ASSERT_GT(full.size(), 14u);
    std::stringstream truncated(full.substr(0, 14));

    EXPECT_THROW(core.import_model(truncated, "TP_GPU", tp_config()), ov::Exception);
}

}  // namespace
}  // namespace ov::tp_gpu::tests
