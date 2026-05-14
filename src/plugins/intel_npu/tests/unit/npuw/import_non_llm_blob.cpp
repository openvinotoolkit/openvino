// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <iostream>

#include "compiled_model.hpp"
#include "intel_npu/config/config.hpp"
#include "intel_npu/config/npuw.hpp"
#include "model_builder.hpp"
#include "openvino/openvino.hpp"
#include "openvino/util/codec_xor.hpp"
#include "orc.hpp"
#include "serialization.hpp"

using ov::test::npuw::ModelBuilder;

class ImportNonLLMBlobTestNPUW : public ::testing::TestWithParam<ov::AnyMap> {
public:
    void SetUp() override {
        ModelBuilder mb;
        m_ov_model = mb.get_model_with_repeated_blocks_with_weightless_cache();
        m_props = {{"NPU_USE_NPUW", "YES"}, {"NPUW_DEVICES", "CPU"}};
    }

protected:
    static ov::EncryptionCallbacks xor_callbacks() {
        return {ov::util::codec_xor, ov::util::codec_xor};
    }

    static ov::Tensor make_input_tensor(const ov::Output<const ov::Node>& input) {
        ov::Tensor tensor(input.get_element_type(), input.get_shape());

        if (input.get_element_type() == ov::element::i32) {
            auto* data = tensor.data<int32_t>();
            for (std::size_t i = 0; i < tensor.get_size(); ++i) {
                data[i] = static_cast<int32_t>(i % 17);
            }
        } else if (input.get_element_type() == ov::element::i64) {
            auto* data = tensor.data<int64_t>();
            for (std::size_t i = 0; i < tensor.get_size(); ++i) {
                data[i] = static_cast<int64_t>(i % 17);
            }
        } else if (input.get_element_type() == ov::element::f32) {
            auto* data = tensor.data<float>();
            for (std::size_t i = 0; i < tensor.get_size(); ++i) {
                data[i] = static_cast<float>(i) * 0.25f;
            }
        } else if (input.get_element_type() == ov::element::f16) {
            auto* data = tensor.data<ov::float16>();
            for (std::size_t i = 0; i < tensor.get_size(); ++i) {
                data[i] = ov::float16(static_cast<float>(i) * 0.25f);
            }
        } else {
            std::memset(tensor.data(), 0, tensor.get_byte_size());
        }

        return tensor;
    }

    static std::vector<ov::Tensor> infer_outputs(ov::CompiledModel& model) {
        auto request = model.create_infer_request();
        for (const auto& input : model.inputs()) {
            request.set_tensor(input, make_input_tensor(input));
        }
        request.infer();

        std::vector<ov::Tensor> outputs;
        outputs.reserve(model.outputs().size());
        for (const auto& output : model.outputs()) {
            auto result = request.get_tensor(output);
            ov::Tensor copy(result.get_element_type(), result.get_shape());
            result.copy_to(copy);
            outputs.push_back(copy);
        }
        return outputs;
    }

    static void expect_outputs_equal(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual) {
        ASSERT_EQ(expected.size(), actual.size());
        for (std::size_t i = 0; i < expected.size(); ++i) {
            EXPECT_EQ(expected[i].get_element_type(), actual[i].get_element_type());
            EXPECT_EQ(expected[i].get_shape(), actual[i].get_shape());
            ASSERT_EQ(expected[i].get_byte_size(), actual[i].get_byte_size());
            EXPECT_EQ(std::memcmp(expected[i].data(), actual[i].data(), expected[i].get_byte_size()), 0);
        }
    }

    ModelBuilder model_builder;
    std::shared_ptr<ov::Model> m_ov_model;
    ov::AnyMap m_props;
    ov::Core m_core;
};

TEST_P(ImportNonLLMBlobTestNPUW, CacheModeOptimizeSpeed) {
    ov::AnyMap wai_props = GetParam();
    m_props.insert(wai_props.begin(), wai_props.end());
    m_props["CACHE_MODE"] = "OPTIMIZE_SPEED";

    auto compiled = m_core.compile_model(m_ov_model, "NPU", m_props);
    auto compiled_outputs = infer_outputs(compiled);

    std::stringstream blob;
    compiled.export_model(blob);
    EXPECT_TRUE(ov::npuw::orc::is_orc(blob).has_value());

    EXPECT_NO_THROW({
        auto imported = m_core.import_model(blob, "NPU", m_props);
        auto imported_outputs = infer_outputs(imported);
        expect_outputs_equal(compiled_outputs, imported_outputs);
    });
}

TEST_P(ImportNonLLMBlobTestNPUW, CacheModeOptimizeSpeedEncrypted) {
    ov::AnyMap wai_props = GetParam();
    m_props.insert(wai_props.begin(), wai_props.end());
    m_props["CACHE_MODE"] = "OPTIMIZE_SPEED";
    m_props[ov::cache_encryption_callbacks.name()] = xor_callbacks();

    auto compiled = m_core.compile_model(m_ov_model, "NPU", m_props);
    auto compiled_outputs = infer_outputs(compiled);

    std::stringstream blob;
    compiled.export_model(blob);
    EXPECT_TRUE(ov::npuw::orc::is_orc(blob).has_value());

    EXPECT_NO_THROW({
        auto imported = m_core.import_model(blob, "NPU", m_props);
        auto imported_outputs = infer_outputs(imported);
        expect_outputs_equal(compiled_outputs, imported_outputs);
    });
}

TEST_P(ImportNonLLMBlobTestNPUW, CacheModeOptimizeSpeedEncryptedRequiresDecryptCallback) {
    ov::AnyMap wai_props = GetParam();
    m_props.insert(wai_props.begin(), wai_props.end());
    m_props["CACHE_MODE"] = "OPTIMIZE_SPEED";
    m_props[ov::cache_encryption_callbacks.name()] = ov::EncryptionCallbacks{ov::util::codec_xor, nullptr};

    auto compiled = m_core.compile_model(m_ov_model, "NPU", m_props);

    std::stringstream blob;
    compiled.export_model(blob);
    EXPECT_TRUE(ov::npuw::orc::is_orc(blob).has_value());

    auto import_props = m_props;
    import_props.erase(ov::cache_encryption_callbacks.name());
    try {
        auto imported = m_core.import_model(blob, "NPU", import_props);
        (void)imported;
        FAIL() << "Expected encrypted ORC import to require a decrypt callback";
    } catch (const ov::Exception& ex) {
        EXPECT_NE(std::string(ex.what()).find("Blob is encrypted, but no decryption callback was provided"),
                  std::string::npos)
            << ex.what();
    }
}

TEST_P(ImportNonLLMBlobTestNPUW, CacheModeOptimizeSpeedEnsureCompatibility) {
    ov::AnyMap wai_props = GetParam();
    m_props.insert(wai_props.begin(), wai_props.end());
    m_props["CACHE_MODE"] = "OPTIMIZE_SPEED";
    m_props["NPUW_ENSURE_COMPATIBILITY"] = "YES";

    auto compiled = m_core.compile_model(m_ov_model, "NPU", m_props);
    std::stringstream blob;
    // This synthetic repeated-block model does not satisfy the phase-0 compatibility contract:
    // even with NPU-only placement it still exercises non-versioned behavior-owned state today,
    // so ENSURE_COMPATIBILITY is expected to reject export rather than produce a misleading blob.
    EXPECT_THROW(compiled.export_model(blob), ov::Exception);
}

TEST_P(ImportNonLLMBlobTestNPUW, CacheModeOptimizeSizeWithModelPtr) {
    ov::AnyMap wai_props = GetParam();
    m_props.insert(wai_props.begin(), wai_props.end());
    m_props["CACHE_MODE"] = "OPTIMIZE_SIZE";

    auto compiled = m_core.compile_model(m_ov_model, "NPU", m_props);
    auto compiled_outputs = infer_outputs(compiled);

    std::stringstream blob;
    compiled.export_model(blob);
    EXPECT_TRUE(ov::npuw::orc::is_orc(blob).has_value());

    EXPECT_NO_THROW({
        auto import_props = m_props;
        import_props[ov::hint::model.name()] = std::static_pointer_cast<const ov::Model>(m_ov_model);
        auto imported = m_core.import_model(blob, "NPU", import_props);
        auto imported_outputs = infer_outputs(imported);
        expect_outputs_equal(compiled_outputs, imported_outputs);
    });
}

TEST_P(ImportNonLLMBlobTestNPUW, CacheModeOptimizeSizeWithModelPtrEncryptedDifferentDecryptSize) {
    ov::AnyMap wai_props = GetParam();
    m_props.insert(wai_props.begin(), wai_props.end());
    m_props["CACHE_MODE"] = "OPTIMIZE_SIZE";
    m_props[ov::cache_encryption_callbacks.name()] =
        ov::EncryptionCallbacks{[](const std::string& unencrypted_blob) {
                                    std::string copy_blob = unencrypted_blob;
                                    copy_blob += "<encrypt-tail>";
                                    return ov::util::codec_xor(copy_blob);
                                },
                                [](const std::string& encrypted_blob) {
                                    std::string decrypted_blob = ov::util::codec_xor(encrypted_blob);
                                    decrypted_blob += "<decrypt-tail>";
                                    return decrypted_blob;
                                }};

    auto compiled = m_core.compile_model(m_ov_model, "NPU", m_props);

    std::stringstream blob;
    compiled.export_model(blob);
    EXPECT_TRUE(ov::npuw::orc::is_orc(blob).has_value());

    auto import_props = m_props;
    import_props[ov::hint::model.name()] = std::static_pointer_cast<const ov::Model>(m_ov_model);
    EXPECT_NO_THROW({
        auto imported = m_core.import_model(blob, "NPU", import_props);
        (void)imported;
    });
}

TEST_P(ImportNonLLMBlobTestNPUW, CacheModeOptimizeSizeWithModelPtrEnsureCompatibility) {
    ov::AnyMap wai_props = GetParam();
    m_props.insert(wai_props.begin(), wai_props.end());
    m_props["CACHE_MODE"] = "OPTIMIZE_SIZE";
    m_props["NPUW_ENSURE_COMPATIBILITY"] = "YES";

    auto compiled = m_core.compile_model(m_ov_model, "NPU", m_props);
    std::stringstream blob;
    EXPECT_THROW(compiled.export_model(blob), ov::Exception);
}

using ImportNonLLMNonWAIBlobTestNPUW = ImportNonLLMBlobTestNPUW;
TEST_P(ImportNonLLMNonWAIBlobTestNPUW, CacheModeOptimizeSizeNoModelPtr) {
    ov::AnyMap wai_props = GetParam();
    m_props.insert(wai_props.begin(), wai_props.end());
    m_props["CACHE_MODE"] = "OPTIMIZE_SIZE";

    auto compiled = m_core.compile_model(m_ov_model, "NPU", m_props);
    auto compiled_outputs = infer_outputs(compiled);

    std::stringstream blob;
    compiled.export_model(blob);
    EXPECT_TRUE(ov::npuw::orc::is_orc(blob).has_value());

    EXPECT_NO_THROW({
        auto imported = m_core.import_model(blob, "NPU", m_props);
        auto imported_outputs = infer_outputs(imported);
        expect_outputs_equal(compiled_outputs, imported_outputs);
    });
}

TEST_P(ImportNonLLMNonWAIBlobTestNPUW, CacheModeOptimizeSizeNoModelPtrEnsureCompatibility) {
    ov::AnyMap wai_props = GetParam();
    m_props.insert(wai_props.begin(), wai_props.end());
    m_props["CACHE_MODE"] = "OPTIMIZE_SIZE";
    m_props["NPUW_ENSURE_COMPATIBILITY"] = "YES";

    auto compiled = m_core.compile_model(m_ov_model, "NPU", m_props);
    std::stringstream blob;
    EXPECT_THROW(compiled.export_model(blob), ov::Exception);
}

using ImportNonLLMWAIBlobTestNPUW = ImportNonLLMBlobTestNPUW;
TEST_P(ImportNonLLMWAIBlobTestNPUW, CacheModeOptimizeSizeNoModelPtr) {
    ov::AnyMap wai_props = GetParam();
    m_props.insert(wai_props.begin(), wai_props.end());
    m_props["CACHE_MODE"] = "OPTIMIZE_SIZE";

    auto compiled = m_core.compile_model(m_ov_model, "NPU", m_props);

    std::stringstream blob;
    compiled.export_model(blob);
    EXPECT_TRUE(ov::npuw::orc::is_orc(blob).has_value());
    try {
        auto imported = m_core.import_model(blob, "NPU", m_props);
        FAIL() << "Expected import to throw when WAI weightless blob is imported without MODEL_PTR/WEIGHTS_PATH";
    } catch (const ov::Exception& ex) {
        const std::string what = ex.what();
        const bool has_expected_text = what.find("Blob is weightless") != std::string::npos &&
                                       what.find("WEIGHTS_PATH") != std::string::npos &&
                                       what.find("MODEL_PTR") != std::string::npos;
        EXPECT_TRUE(has_expected_text) << "Unexpected exception message: " << what;
    }
}

TEST_P(ImportNonLLMWAIBlobTestNPUW, CacheModeOptimizeSizeNoModelPtrEnsureCompatibility) {
    ov::AnyMap wai_props = GetParam();
    m_props.insert(wai_props.begin(), wai_props.end());
    m_props["CACHE_MODE"] = "OPTIMIZE_SIZE";
    m_props["NPUW_ENSURE_COMPATIBILITY"] = "YES";

    auto compiled = m_core.compile_model(m_ov_model, "NPU", m_props);

    std::stringstream blob;
    EXPECT_THROW(compiled.export_model(blob), ov::Exception);
}

INSTANTIATE_TEST_SUITE_P(Only_NPU_USE_NPUW, ImportNonLLMBlobTestNPUW, testing::Values(ov::AnyMap{}));
INSTANTIATE_TEST_SUITE_P(NPUW_FOLD, ImportNonLLMBlobTestNPUW, testing::Values(ov::AnyMap{{"NPUW_FOLD", "YES"}}));
INSTANTIATE_TEST_SUITE_P(NPUW_CWAI, ImportNonLLMBlobTestNPUW, testing::Values(ov::AnyMap{{"NPUW_CWAI", "YES"}}));

INSTANTIATE_TEST_SUITE_P(NPUW_NON_WAI,
                         ImportNonLLMNonWAIBlobTestNPUW,
                         testing::Values(ov::AnyMap{}, ov::AnyMap{{"NPUW_FUNCALL_FOR_ALL", "YES"}}));
INSTANTIATE_TEST_SUITE_P(NPUW_WAI,
                         ImportNonLLMWAIBlobTestNPUW,
                         testing::Values(ov::AnyMap{{"NPUW_FOLD", "YES"}}, ov::AnyMap{{"NPUW_CWAI", "YES"}}));
