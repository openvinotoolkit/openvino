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

using ov::test::npuw::ModelBuilder;

class ImportNonLLMBlobTestNPUW : public ::testing::TestWithParam<ov::AnyMap> {
public:
    void SetUp() override {
        ModelBuilder mb;
        m_ov_model = mb.get_model_with_repeated_blocks_with_weightless_cache();
        m_props = {{"NPU_USE_NPUW", "YES"}, {"NPUW_DEVICES", "CPU"}};
    }

protected:
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

    std::stringstream blob;
    compiled.export_model(blob);

    EXPECT_NO_THROW({
        auto imported = m_core.import_model(blob, "NPU", m_props);
        imported.create_infer_request();
    });
}

TEST_P(ImportNonLLMBlobTestNPUW, CacheModeOptimizeSizeWithModelPtr) {
    ov::AnyMap wai_props = GetParam();
    m_props.insert(wai_props.begin(), wai_props.end());
    m_props["CACHE_MODE"] = "OPTIMIZE_SIZE";

    auto compiled = m_core.compile_model(m_ov_model, "NPU", m_props);

    std::stringstream blob;
    compiled.export_model(blob);

    EXPECT_NO_THROW({
        auto import_props = m_props;
        import_props[ov::hint::model.name()] = std::static_pointer_cast<const ov::Model>(m_ov_model);
        auto imported = m_core.import_model(blob, "NPU", import_props);
        imported.create_infer_request();
    });
}

using ImportNonLLMNonWAIBlobTestNPUW = ImportNonLLMBlobTestNPUW;
TEST_P(ImportNonLLMNonWAIBlobTestNPUW, CacheModeOptimizeSizeNoModelPtr) {
    ov::AnyMap wai_props = GetParam();
    m_props.insert(wai_props.begin(), wai_props.end());
    m_props["CACHE_MODE"] = "OPTIMIZE_SIZE";

    auto compiled = m_core.compile_model(m_ov_model, "NPU", m_props);

    std::stringstream blob;
    compiled.export_model(blob);

    EXPECT_NO_THROW({
        auto imported = m_core.import_model(blob, "NPU", m_props);
        imported.create_infer_request();
    });
}

using ImportNonLLMWAIBlobTestNPUW = ImportNonLLMBlobTestNPUW;
TEST_P(ImportNonLLMWAIBlobTestNPUW, CacheModeOptimizeSizeNoModelPtr) {
    ov::AnyMap wai_props = GetParam();
    m_props.insert(wai_props.begin(), wai_props.end());
    m_props["CACHE_MODE"] = "OPTIMIZE_SIZE";

    auto compiled = m_core.compile_model(m_ov_model, "NPU", m_props);

    std::stringstream blob;
    compiled.export_model(blob);
    try {
        auto imported = m_core.import_model(blob, "NPU", m_props);
        FAIL() << "Expected import to throw when WAI weightless blob is imported without MODEL_PTR/WEIGHTS_PATH";
    } catch (const ov::Exception& ex) {
        const std::string what = ex.what();
        const bool has_expected_text =
            what.find("Blob is weightless") != std::string::npos &&
            what.find("WEIGHTS_PATH") != std::string::npos &&
            what.find("MODEL_PTR") != std::string::npos;
        EXPECT_TRUE(has_expected_text) << "Unexpected exception message: " << what;
    }
}

INSTANTIATE_TEST_SUITE_P(Only_NPU_USE_NPUW, ImportNonLLMBlobTestNPUW,
    testing::Values(ov::AnyMap{}));
INSTANTIATE_TEST_SUITE_P(NPUW_FOLD, ImportNonLLMBlobTestNPUW,
    testing::Values(ov::AnyMap{{"NPUW_FOLD", "YES"}}));
INSTANTIATE_TEST_SUITE_P(NPUW_CWAI, ImportNonLLMBlobTestNPUW,
    testing::Values(ov::AnyMap{{"NPUW_CWAI", "YES"}}));

INSTANTIATE_TEST_SUITE_P(NPUW_NON_WAI, ImportNonLLMNonWAIBlobTestNPUW,
    testing::Values(ov::AnyMap{}, ov::AnyMap{{"NPUW_FUNCALL_FOR_ALL", "YES"}}));
INSTANTIATE_TEST_SUITE_P(NPUW_WAI, ImportNonLLMWAIBlobTestNPUW,
    testing::Values(ov::AnyMap{{"NPUW_FOLD", "YES"}}, ov::AnyMap{{"NPUW_CWAI", "YES"}}));
