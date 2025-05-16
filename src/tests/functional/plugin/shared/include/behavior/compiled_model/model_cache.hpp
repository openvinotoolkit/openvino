// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace test {
namespace behavior {

typedef std::tuple<bool, bool, ov::element::Type, ov::element::Type, const char*> WeightlessCacheAccuracyTestParams;

class WeightlessCacheAccuracy : public ::testing::Test, public ::testing::WithParamInterface<WeightlessCacheAccuracyTestParams> {
public:
    static std::string get_test_case_name(const ::testing::TestParamInfo<WeightlessCacheAccuracyTestParams>& obj);

protected:
    std::shared_ptr<ov::Model> m_model;
    std::string m_xml_path;
    std::string m_bin_path;
    std::string m_cache_path;
    std::string m_cache_dir;
    const char* m_target_device;
    bool m_use_compile_model_api;
    bool m_do_encryption;
    ov::element::Type m_inference_mode;
    ov::element::Type m_model_dtype;

    void SetUp() override;
    void TearDown() override;
    void run();
};

class WeightlessCacheAccuracyLowPrecision : public WeightlessCacheAccuracy {};

static const std::vector<ov::element::Type> inference_modes = {
    ov::element::f32,
    ov::element::f16,
};

static const std::vector<ov::element::Type> model_dtypes = {
    ov::element::f32,
    ov::element::f16,
    ov::element::bf16,
};

static const std::vector<ov::element::Type> low_precision_dtypes = {
    ov::element::u8,
    ov::element::u4,
    ov::element::i4,
};

}  // namespace behavior
}  // namespace test
}  // namespace ov
