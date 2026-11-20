// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#if defined(CPU_DEBUG_CAPS) && defined(OPENVINO_ARCH_X86_64)
#    include <cstddef>
#    include <cstdint>
#    include <cstdlib>
#    include <memory>
#    include <optional>
#    include <string>
#    include <vector>

#    include "openvino/openvino.hpp"
#    include "openvino/opsets/opset13.hpp"
#    include "openvino/runtime/intel_cpu/properties.hpp"
#    include "openvino/runtime/system_conf.hpp"

namespace ov::intel_cpu::test {
namespace {
class ScopedEnvironment {
public:
    ScopedEnvironment(const char* name, const char* value) : m_name(name) {
        if (const auto* previous = std::getenv(name)) {
            m_previous = previous;
        }
        set(value);
    }
    ~ScopedEnvironment() {
        set(m_previous ? m_previous->c_str() : nullptr);
    }

private:
    void set(const char* value) const {
#    ifdef _WIN32
        _putenv_s(m_name.c_str(), value ? value : "");
#    else
        if (value) {
            setenv(m_name.c_str(), value, 1);
        } else {
            unsetenv(m_name.c_str());
        }
#    endif
    }
    std::string m_name;
    std::optional<std::string> m_previous;
};

void check_fc(ov::element::Type weight_type, size_t k, bool fractional_zero_point, bool compressed) {
    using namespace ov::opset13;
    constexpr size_t n = 17;
    const size_t group = k == 33 ? 33 : 32;
    const size_t groups = k / group;
    const bool signed_weights = weight_type == ov::element::i4 || weight_type == ov::element::i8;
    std::vector<int32_t> values(n * k);
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<int32_t>(i * 7 % 16) - (signed_weights ? 8 : 0);
    }
    const float zero = (signed_weights ? -2.F : 3.F) + (fractional_zero_point ? 0.25F : 0.F);
    auto weights = Constant::create(weight_type, ov::Shape{n, groups, group}, values);
    auto converted = std::make_shared<Convert>(weights, ov::element::f16);
    auto zero_points = Constant::create(ov::element::f16, ov::Shape{n, groups, 1}, {zero});
    auto scales = Constant::create(ov::element::f16, ov::Shape{n, groups, 1}, {0.0625F});
    auto dequantized = std::make_shared<Multiply>(std::make_shared<Subtract>(converted, zero_points), scales);
    auto flattened =
        std::make_shared<Reshape>(dequantized, Constant::create(ov::element::i64, ov::Shape{2}, {n, k}), false);
    auto input = std::make_shared<Parameter>(ov::element::f16, ov::PartialShape{-1, k});
    auto matmul = std::make_shared<MatMul>(input, flattened, false, true);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{matmul}, ov::ParameterVector{input});
    ov::Core core;
    auto compiled = core.compile_model(model,
                                       "CPU",
                                       {ov::hint::inference_precision(ov::element::f16),
                                        ov::hint::dynamic_quantization_group_size(32),
                                        ov::inference_num_threads(2),
                                        ov::num_streams(1)});
    auto request = compiled.create_infer_request();
    for (const size_t m : {31U, 1U, 2U, 1U}) {
        ov::Tensor source(ov::element::f16, ov::Shape{m, k});
        for (size_t i = 0; i < source.get_size(); ++i) {
            source.data<ov::float16>()[i] = ov::float16((static_cast<int>(i * 17 % 31) - 15) * 0.0625F);
        }
        request.set_input_tensor(source);
        request.infer();
        const auto output = request.get_output_tensor();
        for (size_t row = 0; row < m; ++row) {
            for (size_t column = 0; column < n; ++column) {
                float sum = 0;
                for (size_t channel = 0; channel < k; ++channel) {
                    const ov::float16 difference(static_cast<float>(values[column * k + channel]) - zero);
                    const ov::float16 value(static_cast<float>(difference) * 0.0625F);
                    sum +=
                        static_cast<float>(source.data<ov::float16>()[row * k + channel]) * static_cast<float>(value);
                }
                ASSERT_EQ(output.data<ov::float16>()[row * n + column], ov::float16(sum));
            }
        }
    }
    size_t fc_count = 0;
    for (const auto& node : compiled.get_runtime_model()->get_ordered_ops()) {
        const auto& info = node->get_rt_info();
        const auto layer_type = info.find("layerType");
        if (layer_type == info.end() || layer_type->second.as<std::string>() != "FullyConnected") {
            continue;
        }
        ++fc_count;
        EXPECT_EQ(node->input_value(1).get_element_type().is_integral_number(), compressed);
    }
    EXPECT_EQ(fc_count, 1U);
}
}  // namespace

TEST(FP16CompressedFC, DynamicShapesAndDecodeCache) {
    if (!ov::with_cpu_x86_avx512_core_fp16()) {
        GTEST_SKIP();
    }
    const ScopedEnvironment route("OV_CPU_ENABLE_DNNL_MAMTUL_FOR_FC", "1");
    const ScopedEnvironment selection("OV_FC_DECODE_CACHE_NODES", nullptr);
    for (const char* budget : {"0", "1"}) {
        const ScopedEnvironment cache("OV_FC_DECODE_CACHE_MIB", budget);
        for (const auto type : {ov::element::i4, ov::element::u4, ov::element::i8, ov::element::u8}) {
            check_fc(type, 64, false, true);
        }
    }
}

TEST(FP16CompressedFC, UnsupportedDecompressionKeepsMaterializedFallback) {
    if (!ov::with_cpu_x86_avx512_core_fp16()) {
        GTEST_SKIP();
    }
    const ScopedEnvironment route("OV_CPU_ENABLE_DNNL_MAMTUL_FOR_FC", "1");
    const ScopedEnvironment cache("OV_FC_DECODE_CACHE_MIB", "1");
    const ScopedEnvironment selection("OV_FC_DECODE_CACHE_NODES", nullptr);
    for (const auto type : {ov::element::i4, ov::element::u4}) {
        check_fc(type, 33, false, false);
        check_fc(type, 64, true, false);
    }
}
}  // namespace ov::intel_cpu::test
#endif
