// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "cpu/x64/cpu_isa_traits.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/openvino.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "openvino/runtime/intel_cpu/properties.hpp"
#include "ov_ops/fully_connected_compressed.hpp"

namespace ov::intel_cpu {
namespace {

TEST(FullyConnectedCompressedFP16Test, IntegerOutputsPreserveCompressedWeightsAcrossShapeChanges) {
    if (!dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_fp16)) {
        GTEST_SKIP() << "FP16 inference requires AVX512-FP16.";
    }
    using Constant = ov::op::v0::Constant;
    ov::Core core;
    for (auto weights_type : {ov::element::u4, ov::element::u8}) {
        for (auto output_type : {ov::element::i8, ov::element::u8}) {
            for (uint64_t dq_group_size : {uint64_t{0}, uint64_t{32}}) {
                for (bool with_bias : {false, true}) {
                    SCOPED_TRACE(testing::Message() << "weights=" << weights_type << " output=" << output_type
                                                    << " dq_group_size=" << dq_group_size << " bias=" << with_bias);
                    auto input =
                        std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, -1, 256});
                    auto bias = with_bias ? Constant::create(ov::element::f16, {64}, {1.0F})
                                          : std::make_shared<Constant>(ov::element::dynamic, ov::Shape{0});
                    auto fc = std::make_shared<ov::op::internal::FullyConnectedCompressed>(
                        input,
                        Constant::create(weights_type, {64, 256}, {1}),
                        bias,
                        Constant::create(ov::element::f32, {64, 1}, {0.125F}),
                        Constant::create(weights_type, {64, 1}, {0}),
                        output_type);
                    auto model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input});
                    auto compiled = core.compile_model(model,
                                                       "CPU",
                                                       {ov::hint::inference_precision(ov::element::f16),
                                                        ov::hint::dynamic_quantization_group_size(dq_group_size),
                                                        ov::num_streams(1),
                                                        ov::inference_num_threads(2)});
                    bool found_fc = false;
                    for (const auto& node : compiled.get_runtime_model()->get_ordered_ops()) {
                        const auto layer_type = node->get_rt_info().find(ov::exec_model_info::LAYER_TYPE);
                        if (layer_type != node->get_rt_info().end() &&
                            layer_type->second.as<std::string>() == "FullyConnected") {
                            found_fc = true;
                            ASSERT_TRUE(node->get_input_element_type(1).is_integral());
                        }
                    }
                    ASSERT_TRUE(found_fc);
                    auto request = compiled.create_infer_request();
                    constexpr std::array<int, 4> expected_without_bias{8, 16, 24, 32};
                    for (size_t rows : {size_t{17}, size_t{1}, size_t{65}, size_t{1}}) {
                        ov::Tensor data(ov::element::f16, {1, rows, 256});
                        for (size_t row = 0; row < rows; ++row) {
                            std::fill_n(data.data<ov::float16>() + row * 256,
                                        256,
                                        ov::float16(0.25F * static_cast<float>(1 + row % 4)));
                        }
                        request.set_input_tensor(data);
                        request.infer();
                        const auto output = request.get_output_tensor();
                        ASSERT_EQ(output.get_element_type(), output_type);
                        ASSERT_EQ(output.get_shape(), (ov::Shape{1, rows, 64}));
                        for (size_t i = 0; i < output.get_size(); ++i) {
                            const int expected = expected_without_bias[(i / 64) % 4] + (with_bias ? 1 : 0);
                            const int actual =
                                output_type == ov::element::i8 ? output.data<int8_t>()[i] : output.data<uint8_t>()[i];
                            ASSERT_EQ(actual, expected) << "rows=" << rows << " element=" << i;
                        }
                    }
                }
            }
        }
    }
}

}  // namespace
}  // namespace ov::intel_cpu
