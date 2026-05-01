// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/snippets/aarch64/pass/lowered/adjust_gemm_copy_b_loop_ports.hpp"

#include <cstdint>
#include <memory>
#include <vector>

#include "lir_test_utils.hpp"
#include "openvino/opsets/opset10_decl.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_info.hpp"
#include "snippets/lowered/loop_port.hpp"
#include "snippets/op/result.hpp"
#include "transformations/snippets/aarch64/op/gemm_cpu.hpp"
#include "transformations/snippets/aarch64/op/gemm_utils.hpp"

namespace ov::test::snippets {

using namespace ov::intel_cpu;
using namespace ov::snippets::lowered;
using LoopPortDesc = UnifiedLoopInfo::LoopPortDesc;
using PortDescriptor = ov::snippets::modifier::MemoryAccess::PortDescriptor;
using PortType = LoopPort::Type;

TEST(AdjustGemmCopyBLoopPorts, RescalesFinalizationOffset) {
    constexpr size_t K = 73;
    constexpr size_t N = 130;
    constexpr int64_t original_ptr_increment = 2;
    constexpr int64_t original_finalization_offset = -20;
    const auto precision = ov::element::f32;
    const auto data_size = static_cast<int64_t>(precision.size());
    const auto n_step = aarch64::gemm_utils::repacking::get_rhs_packed_n_step(precision);
    const auto expected_ptr_increment = static_cast<int64_t>(
        aarch64::gemm_utils::repacking::get_rhs_packed_offset(precision, n_step, K) / (n_step * precision.size()));
    const auto expected_finalization_offset =
        expected_ptr_increment * (original_finalization_offset / original_ptr_increment);

    ov::snippets::lowered::Config lir_config;
    lir_config.m_manual_build_support = true;
    auto linear_ir = std::make_shared<LinearIR>(lir_config);

    const ov::PartialShape input_shape_a{2, 2, 96, K};
    const ov::PartialShape input_shape_b{2, 2, K, N};
    auto data_a = linear_ir->push_node<ov::opset10::Parameter>(precision, input_shape_a);
    auto data_b = linear_ir->push_node<ov::opset10::Parameter>(precision, input_shape_b);
    auto gemm = linear_ir->push_node<aarch64::GemmCPU>(data_a.second,
                                                       data_b.second,
                                                       PortDescriptor{},
                                                       PortDescriptor{},
                                                       PortDescriptor{});
    init_expr_descriptors(*gemm.first);
    linear_ir->push_node<ov::snippets::op::Result>(gemm.second);

    const std::vector<LoopPort> entries{
        LoopPort::create<PortType::NotProcessed>((*gemm.first)->get_input_port(0)),
        LoopPort::create<PortType::Incremented>((*gemm.first)->get_input_port(1), 0),
    };
    const std::vector<LoopPort> exits{LoopPort::create<PortType::Incremented>((*gemm.first)->get_output_port(0), 0)};
    auto loop_info = std::make_shared<UnifiedLoopInfo>(
        N,
        n_step,
        entries,
        exits,
        std::vector<LoopPortDesc>{LoopPortDesc(0, 0, data_size),
                                  LoopPortDesc(original_ptr_increment, original_finalization_offset, data_size)},
        std::vector<LoopPortDesc>{LoopPortDesc(1, -static_cast<int64_t>(n_step), data_size)},
        false);

    ASSERT_TRUE(ov::intel_cpu::pass::aarch64::AdjustGemmCopyBLoopPorts::update_loop_info(loop_info));

    const auto& input_descs = loop_info->get_input_port_descs();
    ASSERT_EQ(input_descs.size(), 2);
    EXPECT_EQ(input_descs[1].ptr_increment, expected_ptr_increment);
    EXPECT_EQ(input_descs[1].finalization_offset, expected_finalization_offset);
    EXPECT_EQ(input_descs[1].data_size, data_size);
}

}  // namespace ov::test::snippets
