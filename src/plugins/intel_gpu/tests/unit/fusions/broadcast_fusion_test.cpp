// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "fusion_test_common.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/broadcast.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/quantize.hpp>
#include <intel_gpu/primitives/data.hpp>

#include <cmath>

using namespace cldnn;
using namespace ::tests;

namespace {
struct broadcast_test_params {
    ov::PartialShape input_size1;  // input for broadcast
    ov::PartialShape input_size2;  // other input connected to output of broadcast
    data_types input_type1;        // input data-type of 'input_size1'
    data_types input_type2;
    format input_format;
    data_types default_type;
    format default_format;
    size_t expected_fused_primitives;
    size_t expected_not_fused_primitives;
};

class BroadcastFusingTest : public ::BaseFusingTest<broadcast_test_params> {
public:
    void execute(broadcast_test_params& p, bool count_reorder = false) {
        auto input_prim = get_mem(get_input_layout1(p));
        auto input_prim2 = get_mem(get_input_layout2(p));

        network network_not_fused(this->engine, this->topology_non_fused, cfg_not_fused);
        network network_fused(this->engine, this->topology_fused, cfg_fused);

        auto inputs = network_fused.get_input_ids();
        if (std::find(inputs.begin(), inputs.end(), "input") != inputs.end()) {
            network_fused.set_input_data("input", input_prim);
            network_not_fused.set_input_data("input", input_prim);
        }
        if (std::find(inputs.begin(), inputs.end(), "input2") != inputs.end()) {
            network_fused.set_input_data("input2", input_prim2);
            network_not_fused.set_input_data("input2", input_prim2);
        }

        compare(network_not_fused, network_fused, p, count_reorder);
    }

    layout get_input_layout1(broadcast_test_params& p) {
        return layout{ p.input_size1, p.input_type1, p.input_format };
    }

    layout get_input_layout2(broadcast_test_params& p) {
        return layout{ p.input_size2, p.input_type2, p.input_format };
    }
};
}  // namespace

#define CASE_BROADCAST_FP16_1         { 1, 16, 4, 4 }, { 2, 16, 4, 4 },        data_types::f16, data_types::f16, format::bfyx,           data_types::f16,  format::bfyx
#define CASE_BROADCAST_FP16_2         { 2, 1,  4, 4, 4 }, { 2, 16, 4, 4, 4 },  data_types::f16, data_types::f16, format::bfzyx,          data_types::f16,  format::bfzyx
#define CASE_BROADCAST_FP16_3         { 2, 16, 4, 4, 1 }, { 2, 16, 4, 4, 8 },  data_types::f16, data_types::f16, format::bfzyx,          data_types::f16,  format::bfzyx

#define CASE_BROADCAST_FP16_1_BLK     { 2, 16, 4, 1 }, { 2, 16, 4, 4 },        data_types::f16, data_types::f16, format::b_fs_yx_fsv16,  data_types::f16,  format::bfyx
#define CASE_BROADCAST_FP16_2_BLK     { 1, 16, 4, 4 }, { 2, 16, 4, 4 },        data_types::f16, data_types::f16, format::b_fs_yx_fsv16,  data_types::f16,  format::bfyx
#define CASE_BROADCAST_FP16_3_BLK     { 2, 16, 4, 1 }, { 2, 16, 4, 4 },        data_types::u8,  data_types::i8,  format::b_fs_yx_fsv32,  data_types::f16,  format::bfyx

#define CASE_BROADCAST_FP16_OPT_1     { 21, 2, 1, 5 }, { 21, 2, 13, 5 },       data_types::f16, data_types::f16, format::bfyx,           data_types::f16,  format::bfyx
#define CASE_BROADCAST_FP16_OPT_2     { 21, 2, 1, 1 }, { 21, 2, 13, 1 },       data_types::f16, data_types::f16, format::bfyx,           data_types::f16,  format::bfyx

class broadcast_fused_prims : public BroadcastFusingTest {};
TEST_P(broadcast_fused_prims, broadcast_activation_with_broadcast) {
    auto p = GetParam();
    const auto quantize_dt = data_types::i8;
    create_topologies(
        input_layout("input", get_input_layout1(p)),
        input_layout("input2", get_input_layout2(p)),
        broadcast("broadcast", input_info("input"), get_input_layout2(p).get_shape(), ov::AxisSet({}), ov::op::BroadcastType::NUMPY),
        eltwise("eltwise", {input_info("broadcast"), input_info("input2")}, eltwise_mode::sum, p.default_type),
        data("in_lo", get_mem(get_single_element_layout(p), -1.9)),
        data("in_hi", get_mem(get_single_element_layout(p), 1.8)),
        data("out_lo", get_mem(get_single_element_layout(p), -128)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        quantize("quantize", input_info("eltwise"), input_info("in_lo"), input_info("in_hi"), input_info("out_lo"), input_info("out_hi"), 256, quantize_dt),
        activation("activation", input_info("quantize"), activation_func::relu),
        reorder("out", input_info("activation"), p.default_format, data_types::f32));

    tolerance = default_tolerance(quantize_dt);
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, broadcast_fused_prims, ::testing::ValuesIn(std::vector<broadcast_test_params>{
    broadcast_test_params{ CASE_BROADCAST_FP16_1, 4, 6 },
    broadcast_test_params{ CASE_BROADCAST_FP16_2, 4, 6 },
    broadcast_test_params{ CASE_BROADCAST_FP16_3, 4, 6 },

    broadcast_test_params{ CASE_BROADCAST_FP16_1_BLK, 4, 6 },
    broadcast_test_params{ CASE_BROADCAST_FP16_2_BLK, 4, 6 },
    broadcast_test_params{ CASE_BROADCAST_FP16_3_BLK, 4, 6 },

    broadcast_test_params{ CASE_BROADCAST_FP16_OPT_1, 4, 6 },
    broadcast_test_params{ CASE_BROADCAST_FP16_OPT_2, 4, 6 },
}));
