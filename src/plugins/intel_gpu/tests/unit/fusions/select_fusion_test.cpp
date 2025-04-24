// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "fusion_test_common.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/select.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include "select_inst.h"

#include <cmath>

using namespace cldnn;
using namespace ::tests;

namespace {
struct select_test_params {
    ov::Shape input_shape;
    ov::Shape mask_shape;
    data_types input_type;
    data_types output_type;
    format input_format;
    format output_format;
    size_t expected_fused_primitives;
    size_t expected_not_fused_primitives;
};

class SelectFusingTest : public ::BaseFusingTest<select_test_params> {
public:
    void execute(select_test_params& p, bool count_reorder = false) {
        cfg_not_fused.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        cfg_fused.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        cfg_fused.set_property(ov::intel_gpu::optimize_data(true));
 
        tests::random_generator rg;
        auto mask_mem   = get_mem(get_mask_layout(p), 0, 1);
        auto input1_mem = get_mem(get_input_layout(p));
        auto input2_mem = get_mem(get_input_layout(p));

        network network_not_fused(this->engine, this->topology_non_fused, cfg_not_fused);
        network network_fused(this->engine, this->topology_fused, cfg_fused);

        auto inputs = network_fused.get_input_ids();
        network_fused.set_input_data("mask", mask_mem);
        network_fused.set_input_data("input1", input1_mem);
        network_fused.set_input_data("input2", input2_mem);
        network_not_fused.set_input_data("mask", mask_mem);
        network_not_fused.set_input_data("input1", input1_mem);
        network_not_fused.set_input_data("input2", input2_mem);
        compare(network_not_fused, network_fused, p, count_reorder);
    }

    layout get_input_layout(select_test_params& p) {
        return layout{ p.input_shape, p.input_type, p.input_format };
    }
    layout get_mask_layout(select_test_params& p) {
        return layout{ p.mask_shape, data_types::i8, p.input_format };
    }
};
}  // namespace

#define CASE_SELECT_FP32_TO_I8_0  {2, 16, 4, 4}, {2, 16, 4, 4},  data_types::f32, data_types::i8,  format::bfyx, format::bfyx
#define CASE_SELECT_FP32_TO_U8_0  {2, 16, 4, 4}, {2, 16, 4, 4},  data_types::f32, data_types::u8,  format::bfyx, format::bfyx
#define CASE_SELECT_FP32_TO_F16_0 {2, 16, 17, 4}, {2, 16, 1, 4}, data_types::f32, data_types::f16, format::bfyx, format::bfyx
#define CASE_SELECT_FP16_TO_I8_0  {2, 16, 4, 4}, {2, 16, 4, 4},  data_types::f16, data_types::i8,  format::bfyx, format::bfyx
#define CASE_SELECT_FP16_TO_U8_0  {2, 16, 4, 4}, {2, 16, 4, 4},  data_types::f16, data_types::u8,  format::bfyx, format::bfyx
#define CASE_SELECT_FP16_TO_I8_1  {2, 16, 4, 4}, {2, 16, 4, 4},  data_types::f16, data_types::i8,  format::bfyx, format::bfzyx
#define CASE_SELECT_FP16_TO_U8_1  {2, 16, 4, 4}, {2, 16, 4, 4},  data_types::f16, data_types::u8,  format::bfyx, format::bfzyx

class select_reorder_fusion : public SelectFusingTest {};
TEST_P(select_reorder_fusion, basic) {
    auto p = GetParam();
    create_topologies(
        cldnn::input_layout("mask", get_mask_layout(p)),
        cldnn::input_layout("input1", get_input_layout(p)),
        cldnn::input_layout("input2", get_input_layout(p)),
        cldnn::reorder("mask_convert", input_info("mask"), p.input_format, p.input_type),
        cldnn::select("select", input_info("input1"), input_info("input2"), input_info("mask_convert")),
        cldnn::reorder("out", input_info("select"), p.output_format, p.output_type, std::vector<float>(), cldnn::reorder_mean_mode::subtract, cldnn::padding(), true)
    );
    tolerance = 1e-5f;
    execute(p, true);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, select_reorder_fusion, ::testing::ValuesIn(std::vector<select_test_params>{
    select_test_params{ CASE_SELECT_FP32_TO_F16_0, 5, 6},
    select_test_params{ CASE_SELECT_FP32_TO_I8_0, 5, 6},
    select_test_params{ CASE_SELECT_FP32_TO_U8_0, 5, 6},
    select_test_params{ CASE_SELECT_FP16_TO_I8_0, 5, 6},
    select_test_params{ CASE_SELECT_FP16_TO_U8_0, 5, 6},
    select_test_params{ CASE_SELECT_FP16_TO_I8_1, 6, 6}, // reorder should not be fused
    select_test_params{ CASE_SELECT_FP16_TO_U8_1, 6, 6},
}));

class select_reorder_fusion_dynamic : public SelectFusingTest {};
TEST_P(select_reorder_fusion_dynamic, basic) {
    auto p = GetParam();
    create_topologies(
        cldnn::input_layout("mask", layout{ ov::PartialShape::dynamic(p.mask_shape.size()), data_types::i8, p.input_format }),
        cldnn::input_layout("input1", layout {ov::PartialShape::dynamic(p.input_shape.size()), p.input_type, p.input_format }),
        cldnn::input_layout("input2", layout {ov::PartialShape::dynamic(p.input_shape.size()), p.input_type, p.input_format }),
        cldnn::reorder("mask_convert", input_info("mask"), p.input_format, p.input_type),
        cldnn::select("select", input_info("input1"), input_info("input2"), input_info("mask_convert")),
        cldnn::reorder("out", input_info("select"), p.output_format, p.output_type, std::vector<float>(), cldnn::reorder_mean_mode::subtract, cldnn::padding(), true)
    );
    tolerance = 1e-5f;
    execute(p, true);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, select_reorder_fusion_dynamic, ::testing::ValuesIn(std::vector<select_test_params>{
    select_test_params{ CASE_SELECT_FP32_TO_F16_0, 5, 6},
    select_test_params{ CASE_SELECT_FP32_TO_I8_0, 5, 6},
    select_test_params{ CASE_SELECT_FP16_TO_I8_0, 5, 6},
    select_test_params{ CASE_SELECT_FP16_TO_I8_1, 6, 6}, // reorder should not be fused
}));
