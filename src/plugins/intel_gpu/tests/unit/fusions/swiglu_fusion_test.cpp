// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "fusion_test_common.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/quantize.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/swiglu.hpp>
#include <intel_gpu/primitives/reorder.hpp>

#include <cmath>

using namespace cldnn;
using namespace ::tests;

namespace {
struct swiglu_test_params {
    tensor input_size;
    tensor eltwise_size;
    size_t split_length;
    data_types input_type;
    format input_format;
    size_t expected_fused_primitives;
    size_t expected_fused_primitives_onednn;
    size_t expected_not_fused_primitives;
};

class SwiGLUFusingTest : public ::BaseFusingTest<swiglu_test_params> {
public:
    void execute(swiglu_test_params& p) {
        if (engine.get_device_info().supports_immad)
            p.expected_fused_primitives = p.expected_fused_primitives_onednn;
        auto input_prim = get_mem(get_input_layout(p));

        network network_not_fused(this->engine, this->topology_non_fused, cfg_not_fused);
        network network_fused(this->engine, this->topology_fused, cfg_fused);

        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);

        compare(network_not_fused, network_fused, p);
    }

    layout get_input_layout(swiglu_test_params& p) {
        return layout{ p.input_type, p.input_format, p.input_size };
    }
};
}  // namespace


/* ----------------------------------------------------------------------------------------------------- */
/* --------------------------------------- SwiGLU cases ------------------------------------------------ */
/* ----------------------------------------------------------------------------------------------------- */

#define CASE_SwiGLU_F32_1      { 1, 3, 4, 16 },   { 1, 1, 1, 1  },    2, data_types::f32, format::bfyx
#define CASE_SwiGLU_F32_2      { 2, 3, 4, 16 },   { 2, 3, 2, 16 },    2, data_types::f32, format::bfyx
#define CASE_SwiGLU_3D_F32_1   { 1, 3, 4, 5, 6 }, { 1, 3, 2, 5, 6 },  2, data_types::f32, format::bfzyx
#define CASE_SwiGLU_3D_F32_2   { 2, 3, 4, 5, 6 }, { 1, 1, 1, 1  },    2, data_types::f32, format::bfzyx
#define CASE_SwiGLU_F16_1      { 1, 3, 4, 16 },   { 1, 3, 2, 16 },    2, data_types::f16, format::bfyx
#define CASE_SwiGLU_F16_2      { 2, 3, 4, 16 },   { 1, 1, 1, 1  },    2, data_types::f16, format::bfyx
#define CASE_SwiGLU_3D_F16_1   { 1, 3, 4, 5, 6 }, { 1, 1, 1, 1  },    2, data_types::f16, format::bfzyx
#define CASE_SwiGLU_3D_F16_2   { 2, 3, 4, 5, 6 }, { 2, 3, 2, 5, 6 },  2, data_types::f16, format::bfzyx

class swiglu_activation : public SwiGLUFusingTest {};
TEST_P(swiglu_activation, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        swiglu("swiglu", input_info("input"), -1, p.split_length, ov::op::internal::GLU::GluType::Swish, 0, tensor()),
        activation("act", input_info("swiglu"), activation_func::relu),
        reorder("reorder_bfyx", input_info("act"), format::bfyx, data_types::f32)
    );

    tolerance = (p.input_type == data_types::f32) ? 1e-5f : 0.1f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, swiglu_activation, ::testing::ValuesIn(std::vector<swiglu_test_params>{
    swiglu_test_params{ CASE_SwiGLU_F32_1, 2, 2, 3 },
    swiglu_test_params{ CASE_SwiGLU_F32_2, 2, 2, 3 },
    swiglu_test_params{ CASE_SwiGLU_3D_F32_1, 2, 2, 3 },
    swiglu_test_params{ CASE_SwiGLU_3D_F32_2, 2, 2, 3 },
    swiglu_test_params{ CASE_SwiGLU_F16_1, 2, 2, 3 },
    swiglu_test_params{ CASE_SwiGLU_F16_2, 2, 2, 3 },
    swiglu_test_params{ CASE_SwiGLU_3D_F16_1, 2, 2, 3 },
    swiglu_test_params{ CASE_SwiGLU_3D_F16_2, 2, 2, 3 },
}));

class swiglu_eltwise : public SwiGLUFusingTest {};
TEST_P(swiglu_eltwise, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", layout{ p.input_type, p.input_format, p.input_size }),
        swiglu("swiglu", input_info("input"), -1, p.split_length, ov::op::internal::GLU::GluType::Swish, 0, tensor()),
        data("eltw_data", get_mem(layout{ p.input_type, p.input_format, p.eltwise_size })),
        eltwise("eltw", { input_info("swiglu"), input_info("eltw_data") }, eltwise_mode::prod, p.input_type),
        reorder("reorder_bfyx", input_info("eltw"), p.input_format, data_types::f32)
    );

    tolerance = (p.input_type == data_types::f32) ? 1e-5f : 0.1f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, swiglu_eltwise, ::testing::ValuesIn(std::vector<swiglu_test_params>{
    swiglu_test_params{ CASE_SwiGLU_F32_1, 2, 2, 3 },
    swiglu_test_params{ CASE_SwiGLU_F32_2, 2, 2, 3 },
    swiglu_test_params{ CASE_SwiGLU_3D_F32_1, 2, 2, 3 },
    swiglu_test_params{ CASE_SwiGLU_3D_F32_2, 2, 2, 3 },
    swiglu_test_params{ CASE_SwiGLU_F16_1, 2, 2, 3 },
    swiglu_test_params{ CASE_SwiGLU_F16_2, 2, 2, 3 },
    swiglu_test_params{ CASE_SwiGLU_3D_F16_1, 2, 2, 3 },
    swiglu_test_params{ CASE_SwiGLU_3D_F16_2, 2, 2, 3 },
}));
