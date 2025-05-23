// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "fusion_test_common.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/quantize.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/rms.hpp>
#include <intel_gpu/primitives/reorder.hpp>

#include <cmath>

using namespace cldnn;
using namespace ::tests;

namespace {
struct rms_test_params {
    tensor input_size;
    tensor gamma_size;
    tensor elwise_size;
    data_types input_type;
    format input_format;
    size_t expected_fused_primitives;
    size_t expected_fused_primitives_onednn;
    size_t expected_not_fused_primitives;
};

class RMSFusingTest : public ::BaseFusingTest<rms_test_params> {
public:
    void execute(rms_test_params& p) {
        if (engine.get_device_info().supports_immad)
            p.expected_fused_primitives = p.expected_fused_primitives_onednn;
        auto input_prim = get_mem(get_input_layout(p));
        auto gamma_prim = get_mem(get_gamma_layout(p));

        network network_not_fused(this->engine, this->topology_non_fused, cfg_not_fused);
        network network_fused(this->engine, this->topology_fused, cfg_fused);

        network_fused.set_input_data("input", input_prim);
        network_fused.set_input_data("gamma", gamma_prim);
        network_not_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("gamma", gamma_prim);

        compare(network_not_fused, network_fused, p);
    }

    layout get_input_layout(rms_test_params& p) {
        return layout{ p.input_type, p.input_format, p.input_size };
    }

    layout get_gamma_layout(rms_test_params& p) {
        return layout{ p.input_type, p.input_format, p.gamma_size };
    }
};
}  // namespace


/* ----------------------------------------------------------------------------------------------------- */
/* --------------------------------------- RMS cases --------------------------------------------------- */
/* ----------------------------------------------------------------------------------------------------- */

#define CASE_RMS_F32_1      { 1, 16, 8, 8 },    { 1, 1, 1, 8 },     { 1, 16, 8, 8 },    data_types::f32, format::bfyx
#define CASE_RMS_F32_2      { 2, 16, 8, 8 },    { 1, 1, 1, 8 },     { 2, 16, 8, 8 },    data_types::f32, format::bfyx
#define CASE_RMS_3D_F32_1   { 1, 16, 8, 8, 8 }, { 1, 1, 1, 1, 8 },  { 1, 16, 8, 8, 8 }, data_types::f32, format::bfzyx
#define CASE_RMS_3D_F32_2   { 2, 16, 8, 8, 8 }, { 1, 1, 1, 1, 8 },  { 2, 16, 8, 8, 8 }, data_types::f32, format::bfzyx
#define CASE_RMS_F16_1      { 1, 16, 8, 8 },    { 1, 1, 1, 8 },     { 1, 16, 8, 8 },    data_types::f16, format::bfyx
#define CASE_RMS_F16_2      { 2, 16, 8, 8 },    { 1, 1, 1, 8 },     { 2, 16, 8, 8 },    data_types::f16, format::bfyx
#define CASE_RMS_3D_F16_1   { 1, 16, 8, 8, 8 }, { 1, 1, 1, 1, 8 },  { 1, 16, 8, 8, 8 }, data_types::f16, format::bfzyx
#define CASE_RMS_3D_F16_2   { 2, 16, 8, 8, 8 }, { 1, 1, 1, 1, 8 },  { 2, 16, 8, 8, 8 }, data_types::f16, format::bfzyx

class rms_activation : public RMSFusingTest {};
TEST_P(rms_activation, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        input_layout("gamma", get_gamma_layout(p)),
        rms("rms", input_info("input"), input_info("gamma"), 1e-10f),
        activation("act", input_info("rms"), activation_func::relu),
        reorder("reorder_bfyx", input_info("act"), format::bfyx, data_types::f32)
    );

    tolerance = (p.input_type == data_types::f32) ? 1e-5f : 0.1f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, rms_activation, ::testing::ValuesIn(std::vector<rms_test_params>{
    rms_test_params{ CASE_RMS_F32_1, 3, 3, 4 },
    rms_test_params{ CASE_RMS_F32_2, 3, 3, 4 },
    rms_test_params{ CASE_RMS_3D_F32_1, 3, 3, 4 },
    rms_test_params{ CASE_RMS_3D_F32_2, 3, 3, 4 },
    rms_test_params{ CASE_RMS_F16_1, 3, 3, 4 },
    rms_test_params{ CASE_RMS_F16_2, 3, 3, 4 },
    rms_test_params{ CASE_RMS_3D_F16_1, 3, 3, 4 },
    rms_test_params{ CASE_RMS_3D_F16_2, 3, 3, 4 },
}));

class rms_eltwise : public RMSFusingTest {};
TEST_P(rms_eltwise, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", layout{ p.input_type, p.input_format, p.input_size }),
        input_layout("gamma", layout{ p.input_type, p.input_format, p.gamma_size }),
        rms("rms", input_info("input"), input_info("gamma"), 1e-10f),
        data("eltw_data", get_mem(layout{ p.input_type, p.input_format, p.elwise_size })),
        eltwise("eltw", { input_info("rms"), input_info("eltw_data") }, eltwise_mode::sum, p.input_type),
        reorder("reorder_bfyx", input_info("eltw"), p.input_format, data_types::f32)
    );

    tolerance = (p.input_type == data_types::f32) ? 1e-5f : 0.1f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, rms_eltwise, ::testing::ValuesIn(std::vector<rms_test_params>{
    rms_test_params{ CASE_RMS_F32_1, 3, 3, 4 },
    rms_test_params{ CASE_RMS_F32_2, 3, 3, 4 },
    rms_test_params{ CASE_RMS_3D_F32_1, 3, 3, 4 },
    rms_test_params{ CASE_RMS_3D_F32_2, 3, 3, 4 },
    rms_test_params{ CASE_RMS_F16_1, 3, 3, 4 },
    rms_test_params{ CASE_RMS_F16_2, 3, 3, 4 },
    rms_test_params{ CASE_RMS_3D_F16_1, 3, 3, 4 },
    rms_test_params{ CASE_RMS_3D_F16_2, 3, 3, 4 },
}));
