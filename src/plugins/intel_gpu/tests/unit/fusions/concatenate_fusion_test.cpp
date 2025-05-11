// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "fusion_test_common.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/quantize.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/data.hpp>

#include <cmath>

using namespace cldnn;
using namespace ::tests;

struct concat_test_params {
    ov::PartialShape in_shape;
    data_types data_type;
    format input_format;
    data_types default_type;
    format default_format;
    size_t expected_fused_primitives;
    size_t expected_not_fused_primitives;
    std::string kernel_name;
};

#ifdef ENABLE_ONEDNN_FOR_GPU
namespace {
class ConcatOneDNNFusingTest : public ::BaseFusingTest<concat_test_params> {
public:
    void execute(concat_test_params& p) {
        // Onednn post operation has issue in a machine that does not support imad.
        if (!engine.get_device_info().supports_immad)
            return;

        auto input0_prim = get_mem(get_input_layout(p));
        auto input1_prim = get_mem(get_input_layout(p));

        ov::intel_gpu::ImplementationDesc onednn_impl = { p.input_format, "", impl_types::onednn };
        ov::intel_gpu::ImplementationDesc cldnn_impl = { p.input_format, "", impl_types::ocl };

        // for onednn fusing test, topology_non_fused means cldnn, topology_fused is onednn
        ExecutionConfig cldnn_cfg = get_test_default_config(engine,
                                  {ov::intel_gpu::queue_type(QueueTypes::in_order),
                                  ov::intel_gpu::optimize_data(true),
                                  ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "concat", cldnn_impl } })});
        ExecutionConfig onednn_cfg = get_test_default_config(engine,
                                   {ov::intel_gpu::queue_type(QueueTypes::in_order),
                                   ov::intel_gpu::optimize_data(true),
                                   ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "concat", onednn_impl } })});
        network network_fused_cldnn(this->engine, this->topology_non_fused, cldnn_cfg);
        network network_fused_onednn(this->engine, this->topology_fused, onednn_cfg);

        network_fused_cldnn.set_input_data("input0", input0_prim);
        network_fused_cldnn.set_input_data("input1", input1_prim);
        network_fused_onednn.set_input_data("input0", input0_prim);
        network_fused_onednn.set_input_data("input1", input1_prim);

        ASSERT_FALSE(network_fused_cldnn.get_primitives_info().empty());
        ASSERT_FALSE(network_fused_onednn.get_primitives_info().empty());

        auto find_and_check = [&](primitive_info& p) -> bool {
            if (p.original_id == "concat" || p.original_id == "reorder_bfyx")
                return true;
            return false;
        };

        auto pi_fused_onednn = network_fused_onednn.get_primitives_info();
        auto pi_fused_cldnn = network_fused_cldnn.get_primitives_info();
        auto info_fused_onednn = std::find_if(pi_fused_onednn.begin(), pi_fused_onednn.end(), find_and_check);
        auto info_fused_cldnn = std::find_if(pi_fused_cldnn.begin(), pi_fused_cldnn.end(), find_and_check);

        ASSERT_TRUE(info_fused_onednn != pi_fused_onednn.end());
        ASSERT_TRUE(info_fused_cldnn != pi_fused_cldnn.end());

        compare(network_fused_cldnn, network_fused_onednn, p);
    }

    layout get_input_layout(concat_test_params& p) {
        return layout{ p.in_shape, p.data_type, p.input_format };
    }

    layout get_per_channel_layout(concat_test_params& p) {
        return layout{ { 1, p.in_shape[1] }, p.default_type, p.default_format };
    }
};
}  // namespace

/* ----------------------------------------------------------------------------------------------------- */
/* --------------------------------------- Concat cases ------------------------------------------------ */
/* ----------------------------------------------------------------------------------------------------- */
#define CASE_CONCAT_F32_1 { 1, 8, 4, 4 }, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_CONCAT_F16_1 { 1, 8, 4, 4 }, data_types::f16, format::bfyx, data_types::f16, format::bfyx

class concat_onednn_activation : public ConcatOneDNNFusingTest {};
TEST_P(concat_onednn_activation, along_f) {
    auto p = GetParam();
    create_topologies(
        input_layout("input0", get_input_layout(p)),
        input_layout("input1", get_input_layout(p)),
        concatenation("concat",
                      { input_info("input0"), input_info("input1") },
                      1,
                      data_types::f16),
        activation("act", input_info("concat"), activation_func::relu),
        reorder("reorder_bfyx", input_info("act"), cldnn::format::bfyx, p.default_type)
    );

    tolerance = 1.f;
    execute(p);
}

class concat_onednn_eltwise : public ConcatOneDNNFusingTest {};
TEST_P(concat_onednn_eltwise, along_f) {
    auto p = GetParam();
    layout data_layout({ 1, p.in_shape[1] * 2 }, p.default_type, p.default_format);

    create_topologies(
        input_layout("input0", get_input_layout(p)),
        input_layout("input1", get_input_layout(p)),
        data("scale_data", get_mem(data_layout, 1.0f / tensor{ 1, 1, 4, 4 }.count())),
        concatenation("concat",
                      { input_info("input0"), input_info("input1") },
                      1,
                      data_types::f16),
        eltwise("scale", { input_info("concat"), input_info("scale_data") }, eltwise_mode::prod, p.default_type),
        reorder("reorder_bfyx", input_info("scale"), cldnn::format::bfyx, p.default_type)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, concat_onednn_activation, ::testing::ValuesIn(std::vector<concat_test_params>{
    concat_test_params{ CASE_CONCAT_F16_1, 4, 4, "" },
}));

INSTANTIATE_TEST_SUITE_P(fusings_gpu, concat_onednn_eltwise, ::testing::ValuesIn(std::vector<concat_test_params>{
    concat_test_params{ CASE_CONCAT_F32_1, 4, 4, "" },
    concat_test_params{ CASE_CONCAT_F16_1, 4, 4, "" },
}));
#endif

namespace {
class ConcatFusingTest : public ::BaseFusingTest<concat_test_params> {
public:
    void execute(concat_test_params& p) {
        auto input0_prim = get_mem(get_input_layout(p));
        auto input1_prim = get_mem(get_input_layout(p));

        network network_not_fused(this->engine, this->topology_non_fused, cfg_not_fused);
        network network_fused(this->engine, this->topology_fused, cfg_fused);

        network_not_fused.set_input_data("input0", input0_prim);
        network_not_fused.set_input_data("input1", input1_prim);

        network_fused.set_input_data("input0", input0_prim);
        network_fused.set_input_data("input1", input1_prim);

        ASSERT_FALSE(network_not_fused.get_primitives_info().empty());
        ASSERT_FALSE(network_fused.get_primitives_info().empty());

        auto find_and_check = [&](primitive_info& p) -> bool {
            return p.original_id == "concat" || p.original_id == "reorder_bfyx";
        };

        auto pi_fused = network_fused.get_primitives_info();
        auto pi_not_fused = network_not_fused.get_primitives_info();

        auto info_fused = std::find_if(pi_fused.begin(), pi_fused.end(), find_and_check);
        auto info_not_fused = std::find_if(pi_not_fused.begin(), pi_not_fused.end(), find_and_check);

        ASSERT_TRUE(info_fused != pi_fused.end());
        ASSERT_TRUE(info_not_fused != pi_not_fused.end());

        compare(network_not_fused, network_fused, p);
    }

    layout get_input_layout(concat_test_params& p) {
        return layout{ p.in_shape, p.data_type, p.input_format };
    }

    layout get_per_channel_layout(concat_test_params& p) {
        return layout{ { 1, p.in_shape[1] }, p.default_type, p.default_format };
    }
};
}  // namespace


/* ----------------------------------------------------------------------------------------------------- */
/* --------------------------------------- Concat cases ------------------------------------------------ */
/* ----------------------------------------------------------------------------------------------------- */
#define CASE_CONCAT_F32_1 { 1, 8, 4, 4 }, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_CONCAT_F16_1 { 1, 8, 4, 4 }, data_types::f16, format::bfyx, data_types::f16, format::bfyx

class concat_activation : public ConcatFusingTest {};
TEST_P(concat_activation, along_f) {
    auto p = GetParam();

    if (engine.get_device_info().supports_immad)
        p.expected_fused_primitives++;

    create_topologies(
        input_layout("input0", get_input_layout(p)),
        input_layout("input1", get_input_layout(p)),
        concatenation("concat", { input_info("input0"), input_info("input1") }, 1, p.data_type),
        activation("act1", input_info("concat"), activation_func::round_half_to_even),
        activation("act2", input_info("act1"), activation_func::clamp, { -0.5f, 0.5f }),
        reorder("reorder_bfyx", input_info("act2"), cldnn::format::bfyx, p.default_type)
    );

    tolerance = default_tolerance(p.data_type);
    execute(p);
}

class concat_eltwise_with_broadcast : public ConcatFusingTest {};
TEST_P(concat_eltwise_with_broadcast, along_f) {
    auto p = GetParam();
    layout data_layout({ 1, p.in_shape[1] * 2 }, p.default_type, p.default_format);
    create_topologies(
        input_layout("input0", get_input_layout(p)),
        input_layout("input1", get_input_layout(p)),
        data("scale_data", get_mem(data_layout, 1.0f / tensor{ 1, 1, 4, 4 }.count())),
        concatenation("concat", { input_info("input0"), input_info("input1") }, 1, p.data_type),
        eltwise("scale", { input_info("concat"), input_info("scale_data") }, eltwise_mode::prod, p.default_type),
        reorder("reorder_bfyx", input_info("scale"), cldnn::format::bfyx, p.default_type)
    );

    tolerance = default_tolerance(p.data_type);
    execute(p);
}

class concat_eltwise_wo_broadcast : public ConcatFusingTest {};
TEST_P(concat_eltwise_wo_broadcast, along_f) {
    auto p = GetParam();
    ov::PartialShape concatenated_shape = p.in_shape;
    concatenated_shape[1] *= 2;
    layout data_layout(concatenated_shape, p.default_type, p.default_format);
    create_topologies(
        input_layout("input0", get_input_layout(p)),
        input_layout("input1", get_input_layout(p)),
        data("scale_data", get_mem(data_layout, 1.0f / tensor{ 1, 1, 4, 4 }.count())),
        concatenation("concat", { input_info("input0"), input_info("input1") }, 1, p.data_type),
        eltwise("scale", { input_info("concat"), input_info("scale_data") }, eltwise_mode::prod, p.default_type),
        reorder("reorder_bfyx", input_info("scale"), cldnn::format::bfyx, p.default_type)
    );

    tolerance = default_tolerance(p.data_type);
    execute(p);
}

class concat_quantize : public ConcatFusingTest {};
TEST_P(concat_quantize, along_f) {
    auto p = GetParam();
    create_topologies(
        input_layout("input0", get_input_layout(p)),
        input_layout("input1", get_input_layout(p)),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), 0)),
        data("out_hi", get_mem(get_single_element_layout(p), 255)),
        concatenation("concat", { input_info("input0"), input_info("input1") }, 1, p.data_type),
        quantize("quantize", input_info("concat"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 256, data_types::u8),
        reorder("reorder_bfyx", input_info("quantize"), cldnn::format::bfyx, p.default_type)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, concat_activation, ::testing::ValuesIn(std::vector<concat_test_params>{
    concat_test_params{ CASE_CONCAT_F32_1, 3, 5, "" },
    concat_test_params{ CASE_CONCAT_F16_1, 3, 5, "" },
}));

INSTANTIATE_TEST_SUITE_P(fusings_gpu, concat_eltwise_with_broadcast, ::testing::ValuesIn(std::vector<concat_test_params>{
    concat_test_params{ CASE_CONCAT_F32_1, 4, 4, "" },
    concat_test_params{ CASE_CONCAT_F16_1, 4, 4, "" },
}));

INSTANTIATE_TEST_SUITE_P(fusings_gpu, concat_eltwise_wo_broadcast, ::testing::ValuesIn(std::vector<concat_test_params>{
    concat_test_params{ CASE_CONCAT_F32_1, 4, 4, "" },
    concat_test_params{ CASE_CONCAT_F16_1, 4, 4, "" },
}));

INSTANTIATE_TEST_SUITE_P(fusings_gpu, concat_quantize, ::testing::ValuesIn(std::vector<concat_test_params>{
    concat_test_params{ CASE_CONCAT_F32_1, 4, 4, "" },
    concat_test_params{ CASE_CONCAT_F16_1, 4, 4, "" },
}));
