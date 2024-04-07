// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "fusion_test_common.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/quantize.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/fully_connected.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/crop.hpp>

#include <cmath>

using namespace cldnn;
using namespace ::tests;

namespace {
struct fully_connected_test_params {
    ov::PartialShape in_shape;
    ov::PartialShape out_shape;
    ov::PartialShape weights_shape;
    data_types data_type;
    format input_format;
    data_types weights_type;
    format weights_format;
    data_types default_type;
    format default_format;
    size_t expected_fused_primitives;
    size_t expected_not_fused_primitives;
};

class FullyConnectedFusingTest : public ::BaseFusingTest<fully_connected_test_params> {
public:

    void execute(fully_connected_test_params& p, bool is_dynamic = false) {
        cfg_not_fused.set_property(ov::intel_gpu::allow_new_shape_infer(is_dynamic));
        cfg_fused.set_property(ov::intel_gpu::allow_new_shape_infer(is_dynamic));
        auto input_prim = this->get_mem(get_input_layout(p));
        network network_not_fused(this->engine, this->topology_non_fused, this->cfg_not_fused);
        network network_fused(this->engine, this->topology_fused, this->cfg_fused);
        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);

        this->compare(network_not_fused, network_fused, p);
    }

    layout get_input_layout(fully_connected_test_params& p) {
        return layout{ p.in_shape, p.data_type, p.input_format,};
    }

    layout get_per_channel_layout(fully_connected_test_params& p) {
        return layout{ ov::PartialShape{1, p.out_shape[1]}, p.default_type, p.default_format };
    }

    size_t get_output_dim_size(fully_connected_test_params& p) {
        return p.out_shape.size();
    }

    layout get_weights_layout(fully_connected_test_params& p) {
        return layout{ p.weights_shape, p.weights_type, p.weights_format };
    }

    size_t get_input_weights_rank(fully_connected_test_params& p) {
        return p.weights_shape.size();
    }

    layout get_bias_layout(fully_connected_test_params& p) {
        auto bias_shape = p.out_shape.size() == 3 ? ov::PartialShape{1, 1, p.out_shape[2]} : ov::PartialShape{1, p.out_shape[1]};
        return layout{ bias_shape, p.default_type, p.default_format };
    }

    layout get_scale_layout(fully_connected_test_params& p, size_t group_size = 1) {
        if (p.weights_type == data_types::u8 || p.weights_type == data_types::i8) {
            auto scale_shape = p.out_shape.size() == 3 ? ov::PartialShape{p.out_shape[2]} : ov::PartialShape{p.out_shape[1]};
            return layout{ scale_shape, p.default_type, p.default_format };
        } else {
            auto groups_num = p.in_shape.size() == 3 ? p.in_shape[2] / group_size : p.in_shape[1] / group_size;
            auto scale_shape = p.out_shape.size() == 3 ? ov::PartialShape{p.out_shape[2], groups_num} : ov::PartialShape{p.out_shape[1], groups_num};
            return layout{ scale_shape, p.default_type, p.default_format };
        }
    }
};


#ifdef ENABLE_ONEDNN_FOR_GPU
class FullyConnectedFusingTestOneDNN : public BaseFusingTest<fully_connected_test_params> {
public:
    void execute(fully_connected_test_params& p, bool is_caching_test = false, bool is_dynamic = false) {
        // Onednn post operation has issue in a machine that does not support imad.
        if (!engine.get_device_info().supports_immad)
            return;

        auto input_prim = p.data_type == data_types::u8 ? get_mem(get_input_layout(p), 0, 10) : get_mem(get_input_layout(p));

        auto impl_forcing = cfg_fused.get_property(ov::intel_gpu::force_implementations);
        auto forcing_format = p.input_format;
        for (auto& forcing : impl_forcing)
            if (forcing.first == "fc_prim")
                forcing_format = forcing.second.output_format;

        ov::intel_gpu::ImplementationDesc fc_impl = { forcing_format, "", impl_types::onednn };
        cfg_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "fc_prim", fc_impl } }));
        cfg_fused.set_property(ov::intel_gpu::allow_new_shape_infer(is_dynamic));

        network::ptr network_not_fused = get_network(this->engine, this->topology_non_fused, cfg_not_fused, get_test_stream_ptr(cfg_not_fused), is_caching_test);
        network::ptr network_fused = get_network(this->engine, this->topology_fused, cfg_fused, get_test_stream_ptr(cfg_fused), is_caching_test);
        network_fused->set_input_data("input", input_prim);
        network_not_fused->set_input_data("input", input_prim);

        compare(*network_not_fused, *network_fused, p);
    }

    layout get_input_layout(fully_connected_test_params& p) {
        return layout{ p.in_shape, p.data_type, p.input_format,};
    }

    layout get_per_channel_layout(fully_connected_test_params& p) {
        return layout{ ov::PartialShape{1, p.out_shape[1]}, p.default_type, p.default_format };
    }

    size_t get_output_dim_size(fully_connected_test_params& p) {
        return p.out_shape.size();
    }

    layout get_weights_layout(fully_connected_test_params& p) {
        return layout{ p.weights_shape, p.weights_type, p.weights_format };
    }

    size_t get_input_weights_rank(fully_connected_test_params& p) {
        return p.weights_shape.size();
    }

    layout get_bias_layout(fully_connected_test_params& p) {
        auto bias_shape = p.out_shape.size() == 3 ? ov::PartialShape{1, 1, p.out_shape[2]} : ov::PartialShape{1, p.out_shape[1]};
        return layout{ bias_shape, p.default_type, p.default_format };
    }

    layout get_output_layout(fully_connected_test_params& p) {
        return layout{ p.out_shape, p.data_type, p.input_format };
    }
};
#endif  // ENABLE_ONEDNN_FOR_GPU

}  // namespace

// in_shape; out_shape; kernel;  data_type; input_format; weights_type; weights_format; default_type; default_format;
#define CASE_FC_FP32_1 { 1, 3 }, { 1, 4 }, { 4, 3 }, data_types::f32, format::bfyx, data_types::f32, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_FP32_2 { 2, 3 }, { 2, 4 }, { 4, 3 }, data_types::f32, format::yxfb, data_types::f32, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_FP32_3 { 2, 32 }, { 2, 16 }, { 16, 32 }, data_types::f32, format::bfyx, data_types::i8, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_FP32_3D_1 { 5, 3, 3 }, { 5, 3, 5 }, { 5, 3, 1 }, data_types::f32, format::bfyx, data_types::f32, format::os_iyx_osv16, data_types::f32, format::bfyx
#define CASE_FC_FP32_3D_2 { 2, 1, 1 }, { 2, 1, 32 }, { 32, 1, 1 }, data_types::f32, format::bfyx, data_types::f32, format::os_iyx_osv16, data_types::f32, format::bfyx
#define CASE_FC_FP32_3D_3 { 2, 32, 32 }, { 2, 32, 16 }, { 16, 32, 1 }, data_types::f32, format::bfyx, data_types::f32, format::os_iyx_osv16, data_types::f32, format::bfyx

#define DYN_CASE_FC_FP32_3D_1 { 5, 3, 3 }, { 5, 3, 5 }, { 5, 3 }, data_types::f32, format::bfyx, data_types::f32, format::os_iyx_osv16, data_types::f32, format::bfyx
#define DYN_CASE_FC_FP32_3D_2 { 2, 1, 1 }, { 2, 1, 32 }, { 32, 1 }, data_types::f32, format::bfyx, data_types::f32, format::os_iyx_osv16, data_types::f32, format::bfyx
#define DYN_CASE_FC_FP32_3D_3 { 2, 32, 32 }, { 2, 32, 16 }, { 16, 32 }, data_types::f32, format::bfyx, data_types::f32, format::os_iyx_osv16, data_types::f32, format::bfyx

#define CASE_FC_U8S8_1 { 1, 3 }, { 1, 4 }, { 4, 3 }, data_types::u8, format::bfyx, data_types::i8, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_U8S8_2 { 2, 3 }, { 2, 4 }, { 4, 3 }, data_types::u8, format::b_fs_yx_fsv4, data_types::i8, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_U8S8_3 { 2, 32 }, { 2, 16 }, { 16, 32 }, data_types::u8, format::b_fs_yx_fsv4, data_types::i8, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_U8S8_4 { 1, 3 }, { 1, 3 }, { 3, 3 }, data_types::u8, format::bfyx, data_types::i8, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_U8S8_3D_1 { 2, 32, 3 }, { 2, 32, 16 }, { 16, 3, 1 }, data_types::u8, format::bfyx, data_types::i8, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_U8S8_3D_2 { 1, 1, 3 }, { 1, 1, 32 }, { 32, 3, 1 }, data_types::u8, format::bfyx, data_types::i8, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_U8S8_3D_3 { 2, 3, 1 }, { 2, 3, 15 }, { 15, 1, 1 }, data_types::u8, format::bfyx, data_types::i8, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_U8S8_3D_4 { 1, 512, 1024 }, { 1, 384, 1024 }, { 1024, 1024, 1 }, data_types::u8, format::bfyx, data_types::i8, format::oiyx, data_types::f32, format::bfyx

#define CASE_FC_FP16_1 { 1, 3 }, { 1, 4 }, { 4, 3 }, data_types::f16, format::bfyx, data_types::f16, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_FP16_2 { 2, 3 }, { 2, 4 }, { 4, 3 }, data_types::f16, format::bfyx, data_types::f16, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_FP16_3 { 2, 32 }, { 2, 16 }, { 16, 32 }, data_types::f16, format::bfyx, data_types::f16, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_FP16_4 { 128, 76 }, { 128, 768 }, { 768, 76 }, data_types::f16, format::bfyx, data_types::f16, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_FP16_5 { 1, 128, 76 }, { 1, 128, 768 }, { 1, 768, 76 }, data_types::f16, format::bfyx, data_types::f16, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_FP16_6 { 2, 1, 76 }, { 2, 1, 768 }, { 768, 76, 1 }, data_types::f16, format::bfyx, data_types::f16, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_FP16_7 { 2, 128, 76 }, { 2, 128, 768 }, { 768, 76, 1 }, data_types::f16, format::bfyx, data_types::f16, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_FP16_3D_1 { 2, 32, 3 }, { 2, 32, 16 }, { 16, 3, 1 }, data_types::f16, format::bfyx, data_types::f16, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_FP16_3D_2 { 1, 1, 3 }, { 1, 1, 32 }, { 32, 3, 1 }, data_types::f16, format::bfyx, data_types::f16, format::oiyx, data_types::f32, format::bfyx

#define CASE_FC_FP16_INT4_COMP_1 { 1, 128 }, { 1, 128 }, { 128, 128 }, data_types::f16, format::bfyx, data_types::u4, format::oiyx, data_types::f16, format::bfyx

/* ----------------------------------------------------------------------------------------------------- */
/* ---------------------------------------- FC cases --------------------------------------------------- */
/* ----------------------------------------------------------------------------------------------------- */
class fc_fp32_activation : public FullyConnectedFusingTest {};
TEST_P(fc_fp32_activation, basic) {
    // TODO: Fix me, refer PR(#15873)
    if (engine.get_device_info().supports_immad)
        return;
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_bias_layout(p))),
        fully_connected("fc_prim", input_info("input"), "weights", "bias", padding(), get_output_dim_size(p), get_input_weights_rank(p)),
        activation("activation", input_info("fc_prim"), activation_func::abs),
        reorder("reorder_bfyx", input_info("activation"), p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}
INSTANTIATE_TEST_SUITE_P(fusings_gpu, fc_fp32_activation, ::testing::ValuesIn(std::vector<fully_connected_test_params>{
    fully_connected_test_params{ CASE_FC_FP32_1, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP32_2, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP32_3, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP32_3D_1, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP32_3D_2, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP32_3D_3, 2, 3 },
}));

class fc_fp32_activation_dynamic : public FullyConnectedFusingTest {};
TEST_P(fc_fp32_activation_dynamic, basic) {
    auto p = GetParam();
    auto test_input_layout = get_input_layout(p);
    auto dynamic_input_layout = layout{ov::PartialShape::dynamic(test_input_layout.get_partial_shape().size()), test_input_layout.data_type, test_input_layout.format};
    create_topologies(
        input_layout("input", dynamic_input_layout),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_bias_layout(p))),
        fully_connected("fc_prim", input_info("input"), "weights", "bias", padding(), get_output_dim_size(p), get_input_weights_rank(p)),
        activation("activation", input_info("fc_prim"), activation_func::abs),
        reorder("reorder_bfyx", input_info("activation"), p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p, true);
}
INSTANTIATE_TEST_SUITE_P(fusings_gpu, fc_fp32_activation_dynamic, ::testing::ValuesIn(std::vector<fully_connected_test_params>{
    fully_connected_test_params{ CASE_FC_FP32_1, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP32_2, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP32_3, 2, 3 },
    fully_connected_test_params{ DYN_CASE_FC_FP32_3D_1, 2, 3 },
    fully_connected_test_params{ DYN_CASE_FC_FP32_3D_2, 2, 3 },
    fully_connected_test_params{ DYN_CASE_FC_FP32_3D_3, 2, 3 },
}));

class fc_fp32_bias : public FullyConnectedFusingTest {};
TEST_P(fc_fp32_bias, basic) {
    // TODO: Fix me, refer PR(#15873)
    if (engine.get_device_info().supports_immad)
        return;
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_bias_layout(p))),
        fully_connected("fc_prim", input_info("input"), "weights", "", padding(), get_output_dim_size(p), get_input_weights_rank(p)),
        eltwise("bias_add", { input_info("fc_prim"), input_info("bias") }, eltwise_mode::sum),
        reorder("reorder_bfyx", input_info("bias_add"), p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, fc_fp32_bias, ::testing::ValuesIn(std::vector<fully_connected_test_params>{
    fully_connected_test_params{ CASE_FC_FP32_1, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP32_2, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP32_3, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP32_3D_1, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP32_3D_2, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP32_3D_3, 2, 3 },
}));

class fc_fp32_bias_dynamic : public FullyConnectedFusingTest {};
TEST_P(fc_fp32_bias_dynamic, basic) {
    auto p = GetParam();
    auto test_input_layout = get_input_layout(p);
    auto dynamic_input_layout = layout{ov::PartialShape::dynamic(test_input_layout.get_partial_shape().rank()), test_input_layout.data_type, test_input_layout.format};
    create_topologies(
        input_layout("input", dynamic_input_layout),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_bias_layout(p))),
        fully_connected("fc_prim", input_info("input"), "weights", "", padding(), get_output_dim_size(p), get_input_weights_rank(p)),
        eltwise("bias_add", { input_info("fc_prim"), input_info("bias") }, eltwise_mode::sum),
        reorder("reorder_bfyx", input_info("bias_add"), p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p, true);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, fc_fp32_bias_dynamic, ::testing::ValuesIn(std::vector<fully_connected_test_params>{
    fully_connected_test_params{ CASE_FC_FP32_1, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP32_2, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP32_3, 2, 3 },
    fully_connected_test_params{ DYN_CASE_FC_FP32_3D_1, 2, 3 },
    fully_connected_test_params{ DYN_CASE_FC_FP32_3D_2, 2, 3 },
    fully_connected_test_params{ DYN_CASE_FC_FP32_3D_3, 2, 3 },
}));

class fc_compressed_int8_bias_dynamic : public FullyConnectedFusingTest {};
TEST_P(fc_compressed_int8_bias_dynamic, basic) {
    auto p = GetParam();
    auto test_input_layout = get_input_layout(p);
    auto dynamic_input_layout = layout{ov::PartialShape::dynamic(test_input_layout.get_partial_shape().rank()), test_input_layout.data_type, test_input_layout.format};

    auto fc_prim = fully_connected("fc_prim", input_info("input"), "weights", "", "scale", "", data_types::f16, padding(), get_output_dim_size(p), get_input_weights_rank(p));
    fc_prim.decompression_zero_point_scalar = 8.0f;

    create_topologies(
        input_layout("input", dynamic_input_layout),
        data("weights", get_mem(get_weights_layout(p))),
        data("scale", get_mem(get_scale_layout(p, 128))),
        data("bias", get_mem(get_bias_layout(p))),
        fc_prim,
        eltwise("bias_add", { input_info("fc_prim"), input_info("bias") }, eltwise_mode::sum),
        reorder("reorder_bfyx", input_info("bias_add"), p.default_format, data_types::f32)
    );

    tolerance = 1.0f;
    execute(p, true);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, fc_compressed_int8_bias_dynamic, ::testing::ValuesIn(std::vector<fully_connected_test_params>{
    fully_connected_test_params{ CASE_FC_FP16_INT4_COMP_1, 2, 3 },
}));

class fc_int8_eltwise_dynamic_residual : public FullyConnectedFusingTest {};
TEST_P(fc_int8_eltwise_dynamic_residual, basic) {
    // The basic purpose of this test is to check crash in fake aligned shape check
    if (engine.get_device_info().supports_immad)
        return;
    auto p = GetParam();
    auto test_input_layout = get_input_layout(p);
    auto dynamic_input_layout = layout{ov::PartialShape::dynamic(test_input_layout.get_partial_shape().rank()), test_input_layout.data_type, test_input_layout.format};
    create_topologies(
        input_layout("input", dynamic_input_layout),
        data("weight", get_mem(get_weights_layout(p))),
        reorder("reorder", input_info("input"), p.default_format, data_types::i8),
        eltwise("mul", { input_info("reorder"), input_info("input") }, eltwise_mode::div),
        fully_connected("fc", input_info("mul"), "weight", "", data_types::i8, padding(), get_output_dim_size(p), get_input_weights_rank(p)),
        eltwise("add", { input_info("fc"), input_info("reorder") }, eltwise_mode::sum),
        reorder("reorder_bfyx", input_info("add"), p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p, true);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, fc_int8_eltwise_dynamic_residual, ::testing::ValuesIn(std::vector<fully_connected_test_params>{
    fully_connected_test_params{ CASE_FC_U8S8_4, 3, 4 },
}));


class fc_int8_eltwise : public FullyConnectedFusingTest {};
TEST_P(fc_int8_eltwise, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_bias_layout(p))),
        data("eltwise_data", get_mem(get_per_channel_layout(p), 1, 9)),
        fully_connected("fc_prim", input_info("input"), "weights", "bias", padding(), get_output_dim_size(p), get_input_weights_rank(p)),
        eltwise("eltwise", { input_info("fc_prim"), input_info("eltwise_data") }, eltwise_mode::sum),
        reorder("reorder_bfyx", input_info("eltwise"), p.default_format, data_types::f32)
    );

    tolerance = default_tolerance(p.data_type);
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, fc_int8_eltwise, ::testing::ValuesIn(std::vector<fully_connected_test_params>{
    fully_connected_test_params{ CASE_FC_U8S8_1, 2, 3 },
    fully_connected_test_params{ CASE_FC_U8S8_2, 2, 3 },
    fully_connected_test_params{ CASE_FC_U8S8_3, 2, 3 },
}));

class fc_int8_quantize_u8 : public FullyConnectedFusingTest {};
TEST_P(fc_int8_quantize_u8, basic) {
    // TODO: Fix me, refer PR(#15873)
    if (engine.get_device_info().supports_immad)
        return;
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_bias_layout(p))),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), 0)),
        data("out_hi", get_mem(get_single_element_layout(p), 255)),
        fully_connected("fc_prim", input_info("input"), "weights", "bias", data_types::f32, padding(), get_output_dim_size(p), get_input_weights_rank(p)),
        quantize("quantize", input_info("fc_prim"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 256, data_types::u8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, fc_int8_quantize_u8, ::testing::ValuesIn(std::vector<fully_connected_test_params>{
    fully_connected_test_params{ CASE_FC_U8S8_1, 2, 3 },
    fully_connected_test_params{ CASE_FC_U8S8_2, 2, 3 },
    fully_connected_test_params{ CASE_FC_U8S8_3, 2, 3 },
    fully_connected_test_params{ CASE_FC_U8S8_3D_1, 2, 3 },
    fully_connected_test_params{ CASE_FC_U8S8_3D_2, 2, 3 },
    fully_connected_test_params{ CASE_FC_U8S8_3D_3, 2, 3 },
}));

class fc_int8_eltwise_quantize_i8 : public FullyConnectedFusingTest {};
TEST_P(fc_int8_eltwise_quantize_i8, basic) {
    // TODO: Fix me, refer PR(#15873)
    if (engine.get_device_info().supports_immad)
        return;
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_bias_layout(p))),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -127)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        data("eltwise_data", get_mem(get_per_channel_layout(p), 1.0f / get_weights_layout(p).count() / 255)),
        fully_connected("fc_prim", input_info("input"), "weights", "bias", data_types::f32, padding(), get_output_dim_size(p), get_input_weights_rank(p)),
        eltwise("eltwise", { input_info("fc_prim"), input_info("eltwise_data") }, eltwise_mode::prod),
        quantize("quantize", input_info("eltwise"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 255, data_types::i8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, fc_int8_eltwise_quantize_i8, ::testing::ValuesIn(std::vector<fully_connected_test_params>{
    fully_connected_test_params{ CASE_FC_U8S8_1, 2, 4 },
    fully_connected_test_params{ CASE_FC_U8S8_2, 2, 4 },
    fully_connected_test_params{ CASE_FC_U8S8_3, 2, 4 },
    fully_connected_test_params{ CASE_FC_U8S8_3D_1, 2, 4 },
    fully_connected_test_params{ CASE_FC_U8S8_3D_2, 2, 4 },
    fully_connected_test_params{ CASE_FC_U8S8_3D_3, 2, 4 },
}));

class fc_int8_eltwise_activation_quantize_i8 : public FullyConnectedFusingTest {};
TEST_P(fc_int8_eltwise_activation_quantize_i8, basic) {
    // TODO: Fix me, refer PR(#15873)
    if (engine.get_device_info().supports_immad)
        return;
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_bias_layout(p))),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -127)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        data("eltwise_data", get_mem(get_per_channel_layout(p), 1.0f / get_weights_layout(p).count() / 255)),
        fully_connected("fc_prim", input_info("input"), "weights", "bias", data_types::f32, padding(), get_output_dim_size(p), get_input_weights_rank(p)),
        eltwise("eltwise", { input_info("fc_prim"), input_info("eltwise_data") }, eltwise_mode::prod),
        activation("activation_eltwise", input_info("eltwise"), activation_func::exp),
        quantize("quantize", input_info("activation_eltwise"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 255, data_types::i8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, fc_int8_eltwise_activation_quantize_i8, ::testing::ValuesIn(std::vector<fully_connected_test_params>{
    fully_connected_test_params{ CASE_FC_U8S8_1, 2, 5 },
    fully_connected_test_params{ CASE_FC_U8S8_2, 2, 5 },
    fully_connected_test_params{ CASE_FC_U8S8_3, 2, 5 },

    fully_connected_test_params{ CASE_FC_U8S8_3D_1, 2, 5 },
    fully_connected_test_params{ CASE_FC_U8S8_3D_2, 2, 5 },
    fully_connected_test_params{ CASE_FC_U8S8_3D_3, 2, 5 },

    fully_connected_test_params{ CASE_FC_FP32_3D_1, 3, 5 },
    fully_connected_test_params{ CASE_FC_FP32_3D_2, 3, 5 },
    fully_connected_test_params{ CASE_FC_FP32_3D_3, 3, 5 },
}));

#ifdef ENABLE_ONEDNN_FOR_GPU

// FC onednn sum case
class fc_int8_inputs_fused_fp32_sum : public FullyConnectedFusingTestOneDNN {
public:
    void run_test(bool is_caching_test = false) {
        auto p = GetParam();
        auto shift_layout = layout{ ov::PartialShape{p.weights_shape[0]}, p.default_type, p.default_format };

        create_topologies(
            input_layout("input", get_input_layout(p)),
            data("weights", get_mem(get_weights_layout(p))),
            data("bias", get_mem(get_bias_layout(p))),
            data("shift_data", get_mem(shift_layout, 1)),
            fully_connected("fc_prim", input_info("input"), "weights", "bias", cldnn::data_types::f32, padding(), get_output_dim_size(p), get_input_weights_rank(p)),
            eltwise("shift", { input_info("fc_prim"), input_info("shift_data") }, eltwise_mode::sum, cldnn::data_types::f32),
            crop("crop", input_info("shift"), get_output_layout(p).get_tensor(), { 0, 0, 0, 0 }),
            reorder("reorder_bfyx", input_info("crop"), p.default_format, data_types::f32)
        );

        tolerance = 1.f;
        execute(p, is_caching_test);
    }
};

TEST_P(fc_int8_inputs_fused_fp32_sum, basic) {
    run_test(false);
}

TEST_P(fc_int8_inputs_fused_fp32_sum, basic_cached) {
    run_test(true);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, fc_int8_inputs_fused_fp32_sum, ::testing::ValuesIn(std::vector<fully_connected_test_params>{
    // OneDNN has issue with small shapes - ticket 7064
    // fully_connected_test_params{ CASE_FC_U8S8_3D_1, 2, 4 },
    // fully_connected_test_params{ CASE_FC_U8S8_3D_2, 2, 4 },
    fully_connected_test_params{ CASE_FC_U8S8_3D_4, 2, 4 },
}));


class fc_fp16_eltwise_add : public FullyConnectedFusingTestOneDNN {
public:
    void run_test(bool is_caching_test = false) {
        auto p = GetParam();
        create_topologies(
            input_layout("input", get_input_layout(p)),
            data("weights", get_mem(get_weights_layout(p))),
            data("bias", get_mem(get_bias_layout(p))),
            data("eltwise_data", get_mem(get_per_channel_layout(p), 1, 9)),
            fully_connected("fc_prim", input_info("input"), "weights", "bias", padding(), get_output_dim_size(p)),
            eltwise("eltwise", { input_info("fc_prim"), input_info("eltwise_data") }, eltwise_mode::sum),
            reorder("reorder_bfyx", input_info("eltwise"), p.default_format, data_types::f32)
        );

        tolerance = 1e-2f;
        execute(p, is_caching_test);
    }
};

TEST_P(fc_fp16_eltwise_add, basic) {
    run_test(false);
}

TEST_P(fc_fp16_eltwise_add, basic_cached) {
    run_test(true);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, fc_fp16_eltwise_add, ::testing::ValuesIn(std::vector<fully_connected_test_params>{
    fully_connected_test_params{ CASE_FC_FP16_1, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP16_2, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP16_3, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP16_4, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP16_5, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP16_6, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP16_7, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP16_3D_1, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP16_3D_2, 2, 3 },
}));

class fc_fp16_eltwise_add_dynamic : public FullyConnectedFusingTestOneDNN {};
TEST_P(fc_fp16_eltwise_add_dynamic, basic) {
    auto p = GetParam();
    auto test_input_layout = get_input_layout(p);
    auto dynamic_input_layout = layout{ov::PartialShape::dynamic(test_input_layout.get_partial_shape().size()), test_input_layout.data_type, test_input_layout.format};
    create_topologies(
        input_layout("input", dynamic_input_layout),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_bias_layout(p))),
        data("eltwise_data", get_mem(get_per_channel_layout(p), 1, 9)),
        fully_connected("fc_prim", input_info("input"), "weights", "bias", padding(), get_output_dim_size(p)),
        eltwise("eltwise", { input_info("fc_prim"), input_info("eltwise_data") }, eltwise_mode::sum),
        reorder("reorder_bfyx", input_info("eltwise"), p.default_format, data_types::f32)
    );

    tolerance = 1e-2f;
    execute(p, false, true);
}

INSTANTIATE_TEST_SUITE_P(DISABLED_fusings_gpu, fc_fp16_eltwise_add_dynamic, ::testing::ValuesIn(std::vector<fully_connected_test_params>{
    fully_connected_test_params{ CASE_FC_FP16_3, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP16_4, 2, 3 },
}));

class fc_fp16_eltwise_sub : public FullyConnectedFusingTestOneDNN {
public:
    void run_test(bool is_caching_test = false) {
        auto p = GetParam();
        create_topologies(
            input_layout("input", get_input_layout(p)),
            data("weights", get_mem(get_weights_layout(p))),
            data("bias", get_mem(get_bias_layout(p))),
            data("eltwise_data", get_mem(get_per_channel_layout(p), 1, 9)),
            fully_connected("fc_prim", input_info("input"), "weights", "bias", padding(), get_output_dim_size(p)),
            eltwise("eltwise", { input_info("fc_prim"), input_info("eltwise_data") }, eltwise_mode::sub),
            reorder("reorder_bfyx", input_info("eltwise"), p.default_format, data_types::f32)
        );

        tolerance = 1e-1f;
        execute(p, is_caching_test);
    }
};

TEST_P(fc_fp16_eltwise_sub, basic) {
    run_test(false);
}

TEST_P(fc_fp16_eltwise_sub, basic_cached) {
    run_test(true);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, fc_fp16_eltwise_sub, ::testing::ValuesIn(std::vector<fully_connected_test_params>{
    fully_connected_test_params{ CASE_FC_FP16_1, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP16_2, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP16_3, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP16_3D_1, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP16_3D_2, 2, 3 },
}));

class fc_fp16_eltwise_prod : public FullyConnectedFusingTestOneDNN {
public:
    void run_test(bool is_caching_test = false) {
        auto p = GetParam();
        create_topologies(
            input_layout("input", get_input_layout(p)),
            data("weights", get_mem(get_weights_layout(p))),
            data("bias", get_mem(get_bias_layout(p))),
            data("eltwise_data", get_mem(get_per_channel_layout(p), 1, 9)),
            fully_connected("fc_prim", input_info("input"), "weights", "bias", padding(), get_output_dim_size(p)),
            eltwise("eltwise", { input_info("fc_prim"), input_info("eltwise_data") }, eltwise_mode::prod),
            reorder("reorder_bfyx", input_info("eltwise"), p.default_format, data_types::f32)
        );

        tolerance = 1e-1f;
        execute(p, is_caching_test);
    }
};

TEST_P(fc_fp16_eltwise_prod, basic) {
    run_test(false);
}

TEST_P(fc_fp16_eltwise_prod, basic_cached) {
    run_test(true);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, fc_fp16_eltwise_prod, ::testing::ValuesIn(std::vector<fully_connected_test_params>{
    fully_connected_test_params{ CASE_FC_FP16_1, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP16_2, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP16_3, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP16_3D_1, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP16_3D_2, 2, 3 },
}));

class fc_fp16_eltwise_sum : public FullyConnectedFusingTestOneDNN {
public:
    void run_test(bool is_caching_test = false) {
        auto p = GetParam();
        create_topologies(
            input_layout("input", get_input_layout(p)),
            data("weights", get_mem(get_weights_layout(p))),
            data("bias", get_mem(get_bias_layout(p))),
            data("eltwise_data", get_mem(get_output_layout(p))),
            fully_connected("fc_prim", input_info("input"), "weights", "bias", padding(), get_output_dim_size(p)),
            eltwise("sum", { input_info("fc_prim"), input_info("eltwise_data") }, eltwise_mode::sum),
            reorder("reorder_bfyx", input_info("sum"), p.default_format, data_types::f32)
        );

        tolerance = 1e-1f;
        execute(p, is_caching_test);
    }
};

TEST_P(fc_fp16_eltwise_sum, basic) {
    run_test(false);
}

TEST_P(fc_fp16_eltwise_sum, basic_cached) {
    run_test(true);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, fc_fp16_eltwise_sum, ::testing::ValuesIn(std::vector<fully_connected_test_params>{
    fully_connected_test_params{ CASE_FC_FP16_1, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP16_2, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP16_3, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP16_3D_1, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP16_3D_2, 2, 3 },
}));

class fc_fp32_activation_prelu : public FullyConnectedFusingTestOneDNN {};
TEST_P(fc_fp32_activation_prelu, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_bias_layout(p))),
        data("data", get_mem(get_bias_layout(p))),
        fully_connected("fc_prim", input_info("input"), "weights", "bias", padding(), get_output_dim_size(p), get_input_weights_rank(p)),
        activation("activation", input_info("fc_prim"), "data", activation_func::relu_negative_slope),
        reorder("reorder_bfyx", input_info("activation"), p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, fc_fp32_activation_prelu, ::testing::ValuesIn(std::vector<fully_connected_test_params>{
    fully_connected_test_params{ CASE_FC_FP32_1, 3, 3 }
}));

class fc_fp32_activation_relu : public FullyConnectedFusingTestOneDNN {};
TEST_P(fc_fp32_activation_relu, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_bias_layout(p))),
        fully_connected("fc_prim", input_info("input"), "weights", "bias", padding(), get_output_dim_size(p), get_input_weights_rank(p)),
        activation("activation", input_info("fc_prim"), activation_func::relu_negative_slope),
        reorder("reorder_bfyx", input_info("activation"), p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, fc_fp32_activation_relu, ::testing::ValuesIn(std::vector<fully_connected_test_params>{
    fully_connected_test_params{ CASE_FC_FP32_1, 2, 3 }
}));
#endif
