// Copyright (C) 2018-2022 Intel Corporation
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

    void execute(fully_connected_test_params& p) {
        auto input_prim = this->get_mem(get_input_layout(p));
        network network_not_fused(this->engine, this->topology_non_fused, this->bo_not_fused);
        network network_fused(this->engine, this->topology_fused, this->bo_fused);
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

    layout get_bias_layout(fully_connected_test_params& p) {
        auto bias_shape = p.out_shape.size() == 3 ? ov::PartialShape{1, 1, p.out_shape[2]} : ov::PartialShape{1, p.out_shape[1]};
        return layout{ bias_shape, p.default_type, p.default_format };
    }
};


#ifdef ENABLE_ONEDNN_FOR_GPU
class FullyConnectedFusingTestOneDNN : public BaseFusingTest<fully_connected_test_params> {
public:
    void execute(fully_connected_test_params& p) {
        // Onednn post operation has issue in a machine that does not support imad.
        if (!engine.get_device_info().supports_immad)
            return;

        auto input_prim = p.data_type == data_types::u8 ? get_mem(get_input_layout(p), 0, 10) : get_mem(get_input_layout(p));

        auto impl_forcing_bo = bo_fused.get<build_option_type::force_implementations>();
        const auto& impl_forcing = impl_forcing_bo->forcing;

        auto forcing_format = p.input_format;
        for (auto& forcing : impl_forcing)
            if (forcing.first == "fc_prim")
                forcing_format = forcing.second.output_format;

        implementation_desc conv_impl = { forcing_format, "", impl_types::onednn };
        bo_fused.set_option(build_option::force_implementations({ { "fc_prim", conv_impl } }));

        network network_not_fused(this->engine, this->topology_non_fused, bo_not_fused);
        network network_fused(this->engine, this->topology_fused, bo_fused);
        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);

        compare(network_not_fused, network_fused, p);
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

#define CASE_FC_U8S8_1 { 1, 3 }, { 1, 4 }, { 4, 3 }, data_types::u8, format::bfyx, data_types::i8, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_U8S8_2 { 2, 3 }, { 2, 4 }, { 4, 3 }, data_types::u8, format::b_fs_yx_fsv4, data_types::i8, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_U8S8_3 { 2, 32 }, { 2, 16 }, { 16, 32 }, data_types::u8, format::b_fs_yx_fsv4, data_types::i8, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_U8S8_3D_1 { 2, 32, 3 }, { 2, 32, 16 }, { 16, 3, 1 }, data_types::u8, format::bfyx, data_types::i8, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_U8S8_3D_2 { 1, 1, 3 }, { 1, 1, 32 }, { 32, 3, 1 }, data_types::u8, format::bfyx, data_types::i8, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_U8S8_3D_3 { 2, 3, 1 }, { 2, 3, 15 }, { 15, 1, 1 }, data_types::u8, format::bfyx, data_types::i8, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_U8S8_3D_4 { 1, 512, 1024 }, { 1, 384, 1024 }, { 1024, 1024, 1 }, data_types::u8, format::bfyx, data_types::i8, format::oiyx, data_types::f32, format::bfyx

/* ----------------------------------------------------------------------------------------------------- */
/* ---------------------------------------- FC cases --------------------------------------------------- */
/* ----------------------------------------------------------------------------------------------------- */
class fc_fp32_activation : public FullyConnectedFusingTest {};
TEST_P(fc_fp32_activation, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_bias_layout(p))),
        fully_connected("fc_prim", "input", "weights", "bias", padding(), get_output_dim_size(p)),
        activation("activation", "fc_prim", activation_func::abs),
        reorder("reorder_bfyx", "activation", p.default_format, data_types::f32)
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

class fc_fp32_bias : public FullyConnectedFusingTest {};
TEST_P(fc_fp32_bias, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_bias_layout(p))),
        fully_connected("fc_prim", "input", "weights", "", padding(), get_output_dim_size(p)),
        eltwise("bias_add", { "fc_prim", "bias" }, eltwise_mode::sum),
        reorder("reorder_bfyx", "bias_add", p.default_format, data_types::f32)
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

class fc_int8_quantize_u8 : public FullyConnectedFusingTest {};
TEST_P(fc_int8_quantize_u8, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_bias_layout(p))),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), 0)),
        data("out_hi", get_mem(get_single_element_layout(p), 255)),
        fully_connected("fc_prim", "input", "weights", "bias", data_types::f32, padding(), get_output_dim_size(p)),
        quantize("quantize", "fc_prim", "in_lo", "in_hi", "out_lo", "out_hi", 256, data_types::u8),
        reorder("reorder_bfyx", "quantize", p.default_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu_fc, fc_int8_quantize_u8, ::testing::ValuesIn(std::vector<fully_connected_test_params>{
    fully_connected_test_params{ CASE_FC_U8S8_1, 2, 3 },
    fully_connected_test_params{ CASE_FC_U8S8_2, 2, 3 },
    fully_connected_test_params{ CASE_FC_U8S8_3, 2, 3 },
    fully_connected_test_params{ CASE_FC_U8S8_3D_1, 2, 3 },
    fully_connected_test_params{ CASE_FC_U8S8_3D_2, 2, 3 },
    fully_connected_test_params{ CASE_FC_U8S8_3D_3, 2, 3 },
}));

class fc_int8_eltwise_quantize_i8 : public FullyConnectedFusingTest {};
TEST_P(fc_int8_eltwise_quantize_i8, basic) {
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
        fully_connected("fc_prim", "input", "weights", "bias", data_types::f32, padding(), get_output_dim_size(p)),
        eltwise("eltwise", {"fc_prim", "eltwise_data"}, eltwise_mode::prod),
        quantize("quantize", "eltwise", "in_lo", "in_hi", "out_lo", "out_hi", 255, data_types::i8),
        reorder("reorder_bfyx", "quantize", p.default_format, data_types::f32)
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
        fully_connected("fc_prim", "input", "weights", "bias", data_types::f32, padding(), get_output_dim_size(p)),
        eltwise("eltwise", {"fc_prim", "eltwise_data"}, eltwise_mode::prod),
        activation("activation_eltwise", "eltwise", activation_func::exp),
        quantize("quantize", "activation_eltwise", "in_lo", "in_hi", "out_lo", "out_hi", 255, data_types::i8),
        reorder("reorder_bfyx", "quantize", p.default_format, data_types::f32)
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
class fc_int8_inputs_fused_fp32_sum : public FullyConnectedFusingTestOneDNN {};
TEST_P(fc_int8_inputs_fused_fp32_sum, basic) {
    auto p = GetParam();
    auto shift_layout = layout{ ov::PartialShape{p.weights_shape[0]}, p.default_type, p.default_format };

    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_bias_layout(p))),
        data("shift_data", get_mem(shift_layout, 1)),
        fully_connected("fc_prim", "input", "weights", "bias", cldnn::data_types::f32, padding(), get_output_dim_size(p)),
        eltwise("shift", { "fc_prim", "shift_data" }, eltwise_mode::sum, cldnn::data_types::f32),
        crop("crop", "shift", get_output_layout(p).get_tensor(), { 0, 0, 0, 0 }),
        reorder("reorder_bfyx", "crop", p.default_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, fc_int8_inputs_fused_fp32_sum, ::testing::ValuesIn(std::vector<fully_connected_test_params>{
    // OneDNN has issue with small shapes - ticket 7064
    // fully_connected_test_params{ CASE_FC_U8S8_3D_1, 2, 4 },
    // fully_connected_test_params{ CASE_FC_U8S8_3D_2, 2, 4 },
    fully_connected_test_params{ CASE_FC_U8S8_3D_4, 2, 4 },
}));
#endif
