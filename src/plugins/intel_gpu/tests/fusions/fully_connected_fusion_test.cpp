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
    tensor in_shape;
    tensor out_shape;
    tensor kernel;
    tensor stride;
    tensor pad;
    tensor dilation;
    uint32_t groups;
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
        auto pad = p.pad;
        std::vector<int> pad_ = { 0, 0, pad.spatial[0], pad.spatial[1] };
        return layout{ p.data_type, p.input_format, p.in_shape, padding{ pad_ } };
    }

    layout get_per_channel_layout(fully_connected_test_params& p) {
        return layout{ p.default_type, p.default_format, tensor{ 1, p.out_shape.feature[0], 1, 1 } };
    }

    size_t get_output_dim_size(fully_connected_test_params& p) {
        size_t size = 2;
        for (auto i : p.out_shape.spatial) {
            if (i > 1)
                size++;
        }
        return size;
    }

    layout get_weights_layout(fully_connected_test_params& p) {
        cldnn::tensor weights_tensor;
        if (p.out_shape.spatial[1] > 1) {
            // 3d case
            weights_tensor = cldnn::tensor(p.kernel.batch[0], p.kernel.feature[0], 1, 1);
        }
        else {
            weights_tensor = cldnn::tensor(batch(p.out_shape.feature[0]), feature(p.in_shape.feature[0]),
                                        spatial(p.kernel.spatial[0], p.kernel.spatial[1], p.kernel.spatial[2]));
        }
        return layout{ p.weights_type, p.weights_format, weights_tensor };
    }

    layout get_bias_layout(fully_connected_test_params& p) {
        if (p.out_shape.spatial[1] > 1) {
            // 3d case
            return layout{ p.default_type, format::bfyx, tensor{ 1, 1, 1, p.out_shape.spatial[1] } };
        }
        else {
            return layout{ p.default_type, format::bfyx, tensor{ 1, p.out_shape.feature[0], 1, 1 } };
        }
    }
};


#ifdef ENABLE_ONEDNN_FOR_GPU
class FullyConnectedFusingTestOneDNN : public BaseFusingTest<fully_connected_test_params> {
public:
    void execute(fully_connected_test_params& p) {
        // Onednn post operation has issue in a machine that does not support imad.
        if (!engine.get_device_info().supports_imad)
            return;

        auto input_prim = p.data_type == data_types::u8 ? get_mem(get_input_layout(p), 0, 10) : get_mem(get_input_layout(p));

        auto impl_forcing_bo = bo_fused.get<build_option_type::force_implementations>();
        const auto& impl_forcing = impl_forcing_bo->forcing;

        auto forcing_format = p.input_format;
        for (auto& forcing : impl_forcing) {
            if (forcing.first == "conv_prim") {
                forcing_format = forcing.second.output_format;
            }
        }

        implementation_desc conv_impl = { forcing_format, "", impl_types::onednn };
        bo_fused.set_option(build_option::force_implementations({ { "conv_prim", conv_impl } }));

        network network_not_fused(this->engine, this->topology_non_fused, bo_not_fused);
        network network_fused(this->engine, this->topology_fused, bo_fused);
        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);

        compare(network_not_fused, network_fused, p);
        auto find_conv = [](primitive_info& p) -> bool {
            if (p.original_id == "conv_prim")
                return true;
            return false;
        };

        auto pi_fused = network_fused.get_primitives_info();
        auto info_fused = std::find_if(pi_fused.begin(), pi_fused.end(), find_conv);
        if (info_fused != pi_fused.end())
            std::cout << "kernel: " << info_fused->kernel_id << std::endl;
    }

    layout get_input_layout(fully_connected_test_params& p) {
        auto pad = p.pad;
        std::vector<int> pad_ = { 0, 0, pad.spatial[0], pad.spatial[1] };
        return layout{ p.data_type, p.input_format, p.in_shape, padding{ pad_ } };
    }

    layout get_per_channel_layout(fully_connected_test_params& p) {
        return layout{ p.default_type, p.default_format, tensor{1, p.out_shape.feature[0], 1, 1} };
    }

    size_t get_output_dim_size(fully_connected_test_params& p) {
        size_t size = 2;
        for (auto i : p.out_shape.spatial) {
            if (i > 1)
                size++;
        }
        return size;
    }
};
#endif  // ENABLE_ONEDNN_FOR_GPU

}  // namespace

// in_shape; out_shape; kernel; stride; pad; dilation; groups; data_type; input_format; weights_type; weights_format; default_type; default_format;
#define CASE_FC_FP32_1 { 1, 1, 3, 1 }, { 1, 4, 1, 1 }, { 4, 1, 3, 1 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::f32, format::bfyx, data_types::f32, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_FP32_2 { 2, 1, 3, 1 }, { 2, 4, 1, 1 }, { 4, 1, 3, 1 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::f32, format::yxfb, data_types::f32, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_FP32_3 { 2, 32, 1, 1 }, { 2, 16, 1, 1 }, { 16, 32, 1, 1 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::f32, format::bfyx, data_types::i8, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_FP32_3D_1 { 5, 3, 1, 3 }, { 5, 3, 1, 5 }, { 5, 3, 1, 1 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::f32, format::bfyx, data_types::f32, format::os_iyx_osv16, data_types::f32, format::bfyx
#define CASE_FC_FP32_3D_2 { 2, 1, 1, 1 }, { 2, 1, 1, 32 }, { 32, 1, 1, 1 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::f32, format::bfyx, data_types::f32, format::os_iyx_osv16, data_types::f32, format::bfyx
#define CASE_FC_FP32_3D_3 { 2, 32, 1, 32 }, { 2, 32, 1, 16 }, { 16, 32, 1, 1 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::f32, format::bfyx, data_types::f32, format::os_iyx_osv16, data_types::f32, format::bfyx

#define CASE_FC_U8S8_1 { 1, 1, 3, 1 }, { 1, 4, 1, 1 }, { 4, 1, 3, 1 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::u8, format::bfyx, data_types::i8, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_U8S8_2 { 2, 1, 3, 1 }, { 2, 4, 1, 1 }, { 4, 1, 3, 1 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::u8, format::b_fs_yx_fsv4, data_types::i8, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_U8S8_3 { 2, 32, 1, 1 }, { 2, 16, 1, 1 }, { 16, 32, 1, 1 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::u8, format::b_fs_yx_fsv4, data_types::i8, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_U8S8_3D_1 { 2, 32, 1, 3 }, { 2, 32, 1, 16 }, { 16, 3, 1, 1 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::u8, format::bfyx, data_types::i8, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_U8S8_3D_2 { 1, 1, 1, 3 }, { 1, 1, 1, 32 }, { 32, 3, 1, 1 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::u8, format::bfyx, data_types::i8, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_U8S8_3D_3 { 2, 3, 1, 1 }, { 2, 3, 1, 15 }, { 15, 1, 1, 1 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::u8, format::bfyx, data_types::i8, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_U8S8_3D_4 { 1, 512, 1, 1024 }, { 1, 384, 1, 1024 }, { 1024, 1024, 1, 1 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::u8, format::bfyx, data_types::i8, format::oiyx, data_types::f32, format::bfyx

/* ----------------------------------------------------------------------------------------------------- */
/* ---------------------------------------- FC cases --------------------------------------------------- */
/* ----------------------------------------------------------------------------------------------------- */
class fc_fp32_activation : public FullyConnectedFusingTest {};
TEST_P(fc_fp32_activation, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        fully_connected("fc_prim", input_info("input"), "weights", "bias", "", padding(), get_output_dim_size(p)),
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

class fc_fp32_bias : public FullyConnectedFusingTest {};
TEST_P(fc_fp32_bias, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        fully_connected("fc_prim", input_info("input"), "weights", "", "", padding(), get_output_dim_size(p)),
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

class fc_int8_scale : public FullyConnectedFusingTest {};
TEST_P(fc_int8_scale, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("scale_data", {get_mem(get_per_channel_layout(p), 1.0f / p.kernel.count())}),
        fully_connected("fc_prim", input_info("input"), "weights", "bias", data_types::f32, "", padding(), get_output_dim_size(p)),
        scale("scale", input_info("fc_prim"), input_info("scale_data")),
        reorder("reorder_bfyx", input_info("scale"), p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

TEST_P(fc_int8_scale, fp16_scale_out) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("scale_data", {get_mem(get_per_channel_layout(p), 1.0f / p.kernel.count())}),
        fully_connected("fc_prim", input_info("input"), "weights", "bias", data_types::f32, "", padding(), get_output_dim_size(p)),
        scale("scale", input_info("fc_prim"), input_info("scale_data"), optional_data_type{ data_types::f16 }),
        reorder("reorder_bfyx", input_info("scale"), p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, fc_int8_scale, ::testing::ValuesIn(std::vector<fully_connected_test_params>{
    fully_connected_test_params{ CASE_FC_U8S8_1, 2, 3 },
    fully_connected_test_params{ CASE_FC_U8S8_2, 2, 3 },
    fully_connected_test_params{ CASE_FC_U8S8_3, 2, 3 },
    fully_connected_test_params{ CASE_FC_U8S8_3D_1, 2, 3 },
    fully_connected_test_params{ CASE_FC_U8S8_3D_2, 2, 3 },
    fully_connected_test_params{ CASE_FC_U8S8_3D_3, 2, 3 },
}));

class fc_int8_quantize_u8 : public FullyConnectedFusingTest {};
TEST_P(fc_int8_quantize_u8, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("in_lo", {get_mem(get_per_channel_layout(p), min_random, 0)}),
        data("in_hi", {get_mem(get_per_channel_layout(p), 1, max_random)}),
        data("out_lo", {get_mem(get_single_element_layout(p), 0)}),
        data("out_hi", {get_mem(get_single_element_layout(p), 255)}),
        fully_connected("fc_prim", input_info("input"), "weights", "bias", data_types::f32, "", padding(), get_output_dim_size(p)),
        quantize("quantize", input_info("fc_prim"), input_info("in_lo"), input_info("in_hi"), input_info("out_lo"), input_info("out_hi"), 256, data_types::u8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
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

class fc_int8_scale_quantize_i8 : public FullyConnectedFusingTest {};
TEST_P(fc_int8_scale_quantize_i8, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("in_lo", {get_mem(get_per_channel_layout(p), min_random, 0)}),
        data("in_hi", {get_mem(get_per_channel_layout(p), 1, max_random)}),
        data("out_lo", {get_mem(get_single_element_layout(p), -127)}),
        data("out_hi", {get_mem(get_single_element_layout(p), 127)}),
        data("scale_data", {get_mem(get_per_channel_layout(p), 1.0f / p.kernel.count() / 255)}),
        fully_connected("fc_prim", input_info("input"), "weights", "bias", data_types::f32, "", padding(), get_output_dim_size(p)),
        scale("scale", input_info("fc_prim"), input_info("scale_data")),
        quantize("quantize", input_info("scale"), input_info("in_lo"), input_info("in_hi"), input_info("out_lo"), input_info("out_hi"), 255, data_types::i8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, fc_int8_scale_quantize_i8, ::testing::ValuesIn(std::vector<fully_connected_test_params>{
    fully_connected_test_params{ CASE_FC_U8S8_1, 2, 4 },
    fully_connected_test_params{ CASE_FC_U8S8_2, 2, 4 },
    fully_connected_test_params{ CASE_FC_U8S8_3, 2, 4 },
    fully_connected_test_params{ CASE_FC_U8S8_3D_1, 2, 4 },
    fully_connected_test_params{ CASE_FC_U8S8_3D_2, 2, 4 },
    fully_connected_test_params{ CASE_FC_U8S8_3D_3, 2, 4 },
}));

class fc_int8_scale_activation_quantize_i8 : public FullyConnectedFusingTest {};
TEST_P(fc_int8_scale_activation_quantize_i8, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("in_lo", {get_mem(get_per_channel_layout(p), min_random, 0)}),
        data("in_hi", {get_mem(get_per_channel_layout(p), 1, max_random)}),
        data("out_lo", {get_mem(get_single_element_layout(p), -127)}),
        data("out_hi", {get_mem(get_single_element_layout(p), 127)}),
        data("scale_data", {get_mem(get_per_channel_layout(p), 1.0f / p.kernel.count() / 255)}),
        fully_connected("fc_prim", input_info("input"), "weights", "bias", data_types::f32, "", padding(), get_output_dim_size(p)),
        scale("scale", input_info("fc_prim"), input_info("scale_data")),
        activation("activation_scale", input_info("scale"), activation_func::exp),
        quantize("quantize", input_info("activation_scale"), input_info("in_lo"), input_info("in_hi"), input_info("out_lo"), input_info("out_hi"), 255, data_types::i8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, fc_int8_scale_activation_quantize_i8, ::testing::ValuesIn(std::vector<fully_connected_test_params>{
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
    auto shift_layout = layout{ p.default_type, p.default_format, tensor{ 1, 1, 1, p.kernel.batch[0] } };

    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", {get_mem(get_weights_layout(p))}),
        data("bias", {get_mem(get_bias_layout(p))}),
        data("shift_data", {get_mem(shift_layout, 1)}),
        fully_connected("fc_prim", input_info("input"), "weights", "bias", cldnn::data_types::f32, "", padding(), get_output_dim_size(p)),
        eltwise("shift", { input_info("fc_prim"), input_info("shift_data") }, eltwise_mode::sum, cldnn::data_types::f32),
        crop("crop", input_info("shift"), get_output_layout(p).size, { 0, 0, 0, 0 }),
        reorder("reorder_bfyx", input_info("crop"), p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, fc_int8_inputs_fused_fp32_sum, ::testing::ValuesIn(std::vector<fully_connected_test_params>{
    // OneDNN has issue with small shapes - ticket 7064
    // fully_connected_test_params{ CASE_FC_U8S8_3D_1, 2, 4 },
    // fully_connected_test_params{ CASE_FC_U8S8_3D_2, 2, 4 },
    fully_connected_test_params{ CASE_FC_U8S8_3D_4, 2, 4 },
}));
#endif
