// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fusion_test_common.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/convolution.hpp>
#include <intel_gpu/primitives/quantize.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/fully_connected.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/crop.hpp>
#include <intel_gpu/primitives/permute.hpp>
#include <intel_gpu/primitives/concatenation.hpp>

#include <cmath>

using namespace cldnn;
using namespace ::tests;

namespace {

struct convolution_test_params {
    ov::PartialShape in_shape;
    ov::PartialShape out_shape;
    ov::PartialShape weights_shape;
    ov::Strides stride;
    ov::CoordinateDiff pad;
    ov::Strides dilation;
    uint32_t groups;
    data_types data_type;
    format input_format;
    data_types weights_type;
    format weights_format;
    data_types default_type;
    format default_format;
    size_t expected_fused_primitives;
    size_t expected_fused_primitives_onednn;
    size_t expected_not_fused_primitives;
};

struct bc_force_kernel_params {
    ov::PartialShape in_shape;
    ov::PartialShape out_shape;
    ov::PartialShape weights_shape;
    ov::Strides stride;
    ov::CoordinateDiff pad;
    ov::Strides dilation;
    uint32_t groups;
    data_types data_type;
    format input_format;
    data_types weights_type;
    format weights_format;
    data_types default_type;
    format default_format;
    size_t expected_fused_primitives;
    size_t expected_not_fused_primitives;
    std::string kernel_name;
};

struct conv_eltw_test_params {
    ov::PartialShape in_shape;
    ov::PartialShape out_shape;
    ov::PartialShape eltw_shape;
    ov::PartialShape weights_shape;
    ov::Strides stride;
    ov::CoordinateDiff pad;
    ov::Strides dilation;
    uint32_t groups;
    data_types data_type;
    format input_format;
    data_types weights_type;
    format weights_format;
    data_types default_type;
    format default_format;
    size_t expected_fused_primitives;
    size_t expected_fused_primitives_onednn;
    size_t expected_not_fused_primitives;
};

struct conv_activation_onednn_test_params {
    ov::PartialShape in_shape;
    ov::PartialShape out_shape;
    ov::PartialShape weights_shape;
    ov::Strides stride;
    ov::CoordinateDiff pad;
    ov::Strides dilation;
    uint32_t groups;
    data_types data_type;
    format input_format;
    data_types weights_type;
    format weights_format;
    data_types default_type;
    format default_format;
    activation_func activation_function_type;
    size_t expected_fused_primitives;
    size_t expected_fused_primitives_onednn;
    size_t expected_not_fused_primitives;
};

class ConvFusingTest : public BaseFusingTest<convolution_test_params> {
public:
    void execute(convolution_test_params& p, int min=0, int max=0) {
        if(engine.get_device_info().supports_immad)
            p.expected_fused_primitives = p.expected_fused_primitives_onednn;
        cldnn::memory::ptr input_prim;
        if (min == max) {
            input_prim = get_mem(get_input_layout(p));
        } else {
            input_prim = get_mem(get_input_layout(p), min, max);
        }
        network network_not_fused(this->engine, this->topology_non_fused, cfg_not_fused);
        network network_fused(this->engine, this->topology_fused, cfg_fused);
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

    layout get_input_layout(convolution_test_params& p) {
        auto pad = p.pad;
        std::vector<int> pad_ = { 0, 0, static_cast<int>(pad[1]), static_cast<int>(pad[0]) };
        return layout{ p.in_shape, p.data_type, p.input_format, padding{ pad_ } };
    }

    layout get_output_layout(convolution_test_params& p) {
        return layout{ p.out_shape, p.data_type, p.input_format };
    }

    layout get_per_channel_layout(convolution_test_params& p) {
        ov::PartialShape shape(std::vector<ov::Dimension>(p.out_shape.size(), 1));
        shape[1] = p.out_shape[1];
        return layout{ shape, p.default_type, p.default_format };
    }

    layout get_prelu_slope_layout(convolution_test_params& p) {
        return get_per_channel_layout(p);
    }

    layout get_weights_layout(convolution_test_params& p) {
        return layout{ p.weights_shape, p.weights_type, p.weights_format };
    }

    layout get_weights_layout(convolution_test_params& p, cldnn::format f) {
        return layout{ p.weights_shape, p.weights_type, f };
    }

    layout get_weights_zp_layout(convolution_test_params& p) {
        ov::PartialShape shape(std::vector<ov::Dimension>(p.out_shape.size(), 1));
        shape[0] = p.out_shape[1];
        return layout{ shape, p.weights_type, p.default_format };
    }

    layout get_activations_zp_layout(convolution_test_params& p) {
        ov::PartialShape shape(std::vector<ov::Dimension>(p.in_shape.size(), 1));
        shape[1] = p.in_shape[1];
        return layout{ shape, p.data_type, p.default_format };
    }
};

class ConvReorderFusingTest : public BaseFusingTest<convolution_test_params> {
public:
    void execute(convolution_test_params& p, std::map<std::string, std::vector<std::string>> expected_fused_primitives_ids = {}) {
        auto input_prim = get_mem(get_input_layout(p));
        network network_not_fused(this->engine, this->topology_non_fused, cfg_not_fused);
        network network_fused(this->engine, this->topology_fused, cfg_fused);
        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);

        compare(network_not_fused, network_fused, p, true);
        check_fusions_correctness(network_fused, expected_fused_primitives_ids);
    }

    layout get_input_layout(convolution_test_params& p) {
        auto pad = p.pad;
        std::vector<int> pad_ = { 0, 0, static_cast<int>(pad[1]), static_cast<int>(pad[0]) };
        return layout{ p.in_shape, p.data_type, p.input_format, padding{ pad_ } };
    }

    layout get_output_layout(convolution_test_params& p) {
        return layout{ p.out_shape, p.data_type, p.input_format };
    }

    layout get_per_channel_layout(convolution_test_params& p) {
        ov::PartialShape shape(std::vector<ov::Dimension>(p.out_shape.size(), 1));
        shape[1] = p.out_shape[1];
        return layout{ shape, p.default_type, p.default_format };
    }


    layout get_weights_layout(convolution_test_params& p) {
        return layout{p.weights_shape, p.weights_type, p.weights_format};
    }

    layout get_weights_layout(convolution_test_params& p, cldnn::format f) {
        return layout{p.weights_shape, p.weights_type, f};
    }
};

class ConvEltwTest : public ::BaseFusingTest<conv_eltw_test_params> {
public:

    void execute(conv_eltw_test_params& p) {
        if (engine.get_device_info().supports_immad)
            p.expected_fused_primitives = p.expected_fused_primitives_onednn;

        auto input_prim = get_mem(get_input_layout(p));
        network network_not_fused(this->engine, this->topology_non_fused, cfg_not_fused);
        network network_fused(this->engine, this->topology_fused, cfg_fused);
        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);

        compare(network_not_fused, network_fused, p);
        auto find_prim = [](primitive_info& p) -> bool {
            // Add more ids when needed
            if (p.original_id == "conv_prim")
                return true;
            return false;
        };

        auto pi_fused = network_fused.get_primitives_info();
        auto info_fused = std::find_if(pi_fused.begin(), pi_fused.end(), find_prim);
        if (info_fused != pi_fused.end())
            std::cout << "kernel: " << info_fused->kernel_id << std::endl;
    }

    layout get_input_layout(conv_eltw_test_params& p) {
        auto pad = p.pad;
        std::vector<int> pad_ = { 0, 0, static_cast<int>(pad[1]), static_cast<int>(pad[0]) };
        return layout{ p.in_shape, p.data_type, p.input_format, padding{ pad_ } };
    }

    layout get_per_channel_layout(conv_eltw_test_params& p) {
        ov::PartialShape shape(std::vector<ov::Dimension>(p.out_shape.size(), 1));
        shape[1] = p.out_shape[1];
        return layout{ shape, p.default_type, p.default_format };
    }

    layout get_weights_layout(conv_eltw_test_params& p) {
        return layout{p.weights_shape, p.weights_type, p.weights_format};
    }

    layout get_weights_layout(conv_eltw_test_params& p, cldnn::format f) {
        return layout{p.weights_shape, p.weights_type, f};
    }
};

class ConvFusingForceKernelTest : public BaseFusingTest<bc_force_kernel_params> {
    public:
    void execute(bc_force_kernel_params& p) {
        auto input_prim = get_mem(get_input_layout(p));
        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));
        ov::intel_gpu::ImplementationDesc conv_impl = { p.input_format, p.kernel_name, impl_types::ocl };
        config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "conv_prim", conv_impl } }));

        network network_not_fused(this->engine, this->topology_non_fused, cfg_not_fused);
        network network_fused(this->engine, this->topology_fused, config);
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

    layout get_input_layout(bc_force_kernel_params& p) {
        auto pad = p.pad;
        std::vector<int> pad_ = { 0, 0, static_cast<int>(pad[1]), static_cast<int>(pad[0]) };
        return layout{ p.in_shape, p.data_type, p.input_format, padding{ pad_ } };
    }

    layout get_output_layout(bc_force_kernel_params& p) {
        return layout{ p.out_shape, p.data_type, p.input_format };
    }

    layout get_per_channel_layout(bc_force_kernel_params& p) {
        ov::PartialShape shape(std::vector<ov::Dimension>(p.out_shape.size(), 1));
        shape[1] = p.out_shape[1];
        return layout{ shape, p.default_type, p.default_format };
    }

    layout get_weights_layout(bc_force_kernel_params& p) {
        return layout{p.weights_shape, p.weights_type, p.weights_format};
    }

    layout get_weights_layout(bc_force_kernel_params& p, cldnn::format f) {
        return layout{p.weights_shape, p.weights_type, f};
    }
};

#ifdef ENABLE_ONEDNN_FOR_GPU
class WeightsPrimitiveFusingTestOneDNN : public BaseFusingTest<convolution_test_params> {
public:
    void execute(convolution_test_params& p) {
        // Onednn post operation has issue in a machine that does not support imad.
        if (!engine.get_device_info().supports_immad)
            return;
        p.expected_fused_primitives = p.expected_fused_primitives_onednn;

        auto input_prim = p.data_type == data_types::u8 ? get_mem(get_input_layout(p), 0, 10) : get_mem(get_input_layout(p));

        auto impl_forcing = cfg_fused.get_property(ov::intel_gpu::force_implementations);

        ov::intel_gpu::ImplementationDesc conv_impl = { format::any, "", impl_types::onednn };

        auto cfg = cfg_fused;
        cfg.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "conv_prim", conv_impl } }));
        network network_not_fused(this->engine, this->topology_non_fused, cfg_not_fused);
        network network_fused(this->engine, this->topology_fused, cfg);
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

    layout get_input_layout(convolution_test_params& p) {
        return layout{ p.in_shape, p.data_type, p.input_format };
    }

    layout get_output_layout(convolution_test_params& p) {
        return layout{ p.out_shape, p.data_type, p.input_format };
    }

    layout get_per_channel_layout(convolution_test_params& p) {
        ov::PartialShape shape(std::vector<ov::Dimension>(p.out_shape.size(), 1));
        shape[1] = p.out_shape[1];
        return layout{ shape, p.default_type, p.default_format };
    }

    layout get_prelu_slope_layout(convolution_test_params& p) {
        return get_per_channel_layout(p);
    }

    layout get_weights_layout(convolution_test_params& p) {
        return layout{p.weights_shape, p.weights_type, p.weights_format};
    }

    layout get_weights_layout(convolution_test_params& p, cldnn::format f) {
        return layout{p.weights_shape, p.weights_type, f};
    }
};
#endif  // ENABLE_ONEDNN_FOR_GPU

class ConvActivationTestOnednn : public BaseFusingTest<conv_activation_onednn_test_params> {
public:
    void execute(conv_activation_onednn_test_params& p, int min=0, int max=0) {
        if(engine.get_device_info().supports_immad)
            p.expected_fused_primitives = p.expected_fused_primitives_onednn;
        cldnn::memory::ptr input_prim;
        if (min == max) {
            input_prim = get_mem(get_input_layout(p));
        } else {
            input_prim = get_mem(get_input_layout(p), min, max);
        }
        network network_not_fused(this->engine, this->topology_non_fused, cfg_not_fused);
        network network_fused(this->engine, this->topology_fused, cfg_fused);
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

    layout get_input_layout(conv_activation_onednn_test_params& p) {
        return layout{ p.in_shape, p.data_type, p.input_format };
    }

    layout get_output_layout(conv_activation_onednn_test_params& p) {
        return layout{ p.out_shape, p.data_type, p.input_format };
    }

    layout get_per_channel_layout(conv_activation_onednn_test_params& p) {
        ov::PartialShape shape(std::vector<ov::Dimension>(p.out_shape.size(), 1));
        shape[1] = p.out_shape[1];
        return layout{ shape, p.default_type, p.default_format };
    }

    layout get_prelu_slope_layout(conv_activation_onednn_test_params& p) {
        auto r = p.out_shape.size();
        ov::PartialShape shape(std::vector<ov::Dimension>(r, 1));
        shape[1] = p.out_shape[1];
        shape[r-1] = p.out_shape[r-1];
        return layout{ shape, p.default_type, p.default_format };
    }

    layout get_weights_layout(conv_activation_onednn_test_params& p) {
        return layout{p.weights_shape, p.weights_type, p.weights_format};
    }

    layout get_weights_layout(conv_activation_onednn_test_params& p, cldnn::format f) {
        return layout{p.weights_shape, p.weights_type, f};
    }
};

}  // namespace

// in_shape; out_shape; weights_shape; stride; pad; dilation; groups; data_type; input_format; weights_type; weights_format; default_type; default_format;
#define CASE_CONV_FP32_1 { 1, 15, 4, 5 }, { 1, 30, 2, 3 }, { 30, 15, 3, 3 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, data_types::f32, format::bfyx, data_types::f32, format::oiyx, data_types::f32, format::bfyx
#define CASE_CONV_FP32_2 { 1, 16, 4, 5 }, { 1, 32, 2, 3 }, { 32, 16, 3, 3 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::os_is_yx_isv16_osv16, data_types::f32, format::bfyx
#define CASE_CONV_FP32_3 { 1, 16, 4, 5 }, { 1, 32, 4, 5 }, { 32, 16, 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::os_is_yx_isv16_osv16, data_types::f32, format::bfyx
#define CASE_CONV_FP32_4 { 1, 32, 4, 5 }, { 1, 32, 4, 5 }, { 32, 1, 1, 3, 3 }, { 1, 1 }, { 1, 1 }, { 1, 1 }, 32, data_types::f32, format::b_fs_yx_fsv16, data_types::f32,  format::gs_oiyx_gsv16, data_types::f32, format::bfyx
#define CASE_CONV_FP32_5 { 1, 15, 4, 5 }, { 1, 30, 2, 3 }, { 30, 15, 3, 3 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, data_types::f32, format::bfyx, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_CONV_FP32_6 { 1, 16, 4, 5, 4 }, { 1, 16, 2, 3, 2 }, { 16, 16, 3, 3, 3 }, { 1, 1, 1 }, { 0, 0, 0 }, { 1, 1, 1 }, 1, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32, format::os_is_zyx_isv16_osv16, data_types::f32, format::bfzyx
#define CASE_CONV_FP32_7 { 1, 16, 4, 5, 4 }, { 1, 32, 2, 3, 2 }, { 32, 16, 3, 3, 3 }, { 1, 1, 1 }, { 0, 0, 0 }, { 1, 1, 1 }, 1, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32, format::os_is_zyx_isv16_osv16, data_types::f32, format::bfzyx
#define CASE_CONV_FP32_8 { 1, 32, 4, 5, 4 }, { 1, 16, 2, 3, 2 }, { 2, 8, 16, 3, 3, 3 }, { 1, 1, 1 }, { 0, 0, 0 }, { 1, 1, 1 }, 2, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32, format::g_os_is_zyx_isv16_osv16, data_types::f32, format::bfzyx
#define CASE_CONV_FP32_9 { 1, 32, 4, 5, 4 }, { 1, 32, 2, 3, 2 }, { 2, 16, 16, 3, 3, 3 }, { 1, 1, 1 }, { 0, 0, 0 }, { 1, 1, 1 }, 2, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32, format::g_os_is_zyx_isv16_osv16, data_types::f32, format::bfzyx
#define CASE_CONV_FP32_10 { 32, 16, 4, 5, 4 }, { 32, 32, 4, 5, 4 }, { 32, 16, 1, 1, 1 }, { 1, 1, 1 }, { 0, 0, 0 }, { 1, 1, 1 }, 1, data_types::f32, format::bs_fs_zyx_bsv16_fsv16, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_CONV_FP32_11 { 1, 32, 4, 5, 4 }, { 1, 16, 2, 3, 2 }, { 2, 8, 16, 3, 3, 3 }, { 1, 1, 1 }, { 0, 0, 0 }, { 1, 1, 1 }, 2, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32, format::g_os_is_zyx_isv16_osv16, data_types::f32, format::bfzyx
#define CASE_CONV_FP32_12 { 1, 16, 4, 5, 4 }, { 1, 16, 2, 3, 2 }, { 2, 8, 8, 3, 3, 3 }, { 1, 1, 1 }, { 0, 0, 0 }, { 1, 1, 1 }, 2, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32, format::g_os_is_zyx_isv16_osv16, data_types::f32, format::bfzyx
#define CASE_CONV_FP32_13 { 1, 16, 18, 5, 4 }, { 1, 16, 16, 3, 2 }, { 2, 8, 8, 3, 3, 3 }, { 1, 1, 1 }, { 0, 0, 0 }, { 1, 1, 1 }, 2, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32, format::g_os_is_zyx_isv16_osv16, data_types::f32, format::bfzyx
#define CASE_CONV_FP32_14 { 1, 3, 4, 5 }, { 1, 30, 2, 3 }, { 30, 3, 3, 3 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, data_types::f32, format::bfyx, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_CONV_FP32_15 { 1, 6, 4, 4 }, { 1, 16, 4, 4 }, { 16, 6, 3, 3 }, { 1, 1 }, { 1, 1 }, { 1, 1 }, 1, data_types::f32, format::bfyx, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_CONV_FP32_16 { 1, 3, 112, 112, 8 }, { 1, 16, 56, 56, 8 }, { 16, 3, 3, 3, 1 }, { 2, 2, 1 }, { 1, 1, 0 }, { 1, 1, 1 }, 1, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx


#define CASE_CONV_FP16_1 { 1, 15, 4, 5 }, { 1, 30, 2, 3 }, { 30, 15, 3, 3 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, data_types::f16, format::bfyx, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_CONV_FP16_2 { 1, 16, 4, 5 }, { 1, 32, 2, 3 }, { 32, 16, 3, 3 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, data_types::f16, format::b_fs_yx_fsv16, data_types::f16, format::os_is_yx_isv16_osv16, data_types::f16, format::bfyx
#define CASE_CONV_FP16_3 { 1, 16, 4, 5 }, { 1, 32, 4, 5 }, { 32, 16, 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, data_types::f16, format::b_fs_yx_fsv16, data_types::f16, format::os_is_yx_isv16_osv16, data_types::f16, format::bfyx
#define CASE_CONV_FP16_4 { 1, 32, 4, 5 }, { 1, 32, 4, 5 }, { 32, 1, 1, 3, 3 }, { 1, 1 }, { 1, 1 }, { 1, 1 }, 32, data_types::f16, format::b_fs_yx_fsv16, data_types::f16,  format::gs_oiyx_gsv16, data_types::f16, format::bfyx
#define CASE_CONV_FP16_5 { 1, 15, 4, 5 }, { 1, 30, 2, 3 }, { 30, 15, 3, 3 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, data_types::f16, format::bfyx, data_types::i8, format::bfyx, data_types::f16, format::bfyx
#define CASE_CONV_FP16_6 { 1, 16, 4, 5, 4 }, { 1, 16, 2, 3, 2 }, { 16, 16, 3, 3, 3 }, { 1, 1, 1 }, { 0, 0, 0 }, { 1, 1, 1 }, 1, data_types::f16, format::b_fs_zyx_fsv16, data_types::f16, format::os_is_zyx_isv16_osv16, data_types::f16, format::bfzyx
#define CASE_CONV_FP16_7 { 1, 16, 4, 5, 4 }, { 1, 32, 2, 3, 2 }, { 32, 16, 3, 3, 3 }, { 1, 1, 1 }, { 0, 0, 0 }, { 1, 1, 1 }, 1, data_types::f16, format::b_fs_zyx_fsv16, data_types::f16, format::os_is_zyx_isv16_osv16, data_types::f16, format::bfzyx
#define CASE_CONV_FP16_8 { 1, 32, 4, 5, 4 }, { 1, 16, 2, 3, 2 }, { 2, 8, 16, 3, 3, 3 }, { 1, 1, 1 }, { 0, 0, 0 }, { 1, 1, 1 }, 2, data_types::f16, format::b_fs_zyx_fsv16, data_types::f16, format::g_os_is_zyx_isv16_osv16, data_types::f16, format::bfzyx
#define CASE_CONV_FP16_9 { 1, 32, 4, 5, 4 }, { 1, 32, 2, 3, 2 }, { 2, 16, 16, 3, 3, 3 }, { 1, 1, 1 }, { 0, 0, 0 }, { 1, 1, 1 }, 2, data_types::f16, format::b_fs_zyx_fsv16, data_types::f16, format::g_os_is_zyx_isv16_osv16, data_types::f16, format::bfzyx
#define CASE_CONV_FP16_10 { 32, 16, 4, 5, 4 }, { 32, 32, 2, 3, 2 }, { 32, 16, 3, 3, 3 }, { 1, 1, 1 }, { 0, 0, 0 }, { 1, 1, 1 }, 1, data_types::f16, format::bs_fs_zyx_bsv16_fsv16, data_types::f16, format::bfzyx, data_types::f16, format::bfzyx
#define CASE_CONV_FP16_11 { 1, 32, 4, 5, 4 }, { 1, 16, 2, 3, 2 }, { 2, 8, 16, 3, 3, 3 }, { 1, 1, 1 }, { 0, 0, 0 }, { 1, 1, 1 }, 2, data_types::f16, format::b_fs_zyx_fsv16, data_types::f16, format::g_os_is_zyx_isv16_osv16, data_types::f16, format::bfzyx
#define CASE_CONV_FP16_12 { 1, 16, 4, 5, 4 }, { 1, 16, 2, 3, 2 }, { 2, 8, 8, 3, 3, 3 }, { 1, 1, 1 }, { 0, 0, 0 }, { 1, 1, 1 }, 2, data_types::f16, format::b_fs_zyx_fsv16, data_types::f16, format::g_os_is_zyx_isv16_osv16, data_types::f16, format::bfzyx
#define CASE_CONV_FP16_13 { 16, 32, 4, 5 }, { 16, 64, 2, 3 }, { 64, 32, 3, 3 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, data_types::f16, format::fs_b_yx_fsv32, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_CONV_FP16_14 { 1, 32, 55, 1 }, { 1, 32, 55, 1 }, { 32, 1, 1, 3, 1 }, { 1, 1 }, { 1, 1 }, { 1, 1 }, 32, data_types::f16, format::b_fs_yx_fsv16, data_types::f16,  format::gs_oiyx_gsv16, data_types::f16, format::bfyx
#define CASE_CONV_FP16_15 { 1, 39, 55, 1 }, { 1, 39, 55, 1 }, { 39, 1, 1, 3, 1 }, { 1, 1 }, { 1, 1 }, { 1, 1 }, 39, data_types::f16, format::b_fs_yx_fsv16, data_types::f16,  format::gs_oiyx_gsv16, data_types::f16, format::bfyx
#define CASE_CONV_FP16_16 { 1, 3, 112, 112, 8 }, { 1, 32, 56, 56, 8 }, { 32, 3, 3, 3, 1 }, { 2, 2, 1 }, { 1, 1, 0 }, { 1, 1, 1 }, 1, data_types::f16, format::bfzyx, data_types::f16, format::bfzyx, data_types::f16, format::bfzyx

#define CASE_CONV_U8S8_1 { 1, 15, 4, 5 }, { 1, 30, 2, 3 }, { 30, 15, 3, 3 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, data_types::u8, format::bfyx, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_CONV_U8S8_2 { 1, 15, 5, 5 }, { 1, 30, 3, 3 }, { 30, 15, 3, 3 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, data_types::u8, format::bfyx, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_CONV_U8S8_3 { 1, 16, 4, 5 }, { 1, 32, 4, 5 }, { 32, 16, 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, data_types::u8, format::bfyx, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_CONV_U8S8_4 { 1, 17, 4, 5 }, { 1, 17, 4, 5 }, { 17, 1, 1, 3, 3 }, { 1, 1 }, { 1, 1 }, { 1, 1 }, 17, data_types::u8, format::bfyx, data_types::i8, format::goiyx, data_types::f32, format::bfyx
#define CASE_CONV_U8S8_5 { 1, 16, 5, 5 }, { 1, 32, 5, 5 }, { 32, 16, 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, data_types::u8, format::bfyx, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_CONV_U8S8_6 { 1, 17, 4, 5 }, { 1, 17, 4, 5 }, { 17, 1, 1, 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 17, data_types::u8, format::bfyx, data_types::i8, format::goiyx, data_types::f32, format::bfyx
#define CASE_CONV_U8S8_7 { 1, 64, 7, 7 }, { 1, 32, 7, 7 }, { 32, 64, 3, 3 }, { 1, 1 }, { 1, 1 }, { 1, 1 }, 1, data_types::u8, format::bfyx, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_CONV_U8S8_8 { 1, 3, 4, 5 }, { 1, 32, 4, 5 }, { 32, 3, 3, 3 }, { 1, 1 }, { 1, 1 }, { 1, 1 }, 1, data_types::u8, format::bfyx, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_CONV_U8S8_9 { 16, 32, 5, 5 }, { 16, 32, 3, 3 }, { 32, 32, 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, data_types::u8, format::bs_fs_yx_bsv16_fsv16, data_types::i8, format::os_is_yx_osv16_isv16, data_types::f32, format::bfyx
#define CASE_CONV_U8S8_10 { 16, 32, 5, 5 }, { 16, 32, 3, 3 }, { 32, 32, 3, 3 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, data_types::u8, format::bs_fs_yx_bsv16_fsv16, data_types::i8, format::os_is_yx_osv16_isv16, data_types::f32, format::bfyx
#define CASE_CONV_U8S8_11 { 32, 15, 4, 5 }, { 32, 30, 2, 3 }, { 30, 15, 3, 3 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, data_types::u8, format::bfyx, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_CONV_U8S8_12 { 32, 15, 5, 5 }, { 32, 30, 3, 3 }, { 30, 15, 3, 3 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, data_types::u8, format::bfyx, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_CONV_U8S8_13 { 32, 16, 4, 5 }, { 32, 32, 4, 5 }, { 32, 16, 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, data_types::u8, format::bfyx, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_CONV_U8S8_14 { 32, 17, 4, 5 }, { 32, 17, 4, 5 }, { 17, 1, 1, 3, 3 }, { 1, 1 }, { 1, 1 }, { 1, 1 }, 17, data_types::u8, format::bfyx, data_types::i8, format::goiyx, data_types::f32, format::bfyx
#define CASE_CONV_U8S8_15 { 1, 15, 3, 3 }, { 1, 30, 1, 1 }, { 30, 15, 3, 3 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, data_types::u8, format::bfyx, data_types::i8, format::bfyx, data_types::f32, format::bfyx

#define CASE_CONV_S8S8_1 { 1, 15, 4, 5 }, { 1, 30, 2, 3 }, { 30, 15, 3, 3 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, data_types::i8, format::bfyx, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_CONV_S8S8_2 { 1, 15, 5, 5 }, { 1, 30, 3, 3 }, { 30, 15, 3, 3 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, data_types::i8, format::bfyx, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_CONV_S8S8_3 { 1, 16, 4, 5 }, { 1, 32, 4, 5 }, { 32, 16, 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, data_types::i8, format::bfyx, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_CONV_S8S8_4 { 1, 17, 4, 5 }, { 1, 17, 4, 5 }, { 17, 1, 1, 3, 3 }, { 1, 1 }, { 1, 1 }, { 1, 1 }, 17, data_types::i8, format::bfyx, data_types::i8, format::goiyx, data_types::f32, format::bfyx
#define CASE_CONV_S8S8_5 { 1, 16, 5, 5 }, { 1, 32, 5, 5 }, { 32, 16, 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, data_types::i8, format::bfyx, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_CONV_S8S8_6 { 1, 17, 4, 5 }, { 1, 17, 4, 5 }, { 17, 1, 1, 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 17, data_types::i8, format::bfyx, data_types::i8, format::goiyx, data_types::f32, format::bfyx
#define CASE_CONV_S8S8_7  { 1, 64, 7, 7 }, { 1, 32, 7, 7 }, { 32, 64, 3, 3 }, { 1, 1 }, { 1, 1 }, { 1, 1 }, 1, data_types::i8, format::bfyx, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_CONV_S8S8_8 { 1, 3, 4, 5 }, { 1, 32, 4, 5 }, { 32, 3, 3, 3 }, { 1, 1 }, { 1, 1 }, { 1, 1 }, 1, data_types::i8, format::bfyx, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_CONV_S8S8_9 { 16, 32, 5, 5 }, { 16, 32, 3, 3 }, { 32, 32, 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, data_types::i8, format::bs_fs_yx_bsv16_fsv16, data_types::i8, format::os_is_yx_osv16_isv16, data_types::f32, format::bfyx
#define CASE_CONV_S8S8_10 { 16, 32, 5, 5 }, { 16, 32, 3, 3 }, { 32, 32, 3, 3 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, data_types::i8, format::bs_fs_yx_bsv16_fsv16, data_types::i8, format::os_is_yx_osv16_isv16, data_types::f32, format::bfyx
#define CASE_CONV_S8S8_11 { 1, 4, 720, 1280 }, { 1, 4, 720, 1280 }, { 4, 4, 5, 5 }, { 1, 1 }, { 2, 2 }, { 1, 1 }, 1, data_types::i8, format::b_fs_yx_fsv4, data_types::i8, format::os_is_yx_osv16_isv4, data_types::f32, format::bfyx
#define CASE_CONV_S8S8_12 { 32, 15, 4, 5 }, { 32, 30, 2, 3 }, { 30, 15, 3, 3 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, data_types::i8, format::bfyx, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_CONV_S8S8_13 { 32, 15, 5, 5 }, { 32, 30, 3, 3 }, { 30, 15, 3, 3 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, data_types::i8, format::bfyx, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_CONV_S8S8_14 { 32, 16, 4, 5 }, { 32, 32, 4, 5 }, { 32, 16, 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, data_types::i8, format::bfyx, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_CONV_S8S8_15 { 32, 17, 4, 5 }, { 32, 17, 4, 5 }, { 17, 1, 1, 3, 3 }, { 1, 1 }, { 1, 1 }, { 1, 1 }, 17, data_types::i8, format::bfyx, data_types::i8, format::goiyx, data_types::f32, format::bfyx

#define CASE_CONV3D_U8S8_1 { 1, 15, 5, 4, 5 }, { 1, 30, 3, 2, 3 }, { 30, 15, 3, 3, 3 }, { 1, 1, 1 }, { 0, 0, 0 }, { 1, 1, 1 }, 1, data_types::u8, format::bfzyx, data_types::i8, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_CONV3D_U8S8_2 { 1, 15, 5, 5, 5 }, { 1, 30, 3, 3, 3 }, { 30, 15, 3, 3, 3 }, { 1, 1, 1 }, { 0, 0, 0 }, { 1, 1, 1 }, 1, data_types::u8, format::bfzyx, data_types::i8, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_CONV3D_U8S8_3 { 1, 16, 5, 4, 5 }, { 1, 32, 5, 4, 5 }, { 32, 16, 1, 1, 1 }, { 1, 1, 1 }, { 0, 0, 0 }, { 1, 1, 1 }, 1, data_types::u8, format::bfzyx, data_types::i8, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_CONV3D_U8S8_4 { 1, 17, 5, 4, 5 }, { 1, 17, 5, 4, 5 }, { 17, 1, 1, 3, 3, 3 }, { 1, 1, 1 }, { 1, 1, 1 }, { 1, 1, 1 }, 17, data_types::u8, format::bfzyx, data_types::i8, format::goizyx, data_types::f32, format::bfzyx
#define CASE_CONV3D_U8S8_5 { 1, 3, 5, 4, 5 },  { 1, 32, 5, 4, 5 }, { 32, 3, 3, 3, 3 }, { 1, 1, 1 }, { 1, 1, 1 }, { 1, 1, 1 }, 1, data_types::u8, format::bfzyx, data_types::i8, format::bfzyx, data_types::f32, format::bfzyx

#define CASE_CONV3D_S8S8_1 { 1, 15, 5, 4, 5 }, { 1, 30, 3, 2, 3 }, { 30, 15, 3, 3, 3 }, { 1, 1, 1 }, { 0, 0, 0 }, { 1, 1, 1 }, 1, data_types::i8, format::bfzyx, data_types::i8, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_CONV3D_S8S8_2 { 1, 15, 5, 5, 5 }, { 1, 30, 3, 3, 3 }, { 30, 15, 3, 3, 3 }, { 1, 1, 1 }, { 0, 0, 0 }, { 1, 1, 1 }, 1, data_types::i8, format::bfzyx, data_types::i8, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_CONV3D_S8S8_3 { 1, 16, 5, 4, 5 }, { 1, 32, 5, 4, 5 }, { 32, 16, 1, 1, 1 }, { 1, 1, 1 }, { 0, 0, 0 }, { 1, 1, 1 }, 1, data_types::i8, format::bfzyx, data_types::i8, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_CONV3D_S8S8_4 { 1, 17, 5, 4, 5 }, { 1, 17, 5, 4, 5 }, { 17, 1, 1, 3, 3, 3 }, { 1, 1, 1 }, { 1, 1, 1 }, { 1, 1, 1 }, 17, data_types::i8, format::bfzyx, data_types::i8, format::goizyx, data_types::f32, format::bfzyx
#define CASE_CONV3D_S8S8_5 { 1, 3, 5, 4, 5 },  { 1, 18, 5, 4, 5 }, { 18, 3, 3, 3, 3 }, { 1, 1, 1 }, { 1, 1, 1 }, { 1, 1, 1 }, 1, data_types::i8, format::bfzyx, data_types::i8, format::bfzyx, data_types::f32, format::bfzyx

// in_shape; out_shape; eltw_shape; kernel; stride; pad; dilation; groups; data_type; input_format; weights_type; weights_format; default_type; default_format;
#define CASE_CONV_ELTW_FP32_1 { 1, 16, 4, 5 }, { 1, 32, 2, 3 }, { 1, 32, 1, 1 }, { 32, 16, 3, 3 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::oiyx, data_types::f32, format::bfyx
#define CASE_CONV_ELTW_FP32_2 { 1, 16, 4, 5 }, { 1, 32, 2, 3 }, { 1, 1, 1, 1 }, { 32, 16, 3, 3 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::os_is_yx_isv16_osv16, data_types::f32, format::bfyx
#define CASE_CONV_ELTW_FP32_3 { 1, 16, 4, 5 }, { 1, 32, 4, 5 }, { 1, 32, 4, 5 }, { 32, 16, 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::os_is_yx_isv16_osv16, data_types::f32, format::bfyx
#define CASE_CONV_ELTW_FP32_4 { 1, 32, 4, 5 }, { 1, 32, 4, 5 }, { 1, 32, 1, 1 }, { 32, 1, 1, 3, 3 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 32, data_types::f32, format::b_fs_yx_fsv16, data_types::f32,  format::gs_oiyx_gsv16, data_types::f32, format::bfyx
#define CASE_CONV_ELTW_FP32_5 { 1, 32, 4, 5, 4 }, { 1, 32, 2, 3, 2 }, { 1, 32, 2, 1, 1 }, { 2, 16, 16, 3, 3, 3 }, { 1, 1, 1 }, { 0, 0, 0 }, { 1, 1, 1 }, 2, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32, format::g_os_is_zyx_isv16_osv16, data_types::f32, format::bfzyx
#define CASE_CONV_ELTW_FP32_6 { 1, 32, 4, 5, 4 }, { 1, 16, 2, 3, 2 }, { 1, 16, 2, 1, 1 }, { 2, 8, 16, 3, 3, 3 }, { 1, 1, 1 }, { 0, 0, 0 }, { 1, 1, 1 }, 2, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32, format::g_os_is_zyx_isv16_osv16, data_types::f32, format::bfzyx
#define CASE_CONV_ELTW_FP32_7 { 1, 16, 3, 5 }, { 1, 32, 1, 3 }, { 1, 32, 3, 1 }, { 32, 16, 3, 3 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::os_is_yx_isv16_osv16, data_types::f32, format::bfyx
#define CASE_CONV_ELTW_FP32_8 { 1, 32, 3, 5, 4 }, { 1, 16, 1, 3, 2 }, { 1, 1, 2, 1, 1 }, { 2, 8, 16, 3, 3, 3 }, { 1, 1, 1 }, { 0, 0, 0 }, { 1, 1, 1 }, 2, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32, format::g_os_is_zyx_isv16_osv16, data_types::f32, format::bfzyx

#define CASE_CONV_ELTW_i8_1 { 1, 16, 3, 5 }, { 1, 32, 1, 3 }, { 1, 32, 3, 1 }, { 32, 16, 3, 3 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, data_types::i8, format::b_fs_yx_fsv16, data_types::i8, format::os_is_yx_osv16_isv16, data_types::f32, format::bfyx
#define CASE_CONV_ELTW_i8_2 { 1, 16, 3, 5, 3 }, { 1, 32, 2, 4, 2 }, { 1, 1, 2, 4, 2 }, { 32, 16, 2, 2, 2 }, { 1, 1, 1 }, { 0, 0, 0 }, { 1, 1, 1 }, 1, data_types::i8, format::bfzyx, data_types::i8, format::oizyx, data_types::f32, format::bfzyx
#define CASE_CONV_ELTW_i8_3 { 1, 1, 1, 1, 1 }, { 1, 1, 1, 1, 1 }, { 1, 1, 1, 1, 1 }, { 1, 1, 1, 1, 1 }, { 1, 1, 1 }, { 0, 0, 0 }, { 1, 1, 1 }, 1, data_types::i8, format::bfzyx, data_types::i8, format::oizyx, data_types::f32, format::bfzyx
#define CASE_CONV_ELTW_i8_4 { 1, 16, 1, 4 }, { 1, 16, 1, 2 }, { 1, 16, 1, 1 }, { 16, 16, 1, 3 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, data_types::i8, format::b_fs_yx_fsv16, data_types::i8, format::os_is_yx_osv16_isv16, data_types::f32, format::bfyx
#define CASE_CONV_ELTW_i8_5 { 1, 16, 1, 4, 1 }, { 1, 16, 1, 2, 1 }, { 1, 16, 2, 1, 1 }, { 16, 16, 1, 3, 1 }, { 1, 1, 1 }, { 0, 0, 0 }, { 1, 1, 1 }, 1, data_types::i8, format::bfzyx, data_types::i8, format::oizyx, data_types::f32, format::bfzyx

/* ----------------------------------------------------------------------------------------------------- */
/* ---------------------------------------- FP32 convolution cases ------------------------------------- */
/* ----------------------------------------------------------------------------------------------------- */
/* ----------- NOTE: A part of tests is disabled until all FP kernels don't support fusings ------------ */

class conv_fp32_reorder_fsv16_to_bfyx : public ConvFusingTest {};
TEST_P(conv_fp32_reorder_fsv16_to_bfyx, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        reorder("reorder_fsv16", input_info("input"), format::b_fs_yx_fsv16, data_types::f32),
        convolution("conv_prim", input_info("reorder_fsv16"), "weights", "", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        reorder("reorder_bfyx", input_info("conv_prim"), format::bfyx, data_types::f32)
    );

    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_reorder_fsv16_to_bfyx, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_FP32_1, 2, 2, 2 },
    convolution_test_params{ CASE_CONV_FP32_2, 2, 2, 2 },
    convolution_test_params{ CASE_CONV_FP32_3, 2, 2, 2 },
    convolution_test_params{ CASE_CONV_FP32_4, 2, 2, 2 },
    convolution_test_params{ CASE_CONV_FP32_5, 2, 2, 2 },
    convolution_test_params{ CASE_CONV_FP32_14, 2, 2, 2 },

    convolution_test_params{ CASE_CONV_FP16_1, 2, 2, 2 },
    convolution_test_params{ CASE_CONV_FP16_2, 2, 2, 2 },
    convolution_test_params{ CASE_CONV_FP16_3, 2, 2, 2 },
    convolution_test_params{ CASE_CONV_FP16_4, 2, 2, 2 },
    convolution_test_params{ CASE_CONV_FP16_5, 2, 2, 2 },
    convolution_test_params{ CASE_CONV_FP16_13, 2, 2, 2 }
}));

class conv_fp32_reorder_fsv16_to_bfyx_conv : public ConvFusingTest {};
TEST_P(conv_fp32_reorder_fsv16_to_bfyx_conv, basic) {
    auto p = GetParam();

    auto dw_weights_layout = layout{ {p.out_shape[1].get_length(), 1, 1, 3, 3}, p.default_type, format::goiyx };
    ov::Strides dw_stride = { 1, 1 };
    ov::CoordinateDiff dw_pad = { 1, 1 };

    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p), -127, 127)),
        data("weights_dw", get_mem(dw_weights_layout, -127, 127)),
        reorder("reorder_fsv16", input_info("input"), format::b_fs_yx_fsv16, data_types::f32),
        convolution("conv_prim", input_info("reorder_fsv16"), "weights", "", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        reorder("reorder_bfyx", input_info("conv_prim"), format::bfyx, data_types::f32),
        convolution("conv_output", input_info("reorder_bfyx"), "weights_dw", "", p.out_shape[1].get_length(), dw_stride, p.dilation, dw_pad, dw_pad, true),
        activation("activation", input_info("conv_output"), activation_func::abs),
        reorder("reorder_output", input_info("activation"), p.default_format, data_types::f32)
    );

    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_reorder_fsv16_to_bfyx_conv, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_FP32_1, 3, 3, 4 },
    convolution_test_params{ CASE_CONV_FP32_2, 3, 3, 4 },
    convolution_test_params{ CASE_CONV_FP32_3, 3, 3, 4 },
    convolution_test_params{ CASE_CONV_FP32_4, 3, 3, 4 },
    convolution_test_params{ CASE_CONV_FP32_5, 3, 3, 4 },
    convolution_test_params{ CASE_CONV_FP32_14, 3, 3, 4 },

    convolution_test_params{ CASE_CONV_FP16_1, 3, 3, 4 },
    convolution_test_params{ CASE_CONV_FP16_2, 3, 3, 4 },
    convolution_test_params{ CASE_CONV_FP16_3, 3, 3, 4 },
    convolution_test_params{ CASE_CONV_FP16_4, 3, 3, 4 },
    convolution_test_params{ CASE_CONV_FP16_5, 3, 3, 4 },
    convolution_test_params{ CASE_CONV_FP16_13, 3, 3, 4 },
}));


class conv_fp32_activation : public ConvFusingTest {};
TEST_P(conv_fp32_activation, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        activation("activation", input_info("conv_prim"), activation_func::abs),
        reorder("reorder_bfyx", input_info("activation"), p.default_format, data_types::f32)
    );

    if (engine.get_device_info().supports_immad &&
        p.default_type == data_types::f16 &&
        p.weights_format == format::gs_oiyx_gsv16) {
        GTEST_SKIP(); // Issue: 94154
    }

    tolerance = default_tolerance(p.default_type);
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_activation, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_FP32_1, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_FP32_2, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_FP32_3, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_FP32_4, 2, 2, 3 },

    convolution_test_params{ CASE_CONV_FP16_1, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_FP16_2, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_FP16_3, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_FP16_4, 2, 2, 3 },
}));


class conv_fp32_scale : public ConvFusingTest {};
TEST_P(conv_fp32_scale, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f/255.f)),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        eltwise("scale", { input_info("conv_prim"), input_info("scale_data") }, eltwise_mode::prod),
        reorder("reorder_bfyx", input_info("scale"), p.default_format, data_types::f32)
    );

    if (engine.get_device_info().supports_immad &&
        p.default_type == data_types::f16 &&
        p.weights_format == format::gs_oiyx_gsv16) {
        GTEST_SKIP(); // Issue: 94154
    }

    tolerance = default_tolerance(p.default_type);
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_scale, ::testing::ValuesIn(std::vector<convolution_test_params>{
    // convolution_test_params{ CASE_CONV_FP32_1, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_FP32_2, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_FP32_3, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_FP32_4, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_FP32_10, 2, 2, 3 },

    // convolution_test_params{ CASE_CONV_FP16_1, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_FP16_2, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_FP16_3, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_FP16_4, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_FP16_10, 2, 2, 3 },
}));

class conv_duplicated_connection : public ConvFusingTest {};
TEST_P(conv_duplicated_connection, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        eltwise("scale", { input_info("conv_prim"), input_info("conv_prim") }, eltwise_mode::prod),
        reorder("reorder_bfyx", input_info("scale"), p.default_format, data_types::f32)
    );

    tolerance = default_tolerance(p.default_type);
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_duplicated_connection, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_FP32_2, 2, 3, 3 },
}));

class conv_fp32_bias : public ConvFusingTest {};
TEST_P(conv_fp32_bias, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        convolution("conv_prim", input_info("input"), "weights", "", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        eltwise("add_bias", { input_info("conv_prim"), input_info("bias") }, eltwise_mode::sum),
        reorder("reorder_bfyx", input_info("add_bias"), p.default_format, data_types::f32)
    );

    if (engine.get_device_info().supports_immad &&
        p.default_type == data_types::f16 &&
        p.weights_format == format::gs_oiyx_gsv16) {
        GTEST_SKIP(); // Issue: 94154
    }

    tolerance = default_tolerance(p.default_type);
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_bias, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_FP32_1, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_FP32_2, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_FP32_3, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_FP32_4, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_FP32_10, 2, 2, 3 },

    convolution_test_params{ CASE_CONV_FP16_1, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_FP16_2, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_FP16_3, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_FP16_4, 2, 2, 3 },
    // convolution_test_params{ CASE_CONV_FP16_10, 2, 2, 3 }, // Issue: 94154
}));

class conv_fp32_double_bias : public ConvFusingTest {};
TEST_P(conv_fp32_double_bias, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias1", get_mem(get_per_channel_layout(p))),
        data("bias2", get_mem(get_per_channel_layout(p))),
        convolution("conv_prim", input_info("input"), "weights", "", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        eltwise("add_bias1", { input_info("conv_prim"), input_info("bias1") }, eltwise_mode::sum),
        eltwise("add_bias2", { input_info("add_bias1"), input_info("bias2") }, eltwise_mode::sum),
        reorder("reorder_bfyx", input_info("add_bias2"), p.default_format, data_types::f32)
    );

    tolerance = default_tolerance(p.default_type);
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_double_bias, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_U8S8_1, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_S8S8_1, 2, 2, 4 },
}));

class conv_fp32_wrong_bias : public ConvFusingTest {};
TEST_P(conv_fp32_wrong_bias, basic) {
    // Check case when eltwise add dependency has shape [1, 1, X, Y] and X*Y == CONV_OUT_FEATURES
    auto p = GetParam();
    ov::PartialShape eltw_data_shape = get_input_layout(p).get_partial_shape();
    for (size_t i = 0; i < eltw_data_shape.size() - 2; i++) {
        eltw_data_shape[i] = 1;
    }

    auto eltw_data_layout = layout{eltw_data_shape, p.default_type, format::bfyx};
    ASSERT_EQ(p.out_shape[1], eltw_data_layout.count());

    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("non-bias", get_mem(eltw_data_layout)),
        convolution("conv_prim", input_info("input"), "weights", "", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        eltwise("add", { input_info("conv_prim"), input_info("non-bias") }, eltwise_mode::sum),
        reorder("reorder_bfyx", input_info("add"), p.default_format, data_types::f32)
    );

    tolerance = default_tolerance(p.default_type);
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_wrong_bias, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_FP32_15, 3, 3, 3 },
}));

class conv_fp32_add_per_element_planar_const : public ConvFusingTest {};
TEST_P(conv_fp32_add_per_element_planar_const, basic) {
    auto p = GetParam();

    ov::intel_gpu::ImplementationDesc conv_impl = { format::b_fs_yx_fsv16, "convolution_gpu_bfyx_f16" };
    ov::intel_gpu::ImplementationDesc permute_impl = { format::b_fs_yx_fsv16, "" };
    cfg_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "conv_prim", conv_impl },
                                                              { "permute", permute_impl } }));

    auto out_layout = get_output_layout(p);
    out_layout.format = format::bfyx;
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("data", get_mem(out_layout)),
        convolution("conv_prim", input_info("input"), "weights", "", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        eltwise("add", { input_info("conv_prim"), input_info("data") }, eltwise_mode::sum),
        permute("permute", input_info("add"), {3, 2, 1, 0}),
        reorder("reorder_bfyx", input_info("permute"), p.default_format, data_types::f32)
    );

    tolerance = default_tolerance(p.default_type);
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_add_per_element_planar_const, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_FP32_3, 3, 3, 4 },
}));

class conv_fp32_prelu_eltwise : public ConvFusingTest {};
TEST_P(conv_fp32_prelu_eltwise, basic_sum) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("slope_data", get_mem(get_per_channel_layout(p))),
        data("eltwise_data", get_mem(get_output_layout(p))),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        activation("activation", input_info("conv_prim"), "slope_data", activation_func::relu_negative_slope),
        eltwise("eltwise", input_info("activation"), input_info("eltwise_data"), eltwise_mode::sum),
        reorder("reorder_bfyx", input_info("eltwise"), p.default_format, data_types::f32)
    );

    tolerance = default_tolerance(p.data_type);
    if (engine.get_device_info().supports_immad && p.default_type == data_types::f16) {
        tolerance *= 2;
        if (p.weights_format == format::gs_oiyx_gsv16) {
            GTEST_SKIP(); // Issue: 94154
        }
    }
    execute(p);
}

TEST_P(conv_fp32_prelu_eltwise, basic_sum_slope_2) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("slope_data", get_mem(get_prelu_slope_layout(p))),
        data("eltwise_data", get_mem(get_output_layout(p))),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        activation("activation", input_info("conv_prim"), "slope_data", activation_func::relu_negative_slope),
        eltwise("eltwise", input_info("activation"), input_info("eltwise_data"), eltwise_mode::sum),
        reorder("reorder_bfyx", input_info("eltwise"), p.default_format, data_types::f32)
    );

    tolerance = default_tolerance(p.data_type);
    if (engine.get_device_info().supports_immad && p.default_type == data_types::f16) {
        tolerance *= 2;
        if (p.weights_format == format::gs_oiyx_gsv16) {
            GTEST_SKIP(); // Issue: 94154
        }
    }
    execute(p);
}

TEST_P(conv_fp32_prelu_eltwise, basic_prod) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("slope_data", get_mem(get_per_channel_layout(p))),
        data("eltwise_data", get_mem(get_output_layout(p))),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        activation("activation", input_info("conv_prim"), "slope_data", activation_func::relu_negative_slope),
        eltwise("eltwise", input_info("activation"), input_info("eltwise_data"), eltwise_mode::prod),
        reorder("reorder_bfyx", input_info("eltwise"), p.default_format, data_types::f32)
    );

    if (engine.get_device_info().supports_immad &&
        p.default_type == data_types::f16 &&
        p.weights_format == format::gs_oiyx_gsv16) {
        GTEST_SKIP(); // Issue: 94154
    }

    tolerance = default_tolerance(p.data_type);
    execute(p);
}

TEST_P(conv_fp32_prelu_eltwise, basic_prod_slope_2) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("slope_data", get_mem(get_prelu_slope_layout(p))),
        data("eltwise_data", get_mem(get_output_layout(p))),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        activation("activation", input_info("conv_prim"), "slope_data", activation_func::relu_negative_slope),
        eltwise("eltwise", input_info("activation"), input_info("eltwise_data"), eltwise_mode::prod),
        reorder("reorder_bfyx", input_info("eltwise"), p.default_format, data_types::f32)
    );

    tolerance = default_tolerance(p.data_type);
    if (engine.get_device_info().supports_immad && p.default_type == data_types::f16) {
        tolerance *= 4;
        if (p.weights_format == format::gs_oiyx_gsv16) {
            GTEST_SKIP(); // Issue: 94154
        }
    }
    execute(p);
}

TEST_P(conv_fp32_prelu_eltwise, eltw_broadcast_sum) {
    auto p = GetParam();
    tensor eltw_shape = p.default_format.spatial_num() == 2 ? tensor{ 1, 1, 1, 1 } : tensor{ 1, 1, 1, 1, 1 };
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("slope_data", get_mem(get_per_channel_layout(p))),
        data("eltwise_data", get_mem(layout{ p.data_type, p.input_format, eltw_shape })),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        activation("activation", input_info("conv_prim"), "slope_data", activation_func::relu_negative_slope),
        eltwise("eltwise", input_info("activation"), input_info("eltwise_data"), eltwise_mode::sum),
        reorder("reorder_bfyx", input_info("eltwise"), p.default_format, data_types::f32)
    );

    tolerance = default_tolerance(p.data_type);
    if (engine.get_device_info().supports_immad && p.default_type == data_types::f16) {
        tolerance *= 2;
        if (p.weights_format == format::gs_oiyx_gsv16) {
            GTEST_SKIP(); // Issue: 94154
        }
    }
    execute(p);
}

TEST_P(conv_fp32_prelu_eltwise, eltw_broadcast_sum_slope_2) {
    auto p = GetParam();
    tensor eltw_shape = p.default_format.spatial_num() == 2 ? tensor{ 1, 1, 1, 1 } : tensor{ 1, 1, 1, 1, 1 };
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("slope_data", get_mem(get_prelu_slope_layout(p))),
        data("eltwise_data", get_mem(layout{ p.data_type, p.input_format, eltw_shape })),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        activation("activation", input_info("conv_prim"), "slope_data", activation_func::relu_negative_slope),
        eltwise("eltwise", input_info("activation"), input_info("eltwise_data"), eltwise_mode::sum),
        reorder("reorder_bfyx", input_info("eltwise"), p.default_format, data_types::f32)
    );

    if (engine.get_device_info().supports_immad &&
        p.default_type == data_types::f16 &&
        p.weights_format == format::gs_oiyx_gsv16) {
        GTEST_SKIP(); // Issue: 94154
    }

    tolerance = default_tolerance(p.data_type);
    execute(p);
}

TEST_P(conv_fp32_prelu_eltwise, eltw_broadcast_prod) {
    auto p = GetParam();
    tensor eltw_shape = p.default_format.spatial_num() == 2 ? tensor{ 1, 1, 1, 1 } : tensor{ 1, 1, 1, 1, 1 };
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("slope_data", get_mem(get_per_channel_layout(p))),
        data("eltwise_data", get_mem(layout{ p.data_type, p.input_format, eltw_shape })),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        activation("activation", input_info("conv_prim"), "slope_data", activation_func::relu_negative_slope),
        eltwise("eltwise", input_info("activation"), input_info("eltwise_data"), eltwise_mode::prod),
        reorder("reorder_bfyx", input_info("eltwise"), p.default_format, data_types::f32)
    );

    tolerance = default_tolerance(p.data_type);
    if (engine.get_device_info().supports_immad && p.default_type == data_types::f16) {
        tolerance *= 4;
        if (p.weights_format == format::gs_oiyx_gsv16) {
            GTEST_SKIP(); // Issue: 94154
        }
    }
    execute(p);
}

TEST_P(conv_fp32_prelu_eltwise, eltw_broadcast_prod_slope_2) {
    auto p = GetParam();
    tensor eltw_shape = p.default_format.spatial_num() == 2 ? tensor{ 1, 1, 1, 1 } : tensor{ 1, 1, 1, 1, 1 };
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("slope_data", get_mem(get_prelu_slope_layout(p))),
        data("eltwise_data", get_mem(layout{ p.data_type, p.input_format, eltw_shape })),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        activation("activation", input_info("conv_prim"), "slope_data", activation_func::relu_negative_slope),
        eltwise("eltwise", input_info("activation"), input_info("eltwise_data"), eltwise_mode::prod),
        reorder("reorder_bfyx", input_info("eltwise"), p.default_format, data_types::f32)
    );

    if (engine.get_device_info().supports_immad &&
        p.default_type == data_types::f16 &&
        p.weights_format == format::gs_oiyx_gsv16) {
        GTEST_SKIP(); // Issue: 94154
    }

    tolerance = default_tolerance(p.data_type);
    execute(p);
}


TEST_P(conv_fp32_prelu_eltwise, vector_ops) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("slope_data", get_mem(get_per_channel_layout(p))),
        data("eltwise_data", get_mem(get_output_layout(p))),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        activation("activation", input_info("conv_prim"), "slope_data", activation_func::relu_negative_slope),
        eltwise("eltwise", input_info("activation"), input_info("eltwise_data"), eltwise_mode::sum),
        reorder("reorder_bfyx", input_info("eltwise"), p.default_format, data_types::f32)
    );

    ov::intel_gpu::ImplementationDesc conv_impl = { format::b_fs_yx_fsv16, "", impl_types::ocl };
    cfg_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "conv_prim", conv_impl } }));

    tolerance = default_tolerance(p.data_type);
    execute(p);
}

TEST_P(conv_fp32_prelu_eltwise, vector_ops_slope_2) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("slope_data", get_mem(get_prelu_slope_layout(p))),
        data("eltwise_data", get_mem(get_output_layout(p))),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        activation("activation", input_info("conv_prim"), "slope_data", activation_func::relu_negative_slope),
        eltwise("eltwise", input_info("activation"), input_info("eltwise_data"), eltwise_mode::sum),
        reorder("reorder_bfyx", input_info("eltwise"), p.default_format, data_types::f32)
    );

    ov::intel_gpu::ImplementationDesc conv_impl = { format::b_fs_yx_fsv16, "", impl_types::ocl };
    cfg_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "conv_prim", conv_impl } }));

    tolerance = default_tolerance(p.data_type);
    if (engine.get_device_info().supports_immad) {
        tolerance = 1e-2f;
    }
    execute(p);
}

TEST_P(conv_fp32_prelu_eltwise, vector_ops_mixed_types) {
    auto p = GetParam();
    auto slope_type = p.default_type == data_types::f32 ? data_types::f16 : data_types::f32;
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("slope_data", get_mem(layout{ { 1, p.out_shape[1], 1, 1 }, slope_type, p.default_format })),
        data("eltwise_data", get_mem(get_output_layout(p))),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        activation("activation", input_info("conv_prim"), "slope_data", activation_func::relu_negative_slope),
        eltwise("eltwise", input_info("activation"), input_info("eltwise_data"), eltwise_mode::sum),
        reorder("reorder_bfyx", input_info("eltwise"), p.default_format, data_types::f32)
    );

    ov::intel_gpu::ImplementationDesc conv_impl = { format::b_fs_yx_fsv16, "", impl_types::ocl };
    cfg_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "conv_prim", conv_impl } }));

    tolerance = default_tolerance(p.data_type);
    execute(p);
}

TEST_P(conv_fp32_prelu_eltwise, vector_ops_mixed_types_slope_2) {
    auto p = GetParam();
    auto slope_type = p.default_type == data_types::f32 ? data_types::f16 : data_types::f32;
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("slope_data", get_mem(layout{ { 1, p.out_shape[1], 1, p.out_shape[p.out_shape.size() - 1] }, slope_type, p.input_format,  })),
        data("eltwise_data", get_mem(get_output_layout(p))),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        activation("activation", input_info("conv_prim"), "slope_data", activation_func::relu_negative_slope),
        eltwise("eltwise", input_info("activation"), input_info("eltwise_data"), eltwise_mode::sum),
        reorder("reorder_bfyx", input_info("eltwise"), p.default_format, data_types::f32)
    );

    ov::intel_gpu::ImplementationDesc conv_impl = { format::b_fs_yx_fsv16, "", impl_types::ocl };
    cfg_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "conv_prim", conv_impl } }));

    tolerance = default_tolerance(p.data_type);
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_prelu_eltwise, ::testing::ValuesIn(std::vector<convolution_test_params>{
    // convolution_test_params{ CASE_CONV_FP32_1, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_FP32_2, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_FP32_3, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_FP32_4, 2, 2, 4 },

    // convolution_test_params{ CASE_CONV_FP16_1, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_FP16_2, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_FP16_3, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_FP16_4, 2, 2, 4 },
}));

class conv_fp32_multi_eltwise_2 : public ConvFusingTest {};
TEST_P(conv_fp32_multi_eltwise_2, basic) {
    if (engine.get_device_info().supports_immad) {
        return;
    }

    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("eltwise_data", get_mem(get_output_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("weights", get_mem(get_weights_layout(p))),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        eltwise("eltwise1", input_info("conv_prim"), input_info("eltwise_data"), eltwise_mode::sum),
        eltwise("eltwise2", input_info("eltwise1"), input_info("conv_prim"), eltwise_mode::prod),
        reorder("reorder_bfyx", input_info("eltwise2"), p.default_format, data_types::f32)
    );
    ov::intel_gpu::ImplementationDesc conv_impl = { format::b_fs_yx_fsv16, "" };
    cfg_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "conv_prim", conv_impl } }));

    tolerance = default_tolerance(p.default_type);
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_multi_eltwise_2, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_FP32_2, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_FP32_3, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_FP32_4, 2, 2, 4 },

    convolution_test_params{ CASE_CONV_FP16_2, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_FP16_3, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_FP16_4, 2, 2, 4 },
}));


class conv_fp32_multi_eltwise_2_clamp : public ConvFusingTest {};
TEST_P(conv_fp32_multi_eltwise_2_clamp, basic) {
    if (engine.get_device_info().supports_immad) {
        return;
    }

    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("eltwise1_data", get_mem(get_output_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("weights", get_mem(get_weights_layout(p))),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        eltwise("eltwise1", input_info("conv_prim"), input_info("eltwise1_data"), eltwise_mode::sum),
        activation("activation", input_info("eltwise1"), activation_func::clamp, { 0.5f, 2.5f }),
        eltwise("eltwise2", input_info("activation"), input_info("conv_prim"), eltwise_mode::prod),
        reorder("reorder_bfyx", input_info("eltwise2"), p.default_format, data_types::f32)
    );
    ov::intel_gpu::ImplementationDesc conv_impl = { format::b_fs_yx_fsv16, "" };
    cfg_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "conv_prim", conv_impl } }));

    tolerance = default_tolerance(p.default_type);
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_multi_eltwise_2_clamp, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_FP32_2, 2, 2, 5 },
    convolution_test_params{ CASE_CONV_FP32_3, 2, 2, 5 },
    convolution_test_params{ CASE_CONV_FP32_4, 2, 2, 5 },

    convolution_test_params{ CASE_CONV_FP16_2, 2, 2, 5 },
    convolution_test_params{ CASE_CONV_FP16_3, 2, 2, 5 },
    convolution_test_params{ CASE_CONV_FP16_4, 2, 2, 5 },
}));


class conv_fp32_multi_eltwise_4_clamp : public ConvFusingTest {};
TEST_P(conv_fp32_multi_eltwise_4_clamp, basic) {
    if (engine.get_device_info().supports_immad) {
        return;
    }

    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("eltwise1_data", get_mem(get_output_layout(p))),
        data("eltwise2_data", get_mem(get_output_layout(p))),
        data("eltwise4_data", get_mem(get_output_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("weights", get_mem(get_weights_layout(p))),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        eltwise("eltwise1_add", input_info("conv_prim"), input_info("eltwise1_data"), eltwise_mode::sum),
        activation("activation", input_info("eltwise1_add"), activation_func::clamp, { 0.5f, 2.5f }),
        eltwise("eltwise2_mul", input_info("activation"), input_info("conv_prim"), eltwise_mode::prod),
        eltwise("eltwise3_div", input_info("eltwise2_mul"), input_info("eltwise2_data"), eltwise_mode::prod),
        eltwise("eltwise4_add", input_info("eltwise3_div"), input_info("eltwise4_data"), eltwise_mode::sum),
        reorder("reorder_bfyx", input_info("eltwise4_add"), p.default_format, data_types::f32)
    );
    ov::intel_gpu::ImplementationDesc conv_impl = { format::b_fs_yx_fsv16, "" };
    cfg_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "conv_prim", conv_impl } }));

    tolerance = default_tolerance(p.default_type);
    if (p.default_type == data_types::f16) {
        tolerance *= 4.f; // Issue: 94154
    }
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_multi_eltwise_4_clamp, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_FP32_2, 2, 2, 7 },
    convolution_test_params{ CASE_CONV_FP32_3, 2, 2, 7 },
    convolution_test_params{ CASE_CONV_FP32_4, 2, 2, 7 },

    convolution_test_params{ CASE_CONV_FP16_2, 2, 2, 7 },
    convolution_test_params{ CASE_CONV_FP16_3, 2, 2, 7 },
    convolution_test_params{ CASE_CONV_FP16_4, 2, 2, 7 },
}));

class conv_fp32_eltwise_fusing_extend_ops : public ConvFusingTest {};
TEST_P(conv_fp32_eltwise_fusing_extend_ops, pattern01_simple_sub) {
    if (engine.get_device_info().supports_immad) {
        return;
    }

    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("eltwise_data1", get_mem(get_output_layout(p))),
        data("eltwise_data2", get_mem(get_output_layout(p))),
        data("eltwise_data4", get_mem(get_output_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("weights", get_mem(get_weights_layout(p))),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        eltwise("eltwise1_sum", input_info("conv_prim"), input_info("eltwise_data1"), eltwise_mode::sum),
        eltwise("eltwise2_sub", input_info("conv_prim"), input_info("eltwise_data2"), eltwise_mode::sub),
        eltwise("eltwise3_prod", input_info("eltwise1_sum"), input_info("eltwise2_sub"), eltwise_mode::prod),
        eltwise("eltwise4_sum", input_info("eltwise3_prod"), input_info("eltwise_data4"), eltwise_mode::sum),
        concatenation("concat", { input_info("eltwise4_sum"), input_info("eltwise4_sum") }, 1),
        reorder("reorder_bfyx", input_info("concat"), p.default_format, data_types::f32)
    );
    ov::intel_gpu::ImplementationDesc conv_impl = { format::b_fs_yx_fsv16, "" };
    cfg_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "conv_prim", conv_impl } }));

    tolerance = default_tolerance(p.default_type);
    if (p.default_type == data_types::f16) {
        tolerance *= 8.f; // Issue: 94154
    }
    execute(p);
}

TEST_P(conv_fp32_eltwise_fusing_extend_ops, pattern02_sub_scale) {
    if (engine.get_device_info().supports_immad) {
        return;
    }

    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("eltwise_data1", get_mem(get_output_layout(p))),
        data("eltwise_data2", get_mem(get_output_layout(p))),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f/255.f)),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("weights", get_mem(get_weights_layout(p))),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        eltwise("eltwise1_sum", input_info("conv_prim"), input_info("eltwise_data1"), eltwise_mode::sum),
        eltwise("eltwise2_sub", input_info("conv_prim"), input_info("eltwise1_sum"), eltwise_mode::sub),
        eltwise("eltwise3_prod", input_info("eltwise2_sub"), input_info("eltwise_data2"), eltwise_mode::prod),
        eltwise("scale", { input_info("eltwise3_prod"), input_info("scale_data") }, eltwise_mode::prod),
        concatenation("concat", { input_info("scale"), input_info("scale") }, 1),
        reorder("reorder_bfyx", input_info("concat"), p.default_format, data_types::f32)
    );
    ov::intel_gpu::ImplementationDesc conv_impl = { format::b_fs_yx_fsv16, "" };
    cfg_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "conv_prim", conv_impl } }));

    tolerance = default_tolerance(p.default_type);
    execute(p);
}

TEST_P(conv_fp32_eltwise_fusing_extend_ops, pattern03_sub_div) {
    if (engine.get_device_info().supports_immad) {
        return;
    }

    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("eltwise_data1", get_mem(get_output_layout(p))),
        data("eltwise_data2", get_mem(get_output_layout(p), 1.0f)),
        data("eltwise_data3", get_mem(get_output_layout(p))),
        data("eltwise_data4", get_mem(get_output_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("weights", get_mem(get_weights_layout(p))),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        eltwise("eltwise1_sum", input_info("conv_prim"), input_info("eltwise_data1"), eltwise_mode::sum),
        eltwise("eltwise2_div", input_info("eltwise1_sum"), input_info("eltwise_data2"), eltwise_mode::div),
        eltwise("eltwise3_prod", input_info("eltwise2_div"), input_info("eltwise_data3"), eltwise_mode::prod),
        eltwise("eltwise4_sum", input_info("eltwise3_prod"), input_info("eltwise_data4"), eltwise_mode::sum),
        concatenation("concat", { input_info("eltwise4_sum"), input_info("eltwise4_sum") }, 1),
        reorder("reorder_bfyx", input_info("concat"), p.default_format, data_types::f32)
    );
    ov::intel_gpu::ImplementationDesc conv_impl = { format::b_fs_yx_fsv16, "" };
    cfg_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "conv_prim", conv_impl } }));

    tolerance = default_tolerance(p.default_type);
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_eltwise_fusing_extend_ops, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_FP32_2, 3, 3, 7 },
    convolution_test_params{ CASE_CONV_FP32_3, 3, 3, 7 },
    convolution_test_params{ CASE_CONV_FP32_4, 3, 3, 7 },

    convolution_test_params{ CASE_CONV_FP16_2, 3, 3, 7 },
    convolution_test_params{ CASE_CONV_FP16_3, 3, 3, 7 },
    convolution_test_params{ CASE_CONV_FP16_4, 3, 3, 7 },
}));

class conv_fp32_eltwise_fusing_2conv : public ConvFusingTest {};
TEST_P(conv_fp32_eltwise_fusing_2conv, basic) {
    if (engine.get_device_info().supports_immad) {
        return;
    }

    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("bias0", get_mem(get_per_channel_layout(p))),
        data("weights0", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("weights", get_mem(get_weights_layout(p))),
        convolution("conv_prim0", input_info("input"), { "weights0" }, { "bias0" }, p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        eltwise("eltwise1", input_info("conv_prim0"), input_info("conv_prim"), eltwise_mode::sum),
        eltwise("eltwise2", input_info("conv_prim0"), input_info("conv_prim"), eltwise_mode::sum),
        eltwise("eltwise3", input_info("eltwise1"), input_info("eltwise2"), eltwise_mode::prod),
        concatenation("concat", { input_info("eltwise3"), input_info("eltwise3") }, 1),
        reorder("reorder_bfyx", input_info("concat"), p.default_format, data_types::f32)
    );
    ov::intel_gpu::ImplementationDesc conv_impl = { format::b_fs_yx_fsv16, "" };
    cfg_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "conv_prim0", conv_impl }, { "conv_prim", conv_impl }  }));

    tolerance = default_tolerance(p.default_type);
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_eltwise_fusing_2conv, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_FP32_2, 4, 4, 7 },
    convolution_test_params{ CASE_CONV_FP32_3, 4, 4, 7 },
    convolution_test_params{ CASE_CONV_FP32_4, 4, 4, 7 },

    convolution_test_params{ CASE_CONV_FP16_2, 4, 4, 7 },
    convolution_test_params{ CASE_CONV_FP16_3, 4, 4, 7 },
    convolution_test_params{ CASE_CONV_FP16_4, 4, 4, 7 },
}));


class conv_fp32_multi_eltwise_3_fusing : public ConvFusingTest {};
TEST_P(conv_fp32_multi_eltwise_3_fusing, basic) {
    if (engine.get_device_info().supports_immad) {
        return;
    }

    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("eltwise_data1", get_mem(get_output_layout(p))),
        data("eltwise_data2", get_mem(get_output_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("weights", get_mem(get_weights_layout(p))),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        eltwise("eltwise1", input_info("conv_prim"), input_info("eltwise_data1"), eltwise_mode::sum),
        eltwise("eltwise2", input_info("conv_prim"), input_info("eltwise_data2"), eltwise_mode::sum),
        eltwise("eltwise3", input_info("eltwise1"), input_info("eltwise2"), eltwise_mode::prod),
        reorder("reorder_bfyx", input_info("eltwise3"), p.default_format, data_types::f32)
    );
    ov::intel_gpu::ImplementationDesc conv_impl = { format::b_fs_yx_fsv16, "" };
    cfg_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "conv_prim", conv_impl } }));

    tolerance = default_tolerance(p.default_type);
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_multi_eltwise_3_fusing, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_FP32_2, 2, 2, 5 },
    convolution_test_params{ CASE_CONV_FP32_3, 2, 2, 5 },
    convolution_test_params{ CASE_CONV_FP32_4, 2, 2, 5 },

    convolution_test_params{ CASE_CONV_FP16_2, 2, 2, 5 },
    convolution_test_params{ CASE_CONV_FP16_3, 2, 2, 5 },
    convolution_test_params{ CASE_CONV_FP16_4, 2, 2, 5 },
}));



class conv_fp32_multi_eltwise_quantization : public ConvFusingTest {};
TEST_P(conv_fp32_multi_eltwise_quantization, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -127)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        data("eltwise_data1", get_mem(get_output_layout(p))),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        quantize("quantize", input_info("conv_prim"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 255, data_types::i8),
        eltwise("eltwise1", input_info("conv_prim"), input_info("eltwise_data1"), eltwise_mode::sum),
        eltwise("eltwise2", input_info("eltwise1"), input_info("quantize"), eltwise_mode::prod),
        reorder("reorder_bfyx", input_info("eltwise2"), p.default_format, data_types::f32)
    );

    if (engine.get_device_info().supports_immad &&
        p.default_type == data_types::f16 &&
        p.weights_format == format::gs_oiyx_gsv16) {
        GTEST_SKIP(); // Issue: 94154
    }

    tolerance = 1.f;
    if (p.default_type == data_types::f16) {
        tolerance *= 8.f; // Issue: 94154
    }
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_multi_eltwise_quantization, ::testing::ValuesIn(std::vector<convolution_test_params>{
    // convolution_test_params{ CASE_CONV_FP32_2, 4, 4, 5 },
    convolution_test_params{ CASE_CONV_FP32_4, 4, 4, 5 },

    convolution_test_params{ CASE_CONV_FP16_2, 4, 4, 5 },
    convolution_test_params{ CASE_CONV_FP16_3, 4, 4, 5 },
    convolution_test_params{ CASE_CONV_FP16_4, 4, 4, 5 },
}));


class conv_fp32_multi_eltwise_concat : public ConvFusingTest {};
TEST_P(conv_fp32_multi_eltwise_concat, basic) {
    auto p = GetParam();
    data_types output_type = data_types::i8;
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("eltwise_data1", get_mem(get_output_layout(p))),
        data("eltwise_data2", get_mem(get_output_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("weights", get_mem(get_weights_layout(p))),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        eltwise("eltwise1", input_info("conv_prim"), input_info("eltwise_data1"), eltwise_mode::sum),
        eltwise("eltwise2", input_info("conv_prim"), input_info("eltwise_data2"), eltwise_mode::sum),
        concatenation("concat",
            { input_info("eltwise1"), input_info("eltwise2") },
            2,
            output_type),
        reorder("reorder_bfyx", input_info("concat"), p.default_format, data_types::f32)
    );
    ov::intel_gpu::ImplementationDesc conv_impl = { format::b_fs_yx_fsv16, "", impl_types::ocl };
    cfg_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "conv_prim", conv_impl } }));

    tolerance = default_tolerance(output_type);
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_multi_eltwise_concat, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_FP32_2, 5, 5, 5 },
    convolution_test_params{ CASE_CONV_FP32_3, 5, 5, 5 },
    convolution_test_params{ CASE_CONV_FP32_4, 5, 5, 5 },

    convolution_test_params{ CASE_CONV_FP16_2, 5, 5, 5 },
    convolution_test_params{ CASE_CONV_FP16_3, 5, 5, 5 },
    convolution_test_params{ CASE_CONV_FP16_4, 5, 5, 5 },
}));

class conv_fp32_eltwise_b_fs_zyx_fsv16 : public ConvFusingTest {};

TEST_P(conv_fp32_eltwise_b_fs_zyx_fsv16, vector_ops) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("eltwise_data", get_mem(get_output_layout(p))),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        eltwise("eltwise", input_info("conv_prim"), input_info("eltwise_data"), eltwise_mode::sum),
        reorder("reorder_bfyx", input_info("eltwise"), p.default_format, data_types::f32)
    );

    ov::intel_gpu::ImplementationDesc conv_impl = { format::b_fs_zyx_fsv16, "" };
    cfg_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "conv_prim", conv_impl } }));

    tolerance = default_tolerance(p.default_type);
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_eltwise_b_fs_zyx_fsv16, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_FP32_6, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_FP32_7, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_FP32_8, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_FP32_9, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_FP32_11, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_FP32_12, 2, 2, 3 },
    // convolution_test_params{ CASE_CONV_FP32_13, 2, 2, 3 }, - leads to mvn_scale_activation_quantize_i8_eltwise_fp32_quantize_i8.basic/11 test failure

    convolution_test_params{ CASE_CONV_FP16_6, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_FP16_7, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_FP16_8, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_FP16_9, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_FP16_11, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_FP16_12, 2, 2, 3 },
}));

class conv_fp32_quantize_u8_first_conv : public ConvFusingTest {};
TEST_P(conv_fp32_quantize_u8_first_conv, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), 0)),
        data("out_hi", get_mem(get_single_element_layout(p), 255)),
        reorder("reordered_input", input_info("input"), format::b_fs_yx_fsv16, p.data_type),
        convolution("conv_prim", input_info("reordered_input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        quantize("quantize", input_info("conv_prim"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 256, data_types::u8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_quantize_u8_first_conv, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_FP32_14, 2, 2, 3 },
}));

class conv_fp32_quantize_u8 : public ConvFusingTest {};
TEST_P(conv_fp32_quantize_u8, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), 0)),
        data("out_hi", get_mem(get_single_element_layout(p), 255)),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        quantize("quantize", input_info("conv_prim"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 256, data_types::u8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_quantize_u8, ::testing::ValuesIn(std::vector<convolution_test_params>{
    // For now only b_fs_yx_fsv16 supports this case
    convolution_test_params{ CASE_CONV_FP32_2, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_FP32_3, 2, 2, 3 },

    convolution_test_params{ CASE_CONV_FP16_2, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_FP16_3, 2, 2, 3 },
}));

class conv_fp32_scale_quantize_i8 : public ConvFusingTest {};
TEST_P(conv_fp32_scale_quantize_i8, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -127)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f/255.f/255)),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        eltwise("scale", { input_info("conv_prim"), input_info("scale_data") }, eltwise_mode::prod),
        quantize("quantize", input_info("scale"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 255, data_types::i8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );
    // Output elements are in range [-127, 127]
    // 1.0f difference is allowed, since quantize can return different values in ref and scale_shift kernels
    // due to big error of division (in ref kernel).
    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_scale_quantize_i8, ::testing::ValuesIn(std::vector<convolution_test_params>{
    // For now only b_fs_yx_fsv16 supports this case
    convolution_test_params{ CASE_CONV_FP32_2, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_FP32_3, 2, 2, 4 },

    convolution_test_params{ CASE_CONV_FP16_2, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_FP16_3, 2, 2, 4 },
}));

class conv_fp32_scale_activation_quantize_i8 : public ConvFusingTest {};
TEST_P(conv_fp32_scale_activation_quantize_i8, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -127)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f/255.f/255)),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        eltwise("scale", { input_info("conv_prim"), input_info("scale_data") }, eltwise_mode::prod),
        activation("activation_scale", input_info("scale"), activation_func::exp),
        quantize("quantize", input_info("activation_scale"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 255, data_types::i8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_scale_activation_quantize_i8, ::testing::ValuesIn(std::vector<convolution_test_params>{
    // For now only b_fs_yx_fsv16 supports this case
    convolution_test_params{ CASE_CONV_FP32_2, 2, 2, 5 },
    convolution_test_params{ CASE_CONV_FP32_3, 2, 2, 5 },

    convolution_test_params{ CASE_CONV_FP16_2, 2, 2, 5 },
    convolution_test_params{ CASE_CONV_FP16_3, 2, 2, 5 },
}));

class conv_fp32_scale_activation_quantize_u8_eltwise_fp32 : public ConvFusingTest {};
TEST_P(conv_fp32_scale_activation_quantize_u8_eltwise_fp32, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("in_lo", get_mem(get_per_channel_layout(p), 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), 0)),
        data("out_hi", get_mem(get_single_element_layout(p), 255)),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f/255.f/255)),
        data("eltwise_data", get_mem(get_output_layout(p))),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        eltwise("scale", { input_info("conv_prim"), input_info("scale_data") }, eltwise_mode::prod),
        activation("activation_scale", input_info("scale"), activation_func::exp),
        quantize("quantize", input_info("activation_scale"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 256, data_types::u8),
        eltwise("sum", { input_info("quantize"), input_info("eltwise_data") }, eltwise_mode::sum,  p.default_type),
        reorder("reorder_bfyx", input_info("sum"), p.default_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_scale_activation_quantize_u8_eltwise_fp32, ::testing::ValuesIn(std::vector<convolution_test_params>{
    // For now only b_fs_yx_fsv16 supports this case
    convolution_test_params{ CASE_CONV_FP32_2, 2, 2, 6 },
    convolution_test_params{ CASE_CONV_FP32_3, 2, 2, 6 },

    convolution_test_params{ CASE_CONV_FP16_2, 2, 2, 6 },
    convolution_test_params{ CASE_CONV_FP16_3, 2, 2, 6 },
}));

class conv_fp32_scale_activation_quantize_i8_activation : public ConvFusingTest {};
TEST_P(conv_fp32_scale_activation_quantize_i8_activation, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -127)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f/255.f/255)),
        data("slope_data", get_mem(get_per_channel_layout(p))),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        eltwise("scale", { input_info("conv_prim"), input_info("scale_data") }, eltwise_mode::prod),
        activation("activation_scale", input_info("scale"), "slope_data", activation_func::relu_negative_slope),
        quantize("quantize", input_info("activation_scale"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 255, data_types::i8),
        activation("activation_quantize", input_info("quantize"), activation_func::relu),
        reorder("reorder_bfyx", input_info("activation_quantize"), p.default_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_scale_activation_quantize_i8_activation, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_FP32_2, 2, 2, 6 },
    convolution_test_params{ CASE_CONV_FP32_3, 2, 2, 6 },

    convolution_test_params{ CASE_CONV_FP16_2, 2, 2, 6 },
    convolution_test_params{ CASE_CONV_FP16_3, 2, 2, 6 },
}));


class conv_fp32_scale_activation_quantize_i8_eltwise_fp32_quantize_i8 : public ConvFusingTest {};
TEST_P(conv_fp32_scale_activation_quantize_i8_eltwise_fp32_quantize_i8, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_lo1", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("in_hi1", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -127)),
        data("out_lo1", get_mem(get_single_element_layout(p), -127)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        data("out_hi1", get_mem(get_single_element_layout(p), 127)),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f/255.f/255)),
        data("eltwise_data", get_mem(layout{ p.out_shape, data_types::i8, p.input_format })),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        eltwise("scale", { input_info("conv_prim"), input_info("scale_data") }, eltwise_mode::prod),
        activation("activation_scale", input_info("scale"), activation_func::exp),
        quantize("quantize", input_info("activation_scale"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 255, data_types::i8),
        eltwise("sum", { input_info("quantize"), input_info("eltwise_data") }, eltwise_mode::sum, data_types::f32),
        quantize("quantize_1", input_info("sum"), input_info("in_lo1"), input_info("in_hi1"),
                 input_info("out_lo1"), input_info("out_hi1"), 255, data_types::i8),
        reorder("reorder_bfyx", input_info("quantize_1"), p.default_format, data_types::f32)
    );

    tolerance = 2.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_scale_activation_quantize_i8_eltwise_fp32_quantize_i8, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_FP32_2, 2, 2, 7 },
    convolution_test_params{ CASE_CONV_FP32_3, 2, 2, 7 },
}));

class conv_fp32_activation_eltwise_in_u8_fp32 : public ConvFusingTest {};
TEST_P(conv_fp32_activation_eltwise_in_u8_fp32, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("eltwise_data", get_mem(layout{ p.out_shape, data_types::i8, p.input_format })),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        activation("activation", input_info("conv_prim"), activation_func::relu_negative_slope),
        eltwise("sum", { input_info("activation"), input_info("eltwise_data") }, eltwise_mode::sum, data_types::f32),
        reorder("reorder_bfyx", input_info("sum"), p.default_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_activation_eltwise_in_u8_fp32, ::testing::ValuesIn(std::vector<convolution_test_params>{
    // convolution_test_params{ CASE_CONV_FP32_1, 2, 2, 4 }, - eltwise fusing not supported
    convolution_test_params{ CASE_CONV_FP32_2, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_FP32_3, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_FP32_4, 2, 2, 4 },
    // convolution_test_params{ CASE_CONV_FP32_5, 2, 2, 4 }, - eltwise fusing not supported
    convolution_test_params{ CASE_CONV_FP32_6, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_FP32_7, 2, 2, 4 },
    // convolution_test_params{ CASE_CONV_FP32_8, 2, 2, 4 }, - unknown bug
    convolution_test_params{ CASE_CONV_FP32_9, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_FP32_10, 2, 2, 4 },
}));

class conv_fp32_activation_eltwise_diff_sizes : public ConvEltwTest {};
TEST_P(conv_fp32_activation_eltwise_diff_sizes, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("eltwise_data", get_mem(layout{ p.eltw_shape, p.data_type, p.input_format })),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        activation("activation", input_info("conv_prim"), activation_func::relu_negative_slope),
        eltwise("sum", { input_info("activation"), input_info("eltwise_data") }, eltwise_mode::sum, data_types::f32),
        reorder("reorder_bfyx", input_info("sum"), p.default_format, data_types::f32)
    );

    tolerance = default_tolerance(p.default_type);
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_activation_eltwise_diff_sizes, ::testing::ValuesIn(std::vector<conv_eltw_test_params>{
    conv_eltw_test_params{ CASE_CONV_ELTW_FP32_1, 2, 2, 4 },
    conv_eltw_test_params{ CASE_CONV_ELTW_FP32_2, 2, 2, 4 },
    conv_eltw_test_params{ CASE_CONV_ELTW_FP32_3, 2, 2, 4 },
    conv_eltw_test_params{ CASE_CONV_ELTW_FP32_4, 2, 2, 4 },
    conv_eltw_test_params{ CASE_CONV_ELTW_FP32_5, 2, 2, 4 },
    conv_eltw_test_params{ CASE_CONV_ELTW_FP32_6, 2, 2, 4 },
    conv_eltw_test_params{ CASE_CONV_ELTW_FP32_7, 3, 3, 4 },
    conv_eltw_test_params{ CASE_CONV_ELTW_FP32_8, 3, 3, 4 },
}));

class conv_fp32_group_conv_eltwise_sum : public ConvEltwTest {};
TEST_P(conv_fp32_group_conv_eltwise_sum, basic) {
    auto p = GetParam();

    ov::intel_gpu::ImplementationDesc conv_impl = { format::bfyx, "convolution_gpu_bfyx_os_iyx_osv16", impl_types::ocl };
    cfg_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "conv_prim", conv_impl } }));

    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("eltwise_data", get_mem(layout{ p.eltw_shape, p.data_type, p.input_format })),
        convolution("conv_prim", input_info("input"), "weights", "", p.groups, p.stride, p.dilation, p.pad, p.pad, true),
        eltwise("sum", { input_info("conv_prim"), input_info("eltwise_data") }, eltwise_mode::sum, data_types::f32),
        reorder("reorder_bfyx", input_info("sum"), p.default_format, data_types::f32)
    );

    tolerance = default_tolerance(p.default_type);
    execute(p);
}

// in_shape; out_shape; eltw_shape; kernel; stride; pad; dilation; groups; data_type; input_format; weights_type; weights_format; default_type; default_format;
#define CASE_GROUP_CONV_ELTW_FP32_1 { 1, 48, 3, 3 }, { 1, 48, 3, 3 }, { 1, 48, 3, 3 }, { 16, 3, 3, 3, 3 }, { 1, 1 }, { 1, 1 }, { 1, 1 }, 16, data_types::f32, format::bfyx, data_types::f32, format::g_os_iyx_osv16, data_types::f32, format::bfyx

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_group_conv_eltwise_sum, ::testing::ValuesIn(std::vector<conv_eltw_test_params>{
    conv_eltw_test_params{ CASE_GROUP_CONV_ELTW_FP32_1, 3, 3, 3 },
}));

class conv_swap_xy_with_eltwise_diff_sizes : public ConvEltwTest {};
TEST_P(conv_swap_xy_with_eltwise_diff_sizes, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("eltwise_data", get_mem(layout{ p.eltw_shape, p.data_type, p.input_format })),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        activation("activation", input_info("conv_prim"), activation_func::relu_negative_slope),
        eltwise("sum", { input_info("activation"), input_info("eltwise_data") }, eltwise_mode::sum, data_types::f16),
        reorder("reorder_bfyx", input_info("sum"), p.default_format, data_types::f16)
    );

    if (engine.get_device_info().supports_immad &&
        p.default_type == data_types::f16) {
        GTEST_SKIP(); // Issue: 94154
    }

    tolerance = default_tolerance(p.default_type);
    execute(p);
}

// in_shape; out_shape; eltw_shape; kernel; stride; pad; dilation; groups; data_type; input_format; weights_type; weights_format; default_type; default_format;
#define CASE_CONV_ELTW_FP16_SWAP_XY_1 { 1, 16, 5, 1}, { 1, 32, 7, 1 }, { 1, 32, 1, 1}, { 32, 16, 3, 1 }, { 1, 1 }, { 2, 0 }, { 1, 1 }, 1, data_types::f16, format::bfyx, data_types::f16, format::os_iyx_osv16, data_types::f16, format::bfyx
#define CASE_CONV_ELTW_FP16_SWAP_XY_2 { 1, 16, 5, 1}, { 1, 32, 7, 1 }, { 1, 32, 7, 1}, { 32, 16, 3, 1 }, { 1, 1 }, { 2, 0 }, { 1, 1 }, 1, data_types::f16, format::bfyx, data_types::f16, format::os_iyx_osv16, data_types::f16, format::bfyx
#define CASE_CONV_ELTW_FP16_SWAP_XY_3 { 3, 16, 5, 1}, { 3, 32, 7, 1 }, { 1, 32, 1, 1}, { 32, 16, 3, 1 }, { 1, 1 }, { 2, 0 }, { 1, 1 }, 1, data_types::f16, format::bfyx, data_types::f16, format::os_iyx_osv16, data_types::f16, format::bfyx
#define CASE_CONV_ELTW_FP16_SWAP_XY_4 { 3, 16, 5, 1}, { 3, 32, 7, 1 }, { 3, 32, 7, 1}, { 32, 16, 3, 1 }, { 1, 1 }, { 2, 0 }, { 1, 1 }, 1, data_types::f16, format::bfyx, data_types::f16, format::os_iyx_osv16, data_types::f16, format::bfyx

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_swap_xy_with_eltwise_diff_sizes, ::testing::ValuesIn(std::vector<conv_eltw_test_params>{
    conv_eltw_test_params{ CASE_CONV_ELTW_FP16_SWAP_XY_1, 3, 2, 4 },
    conv_eltw_test_params{ CASE_CONV_ELTW_FP16_SWAP_XY_2, 3, 2, 4 },
    conv_eltw_test_params{ CASE_CONV_ELTW_FP16_SWAP_XY_3, 3, 2, 4 },
    conv_eltw_test_params{ CASE_CONV_ELTW_FP16_SWAP_XY_4, 3, 2, 4 },
}));

class conv_scale_activation_eltwise_fp32_quantize_i8 : public ConvEltwTest {};
TEST_P(conv_scale_activation_eltwise_fp32_quantize_i8, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        convolution("conv", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        data("scale_data", get_mem(get_per_channel_layout(p))),
        eltwise("scale", { input_info("conv"), input_info("scale_data") }, eltwise_mode::prod),
        activation("activation", input_info("scale"), activation_func::hyperbolic_tan),
        data("eltwise_data", get_mem(layout{ p.eltw_shape, p.data_type, p.input_format })),
        eltwise("eltw", { input_info("activation"), input_info("eltwise_data") }, eltwise_mode::sum, data_types::f32),
        data("in_low", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_high", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_low", get_mem(get_single_element_layout(p), -127, 127)),
        data("out_high", get_mem(get_single_element_layout(p), -127, 127)),
        quantize("quant", input_info("eltw"), input_info("in_low"), input_info("in_high"),
                 input_info("out_low"), input_info("out_high"), 255, data_types::i8),
        reorder("reorder_bfyx", input_info("quant"), p.default_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_scale_activation_eltwise_fp32_quantize_i8, ::testing::ValuesIn(std::vector<conv_eltw_test_params>{
    conv_eltw_test_params{ CASE_CONV_ELTW_FP32_1, 2, 4, 6 },
    conv_eltw_test_params{ CASE_CONV_ELTW_FP32_2, 2, 4, 6 },
    conv_eltw_test_params{ CASE_CONV_ELTW_FP32_3, 2, 4, 6 },
    conv_eltw_test_params{ CASE_CONV_ELTW_FP32_4, 2, 4, 6 },
    conv_eltw_test_params{ CASE_CONV_ELTW_FP32_5, 3, 4, 6 },
    conv_eltw_test_params{ CASE_CONV_ELTW_FP32_6, 3, 4, 6 },
    conv_eltw_test_params{ CASE_CONV_ELTW_FP32_7, 3, 4, 6 },
    conv_eltw_test_params{ CASE_CONV_ELTW_FP32_8, 3, 4, 6 },
}));

/* ----------------------------------------------------------------------------------------------------- */
/* ---------------------------------------- INT8 convolution cases ------------------------------------- */
/* ----------------------------------------------------------------------------------------------------- */

class conv_int8_scale : public ConvFusingTest {};
TEST_P(conv_int8_scale, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f/255.f)),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        eltwise("scale", { input_info("conv_prim"), input_info("scale_data") }, eltwise_mode::prod),
        reorder("reorder_bfyx", input_info("scale"), p.default_format, data_types::f32)
    );

    tolerance = default_tolerance(p.default_type);
    execute(p);
}

TEST_P(conv_int8_scale, fp16_scale_out) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f/255.f)),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        eltwise("scale", { input_info("conv_prim"), input_info("scale_data") }, eltwise_mode::prod, data_types::f16),
        reorder("reorder_bfyx", input_info("scale"), p.default_format, data_types::f32)
    );

    tolerance = default_tolerance(p.default_type);
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_int8_scale, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_U8S8_1, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_2, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_3, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_4, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_1, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_2, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_3, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_4, 2, 2, 3 },

    convolution_test_params{ CASE_CONV3D_U8S8_1, 2, 2, 3 },
    convolution_test_params{ CASE_CONV3D_U8S8_2, 2, 2, 3 },
    convolution_test_params{ CASE_CONV3D_U8S8_3, 2, 2, 3 },
    convolution_test_params{ CASE_CONV3D_U8S8_4, 2, 2, 3 },
    convolution_test_params{ CASE_CONV3D_U8S8_5, 2, 2, 3 },
    convolution_test_params{ CASE_CONV3D_S8S8_1, 2, 2, 3 },
    convolution_test_params{ CASE_CONV3D_S8S8_2, 2, 2, 3 },
    convolution_test_params{ CASE_CONV3D_S8S8_3, 2, 2, 3 },
    convolution_test_params{ CASE_CONV3D_S8S8_4, 2, 2, 3 },
    convolution_test_params{ CASE_CONV3D_S8S8_5, 2, 2, 3 },
}));

class conv_int8_eltwise : public ConvFusingTest {};
TEST_P(conv_int8_eltwise, fp16_eltwise_out) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f/255.f)),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        eltwise("scale", { input_info("conv_prim"), input_info("scale_data") }, eltwise_mode::prod, data_types::f16),
        reorder("reorder_bfyx", input_info("scale"), p.default_format, data_types::f32)
    );

    tolerance = default_tolerance(p.default_type);
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_int8_eltwise, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_U8S8_1, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_2, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_3, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_4, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_1, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_2, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_3, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_4, 2, 2, 3 },

    convolution_test_params{ CASE_CONV3D_U8S8_1, 2, 2, 3 },
    convolution_test_params{ CASE_CONV3D_U8S8_2, 2, 2, 3 },
    convolution_test_params{ CASE_CONV3D_U8S8_3, 2, 2, 3 },
    convolution_test_params{ CASE_CONV3D_U8S8_4, 2, 2, 3 },
    convolution_test_params{ CASE_CONV3D_U8S8_5, 2, 2, 3 },
    convolution_test_params{ CASE_CONV3D_S8S8_1, 2, 2, 3 },
    convolution_test_params{ CASE_CONV3D_S8S8_2, 2, 2, 3 },
    convolution_test_params{ CASE_CONV3D_S8S8_3, 2, 2, 3 },
    convolution_test_params{ CASE_CONV3D_S8S8_4, 2, 2, 3 },
    convolution_test_params{ CASE_CONV3D_S8S8_5, 2, 2, 3 },
}));

class conv_int8_prelu_eltwise : public ConvFusingTest {};
TEST_P(conv_int8_prelu_eltwise, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("slope_data", get_mem(get_per_channel_layout(p))),
        data("eltwise_data", get_mem(get_output_layout(p))),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        activation("activation", input_info("conv_prim"), "slope_data", activation_func::relu_negative_slope),
        eltwise("eltwise", input_info("activation"), input_info("eltwise_data"), eltwise_mode::sum),
        reorder("reorder_bfyx", input_info("eltwise"), p.default_format, data_types::f32)
    );

    tolerance = default_tolerance(p.data_type);
    execute(p);
}

TEST_P(conv_int8_prelu_eltwise, basic_slope_2) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("slope_data", get_mem(get_prelu_slope_layout(p))),
        data("eltwise_data", get_mem(get_output_layout(p))),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        activation("activation", input_info("conv_prim"), "slope_data", activation_func::relu_negative_slope),
        eltwise("eltwise", input_info("activation"), input_info("eltwise_data"), eltwise_mode::sum),
        reorder("reorder_bfyx", input_info("eltwise"), p.default_format, data_types::f32)
    );

    tolerance = default_tolerance(p.data_type);
    execute(p);
}

TEST_P(conv_int8_prelu_eltwise, fsv16) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("slope_data", get_mem(get_per_channel_layout(p))),
        data("eltwise_data", get_mem(get_output_layout(p))),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        activation("activation", input_info("conv_prim"), "slope_data", activation_func::relu_negative_slope),
        eltwise("eltwise", input_info("activation"), input_info("eltwise_data"), eltwise_mode::sum),
        reorder("reorder_bfyx", input_info("eltwise"), p.default_format, data_types::f32)
    );

    if (p.default_format.dimension() == 4) {
        ov::intel_gpu::ImplementationDesc conv_impl = { format::b_fs_yx_fsv16, "" };
        cfg_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "conv_prim", conv_impl } }));
    } else {
        // TODO Add 5D int8 optimized convolution implementations
        return;
    }

    tolerance = default_tolerance(p.data_type);
    execute(p);
}

TEST_P(conv_int8_prelu_eltwise, fsv16_slope_2) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("slope_data", get_mem(get_prelu_slope_layout(p))),
        data("eltwise_data", get_mem(get_output_layout(p))),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        activation("activation", input_info("conv_prim"), "slope_data", activation_func::relu_negative_slope),
        eltwise("eltwise", input_info("activation"), input_info("eltwise_data"), eltwise_mode::sum),
        reorder("reorder_bfyx", input_info("eltwise"), p.default_format, data_types::f32)
    );

    if (p.default_format.dimension() == 4) {
        ov::intel_gpu::ImplementationDesc conv_impl = { format::b_fs_yx_fsv16, "" };
        cfg_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "conv_prim", conv_impl } }));
    } else {
        // TODO Add 5D int8 optimized convolution implementations
        return;
    }

    tolerance = default_tolerance(p.data_type);
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_int8_prelu_eltwise, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_U8S8_1, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_U8S8_2, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_U8S8_3, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_U8S8_4, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_U8S8_7, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_U8S8_8, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_S8S8_1, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_S8S8_2, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_S8S8_3, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_S8S8_4, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_S8S8_7, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_S8S8_8, 2, 2, 4 },

    convolution_test_params{ CASE_CONV3D_U8S8_1, 2, 2, 4 },
    convolution_test_params{ CASE_CONV3D_U8S8_2, 2, 2, 4 },
    convolution_test_params{ CASE_CONV3D_U8S8_3, 2, 2, 4 },
    convolution_test_params{ CASE_CONV3D_U8S8_4, 2, 2, 4 },
    convolution_test_params{ CASE_CONV3D_U8S8_5, 2, 2, 4 },
    convolution_test_params{ CASE_CONV3D_S8S8_1, 2, 2, 4 },
    convolution_test_params{ CASE_CONV3D_S8S8_2, 2, 2, 4 },
    convolution_test_params{ CASE_CONV3D_S8S8_3, 2, 2, 4 },
    convolution_test_params{ CASE_CONV3D_S8S8_4, 2, 2, 4 },
    convolution_test_params{ CASE_CONV3D_S8S8_5, 2, 2, 4 },
}));

class conv_int8_activation_eltwise_quantize : public ConvFusingTest {};
TEST_P(conv_int8_activation_eltwise_quantize, fsv16) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("eltwise_data", get_mem(get_output_layout(p))),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -127)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        activation("activation", input_info("conv_prim"), activation_func::negative),
        eltwise("eltwise", input_info("activation"), input_info("eltwise_data"), eltwise_mode::sum),
        quantize("quantize", input_info("eltwise"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 255, data_types::i8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );

    if (p.default_format.dimension() == 4) {
        ov::intel_gpu::ImplementationDesc conv_impl = { format::b_fs_yx_fsv16, "" };
        cfg_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "conv_prim", conv_impl } }));
    } else {
        // TODO Add 5D int8 optimized convolution implementations
        return;
    }

    tolerance = 1.f;
    execute(p);
}

TEST_P(conv_int8_activation_eltwise_quantize, fsv32) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("eltwise_data", get_mem(get_output_layout(p))),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -127)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        activation("activation", input_info("conv_prim"), activation_func::negative),
        eltwise("eltwise", input_info("activation"), input_info("eltwise_data"), eltwise_mode::sum),
        quantize("quantize", input_info("eltwise"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 255, data_types::i8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );

    if (p.default_format.dimension() == 4) {
        ov::intel_gpu::ImplementationDesc conv_impl = { format::b_fs_yx_fsv32, "" };
        cfg_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "conv_prim", conv_impl } }));
    } else {
        // TODO Add 5D int8 optimized convolution implementations
        return;
    }

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_int8_activation_eltwise_quantize, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_U8S8_1, 2, 2, 5 },
    convolution_test_params{ CASE_CONV_U8S8_2, 2, 2, 5 },
    convolution_test_params{ CASE_CONV_U8S8_3, 2, 2, 5 },
    convolution_test_params{ CASE_CONV_U8S8_4, 2, 2, 5 },
    convolution_test_params{ CASE_CONV_U8S8_7, 2, 2, 5 },
    convolution_test_params{ CASE_CONV_U8S8_8, 2, 2, 5 },
    convolution_test_params{ CASE_CONV_S8S8_1, 2, 2, 5 },
    convolution_test_params{ CASE_CONV_S8S8_2, 2, 2, 5 },
    convolution_test_params{ CASE_CONV_S8S8_3, 2, 2, 5 },
    convolution_test_params{ CASE_CONV_S8S8_4, 2, 2, 5 },
    convolution_test_params{ CASE_CONV_S8S8_7, 2, 2, 5 },
    convolution_test_params{ CASE_CONV_S8S8_8, 2, 2, 5 },
}));

class conv_int8_activation : public ConvFusingTest {};
TEST_P(conv_int8_activation, fsv16) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        activation("activation", input_info("conv_prim"), activation_func::negative),
        reorder("reorder_bfyx", input_info("activation"), p.default_format, data_types::f32)
    );

    if (p.default_format.dimension() == 4) {
        ov::intel_gpu::ImplementationDesc conv_impl = { format::b_fs_yx_fsv16, "convolution_gpu_b_fs_zyx_fsv16_imad" };
        cfg_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "conv_prim", conv_impl } }));
    } else {
        // TODO Add 5D int8 optimized convolution implementations
        return;
    }

    tolerance = default_tolerance(p.default_type);
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_int8_activation, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_U8S8_1, 2, 2, 3 },
}));

class conv_int8_activation_eltwise : public ConvFusingTest {};
TEST_P(conv_int8_activation_eltwise, fsv16) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("eltwise_data", get_mem(get_output_layout(p))),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        activation("activation", input_info("conv_prim"), activation_func::negative),
        eltwise("eltwise", input_info("activation"), input_info("eltwise_data"), eltwise_mode::sum),
        reorder("reorder_bfyx", input_info("eltwise"), p.default_format, data_types::f32)
    );

    if (p.default_format.dimension() == 4) {
        ov::intel_gpu::ImplementationDesc conv_impl = { format::b_fs_yx_fsv16, "" };
        cfg_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "conv_prim", conv_impl } }));
    } else {
        // TODO Add 5D int8 optimized convolution implementations
        return;
    }

    tolerance = default_tolerance(p.default_type);
    execute(p);
}

TEST_P(conv_int8_activation_eltwise, fsv32) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("eltwise_data", get_mem(get_output_layout(p))),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        activation("activation", input_info("conv_prim"), activation_func::negative),
        eltwise("eltwise", input_info("activation"), input_info("eltwise_data"), eltwise_mode::sum),
        reorder("reorder_bfyx", input_info("eltwise"), p.default_format, data_types::f32)
    );

    if (p.default_format.dimension() == 4) {
        ov::intel_gpu::ImplementationDesc conv_impl = { format::b_fs_yx_fsv32, "" };
        cfg_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "conv_prim", conv_impl } }));
    } else {
        // TODO Add 5D int8 optimized convolution implementations
        return;
    }

    tolerance = default_tolerance(p.default_type);
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_int8_activation_eltwise, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_U8S8_1, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_U8S8_2, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_U8S8_3, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_U8S8_4, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_U8S8_7, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_U8S8_8, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_S8S8_1, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_S8S8_2, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_S8S8_3, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_S8S8_4, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_S8S8_7, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_S8S8_8, 2, 2, 4 },
}));

class conv_int8_quantize_u8 : public ConvFusingTest {};
TEST_P(conv_int8_quantize_u8, per_channel) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), 0)),
        data("out_hi", get_mem(get_single_element_layout(p), 255)),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        quantize("quantize", input_info("conv_prim"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 256, data_types::u8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

TEST_P(conv_int8_quantize_u8, per_tensor) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("in_lo", get_mem(get_single_element_layout(p), -10)),
        data("in_hi", get_mem(get_single_element_layout(p), 10)),
        data("out_lo", get_mem(get_single_element_layout(p), 0)),
        data("out_hi", get_mem(get_single_element_layout(p), 255)),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        quantize("quantize", input_info("conv_prim"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 256, data_types::u8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_int8_quantize_u8, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_U8S8_1, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_2, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_3, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_4, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_1, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_2, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_3, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_4, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_8, 2, 2, 3 },

    convolution_test_params{ CASE_CONV3D_U8S8_1, 2, 2, 3 },
    convolution_test_params{ CASE_CONV3D_U8S8_2, 2, 2, 3 },
    convolution_test_params{ CASE_CONV3D_U8S8_3, 2, 2, 3 },
    convolution_test_params{ CASE_CONV3D_U8S8_4, 2, 2, 3 },
    convolution_test_params{ CASE_CONV3D_U8S8_5, 2, 2, 3 },
    convolution_test_params{ CASE_CONV3D_S8S8_1, 2, 2, 3 },
    convolution_test_params{ CASE_CONV3D_S8S8_2, 2, 2, 3 },
    convolution_test_params{ CASE_CONV3D_S8S8_3, 2, 2, 3 },
    convolution_test_params{ CASE_CONV3D_S8S8_4, 2, 2, 3 },
    convolution_test_params{ CASE_CONV3D_S8S8_5, 2, 2, 3 },
}));

class conv_int8_scale_quantize_i8 : public ConvFusingTest {};
TEST_P(conv_int8_scale_quantize_i8, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -127)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f/255.f/255)),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        eltwise("scale", { input_info("conv_prim"), input_info("scale_data") }, eltwise_mode::prod),
        quantize("quantize", input_info("scale"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 255, data_types::i8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );
    // Output elements are in range [-127, 127]
    // 1.0f difference is allowed, since quantize can return different values in ref and scale_shift kernels
    // due to big error of division (in ref kernel).
    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_int8_scale_quantize_i8, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_U8S8_1, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_U8S8_2, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_U8S8_3, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_U8S8_4, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_U8S8_9, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_U8S8_10, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_S8S8_1, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_S8S8_2, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_S8S8_3, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_S8S8_4, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_S8S8_9, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_S8S8_10, 2, 2, 4 },

    convolution_test_params{ CASE_CONV3D_U8S8_1, 2, 2, 4 },
    convolution_test_params{ CASE_CONV3D_U8S8_2, 2, 2, 4 },
    convolution_test_params{ CASE_CONV3D_U8S8_3, 2, 2, 4 },
    convolution_test_params{ CASE_CONV3D_U8S8_4, 2, 2, 4 },
    convolution_test_params{ CASE_CONV3D_U8S8_5, 2, 2, 4 },
    convolution_test_params{ CASE_CONV3D_S8S8_1, 2, 2, 4 },
    convolution_test_params{ CASE_CONV3D_S8S8_2, 2, 2, 4 },
    convolution_test_params{ CASE_CONV3D_S8S8_3, 2, 2, 4 },
    convolution_test_params{ CASE_CONV3D_S8S8_4, 2, 2, 4 },
    convolution_test_params{ CASE_CONV3D_S8S8_5, 2, 2, 4 },
}));

class conv_int8_scale_quantize_i8_conv_b_fs_yx_fsv4_int8 : public ConvFusingTest {};
TEST_P(conv_int8_scale_quantize_i8_conv_b_fs_yx_fsv4_int8, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -127)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f / 255.0f)),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        eltwise("scale", { input_info("conv_prim"), input_info("scale_data") }, eltwise_mode::prod),
        quantize("quantize", input_info("scale"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 255, data_types::i8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );
    ov::intel_gpu::ImplementationDesc conv_impl = { format::b_fs_yx_fsv4, "convolution_gpu_b_fs_yx_fsv4_int8" };
    cfg_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "conv_prim", conv_impl } }));

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_int8_scale_quantize_i8_conv_b_fs_yx_fsv4_int8, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_S8S8_11, 2, 2, 4 },
}));

class conv_int8_relu_quantize : public ConvFusingTest {};
TEST_P(conv_int8_relu_quantize, i8) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -127)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        activation("relu", input_info("conv_prim"), activation_func::relu),
        quantize("quantize", input_info("relu"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 256, data_types::i8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );
    // Output elements are in range [-127, 127]
    // 1.0f difference is allowed, since quantize can return different values in ref and scale_shift kernels
    // due to big error of division (in ref kernel).
    tolerance = 1.f;
    execute(p);
}

TEST_P(conv_int8_relu_quantize, u8) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("in_lo", get_mem(get_per_channel_layout(p), 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), 0)),
        data("out_hi", get_mem(get_single_element_layout(p), 255)),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        activation("relu", input_info("conv_prim"), activation_func::relu),
        quantize("quantize", input_info("relu"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 256, data_types::u8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_int8_relu_quantize, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_U8S8_1, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_U8S8_2, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_U8S8_3, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_U8S8_4, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_S8S8_1, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_S8S8_2, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_S8S8_3, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_S8S8_4, 2, 2, 4 },

    convolution_test_params{ CASE_CONV3D_U8S8_1, 2, 2, 4 },
    convolution_test_params{ CASE_CONV3D_U8S8_2, 2, 2, 4 },
    convolution_test_params{ CASE_CONV3D_U8S8_3, 2, 2, 4 },
    convolution_test_params{ CASE_CONV3D_U8S8_4, 2, 2, 4 },
    convolution_test_params{ CASE_CONV3D_U8S8_5, 2, 2, 4 },
    convolution_test_params{ CASE_CONV3D_S8S8_1, 2, 2, 4 },
    convolution_test_params{ CASE_CONV3D_S8S8_2, 2, 2, 4 },
    convolution_test_params{ CASE_CONV3D_S8S8_3, 2, 2, 4 },
    convolution_test_params{ CASE_CONV3D_S8S8_4, 2, 2, 4 },
    convolution_test_params{ CASE_CONV3D_S8S8_5, 2, 2, 4 },
}));

class conv_int8_scale_activation_quantize_i8 : public ConvFusingTest {};
TEST_P(conv_int8_scale_activation_quantize_i8, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -127)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f/255.f/255)),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        eltwise("scale", { input_info("conv_prim"), input_info("scale_data") }, eltwise_mode::prod),
        activation("activation_scale", input_info("scale"), activation_func::exp),
        quantize("quantize", input_info("activation_scale"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 255, data_types::i8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );

    tolerance = 2.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_int8_scale_activation_quantize_i8, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_U8S8_1, 2, 2, 5 },
    convolution_test_params{ CASE_CONV_U8S8_2, 2, 2, 5 },
    convolution_test_params{ CASE_CONV_U8S8_3, 2, 2, 5 },
    convolution_test_params{ CASE_CONV_U8S8_4, 2, 2, 5 },
    convolution_test_params{ CASE_CONV_S8S8_1, 2, 2, 5 },
    convolution_test_params{ CASE_CONV_S8S8_2, 2, 2, 5 },
    convolution_test_params{ CASE_CONV_S8S8_3, 2, 2, 5 },
    convolution_test_params{ CASE_CONV_S8S8_4, 2, 2, 5 },

    convolution_test_params{ CASE_CONV3D_U8S8_1, 2, 2, 5 },
    convolution_test_params{ CASE_CONV3D_U8S8_2, 2, 2, 5 },
    convolution_test_params{ CASE_CONV3D_U8S8_3, 2, 2, 5 },
    convolution_test_params{ CASE_CONV3D_U8S8_4, 2, 2, 5 },
    convolution_test_params{ CASE_CONV3D_U8S8_5, 2, 2, 5 },
    convolution_test_params{ CASE_CONV3D_S8S8_1, 2, 2, 5 },
    convolution_test_params{ CASE_CONV3D_S8S8_2, 2, 2, 5 },
    convolution_test_params{ CASE_CONV3D_S8S8_3, 2, 2, 5 },
    convolution_test_params{ CASE_CONV3D_S8S8_4, 2, 2, 5 },
    convolution_test_params{ CASE_CONV3D_S8S8_5, 2, 2, 5 },
}));

class conv_int8_scale_activation_quantize_i8_eltwise_fp32 : public ConvFusingTest {};
TEST_P(conv_int8_scale_activation_quantize_i8_eltwise_fp32, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -127)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f/255.f/255)),
        data("eltwise_data", get_mem(get_output_layout(p))),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        eltwise("scale", { input_info("conv_prim"), input_info("scale_data") }, eltwise_mode::prod),
        activation("activation_scale", input_info("scale"), activation_func::exp),
        quantize("quantize", input_info("activation_scale"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 255, data_types::i8),
        eltwise("sum", { input_info("quantize"), input_info("eltwise_data") }, eltwise_mode::sum,  data_types::f32),
        reorder("reorder_bfyx", input_info("sum"), p.default_format, data_types::f32)
    );

    tolerance = 2.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_int8_scale_activation_quantize_i8_eltwise_fp32, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_U8S8_1, 2, 2, 6 },
    convolution_test_params{ CASE_CONV_U8S8_2, 2, 2, 6 },
    convolution_test_params{ CASE_CONV_U8S8_3, 2, 2, 6 },
    convolution_test_params{ CASE_CONV_U8S8_4, 2, 2, 6 },
    convolution_test_params{ CASE_CONV_S8S8_1, 2, 2, 6 },
    convolution_test_params{ CASE_CONV_S8S8_2, 2, 2, 6 },
    convolution_test_params{ CASE_CONV_S8S8_3, 2, 2, 6 },
    convolution_test_params{ CASE_CONV_S8S8_4, 2, 2, 6 },

    convolution_test_params{ CASE_CONV3D_U8S8_1, 2, 2, 6 },
    convolution_test_params{ CASE_CONV3D_U8S8_2, 2, 2, 6 },
    convolution_test_params{ CASE_CONV3D_U8S8_3, 2, 2, 6 },
    convolution_test_params{ CASE_CONV3D_U8S8_4, 2, 2, 6 },
    convolution_test_params{ CASE_CONV3D_U8S8_5, 2, 2, 6 },
    convolution_test_params{ CASE_CONV3D_S8S8_1, 2, 2, 6 },
    convolution_test_params{ CASE_CONV3D_S8S8_2, 2, 2, 6 },
    convolution_test_params{ CASE_CONV3D_S8S8_3, 2, 2, 6 },
    convolution_test_params{ CASE_CONV3D_S8S8_4, 2, 2, 6 },
    convolution_test_params{ CASE_CONV3D_S8S8_5, 2, 2, 6 },
}));

class conv_int8_scale_activation_quantize_i8_activation : public ConvFusingTest {};
TEST_P(conv_int8_scale_activation_quantize_i8_activation, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -127)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f/255.f/255)),
        data("slope_data", get_mem(get_per_channel_layout(p))),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        eltwise("scale", { input_info("conv_prim"), input_info("scale_data") }, eltwise_mode::prod),
        activation("activation_scale", input_info("scale"), "slope_data", activation_func::relu_negative_slope),
        quantize("quantize", input_info("activation_scale"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 255, data_types::i8),
        activation("activation_quantize", input_info("quantize"), activation_func::relu),
        reorder("reorder_bfyx", input_info("activation_quantize"), p.default_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

TEST_P(conv_int8_scale_activation_quantize_i8_activation, activation_clamp) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -127)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f/255.f/255)),
        data("slope_data", get_mem(get_per_channel_layout(p))),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        eltwise("scale", { input_info("conv_prim"), input_info("scale_data") }, eltwise_mode::prod),
        activation("activation_scale", input_info("scale"), "slope_data", activation_func::relu_negative_slope),
        quantize("quantize", input_info("activation_scale"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 255, data_types::i8),
        activation("activation_quantize", input_info("quantize"), activation_func::clamp, {-136.f, 136.f}),
        reorder("reorder_bfyx", input_info("activation_quantize"), p.default_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_int8_scale_activation_quantize_i8_activation, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_U8S8_1, 2, 2, 6 },
    convolution_test_params{ CASE_CONV_U8S8_2, 2, 2, 6 },
    convolution_test_params{ CASE_CONV_U8S8_3, 2, 2, 6 },
    convolution_test_params{ CASE_CONV_U8S8_4, 2, 2, 6 },
    convolution_test_params{ CASE_CONV_S8S8_1, 2, 2, 6 },
    convolution_test_params{ CASE_CONV_S8S8_2, 2, 2, 6 },
    convolution_test_params{ CASE_CONV_S8S8_3, 2, 2, 6 },
    convolution_test_params{ CASE_CONV_S8S8_4, 2, 2, 6 },

    convolution_test_params{ CASE_CONV3D_U8S8_1, 2, 2, 6 },
    convolution_test_params{ CASE_CONV3D_U8S8_2, 2, 2, 6 },
    convolution_test_params{ CASE_CONV3D_U8S8_3, 2, 2, 6 },
    convolution_test_params{ CASE_CONV3D_U8S8_4, 2, 2, 6 },
    convolution_test_params{ CASE_CONV3D_U8S8_5, 2, 2, 6 },
    convolution_test_params{ CASE_CONV3D_S8S8_1, 2, 2, 6 },
    convolution_test_params{ CASE_CONV3D_S8S8_2, 2, 2, 6 },
    convolution_test_params{ CASE_CONV3D_S8S8_3, 2, 2, 6 },
    convolution_test_params{ CASE_CONV3D_S8S8_4, 2, 2, 6 },
    convolution_test_params{ CASE_CONV3D_S8S8_5, 2, 2, 6 },
}));


class conv_int8_scale_activation_quantize_i8_eltwise_fp32_quantize_i8 : public ConvFusingTest {};
// With some input values accuracy error might be = 2, so the test is disabled.
TEST_P(conv_int8_scale_activation_quantize_i8_eltwise_fp32_quantize_i8, DISABLED_basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_lo1", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("in_hi1", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -127)),
        data("out_lo1", get_mem(get_single_element_layout(p), -127)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        data("out_hi1", get_mem(get_single_element_layout(p), 127)),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f/255.f/255)),
        data("eltwise_data", get_mem(layout{ p.out_shape, data_types::i8, p.input_format })),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        eltwise("scale", { input_info("conv_prim"), input_info("scale_data") }, eltwise_mode::prod),
        activation("activation_scale", input_info("scale"), activation_func::exp),
        quantize("quantize", input_info("activation_scale"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 255, data_types::i8),
        eltwise("sum", { input_info("quantize"), input_info("eltwise_data") }, eltwise_mode::sum, data_types::f32),
        quantize("quantize_1", input_info("sum"), input_info("in_lo1"), input_info("in_hi1"),
                 input_info("out_lo1"), input_info("out_hi1"), 255, data_types::i8),
        reorder("reorder_bfyx", input_info("quantize_1"), p.default_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_int8_scale_activation_quantize_i8_eltwise_fp32_quantize_i8, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_U8S8_1, 2, 2, 7 },
    convolution_test_params{ CASE_CONV_U8S8_2, 2, 2, 7 },
    convolution_test_params{ CASE_CONV_U8S8_3, 2, 2, 7 },
    convolution_test_params{ CASE_CONV_U8S8_4, 2, 2, 7 },
    convolution_test_params{ CASE_CONV_S8S8_1, 2, 2, 7 },
    convolution_test_params{ CASE_CONV_S8S8_2, 2, 2, 7 },
    convolution_test_params{ CASE_CONV_S8S8_3, 2, 2, 7 },
    convolution_test_params{ CASE_CONV_S8S8_4, 2, 2, 7 },

    convolution_test_params{ CASE_CONV3D_U8S8_1, 2, 2, 7 },
    convolution_test_params{ CASE_CONV3D_U8S8_2, 2, 2, 7 },
    convolution_test_params{ CASE_CONV3D_U8S8_3, 2, 2, 7 },
    convolution_test_params{ CASE_CONV3D_U8S8_4, 2, 2, 7 },
    convolution_test_params{ CASE_CONV3D_U8S8_5, 2, 2, 7 },
    convolution_test_params{ CASE_CONV3D_S8S8_1, 2, 2, 7 },
    convolution_test_params{ CASE_CONV3D_S8S8_2, 2, 2, 7 },
    convolution_test_params{ CASE_CONV3D_S8S8_3, 2, 2, 7 },
    convolution_test_params{ CASE_CONV3D_S8S8_4, 2, 2, 7 },
    convolution_test_params{ CASE_CONV3D_S8S8_5, 2, 2, 7 },
}));

class conv_int8_scale_prelu_quantize_i8_eltwise_fp32_quantize_i8_vec : public ConvFusingTest {};
TEST_P(conv_int8_scale_prelu_quantize_i8_eltwise_fp32_quantize_i8_vec, vector_ops) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_lo1", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("in_hi1", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -127)),
        data("out_lo1", get_mem(get_single_element_layout(p), -127)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        data("out_hi1", get_mem(get_single_element_layout(p), 127)),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f/255.f/255)),
        data("slope_data", get_mem(get_per_channel_layout(p))),
        data("eltwise_data", get_mem(layout{ p.out_shape, data_types::i8, format::b_fs_yx_fsv4 })),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        eltwise("scale", { input_info("conv_prim"), input_info("scale_data") }, eltwise_mode::prod),
        activation("activation_scale", input_info("scale"), "slope_data", activation_func::relu_negative_slope),
        quantize("quantize", input_info("activation_scale"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 255, data_types::i8),
        eltwise("sum", { input_info("quantize"), input_info("eltwise_data") }, eltwise_mode::sum, data_types::f32),
        quantize("quantize_1", input_info("sum"), input_info("in_lo1"), input_info("in_hi1"),
                 input_info("out_lo1"), input_info("out_hi1"), 255, data_types::i8),
        reorder("reorder_bfyx", input_info("quantize_1"), p.default_format, data_types::f32)
    );

    ov::intel_gpu::ImplementationDesc conv_impl = { format::b_fs_yx_fsv4, "convolution_gpu_b_fs_yx_fsv4_1x1", impl_types::ocl };
    cfg_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "conv_prim", conv_impl } }));

    tolerance = 1.f;
    execute(p);
}

TEST_P(conv_int8_scale_prelu_quantize_i8_eltwise_fp32_quantize_i8_vec, vector_ops_mixed_types) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_lo1", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("in_hi1", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -127)),
        data("out_lo1", get_mem(get_single_element_layout(p), -127)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        data("out_hi1", get_mem(get_single_element_layout(p), 127)),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f/255.f/255)),
        data("slope_data", get_mem(layout{ { 1, p.out_shape[1], 1, 1 }, data_types::f16, p.default_format })),
        data("eltwise_data", get_mem(layout{ p.out_shape , data_types::u8, format::b_fs_yx_fsv4 })),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        eltwise("scale", { input_info("conv_prim"), input_info("scale_data") }, eltwise_mode::prod),
        activation("activation_scale", input_info("scale"), "slope_data", activation_func::relu_negative_slope),
        quantize("quantize", input_info("activation_scale"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 255, data_types::i8),
        eltwise("sum", { input_info("quantize"), input_info("eltwise_data") }, eltwise_mode::sum, data_types::f32),
        quantize("quantize_1", input_info("sum"), input_info("in_lo1"), input_info("in_hi1"),
                 input_info("out_lo1"), input_info("out_hi1"), 255, data_types::i8),
        reorder("reorder_bfyx", input_info("quantize_1"), p.default_format, data_types::f32)
    );

    ov::intel_gpu::ImplementationDesc conv_impl = { format::b_fs_yx_fsv4, "convolution_gpu_b_fs_yx_fsv4_1x1", impl_types::ocl };
    cfg_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "conv_prim", conv_impl } }));

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_int8_scale_prelu_quantize_i8_eltwise_fp32_quantize_i8_vec, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_U8S8_3, 2, 2, 7 },
    convolution_test_params{ CASE_CONV_U8S8_5, 2, 2, 7 },
    convolution_test_params{ CASE_CONV_S8S8_3, 2, 2, 7 },
    convolution_test_params{ CASE_CONV_S8S8_5, 2, 2, 7 },
}));

class conv_i8_activation_eltwise_diff_sizes : public ConvEltwTest {};
TEST_P(conv_i8_activation_eltwise_diff_sizes, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("eltwise_data", get_mem(layout{ p.eltw_shape, p.data_type, p.input_format })),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        activation("activation", input_info("conv_prim"), activation_func::abs),
        eltwise("sum", { input_info("activation"), input_info("eltwise_data") }, eltwise_mode::sum, data_types::f32),
        reorder("reorder_bfyx", input_info("sum"), p.default_format, data_types::f32)
    );

    tolerance = default_tolerance(p.default_type);
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_i8_activation_eltwise_diff_sizes, ::testing::ValuesIn(std::vector<conv_eltw_test_params>{
    conv_eltw_test_params{ CASE_CONV_ELTW_i8_1, 3, 3, 4 },
    conv_eltw_test_params{ CASE_CONV_ELTW_i8_2, 2, 2, 4 },
    conv_eltw_test_params{ CASE_CONV_ELTW_i8_3, 2, 2, 4 },
    conv_eltw_test_params{ CASE_CONV_ELTW_i8_4, 2, 2, 4 },
    conv_eltw_test_params{ CASE_CONV_ELTW_i8_5, 3, 3, 4 },
}));

/* ----------------------------------------------------------------------------------------------------- */
/* ----------------------------------- Force convolution kernel cases ---------------------------------- */
/* ----------------------------------------------------------------------------------------------------- */

class conv_fp16_activation : public ConvFusingForceKernelTest {};
TEST_P(conv_fp16_activation, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        activation("activation", input_info("conv_prim"), activation_func::abs),
        reorder("reorder_bfyx", input_info("activation"), p.default_format, data_types::f32)
    );

    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp16_activation, ::testing::ValuesIn(std::vector<bc_force_kernel_params>{
    bc_force_kernel_params{ CASE_CONV_FP16_13, 2, 3, "convolution_gpu_fs_byx_fsv32" },
    bc_force_kernel_params{ CASE_CONV_FP16_14, 2, 3, "convolution_gpu_bfyx_f16_depthwise" },
    bc_force_kernel_params{ CASE_CONV_FP16_15, 2, 3, "convolution_gpu_bfyx_f16_depthwise" },
}));


class conv_fp16_scale : public ConvFusingForceKernelTest {};
TEST_P(conv_fp16_scale, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f/255.f)),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        eltwise("scale", { input_info("conv_prim"), input_info("scale_data") }, eltwise_mode::prod),
        reorder("reorder_bfyx", input_info("scale"), p.default_format, data_types::f32)
    );

    tolerance = default_tolerance(p.default_type);
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp16_scale, ::testing::ValuesIn(std::vector<bc_force_kernel_params>{
    bc_force_kernel_params{ CASE_CONV_FP16_13, 2, 3, "convolution_gpu_fs_byx_fsv32" },
    bc_force_kernel_params{ CASE_CONV_FP16_14, 2, 3, "convolution_gpu_bfyx_f16_depthwise" },
    bc_force_kernel_params{ CASE_CONV_FP16_15, 2, 3, "convolution_gpu_bfyx_f16_depthwise" },
}));

class conv_activation_onednn : public ConvActivationTestOnednn {};
TEST_P(conv_activation_onednn, basic) {
    if (!engine.get_device_info().supports_immad)
        return;
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        activation("activation", input_info("conv_prim"), p.activation_function_type),
        reorder("reorder_bfyx", input_info("activation"), p.default_format, data_types::f32)
    );

    tolerance = 1e-4f;
    execute(p);
}
INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_activation_onednn, ::testing::ValuesIn(std::vector<conv_activation_onednn_test_params>{
    conv_activation_onednn_test_params{ CASE_CONV_U8S8_1, activation_func::relu, 2, 2, 3},
    conv_activation_onednn_test_params{ CASE_CONV_U8S8_2, activation_func::relu_negative_slope, 2, 2, 3 },
    conv_activation_onednn_test_params{ CASE_CONV_U8S8_3, activation_func::hard_sigmoid, 2, 2, 3 },
    conv_activation_onednn_test_params{ CASE_CONV_S8S8_1, activation_func::hsigmoid, 2, 2, 3 },
    conv_activation_onednn_test_params{ CASE_CONV_S8S8_2, activation_func::negative, 2, 2, 3 },
    conv_activation_onednn_test_params{ CASE_CONV_S8S8_3, activation_func::sqrt, 2, 2, 3 },
}));

/* ----------------------------------------------------------------------------------------------------- */
/* ---------------------- reorder(bfyx to fs_b_yx_fsv32) + convolution kernel cases -------------------- */
/* ----------------------------------------------------------------------------------------------------- */
#define FSV32_CASE_CONV_FP32_1 { 1, 32, 4, 5 }, { 1, 32, 2, 3 }, { 32, 32, 3, 3 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, data_types::f32, format::bfyx, data_types::f32, format::oiyx, data_types::f32, format::bfyx

// 'reorder_fsv32' is being removed from "remove_redundant_reorders" in the current impl
class conv_fp32_reorder_bfyx_to_fsv32_conv_basic : public ConvReorderFusingTest {};
TEST_P(conv_fp32_reorder_bfyx_to_fsv32_conv_basic, basic) {
    auto p = GetParam();

    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p), -127, 127)),
        reorder("reorder_fsv32", input_info("input"), format::fs_b_yx_fsv32, data_types::f32),
        convolution("conv_prim", input_info("reorder_fsv32"), "weights", "", 1, { 1, 1 }, p.dilation, p.pad, p.pad, false),
        activation("activation", input_info("conv_prim"), activation_func::abs),
        reorder("reorder_out", input_info("activation"), format::bfyx, data_types::f32)
    );

    ov::intel_gpu::ImplementationDesc conv_impl = { format::fs_b_yx_fsv32, "", impl_types::ocl };
    cfg_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "conv_prim", conv_impl } }));

    execute(p);
}
INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_reorder_bfyx_to_fsv32_conv_basic, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ FSV32_CASE_CONV_FP32_1, 3, 3, 4 }
}));

// 'reorder_fsv32' is not being fused in the current impl, since it has 'mean'
class conv_fp32_reorder_bfyx_to_fsv32_conv_mean : public ConvReorderFusingTest {};
TEST_P(conv_fp32_reorder_bfyx_to_fsv32_conv_mean, have_mean) {
    auto p = GetParam();
    memory::ptr mul = engine.allocate_memory({ data_types::f32, format::bfyx, tensor{ 1, 3, 1, 2 } });
    set_values<float>(mul, { 0.5f, 2.5f, -5.0f, 4.3f, 1.2f, -3.5f });

    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("mul", mul),
        data("weights", get_mem(get_weights_layout(p), -127, 127)),
        reorder("reorder_fsv32", input_info("input"), format::fs_b_yx_fsv32, data_types::f32, "mul", reorder_mean_mode::mul),
        convolution("conv_prim", input_info("reorder_fsv32"), "weights", "", 1, { 1, 1 }, p.dilation, p.pad, p.pad, false),
        activation("activation", input_info("conv_prim"), activation_func::abs)
    );

    execute(p);
}
INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_reorder_bfyx_to_fsv32_conv_mean, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ FSV32_CASE_CONV_FP32_1, 4, 4, 4 }
}));

// 'reorder_fsv32' is not being fused in the current impl, since it has 'subtract'
class conv_fp32_reorder_bfyx_to_fsv32_conv_subtract : public ConvReorderFusingTest {};
TEST_P(conv_fp32_reorder_bfyx_to_fsv32_conv_subtract, have_subtract_per_feature) {
    auto p = GetParam();
    const std::vector<float>& values_to_subtract = {
        0.1f, 0.2f, 0.1f, 0.1f, 0.1f, 0.2f, 0.1f, 0.1f,
        0.1f, 0.2f, 0.1f, 0.1f, 0.1f, 0.2f, 0.1f, 0.1f,
        0.1f, 0.2f, 0.1f, 0.1f, 0.1f, 0.2f, 0.1f, 0.1f,
        0.1f, 0.2f, 0.1f, 0.1f, 0.1f, 0.2f, 0.1f, 0.1f
    };

    auto dw_weights_layout = layout{ {p.out_shape[1].get_length(), 1, 1, 2, 2}, p.default_type, format::goiyx };
    ov::Strides dw_stride = { 1, 1 };

    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p), -127, 127)),
        data("weights_dw", get_mem(dw_weights_layout, -127, 127)),
        convolution("conv_prim", input_info("input"), "weights", "", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        reorder("reorder_fsv32", input_info("conv_prim"), format::fs_b_yx_fsv32, data_types::f32, values_to_subtract),
        convolution("conv_output", input_info("reorder_fsv32"), "weights_dw", "", p.out_shape[1].get_length(), dw_stride, p.dilation, p.pad, p.pad, true)
    );

    ov::intel_gpu::ImplementationDesc conv_impl = { format::fs_b_yx_fsv32, "", impl_types::ocl };
    cfg_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "conv_prim", conv_impl } }));

    execute(p);
}
INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_reorder_bfyx_to_fsv32_conv_subtract, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ FSV32_CASE_CONV_FP32_1, 4, 4, 4 }
}));

// 'reorder_fsv32' is not being fused in the current impl, since it has 'fused_activation'
class conv_fp32_reorder_bfyx_to_fsv32_conv_fused_activation : public ConvReorderFusingTest {};
TEST_P(conv_fp32_reorder_bfyx_to_fsv32_conv_fused_activation, have_fused_activation) {
    auto p = GetParam();

    auto dw_weights_layout = layout{ {p.out_shape[1].get_length(), 1, 1, 2, 2}, p.default_type, format::goiyx };
    ov::Strides dw_stride = { 1, 1 };

    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p), -127, 127)),
        data("weights_dw", get_mem(dw_weights_layout, -127, 127)),
        data("actv_params", get_mem(get_per_channel_layout(p), -127, 127)),
        convolution("conv_prim", input_info("input"), "weights", "", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        reorder("reorder_fsv32", input_info("conv_prim"), format::fs_b_yx_fsv32, data_types::f32),
        activation("activation_quantize", input_info("reorder_fsv32"), "actv_params", activation_func::relu_negative_slope),
        convolution("conv_prim2", input_info("activation_quantize"), "weights_dw", "", p.out_shape[1].get_length(), dw_stride, p.dilation, p.pad, p.pad, true),
        activation("activation", input_info("conv_prim2"), activation_func::abs)
    );

    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_reorder_bfyx_to_fsv32_conv_fused_activation, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ FSV32_CASE_CONV_FP32_1, 5, 5, 6 }
}));

// activation will be fused through 'reorder_fsv32' and 'reorder_fsv32' will be fused as well
class conv_fp32_reorder_bfyx_to_fsv32_conv_fused_through_activation : public ConvReorderFusingTest {};
TEST_P(conv_fp32_reorder_bfyx_to_fsv32_conv_fused_through_activation, have_fused_through_activation) {
    auto p = GetParam();

    auto dw_weights_layout = layout{ {p.out_shape[1].get_length(), 1, 1, 2, 2}, p.default_type, format::goiyx, };
    ov::Strides dw_stride = { 1, 1 };

    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p), -127, 127)),
        data("weights_dw", get_mem(dw_weights_layout, -127, 127)),
        convolution("conv_prim", input_info("input"), "weights", "", p.groups, p.stride, p.dilation, p.pad, p.pad, false),
        reorder("reorder_fsv32", input_info("conv_prim"), format::fs_b_yx_fsv32, data_types::f32),
        activation("activation_quantize", input_info("reorder_fsv32"), activation_func::relu),
        convolution("conv_prim2", input_info("activation_quantize"), "weights_dw", "", p.out_shape[1].get_length(), dw_stride, p.dilation, p.pad, p.pad, true),
        activation("activation", input_info("conv_prim2"), activation_func::abs)
    );

    execute(p, {{"conv_prim", {"activation_quantize"}}});
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_reorder_bfyx_to_fsv32_conv_fused_through_activation, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ FSV32_CASE_CONV_FP32_1, 4, 4, 6 }
}));

// 'reorder_fsv32' is being fused even if it has 'padding'
class conv_fp32_reorder_bfyx_to_fsv32_conv_data_padding : public ConvReorderFusingTest {};
TEST_P(conv_fp32_reorder_bfyx_to_fsv32_conv_data_padding, have_data_padding) {
    auto p = GetParam();

    auto dw_weights_layout = layout{ {p.out_shape[1].get_length(), 1, 1, 2, 2}, p.default_type, format::goiyx };
    ov::Strides dw_stride = { 1, 1 };

    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p), -127, 127)),
        data("weights_dw", get_mem(dw_weights_layout, -127, 127)),
        convolution("conv_prim", input_info("input"), "weights", "", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        reorder("reorder_fsv32", input_info("conv_prim"), format::fs_b_yx_fsv32, data_types::f32, std::vector<float>{}, reorder_mean_mode::subtract, padding{ { 0, 0, 1, 1 }, 0 }),
        convolution("conv_prim2", input_info("reorder_fsv32"), "weights_dw", "", p.out_shape[1].get_length(), dw_stride, p.dilation, p.pad, p.pad, true),
        reorder("reorder_out", input_info("conv_prim2"), format::fs_b_yx_fsv32, data_types::f32)
    );

    execute(p);
}
INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_reorder_bfyx_to_fsv32_conv_data_padding, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ FSV32_CASE_CONV_FP32_1, 4, 4, 5 }
}));

class conv_gen9_common_conv_fwd_data_1stconv : public ConvFusingTest {};
TEST_P(conv_gen9_common_conv_fwd_data_1stconv, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        convolution("conv_prim", input_info("input"), { "weights" }, { "bias" }, p.groups, p.stride, p.dilation, p.pad, p.pad, false ),
        activation("activation", input_info("conv_prim"), activation_func::hswish),
        reorder("reorder_bfyx", input_info("activation"), p.default_format, data_types::f32)
    );

    tolerance = default_tolerance(p.default_type);
    if (engine.get_device_info().supports_immad &&
        p.default_type == data_types::f16) {
        tolerance *= 2; // Issue: 94154
    }
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_gen9_common_conv_fwd_data_1stconv, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_FP32_16, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_FP16_16, 2, 2, 3 },
}));

#ifdef ENABLE_ONEDNN_FOR_GPU
class conv_fp16_prelu_onednn : public WeightsPrimitiveFusingTestOneDNN {};
TEST_P(conv_fp16_prelu_onednn, basic_activation_eltwise) {
    auto p = GetParam();

    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("slope_data", get_mem(get_prelu_slope_layout(p))),
        data("eltwise_data", get_mem(get_output_layout(p))),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        activation("activation", input_info("conv_prim"), "slope_data", activation_func::relu_negative_slope),
        eltwise("eltwise", input_info("activation"), input_info("eltwise_data"), eltwise_mode::sum),
        reorder("reorder_bfyx", input_info("eltwise"), p.default_format, data_types::f32)
    );

    tolerance = default_tolerance(p.default_type);
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp16_prelu_onednn, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_FP16_1, 2, 2, 4 },
}));

class conv_int8_eltwise_onednn : public WeightsPrimitiveFusingTestOneDNN {};
TEST_P(conv_int8_eltwise_onednn, u8_eltwise_sum_out) {
    auto p = GetParam();

    auto shift_layout = get_output_layout(p);
    shift_layout.data_type = data_types::f32;

    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p), 0, 2)),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("shift_data", get_mem(shift_layout)),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        eltwise("shift", { input_info("conv_prim"), input_info("shift_data") }, eltwise_mode::sum, data_types::f32),
        reorder("reorder_bfyx", input_info("shift"), p.default_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

TEST_P(conv_int8_eltwise_onednn, u8_eltwise_prod_out) {
    auto p = GetParam();

    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p), -2, 2)),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f/255.f) ),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        eltwise("scale", { input_info("conv_prim"), input_info("scale_data") }, eltwise_mode::prod, data_types::u8),
        reorder("reorder_bfyx", input_info("scale"), p.default_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

static const int NOT_FOR_CLDNN = 0;
INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_int8_eltwise_onednn, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_U8S8_1, NOT_FOR_CLDNN, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_2, NOT_FOR_CLDNN, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_3, NOT_FOR_CLDNN, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_1, NOT_FOR_CLDNN, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_2, NOT_FOR_CLDNN, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_3, NOT_FOR_CLDNN, 2, 3 },

    convolution_test_params{ CASE_CONV_U8S8_11, NOT_FOR_CLDNN, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_12, NOT_FOR_CLDNN, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_13, NOT_FOR_CLDNN, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_12, NOT_FOR_CLDNN, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_13, NOT_FOR_CLDNN, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_14, NOT_FOR_CLDNN, 2, 3 },

    convolution_test_params{ CASE_CONV3D_U8S8_1, NOT_FOR_CLDNN, 2, 3 },
    convolution_test_params{ CASE_CONV3D_U8S8_2, NOT_FOR_CLDNN, 2, 3 },
    convolution_test_params{ CASE_CONV3D_U8S8_3, NOT_FOR_CLDNN, 2, 3 },
    convolution_test_params{ CASE_CONV3D_U8S8_5, NOT_FOR_CLDNN, 2, 3 },
    convolution_test_params{ CASE_CONV3D_S8S8_1, NOT_FOR_CLDNN, 2, 3 },
    convolution_test_params{ CASE_CONV3D_S8S8_2, NOT_FOR_CLDNN, 2, 3 },
    convolution_test_params{ CASE_CONV3D_S8S8_3, NOT_FOR_CLDNN, 2, 3 },
    convolution_test_params{ CASE_CONV3D_S8S8_5, NOT_FOR_CLDNN, 2, 3 },
}));

class conv_fp32_activation_abs_onednn : public WeightsPrimitiveFusingTestOneDNN {};
TEST_P(conv_fp32_activation_abs_onednn, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        activation("activation", input_info("conv_prim"), activation_func::abs),
        reorder("reorder_bfyx", input_info("activation"), p.default_format, data_types::f32)
    );

    if (engine.get_device_info().supports_immad &&
        p.default_type == data_types::f16 &&
        p.weights_format == format::gs_oiyx_gsv16) {
        GTEST_SKIP(); // Issue: 94154
    }

    tolerance = default_tolerance(p.default_type);
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_activation_abs_onednn, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_FP16_1, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_FP16_2, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_FP16_3, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_FP16_4, 2, 2, 3 },
}));

class conv_fp32_activation_mish_onednn : public WeightsPrimitiveFusingTestOneDNN {};
TEST_P(conv_fp32_activation_mish_onednn, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        activation("activation", input_info("conv_prim"), activation_func::mish),
        reorder("reorder_bfyx", input_info("activation"), p.default_format, data_types::f32)
    );

    tolerance = default_tolerance(p.default_type);
    if (engine.get_device_info().supports_immad && p.default_type == data_types::f16) {
        tolerance *= 4;
        if (p.weights_format == format::gs_oiyx_gsv16) {
            GTEST_SKIP(); // Issue: 94154
        }
    }
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_activation_mish_onednn, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_FP16_1, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_FP16_2, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_FP16_3, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_FP16_4, 2, 2, 3 },
}));

class conv_fp32_activation_swish_onednn : public WeightsPrimitiveFusingTestOneDNN {};
TEST_P(conv_fp32_activation_swish_onednn, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        activation("activation", input_info("conv_prim"), activation_func::swish),
        reorder("reorder_bfyx", input_info("activation"), p.default_format, data_types::f32)
    );

    if (engine.get_device_info().supports_immad &&
        p.default_type == data_types::f16 &&
        p.weights_format == format::gs_oiyx_gsv16) {
        GTEST_SKIP(); // Issue: 94154
    }

    tolerance = default_tolerance(p.default_type);
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_activation_swish_onednn, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_FP16_1, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_FP16_2, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_FP16_3, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_FP16_4, 2, 2, 3 },
}));

class conv_fp32_activation_hswish_onednn : public WeightsPrimitiveFusingTestOneDNN {};
TEST_P(conv_fp32_activation_hswish_onednn, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        activation("activation", input_info("conv_prim"), activation_func::hswish),
        reorder("reorder_bfyx", input_info("activation"), p.default_format, data_types::f32)
    );

    tolerance = default_tolerance(p.default_type);
    if (engine.get_device_info().supports_immad && p.default_type == data_types::f16) {
        tolerance *= 8;
        if (p.weights_format == format::gs_oiyx_gsv16) {
            GTEST_SKIP(); // Issue: 94154
        }
    }
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_activation_hswish_onednn, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_FP16_1, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_FP16_2, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_FP16_3, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_FP16_4, 2, 2, 3 },
}));

class conv_fp32_activation_exp_onednn : public WeightsPrimitiveFusingTestOneDNN {};
TEST_P(conv_fp32_activation_exp_onednn, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        activation("activation", input_info("conv_prim"), activation_func::exp),
        reorder("reorder_bfyx", input_info("activation"), p.default_format, data_types::f32)
    );

    if (engine.get_device_info().supports_immad &&
        p.default_type == data_types::f16) {
        GTEST_SKIP(); // Issue: 94154
    }

    tolerance = default_tolerance(p.default_type);
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_activation_exp_onednn, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_FP16_1, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_FP16_2, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_FP16_3, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_FP16_4, 2, 2, 3 },
}));

class conv_int8_quantize_u8_onednn : public WeightsPrimitiveFusingTestOneDNN {};
TEST_P(conv_int8_quantize_u8_onednn, per_channel) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p), -2, 2)),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("in_lo", get_mem(get_per_channel_layout(p), -10, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 0, 10)),
        data("out_lo", get_mem(get_single_element_layout(p), 0)),
        data("out_hi", get_mem(get_single_element_layout(p), 255)),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        quantize("quantize", input_info("conv_prim"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 256, data_types::u8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

TEST_P(conv_int8_quantize_u8_onednn, per_tensor) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p), -2, 2)),
        data("bias", get_mem(get_per_channel_layout(p), 0)),
        data("in_lo", get_mem(get_single_element_layout(p), -10)),
        data("in_hi", get_mem(get_single_element_layout(p), 10)),
        data("out_lo", get_mem(get_single_element_layout(p), 0)),
        data("out_hi", get_mem(get_single_element_layout(p), 255)),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        quantize("quantize", input_info("conv_prim"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 256, data_types::u8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_int8_quantize_u8_onednn, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_U8S8_1, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_2, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_3, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_1, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_2, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_3, 2, 2, 3 },
}));

class conv_int8_activation_eltwise_quantize_onednn : public WeightsPrimitiveFusingTestOneDNN {};
TEST_P(conv_int8_activation_eltwise_quantize_onednn, bsv32_fsv32) {
    auto p = GetParam();
    layout eltwise_layout = get_output_layout(p);
    eltwise_layout.format = format::bs_fs_yx_bsv32_fsv32;
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p), -1, 1)),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("eltwise_data", get_mem(eltwise_layout, -1, 1)),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -127)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        activation("activation", input_info("conv_prim"), activation_func::abs),
        eltwise("eltwise", input_info("activation"), input_info("eltwise_data"), eltwise_mode::sum),
        quantize("quantize", input_info("eltwise"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 255, data_types::i8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );

    ov::intel_gpu::ImplementationDesc conv_impl = { format::bs_fs_yx_bsv32_fsv32, "", impl_types::onednn };
    cfg_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "conv_prim", conv_impl } }));

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_int8_activation_eltwise_quantize_onednn, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_U8S8_1, 2, 2, 5 },
    convolution_test_params{ CASE_CONV_U8S8_2, 2, 2, 5 },
    convolution_test_params{ CASE_CONV_U8S8_3, 2, 2, 5 },
    convolution_test_params{ CASE_CONV_U8S8_4, 2, 2, 5 },
    convolution_test_params{ CASE_CONV_U8S8_7, 2, 2, 5 },
    //convolution_test_params{ CASE_CONV_U8S8_8, 2, 2, 5 },
    convolution_test_params{ CASE_CONV_U8S8_11, 2, 2, 5 },
    convolution_test_params{ CASE_CONV_U8S8_12, 2, 2, 5 },
    convolution_test_params{ CASE_CONV_U8S8_13, 2, 2, 5 },
    convolution_test_params{ CASE_CONV_U8S8_14, 2, 2, 5 },

    convolution_test_params{ CASE_CONV_S8S8_1, 2, 2, 5 },
    convolution_test_params{ CASE_CONV_S8S8_2, 2, 2, 5 },
    convolution_test_params{ CASE_CONV_S8S8_3, 2, 2, 5 },
    convolution_test_params{ CASE_CONV_S8S8_4, 2, 2, 5 },
    convolution_test_params{ CASE_CONV_S8S8_7, 2, 2, 5 },
    //convolution_test_params{ CASE_CONV_S8S8_8, 2, 2, 5 },
    convolution_test_params{ CASE_CONV_S8S8_12, 2, 2, 5 },
    convolution_test_params{ CASE_CONV_S8S8_13, 2, 2, 5 },
    convolution_test_params{ CASE_CONV_S8S8_14, 2, 2, 5 },
    convolution_test_params{ CASE_CONV_S8S8_15, 2, 2, 5 },
}));

class conv_int8_scale_shift_swish_onednn : public WeightsPrimitiveFusingTestOneDNN {};
TEST_P(conv_int8_scale_shift_swish_onednn, bsv32_fsv32) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p), -1, 1)),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f/255.f)),
        data("shift_data", get_mem(get_per_channel_layout(p), 1)),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        eltwise("scale0", { input_info("conv_prim"), input_info("scale_data") }, eltwise_mode::sum),
        eltwise("shift0", { input_info("scale0"), input_info("shift_data") }, eltwise_mode::sum),
        activation("sigmoid", input_info("shift0"), activation_func::swish),
        eltwise("scale1", { input_info("sigmoid"), input_info("scale_data") }, eltwise_mode::sum),
        eltwise("shift1", { input_info("scale1"), input_info("shift_data") }, eltwise_mode::sum),
        reorder("reorder_bfyx", input_info("shift1"), p.default_format, data_types::f32)
    );

    ov::intel_gpu::ImplementationDesc conv_impl = { format::bs_fs_yx_bsv32_fsv32, "", impl_types::onednn };
    cfg_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "conv_prim", conv_impl } }));

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_int8_scale_shift_swish_onednn, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_U8S8_1, 2, 2, 7 },
    convolution_test_params{ CASE_CONV_U8S8_2, 2, 2, 7 },
    convolution_test_params{ CASE_CONV_S8S8_1, 2, 2, 7 },
    convolution_test_params{ CASE_CONV_S8S8_2, 2, 2, 7 },

    convolution_test_params{ CASE_CONV_U8S8_11, 2, 2, 7 },
    convolution_test_params{ CASE_CONV_U8S8_12, 2, 2, 7 },
    convolution_test_params{ CASE_CONV_U8S8_14, 2, 2, 7 },
    convolution_test_params{ CASE_CONV_S8S8_12, 2, 2, 7 },
    convolution_test_params{ CASE_CONV_S8S8_13, 2, 2, 7 },
    convolution_test_params{ CASE_CONV_S8S8_15, 2, 2, 7 },
}));

class conv_int8_eltwise_scale_onednn : public WeightsPrimitiveFusingTestOneDNN {};
TEST_P(conv_int8_eltwise_scale_onednn, u8_eltwise_prod_out_reuse) {
    auto p = GetParam();

    if (!engine.get_device_info().supports_immad)
        return;

    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p), -2, 2)),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("sum_data", get_mem(get_per_channel_layout(p), 1.0f/255.f)),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f/255.f)),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        eltwise("sum", { input_info("conv_prim"), input_info("sum_data") }, eltwise_mode::sum, data_types::f32),
        eltwise("scale", { input_info("sum"), input_info("scale_data") }, eltwise_mode::prod, data_types::f32),
        reorder("reorder_bfyx", input_info("scale"), p.default_format, data_types::f32)
    );

    tolerance = 1.f;

    auto input_prim = get_mem(get_input_layout(p));

    auto forcing_format = p.input_format;
    ov::intel_gpu::ImplementationDesc conv_impl = { forcing_format, "", impl_types::onednn };
    cfg_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "conv_prim", conv_impl } }));

    network network_not_fused(this->engine, this->topology_non_fused, cfg_not_fused);
    network network_fused(this->engine, this->topology_fused, cfg_fused);
    network_fused.set_input_data("input", input_prim);
    network_not_fused.set_input_data("input", input_prim);

    // First network.execute() call
    compare(network_not_fused, network_fused, p);
    // Second network.execute() call to make sure that scales have not been wrongly overwritten within first iteration
    // and don't affect final result of second iteration
    compare(network_not_fused, network_fused, p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_int8_eltwise_scale_onednn, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_U8S8_15, 2, 2, 4 },
}));

/* ----------------------------------------------------------------------------------------------------- */
/* ------------------------------ OneDNN post-ops cases with optimizations ----------------------------- */
/* ----------------------------------------------------------------------------------------------------- */

// Before optimization: eltw_linear + eltw_linear
// After optimization: eltw_linear
// Limitations: no
// DNNL_VERBOSE log without optimization: attr-post-ops:eltwise_linear:12.75:127.5+eltwise_linear:1:-128
// DNNL_VERBOSE log with optimization:    attr-post-ops:eltwise_linear:12.75:-0.5
class post_ops_optimizations_onednn_eltw_linear_eltw_linear : public WeightsPrimitiveFusingTestOneDNN {};
TEST_P(post_ops_optimizations_onednn_eltw_linear_eltw_linear, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("in_lo", get_mem(get_single_element_layout(p), -10)),
        data("in_hi", get_mem(get_single_element_layout(p), 10)),
        data("out_lo", get_mem(get_single_element_layout(p), -128)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        quantize("quantize", input_info("conv_prim"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 256, data_types::i8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, post_ops_optimizations_onednn_eltw_linear_eltw_linear, ::testing::ValuesIn(std::vector<convolution_test_params>{
    // cases with batch = 1
    convolution_test_params{ CASE_CONV_U8S8_1, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_2, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_3, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_1, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_2, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_3, 2, 2, 3 },

    // cases with batch = 16
    convolution_test_params{ CASE_CONV_U8S8_9, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_10, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_9, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_10, 2, 2, 3 },

    // cases with batch = 32
    convolution_test_params{ CASE_CONV_U8S8_11, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_12, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_13, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_14, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_12, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_13, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_14, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_15, 2, 2, 3 },
}));

// Before optimization: eltw_non_linear + eltw_linear
// After optimization: eltw_non_linear
// Limitations: beta = 0 in eltw_linear
// DNNL_VERBOSE log without optimization: attr-post-ops:eltwise_linear:12.75:127.5+eltwise_round+eltwise_linear:2.00784+eltwise_clip:0:512
// DNNL_VERBOSE log with optimization:    attr-post-ops:eltwise_linear:12.75:127.5+eltwise_round:0:0:2.00784+eltwise_clip:0:512
class post_ops_optimizations_onednn_eltw_non_linear_eltw_linear : public WeightsPrimitiveFusingTestOneDNN {};
TEST_P(post_ops_optimizations_onednn_eltw_non_linear_eltw_linear, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("in_lo", get_mem(get_single_element_layout(p), -10)),
        data("in_hi", get_mem(get_single_element_layout(p), 10)),
        data("out_lo", get_mem(get_single_element_layout(p), 0)),
        data("out_hi", get_mem(get_single_element_layout(p), 512)),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        quantize("quantize", input_info("conv_prim"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 256, data_types::f32),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, post_ops_optimizations_onednn_eltw_non_linear_eltw_linear, ::testing::ValuesIn(std::vector<convolution_test_params>{
    // cases with batch = 1
    convolution_test_params{ CASE_CONV_U8S8_1, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_2, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_3, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_1, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_2, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_3, 2, 2, 3 },

    // cases with batch = 16
    convolution_test_params{ CASE_CONV_U8S8_9, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_10, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_9, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_10, 2, 2, 3 },

    // cases with batch = 32
    convolution_test_params{ CASE_CONV_U8S8_11, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_12, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_13, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_14, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_12, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_13, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_14, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_15, 2, 2, 3 },
}));

// Before optimization: binary_add + eltw_linear
// After optimization: binary_add
// Limitations: alpha = 1 and scale = 1 in eltw_linear; binary_add is a constant compile-time buffer
// DNNL_VERBOSE log without optimization: attr-oscale:2 attr-post-ops:binary_add:f32:2+eltwise_linear:1:-127+eltwise_clip:-127:127
// DNNL_VERBOSE log with optimization:    attr-oscale:2 attr-post-ops:binary_add:f32:2+eltwise_clip:-127:127
class post_ops_optimizations_onednn_binary_add_eltw_linear : public WeightsPrimitiveFusingTestOneDNN {};
TEST_P(post_ops_optimizations_onednn_binary_add_eltw_linear, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -127)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        quantize("quantize", input_info("conv_prim"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 255, data_types::i8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, post_ops_optimizations_onednn_binary_add_eltw_linear, ::testing::ValuesIn(std::vector<convolution_test_params>{
    // cases with batch = 1
    convolution_test_params{ CASE_CONV_U8S8_1, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_2, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_3, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_1, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_2, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_3, 2, 2, 3 },

    // cases with batch = 16
    convolution_test_params{ CASE_CONV_U8S8_9, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_10, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_9, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_10, 2, 2, 3 },

    // cases with batch = 32
    convolution_test_params{ CASE_CONV_U8S8_11, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_12, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_13, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_14, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_12, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_13, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_14, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_15, 2, 2, 3 },
}));

// Before optimization: binary_mul + eltw_linear
// After optimization: binary_mul
// Limitations: beta = 0 in eltw_linear; binary_mul is a constant compile-time buffer
// DNNL_VERBOSE log without optimization: attr-oscale:2 attr-post-ops:binary_mul:f32:2+eltwise_linear:2.01575+eltwise_clip:0:512
// DNNL_VERBOSE log with optimization:    attr-oscale:2 attr-post-ops:binary_mul:f32:2+eltwise_clip:0:512
class post_ops_optimizations_onednn_binary_mul_eltw_linear : public WeightsPrimitiveFusingTestOneDNN {};
TEST_P(post_ops_optimizations_onednn_binary_mul_eltw_linear, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("eltwise_data", get_mem(get_per_channel_layout(p), -1, 1)),
        data("in_lo", get_mem(get_per_channel_layout(p), 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), 0)),
        data("out_hi", get_mem(get_single_element_layout(p), 512)),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        eltwise("eltwise", { input_info("conv_prim"), input_info("eltwise_data") }, eltwise_mode::prod),
        quantize("quantize", input_info("eltwise"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 255, data_types::i8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, post_ops_optimizations_onednn_binary_mul_eltw_linear, ::testing::ValuesIn(std::vector<convolution_test_params>{
    // cases with batch = 1
    convolution_test_params{ CASE_CONV_U8S8_1, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_U8S8_2, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_U8S8_3, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_S8S8_1, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_S8S8_2, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_S8S8_3, 2, 2, 4 },

    // cases with batch = 16
    convolution_test_params{ CASE_CONV_U8S8_9, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_U8S8_10, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_S8S8_9, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_S8S8_10, 2, 2, 4 },

    // cases with batch = 32
    convolution_test_params{ CASE_CONV_U8S8_11, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_U8S8_12, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_U8S8_13, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_S8S8_12, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_S8S8_13, 2, 2, 4 },
    convolution_test_params{ CASE_CONV_S8S8_14, 2, 2, 4 },
}));

// Before optimization: o_scale + eltw_linear
// After optimization: o_scale
// Limitations: beta = 0 in eltw_linear
// DNNL_VERBOSE log without optimization: attr-oscale:2 attr-post-ops:eltwise_linear:2.01575+eltwise_clip:0:512
// DNNL_VERBOSE log with optimization:    attr-oscale:2 attr-post-ops:eltwise_clip:0:512
class post_ops_optimizations_onednn_oscale_eltw_linear : public WeightsPrimitiveFusingTestOneDNN {};
TEST_P(post_ops_optimizations_onednn_oscale_eltw_linear, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("in_lo", get_mem(get_per_channel_layout(p), 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), 0)),
        data("out_hi", get_mem(get_single_element_layout(p), 512)),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        quantize("quantize", input_info("conv_prim"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 255, data_types::i8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, post_ops_optimizations_onednn_oscale_eltw_linear, ::testing::ValuesIn(std::vector<convolution_test_params>{
    // cases with batch = 1
    convolution_test_params{ CASE_CONV_U8S8_1, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_2, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_3, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_1, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_2, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_3, 2, 2, 3 },

    // cases with batch = 16
    convolution_test_params{ CASE_CONV_U8S8_9, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_10, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_9, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_10, 2, 2, 3 },

    // cases with batch = 32
    convolution_test_params{ CASE_CONV_U8S8_11, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_12, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_13, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_12, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_13, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_14, 2, 2, 3 },
}));

// Before optimization: eltw_any + sum + eltw_linear
// After optimization: eltw_any + sum
// Limitations: beta = 0 in eltw_linear
// DNNL_VERBOSE log without optimization: attr-post-ops:eltwise_relu+sum:1:0:u8+eltwise_linear:12.7+eltwise_clip:0:127
// DNNL_VERBOSE log with optimization:    attr-post-ops:eltwise_relu:0:0:12.7+sum:12.7:0:u8+eltwise_clip:0:127
class post_ops_optimizations_onednn_eltw_any_sum_eltw_linear : public WeightsPrimitiveFusingTestOneDNN {};
TEST_P(post_ops_optimizations_onednn_eltw_any_sum_eltw_linear, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("in_lo", get_mem(get_single_element_layout(p), 0)),
        data("in_hi", get_mem(get_single_element_layout(p), 10)),
        data("out_lo", get_mem(get_single_element_layout(p), 0)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        data("eltwise_data", get_mem(get_output_layout(p))),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        activation("activation", input_info("conv_prim"), activation_func::relu_negative_slope),
        eltwise("sum", { input_info("activation"), input_info("eltwise_data") }, eltwise_mode::sum),
        quantize("quantize", input_info("sum"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 128, data_types::u8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, post_ops_optimizations_onednn_eltw_any_sum_eltw_linear, ::testing::ValuesIn(std::vector<convolution_test_params>{
    // cases with batch = 1
    convolution_test_params{ CASE_CONV_U8S8_1, 2, 2, 5 },
    convolution_test_params{ CASE_CONV_U8S8_2, 2, 2, 5 },
    convolution_test_params{ CASE_CONV_U8S8_3, 2, 2, 5 },
    convolution_test_params{ CASE_CONV_S8S8_1, 2, 2, 5 },
    convolution_test_params{ CASE_CONV_S8S8_2, 2, 2, 5 },
    convolution_test_params{ CASE_CONV_S8S8_3, 2, 2, 5 },

    // cases with batch = 16
    convolution_test_params{ CASE_CONV_U8S8_10, 2, 2, 5 },
    convolution_test_params{ CASE_CONV_S8S8_10, 2, 2, 5 },

    // cases with batch = 32
    convolution_test_params{ CASE_CONV_U8S8_11, 2, 2, 5 },
    convolution_test_params{ CASE_CONV_U8S8_12, 2, 2, 5 },
    convolution_test_params{ CASE_CONV_U8S8_13, 2, 2, 5 },
    convolution_test_params{ CASE_CONV_U8S8_14, 2, 2, 5 },
    convolution_test_params{ CASE_CONV_S8S8_12, 2, 2, 5 },
    convolution_test_params{ CASE_CONV_S8S8_13, 2, 2, 5 },
    convolution_test_params{ CASE_CONV_S8S8_14, 2, 2, 5 },
    convolution_test_params{ CASE_CONV_S8S8_15, 2, 2, 5 },
}));

// Input range uses in 2 cases: not per-tensor output range or out_lo > out_hi
// Here's out_lo > out_hi and no optimizations
// DNNL_VERBOSE log: attr-post-ops:eltwise_linear:12.75:127.5+eltwise_round+eltwise_linear:-1:127
class post_ops_optimizations_input_range : public WeightsPrimitiveFusingTestOneDNN {};
TEST_P(post_ops_optimizations_input_range, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("in_lo", get_mem(get_single_element_layout(p), -10)),
        data("in_hi", get_mem(get_single_element_layout(p), 10)),
        data("out_lo", get_mem(get_single_element_layout(p), 127)),
        data("out_hi", get_mem(get_single_element_layout(p), -128)),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        quantize("quantize", input_info("conv_prim"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 256, data_types::i8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, post_ops_optimizations_input_range, ::testing::ValuesIn(std::vector<convolution_test_params>{
    // cases with batch = 1
    convolution_test_params{ CASE_CONV_U8S8_1, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_2, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_3, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_1, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_2, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_3, 2, 2, 3 },

    // cases with batch = 16
    convolution_test_params{ CASE_CONV_U8S8_9, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_10, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_9, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_10, 2, 2, 3 },

    // cases with batch = 32
    convolution_test_params{ CASE_CONV_U8S8_11, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_12, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_13, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_U8S8_14, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_12, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_13, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_14, 2, 2, 3 },
    convolution_test_params{ CASE_CONV_S8S8_15, 2, 2, 3 },
}));

struct convolution_eltw_sum_test_params {
    ov::PartialShape in_shape;
    ov::PartialShape out_shape;
    ov::PartialShape weights_shape;
    ov::Strides stride;
    ov::CoordinateDiff pad;
    ov::Strides dilation;
    uint32_t groups;
    data_types data_type;
    format input_format;
    data_types weights_type;
    format weights_format;
    data_types eltw_type;
    format eltw_format;
    data_types out_type;
    format out_format;
    data_types default_type;
    format default_format;
    size_t expected_fused_primitives;
    size_t expected_fused_primitives_onednn;
    size_t expected_not_fused_primitives;
};

class EltwiseSumFusingTestOneDNN : public BaseFusingTest<convolution_eltw_sum_test_params> {
public:
    void execute(convolution_eltw_sum_test_params& p) {
        if (!engine.get_device_info().supports_immad)
            return;
        auto input_prim = p.data_type == data_types::u8 ? get_mem(get_input_layout(p), 0, 10) : get_mem(get_input_layout(p));

        network network_not_fused(this->engine, this->topology_non_fused, cfg_not_fused);
        network network_fused(this->engine, this->topology_fused, cfg_fused);
        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);

        compare(network_not_fused, network_fused, p);

        auto pi_fused = network_fused.get_primitives_info();
        auto info_fused = std::find_if(pi_fused.begin(), pi_fused.end(), [](primitive_info& p) -> bool {
            if (p.original_id == "conv_prim")
                return true;
            return false;
        });

        if (info_fused != pi_fused.end()) {
            std::cout << "kernel: " << info_fused->kernel_id << std::endl;
            EXPECT_TRUE(info_fused->kernel_id.find("jit:ir") != std::string::npos);
        }
    }

    layout get_input_layout(convolution_eltw_sum_test_params& p) {
        return layout{ p.in_shape, p.data_type, p.input_format };
    }

    layout get_per_channel_layout(convolution_eltw_sum_test_params& p) {
        ov::PartialShape shape(std::vector<ov::Dimension>(p.out_shape.size(), 1));
        shape[1] = p.out_shape[1];
        return layout{ shape, p.default_type, p.default_format };
    }

    layout get_weights_layout(convolution_eltw_sum_test_params& p) {
        return layout{ p.weights_shape, p.weights_type, p.weights_format };
    }
};

class onednn_binary_add_full_tensor : public EltwiseSumFusingTestOneDNN {};
TEST_P(onednn_binary_add_full_tensor, basic) {
    auto p = GetParam();
    if (engine.get_device_info().supports_immad)
        p.expected_fused_primitives = p.expected_fused_primitives_onednn;

    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("in_lo1", get_mem(get_single_element_layout(p), 0)),
        data("in_hi1", get_mem(get_single_element_layout(p), 100)),
        data("out_lo1", get_mem(get_single_element_layout(p), 0)),
        data("out_hi1", get_mem(get_single_element_layout(p), 100)),
        data("eltwise_data", get_mem(layout{ p.out_shape, p.eltw_type, p.eltw_format }, 0, 100)),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        activation("activation", input_info("conv_prim"), activation_func::hyperbolic_tan),
        quantize("quantize1", input_info("activation"), input_info("in_lo1"), input_info("in_hi1"),
                 input_info("out_lo1"), input_info("out_hi1"), 256, p.out_type),
        eltwise("sum", { input_info("quantize1"), input_info("eltwise_data") }, eltwise_mode::sum, p.out_type),
        reorder("reorder_bfyx", input_info("sum"), p.default_format, p.default_type)
    );

    tolerance = 1.f;
    execute(p);
}

// in_shape; out_shape; kernel; stride; pad; dilation; groups; data_type; input_format; weights_type; weights_format; eltw_type; eltw_format; out_type; out_format; default_type; default_format;
#define CASE_CONV_ELTW_SUM_BINARY_ADD_1     { 1, 32, 4, 4 }, { 1, 16, 4, 4 }, { 16, 32, 3, 3 }, { 1, 1 }, { 1, 1 }, { 1, 1 }, 1, data_types::u8, format::b_fs_yx_fsv32, data_types::i8, format::bfyx, data_types::u8, format::b_fs_yx_fsv32, data_types::f16, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_CONV_ELTW_SUM_SUM_1            { 1, 32, 4, 4 }, { 1, 16, 4, 4 }, { 16, 32, 3, 3 }, { 1, 1 }, { 1, 1 }, { 1, 1 }, 1, data_types::u8, format::b_fs_yx_fsv32, data_types::i8, format::bfyx, data_types::u8, format::b_fs_yx_fsv32, data_types::u8, format::b_fs_yx_fsv32, data_types::f32, format::bfyx
#define CASE_CONV_ELTW_SUM_SUM_DIFF_DTYPE_1 { 1, 32, 4, 4 }, { 1, 16, 4, 4 }, { 16, 32, 3, 3 }, { 1, 1 }, { 1, 1 }, { 1, 1 }, 1, data_types::u8, format::b_fs_yx_fsv32, data_types::i8, format::bfyx, data_types::i8, format::b_fs_yx_fsv32, data_types::u8, format::b_fs_yx_fsv32, data_types::f32, format::bfyx

INSTANTIATE_TEST_SUITE_P(eltwise_sum_fusings_gpu, onednn_binary_add_full_tensor, ::testing::ValuesIn(std::vector<convolution_eltw_sum_test_params>{
    convolution_eltw_sum_test_params{ CASE_CONV_ELTW_SUM_BINARY_ADD_1, 2, 4, 5 },
    convolution_eltw_sum_test_params{ CASE_CONV_ELTW_SUM_SUM_1, 2, 4, 5 },
    convolution_eltw_sum_test_params{ CASE_CONV_ELTW_SUM_SUM_DIFF_DTYPE_1, 2, 4, 5 },
}));


class onednn_multiple_binary_add_full_tensor : public EltwiseSumFusingTestOneDNN {};
TEST_P(onednn_multiple_binary_add_full_tensor, basic) {
    auto p = GetParam();
    if (engine.get_device_info().supports_immad)
        p.expected_fused_primitives = p.expected_fused_primitives_onednn;

    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("in_lo1", get_mem(get_single_element_layout(p), 0)),
        data("in_hi1", get_mem(get_single_element_layout(p), 100)),
        data("out_lo1", get_mem(get_single_element_layout(p), 0)),
        data("out_hi1", get_mem(get_single_element_layout(p), 100)),
        data("eltwise_data", get_mem(layout{ p.out_shape, p.eltw_type, p.eltw_format }, 0, 100)),
        data("eltwise_data1", get_mem(layout{ p.out_shape, p.eltw_type, p.eltw_format }, 0, 100)),
        data("eltwise_data2", get_mem(layout{ { 1, p.out_shape[1], 1, 1 }, p.eltw_type, format::bfyx}, 0, 100)),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        activation("activation", input_info("conv_prim"), activation_func::hyperbolic_tan),
        quantize("quantize1", input_info("activation"), input_info("in_lo1"), input_info("in_hi1"),
                 input_info("out_lo1"), input_info("out_hi1"), 256, p.out_type),
        eltwise("sum", { input_info("quantize1"), input_info("eltwise_data") }, eltwise_mode::sum, p.out_type), // eltwise sum with full tensor
        eltwise("sum1", { input_info("sum"), input_info("eltwise_data1") }, eltwise_mode::sum, p.out_type),     // eltwise sum with full tensor
        eltwise("sum2", { input_info("sum1"), input_info("eltwise_data2") }, eltwise_mode::sum, p.out_type),    // eltwise sum with broadcasting
        reorder("reorder_bfyx", input_info("sum2"), p.default_format, p.default_type)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(multiple_eltwise_sum_fusings_gpu, onednn_multiple_binary_add_full_tensor, ::testing::ValuesIn(std::vector<convolution_eltw_sum_test_params>{
    convolution_eltw_sum_test_params{ CASE_CONV_ELTW_SUM_BINARY_ADD_1, 2, 4, 7 },
    convolution_eltw_sum_test_params{ CASE_CONV_ELTW_SUM_SUM_1, 2, 4, 7 },
}));

struct implicit_crop_concat_convolution_test_params {
    ov::PartialShape in_shape;
    ov::PartialShape out_shape;
    ov::PartialShape weights_shape;
    ov::Strides stride;
    ov::CoordinateDiff pad;
    ov::Strides dilation;
    uint32_t groups;
    data_types data_type;
    format input_format;
    data_types weights_type;
    format weights_format;
    data_types output_data_type;
    format output_format;
    data_types default_type;
    format default_format;
    size_t expected_fused_primitives;
    size_t expected_not_fused_primitives;
};

class ImplicitCropConcatTestOneDNN: public BaseFusingTest<implicit_crop_concat_convolution_test_params> {
public:
    void execute(implicit_crop_concat_convolution_test_params& p) {
        auto input_prim = p.data_type == data_types::u8 ? get_mem(get_input_layout(p), 0, 10) : get_mem(get_input_layout(p));

        cfg_not_fused = cfg_fused;
        // ov::intel_gpu::ImplementationDesc quantize_impl = { p.output_format, "quantize_gpu_ref", impl_types::ocl };
        cfg_not_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{
            { "quantize1", { p.output_format, "quantize_gpu_scale_shift_opt", impl_types::ocl } },
            { "quantize2", { p.output_format, "quantize_gpu_scale_shift_opt", impl_types::ocl } },
            { "quantize3", { p.output_format, "quantize_gpu_scale_shift_opt", impl_types::ocl } } }));

        network network_not_fused(this->engine, this->topology_non_fused, cfg_not_fused);
        network network_fused(this->engine, this->topology_fused, cfg_fused);
        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);
        compare(network_not_fused, network_fused, p);
    }

    layout get_input_layout(implicit_crop_concat_convolution_test_params& p) {
        return layout{ p.in_shape, p.data_type, p.input_format };
    }

    layout get_per_channel_layout(implicit_crop_concat_convolution_test_params& p) {
        ov::PartialShape shape(std::vector<ov::Dimension>(p.out_shape.size(), 1));
        shape[1] = p.out_shape[1];
        return layout{ shape, p.default_type, p.default_format };
    }

    layout get_weights_layout(implicit_crop_concat_convolution_test_params& p) {
        return layout{ p.weights_shape, p.weights_type, p.weights_format };
    }
};

class implicit_crop_concat_bfyx_input_tensor : public ImplicitCropConcatTestOneDNN {};

TEST_P(implicit_crop_concat_bfyx_input_tensor, basic) {
    auto p = GetParam();

    tensor crop_output = get_input_layout(p).get_tensor();
    crop_output.feature[0] = 1;
    auto crop_offset1 = tensor(batch(0), feature(0), spatial(0, 0));
    auto crop_offset2 = tensor(batch(0), feature(1), spatial(0, 0));
    auto crop_offset3 = tensor(batch(0), feature(2), spatial(0, 0));

    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),

        data("in_lo1", get_mem(get_single_element_layout(p), 0)),
        data("in_hi1", get_mem(get_single_element_layout(p), 100)),
        data("out_lo1", get_mem(get_single_element_layout(p), -127)),
        data("out_hi1", get_mem(get_single_element_layout(p), 127)),

        data("in_lo2", get_mem(get_single_element_layout(p), 0)),
        data("in_hi2", get_mem(get_single_element_layout(p), 100)),
        data("out_lo2", get_mem(get_single_element_layout(p), -127)),
        data("out_hi2", get_mem(get_single_element_layout(p), 127)),

        data("in_lo3", get_mem(get_single_element_layout(p), 0)),
        data("in_hi3", get_mem(get_single_element_layout(p), 100)),
        data("out_lo3", get_mem(get_single_element_layout(p), -127)),
        data("out_hi3", get_mem(get_single_element_layout(p), 127)),

        crop("crop1", input_info("input"), crop_output, crop_offset1),
        quantize("quantize1", input_info("crop1"), input_info("in_lo1"), input_info("in_hi1"),
                 input_info("out_lo1"), input_info("out_hi1"), 255, data_types::i8),
        crop("crop2", input_info("input"), crop_output, crop_offset2),
        quantize("quantize2", input_info("crop2"), input_info("in_lo2"), input_info("in_hi2"),
                 input_info("out_lo2"), input_info("out_hi2"), 255, data_types::i8),
        crop("crop3", input_info("input"), crop_output, crop_offset3),
        quantize("quantize3", input_info("crop3"), input_info("in_lo3"), input_info("in_hi3"),
                 input_info("out_lo3"), input_info("out_hi3"), 255, data_types::i8),
        concatenation("concat", { input_info("quantize1"), input_info("quantize2"), input_info("quantize3") }, 1),
        convolution("conv_prim", input_info("concat"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        reorder("reorder_bfyx", input_info("conv_prim"), p.default_format, p.default_type)
    );

    tolerance = 1.f;
    execute(p);
}

// in_shape; out_shape; kernel; stride; pad; dilation; groups; input_data_type; input_format; weights_type; weights_format; output_data_type; output_format; default_type; default_format;
#define CASE_CROP_FQ_CONCAT_1 { 1, 3, 10, 10 }, { 1, 16, 10, 10 }, { 16, 3, 3, 3 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, data_types::u8, format::bfyx, data_types::i8, format::bfyx, data_types::i8, format::b_fs_yx_fsv16, data_types::f32, format::bfyx

INSTANTIATE_TEST_SUITE_P(implicit_crop_concat_conv_fusings_gpu, implicit_crop_concat_bfyx_input_tensor, ::testing::ValuesIn(std::vector<implicit_crop_concat_convolution_test_params>{
    implicit_crop_concat_convolution_test_params{ CASE_CROP_FQ_CONCAT_1, 5, 9 },
}));


class PermuteOptimizingTestOnednn : public BaseFusingTest<convolution_test_params> {
public:
    void execute(convolution_test_params& p, bool is_permute_optimized = true) {
        if (!engine.get_device_info().supports_immad)
            return;

        p.expected_fused_primitives = p.expected_fused_primitives_onednn;

        cldnn::memory::ptr input_prim = get_mem(get_input_layout(p));

        network network_not_fused(this->engine, this->topology_non_fused, cfg_not_fused);
        network network_fused(this->engine, this->topology_fused, cfg_fused);
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

        auto permute_prim = std::find_if(pi_fused.begin(), pi_fused.end(), [](primitive_info& p) -> bool {
            if (p.original_id == "permute")
                return true;
            return false;
        });

        ASSERT_TRUE(permute_prim != pi_fused.end());
        if (is_permute_optimized) {
            ASSERT_TRUE(permute_prim->kernel_id == "undef");
        } else {
            ASSERT_FALSE(permute_prim->kernel_id == "undef");
        }
    }

    layout get_input_layout(convolution_test_params& p) {
        return layout{ p.in_shape, p.data_type, p.input_format };
    }

    layout get_per_channel_layout(convolution_test_params& p) {
        ov::PartialShape shape(std::vector<ov::Dimension>(p.out_shape.size(), 1));
        shape[1] = p.out_shape[1];
        return layout{ shape, p.default_type, p.default_format };
    }

    layout get_weights_layout(convolution_test_params& p) {
        return layout{ p.weights_shape, p.weights_type, p.weights_format };
    }
};


#define CASE_CONV_FP16_PERMUTE_1 { 1, 4, 5, 3 }, { 1, 30, 3, 2 }, { 30, 3, 3, 3 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, data_types::f16, format::bfyx, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_CONV_FP16_PERMUTE_2 { 1, 15, 5, 4 }, { 1, 30, 3, 2 }, { 30, 15, 3, 3 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, data_types::f16, format::bfyx, data_types::f16, format::bfyx, data_types::f16, format::bfyx

class conv_after_permute_optimizing : public PermuteOptimizingTestOnednn {};
TEST_P(conv_after_permute_optimizing, basic) {
    if (!engine.get_device_info().supports_immad)
        return;

    auto p = GetParam();

    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        permute("permute", input_info("input"), {0, 3, 1, 2}),
        convolution("conv_prim", input_info("permute"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        activation("activation", input_info("conv_prim"), activation_func::abs),
        reorder("reorder_bfyx", input_info("activation"), p.default_format, data_types::f32)
    );

    tolerance = default_tolerance(p.default_type);
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_after_permute_optimizing, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_FP16_PERMUTE_1, 3, 2, 4 },
}));

#define CASE_CONV_INT8_PERMUTE_1 { 1, 4, 3, 5 }, { 1, 30, 2, 3 }, { 30, 5, 3, 3 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, data_types::f16, format::bfyx, data_types::i8, format::bfyx, data_types::f32, format::bfyx

class conv_after_permute_not_optimizing : public PermuteOptimizingTestOnednn {};
TEST_P(conv_after_permute_not_optimizing, basic) {
    if (!engine.get_device_info().supports_immad)
        return;

    GTEST_SKIP(); // Issue: 94154

    auto p = GetParam();

    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        data("in_lo1", get_mem(get_single_element_layout(p), 0)),
        data("in_hi1", get_mem(get_single_element_layout(p), 100)),
        data("out_lo1", get_mem(get_single_element_layout(p), 0)),
        data("out_hi1", get_mem(get_single_element_layout(p), 100)),
        permute("permute", input_info("input"), {0, 3, 1, 2}),
        quantize("quantize1", input_info("permute"), input_info("in_lo1"), input_info("in_hi1"),
                 input_info("out_lo1"), input_info("out_hi1"), 256, data_types::i8),
        convolution("conv_prim", input_info("quantize1"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        activation("activation", input_info("conv_prim"), activation_func::abs),
        reorder("reorder_bfyx", input_info("activation"), p.default_format, data_types::f32)
    );

    tolerance = default_tolerance(p.default_type);
    execute(p, false);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_after_permute_not_optimizing, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_INT8_PERMUTE_1, 3, 3, 5 },
}));

class conv_before_permute_optimizing : public PermuteOptimizingTestOnednn {};
TEST_P(conv_before_permute_optimizing, basic) {
    if (!engine.get_device_info().supports_immad)
        return;

    auto p = GetParam();

    ov::intel_gpu::ImplementationDesc conv_impl = { cldnn::format::type::any, "", impl_types::onednn };
    cfg_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "conv_prim", conv_impl } }));

    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_per_channel_layout(p))),
        convolution("conv_prim", input_info("input"), "weights", "bias", p.groups, p.stride, p.dilation, p.pad, p.pad, format::is_grouped(get_weights_layout(p).format)),
        activation("activation", input_info("conv_prim"), activation_func::abs),
        permute("permute", input_info("activation"), {0, 2, 3, 1}),
        reorder("reorder_bfyx", input_info("permute"), p.default_format, data_types::f32)
    );

    tolerance = default_tolerance(p.default_type);
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_before_permute_optimizing, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_FP16_PERMUTE_2, 3, 2, 4 },
}));

class EltwiseSumWithConstantFullTensorFusingTestOneDNN : public BaseFusingTest<convolution_eltw_sum_test_params> {
public:
    void execute(convolution_eltw_sum_test_params& p) {
        if (!engine.get_device_info().supports_immad)
            return;
        auto input_prim = get_mem(get_weights_layout(p));

        network network_not_fused(this->engine, this->topology_non_fused, cfg_not_fused);
        network network_fused(this->engine, this->topology_fused, cfg_fused);
        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);

        // Multiple executions of network to increase error of result.
        // The output of constant layer will be changed through this iterations bigger.
        for (int i = 0; i < 10; i++) {
            network_not_fused.execute();
            network_fused.execute();
        }

        compare(network_not_fused, network_fused, p);
    }

    layout get_input_layout(convolution_eltw_sum_test_params& p) {
        return layout{ p.in_shape, p.data_type, p.input_format };
    }

    layout get_weights_layout(convolution_eltw_sum_test_params& p) {
        return layout{p.weights_shape, p.weights_type, p.weights_format};
    }
};

// When dependency of eltwise is full tensor constant, use binary add instead of sum as post-op.
class onednn_replace_full_tensor_sum_to_binary_add : public EltwiseSumWithConstantFullTensorFusingTestOneDNN {};
TEST_P(onednn_replace_full_tensor_sum_to_binary_add, basic) {
    auto p = GetParam();
    if (engine.get_device_info().supports_immad)
        p.expected_fused_primitives = p.expected_fused_primitives_onednn;

    create_topologies(
        data("src0", get_mem(get_input_layout(p))),
        input_layout("input", get_weights_layout(p)),  // Input is weights.
        data("eltwise_data", get_mem(layout{ p.out_shape, p.eltw_type, p.eltw_format })),
        convolution("conv_prim", input_info("src0"), { "input" }, {}, p.groups, p.stride, p.dilation, p.pad, p.pad, false),
        eltwise("sum", { input_info("conv_prim"), input_info("eltwise_data") }, eltwise_mode::sum, p.out_type),
        reorder("reorder_bfyx", input_info("sum"), p.default_format, p.default_type)
    );

    tolerance = 0.01f;
    execute(p);
}

// in_shape; out_shape; kernel; stride; pad; dilation; groups; data_type; input_format; weights_type; weights_format; eltw_type; eltw_format; out_type; out_format; default_type; default_format;
#define CASE_CONV_ELTW_SUM_TO_BINARY_ADD { 1, 32, 4, 4 }, { 1, 32, 2, 2 }, { 32, 32, 3, 3 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, data_types::f16, format::bfyx, data_types::f16, format::bfyx, data_types::f16, format::b_fs_yx_fsv16, data_types::f16, format::b_fs_yx_fsv16, data_types::f32, format::bfyx

INSTANTIATE_TEST_SUITE_P(eltwise_sum_fusings_gpu, onednn_replace_full_tensor_sum_to_binary_add, ::testing::ValuesIn(std::vector<convolution_eltw_sum_test_params>{
    convolution_eltw_sum_test_params{ CASE_CONV_ELTW_SUM_TO_BINARY_ADD, 2, 2, 3 },
}));

#endif  // ENABLE_ONEDNN_FOR_GPU
