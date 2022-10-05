// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convolution_inst.h"
#include "eltwise_inst.h"
#include "quantize_inst.h"
#include "primitive_onednn_base.h"
#include "impls/implementation_map.hpp"

#include "kernel_selector_common.h"

#include "utils.hpp"

#include "intel_gpu/runtime/debug_configuration.hpp"

#include <oneapi/dnnl/dnnl.hpp>

#include <algorithm>
#include <memory>
#include "convolution_onednn.hpp"
namespace cldnn {
namespace onednn {

struct convolution_onednn : typed_primitive_onednn_impl<convolution, dnnl::convolution_forward::desc> {
    using parent = typed_primitive_onednn_impl<convolution, dnnl::convolution_forward::desc>;
    using parent::parent;

protected:
    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<convolution_onednn>(*this);
    }

    bool validate_impl(const typed_primitive_inst<convolution>& instance) const override {
        bool res = true;

        auto outer_id = instance.id();
        auto data_type = instance.node.input().get_output_layout().data_type;

        // Integer signed/unsigned is ok for convoluiton
        CLDNN_ERROR_DATA_TYPES_MISMATCH_IGNORE_SIGN(outer_id,
                                                    "Input memory",
                                                    data_type,
                                                    "filter memory",
                                                    instance.weights_memory(0)->get_layout().data_type,
                                                    "");

        return res;
    }

    std::unordered_map<int, dnnl::memory> get_arguments(convolution_inst& instance) const override {
        std::unordered_map<int, dnnl::memory> args = parent::get_arguments(instance);
        auto attrs = instance.get_node().get_onednn_primitive_attributes();

        {
            auto weights = instance.weights_memory(0);
            args.insert({DNNL_ARG_WEIGHTS, weights->get_onednn_memory(_pd.weights_desc(0))});
        }

        if (instance.bias_term()) {
            auto bias = instance.bias_memory(0);
            args.insert({DNNL_ARG_BIAS, bias->get_onednn_memory(_pd.weights_desc(1))});
        }

        if (has_zero_points(DNNL_ARG_SRC, attrs)) {
            auto a_zp = instance.activations_zero_points_memory(0);
            dnnl::memory::desc desc = onednn::layout_to_memory_desc(a_zp->get_layout(), dnnl::memory::format_tag::a, true);
            args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC, a_zp->get_onednn_memory(desc)});

            GPU_DEBUG_GET_INSTANCE(debug_config);
            GPU_DEBUG_IF(debug_config->verbose >= 3) {
                auto dnnl_mem = a_zp->get_onednn_memory(desc);
                void *mapped_ptr = dnnl_mem.map_data();
                if (mapped_ptr) {
                    GPU_DEBUG_COUT << instance.get_node().id() << " activations_zero_points: ";
                    for (size_t i = 0; i < desc.get_size(); ++i) {
                        std::cout << static_cast<int32_t*>(mapped_ptr)[i] << " ";
                    }
                    std::cout << std::endl;
                    dnnl_mem.unmap_data(mapped_ptr);
                }
            }
        }

        if (has_zero_points(DNNL_ARG_WEIGHTS, attrs)) {
            auto w_zp = instance.weights_zero_points_memory(0);
            dnnl::memory::desc desc = onednn::layout_to_memory_desc(w_zp->get_layout(), dnnl::memory::format_tag::a, true);
            args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, w_zp->get_onednn_memory(desc)});

            GPU_DEBUG_GET_INSTANCE(debug_config);
            GPU_DEBUG_IF(debug_config->verbose >= 3) {
                auto dnnl_mem = w_zp->get_onednn_memory(desc);
                void *mapped_ptr = dnnl_mem.map_data();
                if (mapped_ptr) {
                    GPU_DEBUG_COUT << instance.get_node().id() << " weights_zero_points: ";
                    for (size_t i = 0; i < desc.get_size(); ++i) {
                        std::cout << static_cast<int32_t*>(mapped_ptr)[i] << " ";
                    }
                    std::cout << std::endl;
                    dnnl_mem.unmap_data(mapped_ptr);
                }
            }
        }

        return args;
    }

    template <typename T>
    static void set_activation_zero_points_attr(const std::shared_ptr<dnnl::primitive_attr>& attrs, cldnn::data_node& node) {
        int32_t zp_val = DNNL_RUNTIME_S32_VAL;
        bool is_per_tensor = onednn::is_per_tensor<T>(node, zp_val);
        if (is_per_tensor) {
            attrs->set_zero_points(DNNL_ARG_SRC, 0, {zp_val});
        } else {
            memory::ptr s32_mem = onednn::convert_zp_data_to_s32<T>(node.get_attached_memory_ptr());
            node.attach_memory(s32_mem, false);
            attrs->set_zero_points(DNNL_ARG_SRC, 2, {DNNL_RUNTIME_S32_VAL});
        }
    }

    static std::shared_ptr<dnnl::primitive_attr> get_primitive_attributes(const typed_program_node<convolution>& arg) {
        auto attrs = arg.get_onednn_primitive_attributes();

        if (arg.activations_zero_points_term()) {
            auto& a_zp = arg.activations_zero_points();
            auto a_zp_dtype = a_zp.get_output_layout().data_type;

            if (!data_type_traits::is_i8_u8(a_zp_dtype)) {
                throw std::runtime_error("Unsupported data type for activations zero points for oneDNN convolution");
            }

            if (a_zp_dtype == data_types::i8) {
                set_activation_zero_points_attr<data_type_to_type<data_types::i8>::type>(attrs, a_zp.as<data>());
            } else { // if (a_zp_dtype == data_types::u8)
                set_activation_zero_points_attr<data_type_to_type<data_types::u8>::type>(attrs, a_zp.as<data>());
            }
        }

        if (arg.weights_zero_points_term()) {
            throw std::runtime_error("Convolution oneDNN primitive doesn't support asymmetric weights quantization");

            // Commented out since oneDNN doesn't support asymmetric weights quantization
            // auto& w_zp = arg.weights_zero_points();
            // int mask = w_zp.get_output_layout().count() > 1 ? 2 : 0;
            // attrs->set_zero_points(DNNL_ARG_WEIGHTS, mask, {DNNL_RUNTIME_S32_VAL});
        }

        return attrs;
    }

    static kernel_selector::WeightsReorderParams get_weights_reorder(const kernel_impl_params& impl_params, const dnnl::primitive_desc& pd) {
        kernel_selector::WeightsReorderParams weights_reorder_params;
        auto& reorderKS = kernel_selector::ReorderWeightsKernelSelctor::Instance();
        kernel_selector::reorder_weights_params r_params;

        auto cldnn_prim = impl_params.typed_desc<convolution>();
        auto weights_layout = impl_params.get_input_layout(1);
        auto grouped_weights = format::is_grouped(weights_layout.format) || cldnn_prim->grouped_weights_shape;
        cldnn::format out_fmt = onednn::find_format(pd.weights_desc(0), grouped_weights);
        kernel_selector::WeightsLayout reqLayout = to_weights_layout(out_fmt, cldnn_prim->grouped_weights_shape);

        set_params(impl_params, r_params);
        r_params.layerID = cldnn_prim->id + "_reorder_";
        r_params.input = convert_weights_tensor(weights_layout, cldnn_prim->grouped_weights_shape);
        r_params.output = r_params.input.TransformIgnorePadding(reqLayout, r_params.input.GetDType(), cldnn_prim->groups, false);
        r_params.rotate_180 = false;

        kernel_selector::reorder_optional_params op;
        kernel_selector::KernelsData kernels_data = reorderKS.GetBestKernels(r_params, op);

        if (kernels_data.empty()) {
            throw std::runtime_error("No suitable kernel found for weights reorder from " +
                                        kernel_selector::toString(r_params.input.GetLayout()) + " to " +
                                        kernel_selector::toString(r_params.output.GetLayout()));
        }

        weights_reorder_params.engine = kernel_selector::WeightsReorderParams::Engine::GPU;
        weights_reorder_params.clKernel = std::make_shared<kernel_selector::clKernelData>(kernels_data[0].kernels[0]);
        weights_reorder_params.dest = r_params.output;

        return weights_reorder_params;
    }



public:
    static primitive_impl* create(const convolution_node& arg, const kernel_impl_params& impl_params) {
        auto& engine = impl_params.prog.get_engine();
        auto desc = get_convolution_descriptor(impl_params);
        auto attr = get_primitive_attributes(arg);
        dnnl::primitive_desc prim_desc{&desc->data, attr.get(), engine.get_onednn_engine(), nullptr};

        return new convolution_onednn(engine, desc, attr, prim_desc, get_weights_reorder(impl_params, prim_desc));
    }
};

namespace detail {

attach_convolution_onednn::attach_convolution_onednn() {
    std::vector<data_types> dt = {
        data_types::f32,
        data_types::f16,
        data_types::u8,
        data_types::i8,
    };
    std::vector<format::type> fmt = {
        format::bfyx,
        format::bfzyx,
        format::byxf,
        format::bzyxf,
        format::b_fs_yx_fsv2,
        format::b_fs_zyx_fsv2,
        format::b_fs_yx_fsv4,
        format::b_fs_zyx_fsv4,
        format::b_fs_yx_fsv16,
        format::b_fs_zyx_fsv16,
        format::b_fs_zyx_fsv32,
        format::b_fs_yx_fsv32,
        format::bs_fs_yx_bsv16_fsv16,
        format::bs_fs_zyx_bsv16_fsv16,
        format::bs_fs_yx_bsv32_fsv16,
        format::bs_fs_zyx_bsv32_fsv16,
        format::bs_fs_yx_bsv32_fsv32,
        format::bs_fs_zyx_bsv32_fsv32,
        format::bs_fs_yx_bsv4_fsv4,
        format::bs_fs_yx_bsv8_fsv4,
        format::bs_fs_yx_bsv16_fsv4,
        format::bs_fs_yx_bsv16_fsv2,
        format::bs_fs_zyx_bsv8_fsv4,
        format::bs_fs_zyx_bsv16_fsv4,
        format::bs_fs_zyx_bsv16_fsv2,
        format::bs_fs_yx_bsv8_fsv2,
        format::bs_fs_zyx_bsv8_fsv2,
        format::bs_fs_yx_bsv4_fsv2,
    };
    implementation_map<convolution>::add(impl_types::onednn, convolution_onednn::create, dt, fmt);
}

}  // namespace detail
}  // namespace onednn
}  // namespace cldnn
