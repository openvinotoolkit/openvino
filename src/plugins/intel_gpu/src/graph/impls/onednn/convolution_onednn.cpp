// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convolution_inst.h"
#include "eltwise_inst.h"
#include "quantize_inst.h"
#include "primitive_onednn_base.h"
#include "implementation_map.hpp"

#include "impls/ocl/kernel_selector_helper.h"

#include "utils.hpp"

#include "intel_gpu/runtime/debug_configuration.hpp"

#include <oneapi/dnnl/dnnl.hpp>

#include <algorithm>
#include <memory>
#include "convolution_onednn.hpp"
namespace cldnn {
namespace onednn {

struct convolution_onednn : typed_primitive_onednn_impl<convolution> {
    using parent = typed_primitive_onednn_impl<convolution>;
    using parent::parent;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::onednn::convolution_onednn)

protected:
    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<convolution_onednn>(*this);
    }

    std::unordered_map<int, dnnl::memory> get_arguments(convolution_inst& instance) const override {
        std::unordered_map<int, dnnl::memory> args = parent::get_arguments(instance);

        {
            auto weights = instance.weights_memory();
            auto offset = onednn::get_offset(instance.get_input_layout(1), _pd.dnnl::primitive_desc_base::weights_desc(0));
            args.insert({DNNL_ARG_WEIGHTS, weights->get_onednn_memory(_pd.weights_desc(0), offset)});
        }

        if (instance.bias_term()) {
            auto bias = instance.bias_memory();
            auto offset = onednn::get_offset(instance.get_input_layout(2), _pd.dnnl::primitive_desc_base::weights_desc(1));
            args.insert({DNNL_ARG_BIAS, bias->get_onednn_memory(_pd.weights_desc(1), offset)});
        }

        if (instance.activations_zero_points_term()) {
            auto a_zp = instance.activations_zero_points_memory();
            dnnl::memory::desc desc = onednn::layout_to_memory_desc(a_zp->get_layout(), dnnl::memory::format_tag::a, true);
            args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC, a_zp->get_onednn_memory(desc)});

            GPU_DEBUG_GET_INSTANCE(debug_config);
            GPU_DEBUG_IF(debug_config->verbose >= static_cast<int>(ov::intel_gpu::LogLevel::TRACE_DETAIL)) {
                auto dnnl_mem = a_zp->get_onednn_memory(desc);
                void *mapped_ptr = dnnl_mem.map_data();
                if (mapped_ptr) {
                    GPU_DEBUG_TRACE_DETAIL << instance.id() << " activations_zero_points: ";
                    for (size_t i = 0; i < desc.get_size(); ++i) {
                        GPU_DEBUG_TRACE_DETAIL << static_cast<int32_t*>(mapped_ptr)[i] << " ";
                    }
                    GPU_DEBUG_TRACE_DETAIL << std::endl;
                    dnnl_mem.unmap_data(mapped_ptr);
                }
            }
        }

        if (instance.weights_zero_points_term()) {
            throw std::runtime_error("Convolution oneDNN primitive doesn't support asymmetric weights quantization");
            // auto w_zp = instance.weights_zero_points_memory();
            // dnnl::memory::desc desc = onednn::layout_to_memory_desc(w_zp->get_layout(), dnnl::memory::format_tag::a, true);
            // args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, w_zp->get_onednn_memory(desc)});

            // GPU_DEBUG_GET_INSTANCE(debug_config);
            // GPU_DEBUG_IF(debug_config->verbose >= static_cast<int>(ov::intel_gpu::LogLevel::TRACE_DETAIL)) {
            //     auto dnnl_mem = w_zp->get_onednn_memory(desc);
            //     void *mapped_ptr = dnnl_mem.map_data();
            //     if (mapped_ptr) {
            //         GPU_DEBUG_TRACE_DETAIL << instance.id() << " weights_zero_points: ";
            //         for (size_t i = 0; i < desc.get_size(); ++i) {
            //             GPU_DEBUG_TRACE_DETAIL << static_cast<int32_t*>(mapped_ptr)[i] << " ";
            //         }
            //         GPU_DEBUG_TRACE_DETAIL << std::endl;
            //         dnnl_mem.unmap_data(mapped_ptr);
            //     }
            // }
        }

        return args;
    }

    int _zero_point_mask;
    void set_zero_point_mask(int zero_point_mask) {
        _zero_point_mask = zero_point_mask;
    }

    template <typename T>
    static void set_activation_zero_points_attr(const std::shared_ptr<dnnl::primitive_attr>& attrs,
                                                cldnn::data_node& node, int& zero_point_mask) {
        int32_t zp_val = DNNL_RUNTIME_S32_VAL;
        bool is_per_tensor = onednn::is_per_tensor<T>(node, zp_val);
        memory::ptr s32_mem = onednn::convert_zp_data_to_s32<T>(node.get_attached_memory_ptr());
        node.attach_memory(s32_mem, false);
        zero_point_mask = is_per_tensor ? 0 : 2;
        attrs->set_zero_points_mask(DNNL_ARG_SRC, zero_point_mask);
    }

    static std::shared_ptr<dnnl::primitive_attr> get_primitive_attributes(const typed_program_node<convolution>& arg,
                                                                          int& zero_point_mask) {
        auto attrs = arg.get_onednn_primitive_attributes();

        if (arg.activations_zero_points_term()) {
            auto& a_zp = arg.activations_zero_points();
            auto a_zp_dtype = a_zp.get_output_layout().data_type;

            if (!data_type_traits::is_i8_u8(a_zp_dtype)) {
                throw std::runtime_error("Unsupported data type for activations zero points for oneDNN convolution");
            }

            if (a_zp_dtype == data_types::i8) {
                set_activation_zero_points_attr<ov::element_type_traits<data_types::i8>::value_type>(attrs, a_zp.as<data>(), zero_point_mask);
            } else { // if (a_zp_dtype == data_types::u8)
                set_activation_zero_points_attr<ov::element_type_traits<data_types::u8>::value_type>(attrs, a_zp.as<data>(), zero_point_mask);
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

    static std::shared_ptr<WeightsReorderParams> get_weights_reorder(const kernel_impl_params& impl_params, const dnnl::primitive_desc& pd, bool rotate) {
        auto cldnn_prim = impl_params.typed_desc<convolution>();

        auto input_weights_layout = impl_params.get_input_layout(1);
        auto grouped_weights = format::is_grouped(input_weights_layout.format) || cldnn_prim->grouped_weights_shape;
        format out_fmt = onednn::find_format(pd.weights_desc(0), grouped_weights);

        auto output_weights_layout = input_weights_layout;
        output_weights_layout.format = out_fmt;

        return std::make_shared<WeightsReorderParams>(input_weights_layout, output_weights_layout, rotate, grouped_weights);
    }

public:
    void save(BinaryOutputBuffer& ob) const override {
#ifdef ONEDNN_PRIMITIVE_SERIALIZATION
        parent::save(ob);

        ob << _zero_point_mask;

        const dnnl::convolution_forward::primitive_desc *typed_pd
            = reinterpret_cast<const dnnl::convolution_forward::primitive_desc *>(&_pd);

        ob << typed_pd->get_strides();
        ob << typed_pd->get_dilations();
        ob << typed_pd->get_padding_l();
        ob << typed_pd->get_padding_r();
        ob << typed_pd->bias_desc().is_zero();

        std::vector<uint8_t> prim_cache;
        prim_cache = _prim.get_cache_blob();
        ob << prim_cache;
#endif
    }

    void load(BinaryInputBuffer& ib) override {
#ifdef ONEDNN_PRIMITIVE_SERIALIZATION
        parent::load(ib);

        ib >> _zero_point_mask;
        if (_zero_point_mask != -1) {
            _attrs->set_zero_points_mask(DNNL_ARG_SRC, _zero_point_mask);
        }

        const kernel_impl_params* impl_params = reinterpret_cast<kernel_impl_params*>(ib.getKernelImplParams());

        auto input_md = onednn::layout_to_memory_desc(impl_params->get_input_layout(0), dnnl::memory::format_tag::undef);
        auto weights_md = onednn::layout_to_memory_desc(impl_params->get_input_layout(1), dnnl::memory::format_tag::any);
        auto output_md = onednn::layout_to_memory_desc(impl_params->get_output_layout(), dnnl::memory::format_tag::undef);

        dnnl::memory::dims strides;
        dnnl::memory::dims dilates;
        dnnl::memory::dims padding_l;
        dnnl::memory::dims padding_r;
        ib >> strides;
        ib >> dilates;
        ib >> padding_l;
        ib >> padding_r;

        bool zero_bias;
        ib >> zero_bias;

        if (zero_bias) {
            auto prim_desc = std::make_shared<dnnl::convolution_forward::primitive_desc>(
                                    ib.get_engine().get_onednn_engine(),
                                    dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_direct,
                                    input_md, weights_md, output_md,
                                    strides, dilates, padding_l, padding_r,
                                    *_attrs.get());
            _pd = *prim_desc;
        } else {
            auto bias_md = onednn::layout_to_memory_desc(impl_params->get_input_layout(2), dnnl::memory::format_tag::any, true);
            auto prim_desc = std::make_shared<dnnl::convolution_forward::primitive_desc>(
                                    ib.get_engine().get_onednn_engine(),
                                    dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_direct,
                                    input_md, weights_md, bias_md, output_md,
                                    strides, dilates, padding_l, padding_r,
                                    *_attrs.get());
            _pd = *prim_desc;
        }

        _scratchpad_md = _pd.scratchpad_desc();

        std::vector<uint8_t> prim_cache;
        ib >> prim_cache;

        _prim = dnnl::primitive(_pd, prim_cache);
#endif
    }

    static std::unique_ptr<primitive_impl> create(const convolution_node& arg, const kernel_impl_params& impl_params) {
        auto& engine = impl_params.prog->get_engine();
        auto& config = impl_params.prog->get_config();
        int zero_point_mask = -1;
        auto attr = get_primitive_attributes(arg, zero_point_mask);

        auto prim_desc = get_convolution_primitive_descriptor(impl_params, *attr);

        auto conv_onednn_impl = cldnn::make_unique<convolution_onednn>(engine, config, attr, *prim_desc,
                                                get_weights_reorder(impl_params, *prim_desc, arg.get_transposed()));
        conv_onednn_impl->set_zero_point_mask(zero_point_mask);
        return conv_onednn_impl;
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
        format::b_fs_yx_fsv8,
        format::b_fs_zyx_fsv8,
        format::b_fs_yx_fsv16,
        format::b_fs_zyx_fsv16,
        format::b_fs_zyx_fsv32,
        format::b_fs_yx_fsv32,
        format::bs_fs_yx_bsv16_fsv16,
        format::bs_fs_zyx_bsv16_fsv16,
        format::bs_fs_yx_bsv16_fsv32,
        format::bs_fs_zyx_bsv16_fsv32,
        format::bs_fs_yx_bsv32_fsv16,
        format::bs_fs_zyx_bsv32_fsv16,
        format::bs_fs_yx_bsv32_fsv32,
        format::bs_fs_zyx_bsv32_fsv32,
        format::bs_fs_yx_bsv4_fsv4,
        format::bs_fs_yx_bsv8_fsv4,
        format::bs_fs_yx_bsv16_fsv8,
        format::bs_fs_yx_bsv16_fsv4,
        format::bs_fs_yx_bsv16_fsv2,
        format::bs_fs_zyx_bsv8_fsv4,
        format::bs_fs_zyx_bsv16_fsv8,
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

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::onednn::convolution_onednn)
