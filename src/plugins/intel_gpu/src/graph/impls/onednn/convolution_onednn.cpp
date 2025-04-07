// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convolution_onednn.hpp"
#include "convolution_inst.h"
#include "permute_inst.h"
#include "intel_gpu/runtime/format.hpp"
#include "intel_gpu/runtime/layout.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "primitive_onednn_base.h"

#include "utils.hpp"

#include "intel_gpu/runtime/debug_configuration.hpp"

#include <oneapi/dnnl/dnnl.hpp>

#include <algorithm>
#include <memory>
namespace cldnn {
namespace onednn {

static std::shared_ptr<dnnl::convolution_forward::primitive_desc> get_convolution_primitive_descriptor(const kernel_impl_params& impl_params,
                                            const dnnl::primitive_attr& attr = dnnl::primitive_attr(),
                                            dnnl::memory::format_tag tag_in_out = dnnl::memory::format_tag::undef) {
    auto& engine = impl_params.prog->get_engine();
    auto prim = impl_params.typed_desc<convolution>();

    auto input_layout = impl_params.get_input_layout(0);
    auto weights_layout = impl_params.get_input_layout(1);
    auto output_layout = impl_params.get_output_layout();

    dnnl::memory::dims stride(prim->stride.begin(), prim->stride.end());
    dnnl::memory::dims dilation(prim->dilation.begin(), prim->dilation.end());
    dnnl::memory::dims pad_l(prim->padding_begin.begin(), prim->padding_begin.end());
    dnnl::memory::dims pad_r(prim->padding_end.begin(), prim->padding_end.end());

    // issue: it could not find the implementation for 1d kernel GroupConvolution from onednn.
    // root-cause: 3d tensor of input/output is changed to 4d via ngraph.
    //             Creating conv description returns error if two inputs have same tensor of data input and weight.
    //     - original dims of IR
    //       input1: [  1, 280, 1200]      // [number of batches, number of channels, X]
    //       input2: [280,   1,    1, 67]  // [number of output channels, number of input channels, Y, X]
    //       output: [  1, 280, 1200]      // [number of batches, number of kernel output channels, X]
    //     - changed dims
    //       input1: [  1, 280, 1200,  1]
    //       input2: [280,   1,   67,  1]
    //       output: [  1, 280, 1200,  1]
    // WA: Weight tensor will be updated from 4d to 5d.
    auto grouped_weights = format::is_grouped(weights_layout.format) || prim->grouped_weights_shape;
    if (grouped_weights && (input_layout.get_rank() == weights_layout.get_rank())) {
        auto tensor = weights_layout.get_tensor();
        if (tensor.spatial[0] == 1 && tensor.spatial[1] != 1) {
            std::swap(tensor.spatial[0], tensor.spatial[1]);
            weights_layout.set_tensor(tensor);
        }
        weights_layout.format = format::get_default_format(weights_layout.get_rank() + 1, true, true);
    }

    auto input_md = onednn::layout_to_memory_desc(input_layout, tag_in_out);
    auto weights_md = onednn::layout_to_memory_desc(weights_layout, dnnl::memory::format_tag::any);
    auto output_md = onednn::layout_to_memory_desc(output_layout, tag_in_out);

    // adjust_conv_dilation_pad(dilation, stride, pad_l, pad_r, input_md, output_md, weights_md, grouped_weights);
    for (size_t i = 0; i < dilation.size(); i++) {
        dilation[i]--;
        int weights_offset = (grouped_weights ? 3 : 2) + static_cast<int>(i);
        auto os = output_md.get_dims()[2 + i];
        auto is = input_md.get_dims()[2 + i];
        auto ks = weights_md.get_dims()[weights_offset];
        auto kernel_range = 1 + (ks - 1) * (dilation[i] + 1);
        pad_r[i] = (os - 1) * stride[i] - is + kernel_range - pad_l[i];
    }

    // Extend conv parameters in case if spatials rank of output memory doesn't match size of parameters
    int64_t insert_count = static_cast<int64_t>(output_md.get_dims().size()) - 2 - stride.size();
    if (insert_count > 0) {
        stride.insert(stride.end(), insert_count, 1);
        dilation.insert(dilation.end(), insert_count, 0);
        pad_l.insert(pad_l.end(), insert_count, 0);
        pad_r.insert(pad_r.end(), insert_count, 0);
    }

    if (!prim->bias.empty()) {
        auto bias_md = onednn::layout_to_memory_desc(impl_params.get_input_layout(2), dnnl::memory::format_tag::any, true);
        return std::make_shared<dnnl::convolution_forward::primitive_desc>(
            engine.get_onednn_engine(),
            dnnl::prop_kind::forward_inference,
            dnnl::algorithm::convolution_direct,
            input_md,
            weights_md,
            bias_md,
            output_md,
            stride,
            dilation,
            pad_l,
            pad_r,
            attr);
    } else {
        return std::make_shared<dnnl::convolution_forward::primitive_desc>(
            engine.get_onednn_engine(),
            dnnl::prop_kind::forward_inference,
            dnnl::algorithm::convolution_direct,
            input_md,
            weights_md,
            output_md,
            stride,
            dilation,
            pad_l,
            pad_r,
            attr);
    }
}

struct convolution_onednn : typed_primitive_onednn_impl<convolution> {
    using parent = typed_primitive_onednn_impl<convolution>;
    using parent::parent;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::onednn::convolution_onednn)

private:
    int _zero_point_mask;
    dnnl::memory::data_type _wzp_data_type = dnnl::memory::data_type::undef;

protected:
    std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<convolution_onednn>(*this);
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

            // In the case of dynamic model, if choose_impl was executed in runtime,
            // a_zp could be remained as u8 or i8.
            if (a_zp->get_layout().data_type != data_types::i32) {
                auto& conv_node = instance.get_node().as<convolution>();
                auto& a_zp_node = conv_node.activations_zero_points().as<data>();
                a_zp = a_zp_node.get_attached_memory_ptr();
            }

            dnnl::memory::desc desc = onednn::layout_to_memory_desc(a_zp->get_layout(), dnnl::memory::format_tag::a, true);
            args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC, a_zp->get_onednn_memory(desc)});

            GPU_DEBUG_TRACE_DETAIL << instance.id() << " activations_zero_points: "
                << " " << a_zp->get_layout().to_short_string() << std::endl;
        }

        if (instance.weights_zero_points_term()) {
            auto w_zp = instance.weights_zero_points_memory();
            dnnl::memory::desc desc = onednn::layout_to_memory_desc(w_zp->get_layout(), dnnl::memory::format_tag::a, true);
            args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, w_zp->get_onednn_memory(desc)});

            GPU_DEBUG_TRACE_DETAIL << instance.id() << " weights_zero_points: "
                << " " << w_zp->get_layout().to_short_string() << std::endl;
        }

        return args;
    }

    void set_zero_point_mask(int zero_point_mask) {
        _zero_point_mask = zero_point_mask;
    }

    void set_weights_zero_point_data_type(dnnl::memory::data_type data_type) {
        _wzp_data_type = data_type;
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
                                                                            const kernel_impl_params& impl_params,
                                                                            int& zero_point_mask,
                                                                            dnnl::memory::data_type& wzp_data_type) {
        auto attrs = impl_params.attrs_onednn;

        if (arg.activations_zero_points_term()) {
            auto& a_zp = arg.activations_zero_points();
            auto a_zp_dtype = a_zp.get_output_layout().data_type;

            if (!data_type_traits::is_i8_u8(a_zp_dtype) && a_zp_dtype != data_types::i32) {
                throw std::runtime_error("Unsupported data type for activations zero points for oneDNN convolution");
            }

            if (a_zp_dtype == data_types::i8) {
                set_activation_zero_points_attr<ov::element_type_traits<data_types::i8>::value_type>(attrs, a_zp.as<data>(), zero_point_mask);
            } else if (a_zp_dtype == data_types::u8) {
                set_activation_zero_points_attr<ov::element_type_traits<data_types::u8>::value_type>(attrs, a_zp.as<data>(), zero_point_mask);
            } else if (a_zp_dtype == data_types::i32) {
                set_activation_zero_points_attr<ov::element_type_traits<data_types::i32>::value_type>(attrs, a_zp.as<data>(), zero_point_mask);
            }
        }

        if (arg.weights_zero_points_term()) {
            auto& wzp = arg.weights_zero_points();
            auto wzp_layout = wzp.get_output_layout();
            wzp_data_type = convert_data_type(wzp_layout.data_type);
            if (wzp_layout.count() == 1) {
                attrs->set_zero_points(DNNL_ARG_WEIGHTS, 0, dnnl::memory::dims{}, wzp_data_type);
            } else {
                throw std::runtime_error("Convolution oneDNN primitive doesn't support PER_OC weights zero points");
            }
        }

        return attrs;
    }

    static std::shared_ptr<WeightsReorderParams> get_weights_reorder(const kernel_impl_params& impl_params, const dnnl::primitive_desc& pd, bool rotate) {
        auto cldnn_prim = impl_params.typed_desc<convolution>();

        auto source_weights_layout = impl_params.get_input_layout(1);
        auto grouped_weights = format::is_grouped(source_weights_layout.format) || cldnn_prim->grouped_weights_shape;
        auto target_weights_desc = pd.weights_desc(0);

        auto shape_consistent = onednn::keep_weights_reorder_shape_consistent(source_weights_layout, target_weights_desc);
        OPENVINO_ASSERT(shape_consistent, "[GPU] Input shape and output shape of weight reorder should be same.");

        auto source_weights_desc = onednn::layout_to_memory_desc(source_weights_layout);

        const bool weights_format = true;
        auto traits = convert_memory_desc_to_traits(target_weights_desc, weights_format, grouped_weights);

        auto target_weights_layout = source_weights_layout;
        target_weights_layout.format = format(traits);

        return std::make_shared<WeightsReorderParamsOneDNN>(source_weights_layout,
                                                            target_weights_layout,
                                                            source_weights_desc,
                                                            target_weights_desc,
                                                            rotate,
                                                            grouped_weights);
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

        const kernel_impl_params* impl_params = reinterpret_cast<kernel_impl_params*>(ob.getKernelImplParams());
        auto prim = impl_params->typed_desc<convolution>();
        bool has_wzp = !prim->weights_zero_points.empty();
        if (has_wzp) {
            ob << make_data(&_wzp_data_type, sizeof(dnnl::memory::data_type));
        }

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

        auto prim = impl_params->typed_desc<convolution>();
        bool has_wzp = !prim->weights_zero_points.empty();
        if (has_wzp) {
            ib >> make_data(&_wzp_data_type, sizeof(dnnl::memory::data_type));
            _attrs->set_zero_points(DNNL_ARG_WEIGHTS, 0, dnnl::memory::dims{}, _wzp_data_type);
        }

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
        dnnl::memory::data_type wzp_data_type = dnnl::memory::data_type::undef;

        auto attr = get_primitive_attributes(arg, impl_params, zero_point_mask, wzp_data_type);

        auto prim_desc = get_convolution_primitive_descriptor(impl_params, *attr);

        auto conv_onednn_impl = std::make_unique<convolution_onednn>(engine, config, attr, *prim_desc,
                                                get_weights_reorder(impl_params, *prim_desc, arg.get_transposed()));

        conv_onednn_impl->set_zero_point_mask(zero_point_mask);
        conv_onednn_impl->set_weights_zero_point_data_type(wzp_data_type);

        return conv_onednn_impl;
    }
};

std::unique_ptr<primitive_impl> ConvolutionImplementationManager::create_impl(const program_node& node, const kernel_impl_params& params) const {
    assert(node.is_type<convolution>());
    return convolution_onednn::create(static_cast<const convolution_node&>(node), params);
}

in_out_fmts_t ConvolutionImplementationManager::query_formats(const program_node& node) const {
    assert(node.is_type<convolution>());
    std::vector<format::type> in_fmts(node.get_dependencies().size(), format::any);
    std::vector<format::type> out_fmts(node.get_outputs_count(), format::any);

    const auto& conv_node = node.as<convolution>();

    auto prim_desc = get_convolution_primitive_descriptor(*node.get_kernel_impl_params(), dnnl::primitive_attr(), dnnl::memory::format_tag::any);

    for (size_t idx = 0 ; idx < node.get_dependencies().size() ; idx++) {
        if (node.get_dependency(idx).is_constant())
            continue;

        // Conv or deconv gets a preferred format for its data input based on source memory description
        // But an input format for fused post-ops should be same with an output format of conv/deconv
        size_t prim_input = node.get_dependency_index(conv_node.input());
        size_t prim_weights = node.get_primitive()->input_size();

        // Note: did not handle attribute properly. especially for zero-point
        cldnn::format src_fmt = format::any;
        if (idx == prim_input) {
            src_fmt = onednn::find_data_format(prim_desc->src_desc());
        } else if (idx == prim_weights) {
            src_fmt = format::any;
        } else {  // Dep for fused post ops
            src_fmt = onednn::find_data_format(prim_desc->dst_desc());
        }

        // WA: Avoid b_fs_yx_fsv2 because Onednn tag aBcd2b is not declared.
        if (src_fmt == format::b_fs_yx_fsv2)
            src_fmt = format::byxf;

        // WA: shallow convolution needs to set input format by bfyx.
        //     onednn recommended byxf for input format. It will insert reorder before shallow conv.
        if (node.get_input_layout(0).get_partial_shape()[1] == 3) {
            bool can_optimize_permute = false;
            // In permute-conv pattern, check if permute can be optimized
            // when the input memory of permute has been aligned like byxf format.
            // ex) pattern: input (bfyx) -> permute (byxf) -> oneDNN convolution
            //      input layout of permute: bfyx [b:1, f:416, y:416, x:3]
            //     output layout of permute: byxf [b:1, f:3, y:416, x:416]
            // In this case, it can be handled by changing only the shape of permute without the kernel execution.
            if (node.get_output_layout().get_rank() == 4 && node.get_dependency(0).is_type<permute>()) {
                auto& pnode = node.get_dependency(0).as<permute>();
                can_optimize_permute = pnode.get_users().size() == 1
                    && pnode.get_output_layout().data_type == node.get_output_layout().data_type
                    && !pnode.has_fused_primitives()
                    && !pnode.is_output() && pnode.get_input_layout(0).is_static()
                    && pnode.is_reverse_rotating_except_batch();
            }
            if (!can_optimize_permute) {
                src_fmt = format::get_default_format(node.get_input_layout(0).get_rank(), false, false);
            } else {
                // The size of dependencies and users must each be 1.
                // In permute-conv pattern, the preferred format of permute should follow previous node.
                node.get_dependency(0).init_preferred_fmt(1, 1);
                node.get_dependency(0).set_preferred_input_fmt(0, format::bfyx);
                node.get_dependency(0).can_be_optimized(true);
            }
        }

        in_fmts[idx] = src_fmt;
    }

    auto dst_fmt = onednn::find_data_format(prim_desc->dst_desc());
    if (out_fmts[0] == format::any) {
        out_fmts[0] = dst_fmt;
    }

    // WA: Avoid b_fs_yx_fsv2 because Onednn tag aBcd2b is not declared.
    if (out_fmts[0] == format::b_fs_yx_fsv2)
        out_fmts[0] = format::byxf;

    // Errata: Best impl for shallow input conv with zero-point ops is ocl:xe_lp.
    if (in_fmts[0] == format::bfyx) {
        if (conv_node.get_input_layout(0).feature() <= 8 && conv_node.activations_zero_points_term() &&
            conv_node.get_input_layout(0).data_type == data_types::u8 && conv_node.get_output_layout().data_type == data_types::u8) {
            dst_fmt = format::b_fs_yx_fsv32;
        }
    }
    return {in_fmts, out_fmts};
}

}  // namespace onednn
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::onednn::convolution_onednn)
