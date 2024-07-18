// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "deconvolution_inst.h"
#include "impls/onednn/register.hpp"
#include "impls/onednn/utils.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "primitive_onednn_base.h"
#include "impls/registry/implementation_map.hpp"

#include <oneapi/dnnl/dnnl.hpp>

#include <memory>
namespace cldnn {

namespace onednn {

static std::shared_ptr<dnnl::deconvolution_forward::primitive_desc> get_deconvolution_primitive_descriptor(const kernel_impl_params& impl_params,
                                            const dnnl::primitive_attr& attr = dnnl::primitive_attr(),
                                            dnnl::memory::format_tag tag_in_out = dnnl::memory::format_tag::undef) {
    auto& engine = impl_params.prog->get_engine();
    auto prim = impl_params.typed_desc<deconvolution>();

    auto input_layout = impl_params.get_input_layout(0);
    auto weights_layout = impl_params.get_input_layout(1);
    auto output_layout = impl_params.get_output_layout();

    dnnl::memory::dims stride(prim->stride.begin(), prim->stride.end());
    dnnl::memory::dims dilation(input_layout.get_spatial_rank(), 1);
    dnnl::memory::dims pad_l(prim->pad.begin(), prim->pad.end());
    dnnl::memory::dims pad_r(prim->pad.begin(), prim->pad.end());

    auto input_md = onednn::layout_to_memory_desc(input_layout, tag_in_out);
    auto weights_md = onednn::layout_to_memory_desc(weights_layout, dnnl::memory::format_tag::any);
    auto output_md = onednn::layout_to_memory_desc(output_layout, tag_in_out);
    auto grouped_weights = format::is_grouped(weights_layout.format) || prim->grouped_weights_shape;

    for (size_t i = 0; i < dilation.size(); i++) {
        dilation[i]--;
        int weights_offset = (grouped_weights ? 3 : 2) + static_cast<int>(i);
        auto os = output_md.get_dims()[2 + i];
        auto is = input_md.get_dims()[2 + i];
        auto ks = weights_md.get_dims()[weights_offset];
        auto kernel_range = 1 + (ks - 1) * (dilation[i] + 1);
        pad_r[i] = (is - 1) * stride[i] - os + kernel_range - pad_l[i];
    }

    if (!prim->bias.empty()) {
        auto bias_md = onednn::layout_to_memory_desc(impl_params.get_input_layout(2), dnnl::memory::format_tag::any, true);
        return std::make_shared<dnnl::deconvolution_forward::primitive_desc>(
            engine.get_onednn_engine(),
            dnnl::prop_kind::forward_inference,
            dnnl::algorithm::deconvolution_direct,
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
        return std::make_shared<dnnl::deconvolution_forward::primitive_desc>(
            engine.get_onednn_engine(),
            dnnl::prop_kind::forward_inference,
            dnnl::algorithm::deconvolution_direct,
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

struct deconvolution_onednn : typed_primitive_onednn_impl<deconvolution> {
    using parent = typed_primitive_onednn_impl<deconvolution>;
    using parent::parent;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::onednn::deconvolution_onednn)

protected:
    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<deconvolution_onednn>(*this);
    }

    std::unordered_map<int, dnnl::memory> get_arguments(deconvolution_inst& instance) const override {
        std::unordered_map<int, dnnl::memory> args = parent::get_arguments(instance);
        auto& engine = instance.get_network().get_engine();
        auto onednn_engine = engine.get_onednn_engine();

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

        return args;
    }

    static std::shared_ptr<WeightsReorderParams> get_weights_reorder(const kernel_impl_params& impl_params, const dnnl::primitive_desc& pd) {
        auto cldnn_prim = impl_params.typed_desc<deconvolution>();

        auto source_weights_layout = impl_params.get_input_layout(1);
        auto grouped_weights = format::is_grouped(source_weights_layout.format) || cldnn_prim->grouped_weights_shape;
        auto target_weights_desc = pd.weights_desc(0);

        auto shape_consistent = onednn::keep_weights_reorder_shape_consistent(source_weights_layout, target_weights_desc);
        OPENVINO_ASSERT(shape_consistent, "[GPU] Input shape and output shape of weight reorder should be same.");

        auto source_weights_desc = onednn::layout_to_memory_desc(source_weights_layout);

        const bool weights_format = true;
        auto traits = convert_memory_desc_to_traits(target_weights_desc, weights_format, cldnn_prim->grouped_weights_shape);

        auto target_weights_layout = source_weights_layout;
        target_weights_layout.format = format(traits);

        return std::make_shared<WeightsReorderParamsOneDNN>(source_weights_layout,
                                                            target_weights_layout,
                                                            source_weights_desc,
                                                            target_weights_desc,
                                                            false,
                                                            grouped_weights);
    }

public:
    void save(BinaryOutputBuffer& ob) const override {
#ifdef ONEDNN_PRIMITIVE_SERIALIZATION
        parent::save(ob);

        const dnnl::deconvolution_forward::primitive_desc *typed_pd
            = reinterpret_cast<const dnnl::deconvolution_forward::primitive_desc *>(&_pd);

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
            auto prim_desc = std::make_shared<dnnl::deconvolution_forward::primitive_desc>(
                                    ib.get_engine().get_onednn_engine(),
                                    dnnl::prop_kind::forward_inference, dnnl::algorithm::deconvolution_direct,
                                    input_md, weights_md, output_md,
                                    strides, dilates, padding_l, padding_r,
                                    *_attrs.get());
            _pd = *prim_desc;
        } else {
            auto bias_md = onednn::layout_to_memory_desc(impl_params->get_input_layout(2), dnnl::memory::format_tag::any, true);
            auto prim_desc = std::make_shared<dnnl::deconvolution_forward::primitive_desc>(
                                    ib.get_engine().get_onednn_engine(),
                                    dnnl::prop_kind::forward_inference, dnnl::algorithm::deconvolution_direct,
                                    input_md, weights_md, bias_md, output_md,
                                    strides, dilates, padding_l, padding_r,
                                    *_attrs.get());
            _pd = *prim_desc;
        }

        std::vector<uint8_t> prim_cache;
        ib >> prim_cache;

        _scratchpad_md = _pd.scratchpad_desc();

        _prim = dnnl::primitive(_pd, prim_cache);
#endif
    }

    static bool validate(const deconvolution_node& node) {
        if (!is_supported_format(node.get_preferred_input_fmt(0)))
            return false;

        const auto& input_layout = node.get_input_layout(0);
        auto in_dt = input_layout.data_type;
        auto wei_dt = node.weights().get_output_layout().data_type;
        auto out_dt = node.get_output_layout(false).data_type;

        const auto& prim = node.get_primitive();

        if (prim->groups != 1)
            return false;

        auto spatial_dims_num = input_layout.get_spatial_rank();

        if (spatial_dims_num > 3)
            return false;

        bool f16_deconv = everyone_is(data_types::f16, in_dt, wei_dt) && one_of(out_dt, {data_types::f16, data_types::u8, data_types::i8});
        bool f32_deconv = everyone_is(data_types::f32, in_dt, wei_dt) && one_of(out_dt, {data_types::u8, data_types::i8});
        bool u8s8_deconv = one_of(in_dt, {data_types::i8, data_types::u8}) &&
                           wei_dt == data_types::i8 &&
                           one_of(out_dt, {data_types::i32, data_types::f16, data_types::f32, data_types::u8, data_types::i8});

        if (!f16_deconv && !f32_deconv && !u8s8_deconv)
            return false;

        if (!is_supported_post_ops(node))
            return false;

        return true;
    }

    static std::unique_ptr<primitive_impl> create(const deconvolution_node& arg, const kernel_impl_params& impl_params) {
        auto& engine = impl_params.prog->get_engine();
        auto& config = impl_params.prog->get_config();
        auto attr = impl_params.attrs_onednn;
        auto prim_desc = get_deconvolution_primitive_descriptor(impl_params, *attr);

        return cldnn::make_unique<deconvolution_onednn>(engine, config, attr, *prim_desc, get_weights_reorder(impl_params, *prim_desc));
    }
};

struct deconvolution_factory : public cldnn::implementation_factory<deconvolution> {
    std::unique_ptr<primitive_impl> create(const program_node& node, const kernel_impl_params& params) const override {
        OPENVINO_ASSERT(node.is_type<deconvolution>());
        return onednn::deconvolution_onednn::create(static_cast<const deconvolution_node&>(node), params);
    }

    bool validate(const program_node& node) const override {
        OPENVINO_ASSERT(node.is_type<deconvolution>());
        return onednn::deconvolution_onednn::validate(static_cast<const deconvolution_node&>(node));
    }

    in_out_fmts_t query_formats(const program_node& node) const override {
        OPENVINO_ASSERT(node.is_type<deconvolution>());
        std::vector<format::type> in_fmts(node.get_dependencies().size(), format::any);
        std::vector<format::type> out_fmts(node.get_outputs_count(), format::any);

        const auto& deconv_node = node.as<deconvolution>();
        auto prim_desc = onednn::get_deconvolution_primitive_descriptor(*node.get_kernel_impl_params(), dnnl::primitive_attr(), dnnl::memory::format_tag::any);

        for (size_t idx = 0 ; idx < node.get_dependencies().size() ; idx++) {
            if (node.get_dependency(idx).is_constant())
                continue;

            // Conv or deconv gets a preferred format for its data input based on source memory description
            // But an input format for fused post-ops should be same with an output format of conv/deconv
            size_t prim_input = node.get_dependency_index(deconv_node.input());

            // Note: did not handle attribute properly. especially for zero-point
            cldnn::format src_fmt = format::any;
            if (idx == prim_input)
                src_fmt = onednn::find_data_format(prim_desc->src_desc());
            else  // Dep for fused post ops
                src_fmt = onednn::find_data_format(prim_desc->dst_desc());

            in_fmts[idx] = src_fmt;

            auto dst_fmt = onednn::find_data_format(prim_desc->dst_desc());

            if (out_fmts[0] == format::any) {
                out_fmts[0] = dst_fmt;
            }

            GPU_DEBUG_LOG << "select_preferred_formats:" << node.id() << ": " << fmt_to_str(src_fmt) << " --> " << fmt_to_str(dst_fmt)
                          << " For index : " << idx << std::endl;
        }

        return {in_fmts, out_fmts};
    }
};

namespace detail {

attach_deconvolution_onednn::attach_deconvolution_onednn() {
    std::vector<data_types> dt = {
        data_types::f32,
        data_types::f16,
        data_types::u8,
        data_types::i8,
    };
    std::vector<format::type> fmt = {
        format::bfyx,
        format::byxf,
        format::b_fs_yx_fsv16,
        format::b_fs_yx_fsv32,
        format::b_fs_zyx_fsv32,
        format::bs_fs_yx_bsv16_fsv16,
        format::bs_fs_yx_bsv16_fsv32,
        format::bs_fs_yx_bsv32_fsv16,
        format::bs_fs_yx_bsv32_fsv32,
        format::bs_fs_yx_bsv4_fsv4,
        format::bs_fs_yx_bsv8_fsv4,
        format::bs_fs_yx_bsv8_fsv2,
        format::bs_fs_yx_bsv4_fsv2,
    };

    implementation_map<deconvolution>::add(impl_types::onednn, shape_types::static_shape, cldnn::make_unique<deconvolution_factory>(), dt, fmt);
}

}  // namespace detail
}  // namespace onednn
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::onednn::deconvolution_onednn)
