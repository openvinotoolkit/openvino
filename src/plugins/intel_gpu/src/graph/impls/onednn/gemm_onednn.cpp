// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gemm_inst.h"
#include "primitive_onednn_base.h"
#include "implementation_map.hpp"

#include "kernel_selector_common.h"

#include <oneapi/dnnl/dnnl.hpp>

#include <algorithm>
#include <memory>
namespace cldnn {
namespace onednn {

struct gemm_onednn : typed_primitive_onednn_impl<gemm> {
    using parent = typed_primitive_onednn_impl<gemm>;
    using parent::parent;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::onednn::gemm_onednn)

protected:
    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<gemm_onednn>(*this);
    }

    std::unordered_map<int, dnnl::memory> get_arguments(gemm_inst& instance) const override {
        std::unordered_map<int, dnnl::memory> args = parent::get_arguments(instance);
        auto& engine = instance.get_network().get_engine();
        auto dnnl_engine = engine.get_onednn_engine();

        {
            auto& weights = instance.input_memory(1);
            auto offset = onednn::get_offset(instance.get_input_layout(1), _pd.dnnl::primitive_desc_base::weights_desc(0));
            args.insert({DNNL_ARG_WEIGHTS, weights.get_onednn_memory(_pd.weights_desc(0), offset)});
        }

        if (instance.inputs_memory_count() == 3) {
            auto& weights = instance.input_memory(2);
            auto offset = onednn::get_offset(instance.get_input_layout(2), _pd.dnnl::primitive_desc_base::weights_desc(1));
            args.insert({DNNL_ARG_BIAS, weights.get_onednn_memory(_pd.weights_desc(1), offset)});
        }

        return args;
    }

    static dnnl::memory::format_tag transpose_format(dnnl::memory::format_tag fmt) {
        switch (fmt) {
            case dnnl::memory::format_tag::ab: return dnnl::memory::format_tag::ba;
            case dnnl::memory::format_tag::abc: return dnnl::memory::format_tag::acb;
            case dnnl::memory::format_tag::abcd: return dnnl::memory::format_tag::abdc;
            default: throw std::runtime_error("Unsupported fmt in transpose_format gemm function");
        }
    }

    static void get_gemm_primitive_md(const kernel_impl_params& impl_params,
                                      dnnl::memory::data_type& in0_dt,
                                      dnnl::memory::data_type& in1_dt,
                                      dnnl::memory::data_type& out_dt,
                                      dnnl::memory::dims& in0_dims,
                                      dnnl::memory::dims& in1_dims,
                                      dnnl::memory::dims& out_dims,
                                      dnnl::memory::format_tag& in0_fmt,
                                      dnnl::memory::format_tag& in1_fmt,
                                      dnnl::memory::format_tag& out_fmt,
                                      bool gemm_with_bias,
                                      dnnl::memory::data_type& bias_dt,
                                      dnnl::memory::dims& bias_dims,
                                      dnnl::memory::format_tag& bias_fmt) {
        auto prim = impl_params.typed_desc<gemm>();
        auto out_l = impl_params.get_output_layout();

        std::vector<layout> in_layouts { impl_params.get_input_layout(0), impl_params.get_input_layout(1) };
        if (gemm_with_bias) {
            in_layouts.emplace_back(impl_params.get_input_layout(2));
        }

        in_layouts = gemm_inst::transform_input_layouts(prim, in_layouts);
        out_l = gemm_inst::transform_output_layout(prim, in_layouts, out_l);

        const auto& in0_l = in_layouts[0];
        const auto& in1_l = in_layouts[1];

        size_t in0_batched_size = in0_l.count() / (in0_l.spatial(0) * in0_l.spatial(1));
        size_t in1_batched_size = in1_l.count() / (in1_l.spatial(0) * in1_l.spatial(1));
        size_t out_batched_size = out_l.count() / (out_l.spatial(0) * out_l.spatial(1));

        auto batched_dims_can_be_removed = in0_batched_size == 1 && in1_batched_size == 1 && out_batched_size == 1;
        if (gemm_with_bias) {
            const auto& bias_l = in_layouts[2];
            size_t bias_batched_size = bias_l.count() / (bias_l.spatial(0) * bias_l.spatial(1));
            batched_dims_can_be_removed &= bias_batched_size == 1;
        }

        size_t rank = cldnn::format::dimension(out_l.format);

        in0_dt = onednn::convert_data_type(in0_l.data_type);
        in1_dt = onednn::convert_data_type(in1_l.data_type);
        out_dt = onednn::convert_data_type(out_l.data_type);

        in0_dims = onednn::convert_gemm_tensor(in0_l.get_tensor(), rank, batched_dims_can_be_removed);
        in1_dims = onednn::convert_gemm_tensor(in1_l.get_tensor(), rank, batched_dims_can_be_removed);
        out_dims = onednn::convert_gemm_tensor(out_l.get_tensor(), rank, batched_dims_can_be_removed);

        in0_fmt = onednn::convert_gemm_data_format(in0_dims, in0_l.format);
        in1_fmt = onednn::convert_gemm_data_format(in1_dims, in1_l.format);
        out_fmt = onednn::convert_gemm_data_format(out_dims, out_l.format);

        if (prim->transpose_input0) {
            in0_fmt = transpose_format(in0_fmt);
            std::swap(in0_dims[in0_dims.size() - 1], in0_dims[in0_dims.size() - 2]);
        }

        if (prim->transpose_input1) {
            in1_fmt = transpose_format(in1_fmt);
            std::swap(in1_dims[in1_dims.size() - 1], in1_dims[in1_dims.size() - 2]);
        }

        if (gemm_with_bias) {
            auto bias_l = impl_params.get_input_layout(2);
            auto bias_rank = cldnn::format::dimension(bias_l.format);
            bias_dt = onednn::convert_data_type(bias_l.data_type);
            bias_dims = onednn::convert_gemm_tensor(bias_l.get_tensor(), bias_rank, batched_dims_can_be_removed);
            bias_fmt = onednn::convert_gemm_data_format(bias_dims, bias_l.format);
        }
    }

    static std::shared_ptr<dnnl::matmul::primitive_desc> get_gemm_primitive_descriptor(const kernel_impl_params& impl_params,
                                                                                       const dnnl::primitive_attr& attr = dnnl::primitive_attr()) {
        auto& engine = impl_params.prog->get_engine();
        auto prim = impl_params.typed_desc<gemm>();
        auto gemm_with_bias = prim->dependencies().size() == 3;

        dnnl::memory::data_type in0_dt;
        dnnl::memory::data_type in1_dt;
        dnnl::memory::data_type out_dt;
        dnnl::memory::data_type bias_dt;

        dnnl::memory::dims in0_dims;
        dnnl::memory::dims in1_dims;
        dnnl::memory::dims out_dims;
        dnnl::memory::dims bias_dims;

        dnnl::memory::format_tag in0_fmt;
        dnnl::memory::format_tag in1_fmt;
        dnnl::memory::format_tag out_fmt;
        dnnl::memory::format_tag bias_fmt;

        get_gemm_primitive_md(impl_params, in0_dt, in1_dt, out_dt, in0_dims, in1_dims, out_dims, in0_fmt, in1_fmt, out_fmt,
                              gemm_with_bias, bias_dt, bias_dims, bias_fmt);

        dnnl::memory::desc in0_md(in0_dims, in0_dt, in0_fmt);
        dnnl::memory::desc in1_md(in1_dims, in1_dt, in1_fmt);
        dnnl::memory::desc out_md(out_dims, out_dt, out_fmt);

        if (gemm_with_bias) {
            dnnl::memory::desc bias_md(bias_dims, bias_dt, bias_fmt);

            return std::make_shared<dnnl::matmul::primitive_desc>(
                engine.get_onednn_engine(),
                in0_md,
                in1_md,
                bias_md,
                out_md,
                attr);
        } else {
            return std::make_shared<dnnl::matmul::primitive_desc>(
                engine.get_onednn_engine(),
                in0_md,
                in1_md,
                out_md,
                attr);
        }
    }

public:
    void save(BinaryOutputBuffer& ob) const override {
#ifdef ONEDNN_PRIMITIVE_SERIALIZATION
        parent::save(ob);

        const kernel_impl_params* impl_params = reinterpret_cast<kernel_impl_params*>(ob.getKernelImplParams());
        auto prim = impl_params->typed_desc<gemm>();
        bool gemm_with_bias = prim->dependencies().size() == 3;

        ob << gemm_with_bias;

        dnnl::memory::data_type in0_dt;
        dnnl::memory::data_type in1_dt;
        dnnl::memory::data_type out_dt;
        dnnl::memory::data_type bias_dt;

        dnnl::memory::dims in0_dims;
        dnnl::memory::dims in1_dims;
        dnnl::memory::dims out_dims;
        dnnl::memory::dims bias_dims;

        dnnl::memory::format_tag in0_fmt;
        dnnl::memory::format_tag in1_fmt;
        dnnl::memory::format_tag out_fmt;
        dnnl::memory::format_tag bias_fmt;

        get_gemm_primitive_md(*impl_params, in0_dt, in1_dt, out_dt, in0_dims, in1_dims, out_dims, in0_fmt, in1_fmt, out_fmt,
                              gemm_with_bias, bias_dt, bias_dims, bias_fmt);

        ob << make_data(&in0_dt, sizeof(dnnl::memory::data_type));
        ob << make_data(&in1_dt, sizeof(dnnl::memory::data_type));
        ob << make_data(&out_dt, sizeof(dnnl::memory::data_type));

        ob << in0_dims;
        ob << in1_dims;
        ob << out_dims;

        ob << make_data(&in0_fmt, sizeof(dnnl::memory::format_tag));
        ob << make_data(&in1_fmt, sizeof(dnnl::memory::format_tag));
        ob << make_data(&out_fmt, sizeof(dnnl::memory::format_tag));

        if (gemm_with_bias) {
            ob << make_data(&bias_dt, sizeof(dnnl::memory::data_type));
            ob << bias_dims;
            ob << make_data(&bias_fmt, sizeof(dnnl::memory::format_tag));
        }

        std::vector<uint8_t> prim_cache;
        prim_cache = _prim.get_cache_blob();
        ob << prim_cache;
#endif
    }

    void load(BinaryInputBuffer& ib) override {
#ifdef ONEDNN_PRIMITIVE_SERIALIZATION
        parent::load(ib);

        bool gemm_with_bias;
        ib >> gemm_with_bias;

        dnnl::memory::data_type in0_dt = dnnl::memory::data_type::undef;
        dnnl::memory::data_type in1_dt = dnnl::memory::data_type::undef;
        dnnl::memory::data_type out_dt = dnnl::memory::data_type::undef;
        dnnl::memory::data_type bias_dt = dnnl::memory::data_type::undef;

        dnnl::memory::dims in0_dims;
        dnnl::memory::dims in1_dims;
        dnnl::memory::dims out_dims;
        dnnl::memory::dims bias_dims;

        dnnl::memory::format_tag in0_fmt = dnnl::memory::format_tag::undef;
        dnnl::memory::format_tag in1_fmt = dnnl::memory::format_tag::undef;
        dnnl::memory::format_tag out_fmt = dnnl::memory::format_tag::undef;
        dnnl::memory::format_tag bias_fmt = dnnl::memory::format_tag::undef;

        ib >> make_data(&in0_dt, sizeof(dnnl::memory::data_type));
        ib >> make_data(&in1_dt, sizeof(dnnl::memory::data_type));
        ib >> make_data(&out_dt, sizeof(dnnl::memory::data_type));

        ib >> in0_dims;
        ib >> in1_dims;
        ib >> out_dims;

        ib >> make_data(&in0_fmt, sizeof(dnnl::memory::format_tag));
        ib >> make_data(&in1_fmt, sizeof(dnnl::memory::format_tag));
        ib >> make_data(&out_fmt, sizeof(dnnl::memory::format_tag));

        if (gemm_with_bias) {
            ib >> make_data(&bias_dt, sizeof(dnnl::memory::data_type));
            ib >> bias_dims;
            ib >> make_data(&bias_fmt, sizeof(dnnl::memory::format_tag));
        }

        dnnl::memory::desc in0_md(in0_dims, in0_dt, in0_fmt);
        dnnl::memory::desc in1_md(in1_dims, in1_dt, in1_fmt);
        dnnl::memory::desc out_md(out_dims, out_dt, out_fmt);

        if (gemm_with_bias) {
            dnnl::memory::desc bias_md(bias_dims, bias_dt, bias_fmt);

            auto prim_desc = std::make_shared<dnnl::matmul::primitive_desc>(
                ib.get_engine().get_onednn_engine(),
                in0_md,
                in1_md,
                bias_md,
                out_md,
                *_attrs.get());

            _pd = *prim_desc;
        } else {
            auto prim_desc = std::make_shared<dnnl::matmul::primitive_desc>(
                ib.get_engine().get_onednn_engine(),
                in0_md,
                in1_md,
                out_md,
                *_attrs.get());

            _pd = *prim_desc;
        }

        std::vector<uint8_t> prim_cache;
        ib >> prim_cache;

        _scratchpad_md = _pd.scratchpad_desc();

        _prim = dnnl::primitive(_pd, prim_cache);
#endif
    }

    static std::unique_ptr<primitive_impl> create(const gemm_node& arg, const kernel_impl_params& impl_params) {
        auto& engine = impl_params.prog->get_engine();
        auto& config = impl_params.prog->get_config();
        auto attr = arg.get_onednn_primitive_attributes();
        auto prim_desc = get_gemm_primitive_descriptor(impl_params, *attr);

        return cldnn::make_unique<gemm_onednn>(engine, config, attr, *prim_desc);
    }
};

namespace detail {

attach_gemm_onednn::attach_gemm_onednn() {
    std::vector<data_types> dt = {
        data_types::f32,
        data_types::f16,
        data_types::u8,
        data_types::i8,
    };
    std::vector<format::type> fmt = {
        format::bfyx,
        format::byxf,
        format::byfx,
        format::bxfy,
        format::bfzyx,
        format::bfwzyx,
    };
    implementation_map<gemm>::add(impl_types::onednn, gemm_onednn::create, dt, fmt);
}

}  // namespace detail
}  // namespace onednn
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::onednn::gemm_onednn)
