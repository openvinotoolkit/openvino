// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gemm_inst.h"
#include "primitive_onednn_base.h"
#include "impls/implementation_map.hpp"

#include "kernel_selector_common.h"

#include <oneapi/dnnl/dnnl.hpp>

#include <algorithm>
#include <memory>
namespace cldnn {
namespace onednn {

struct gemm_onednn : typed_primitive_onednn_impl<gemm, dnnl::matmul::desc> {
    using parent = typed_primitive_onednn_impl<gemm, dnnl::matmul::desc>;
    using parent::parent;

    DECLARE_OBJECT_TYPE_SERIALIZATION

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
            args.insert({DNNL_ARG_WEIGHTS, weights.get_onednn_memory(_pd.weights_desc(0))});
        }

        if (instance.inputs_memory_count() == 3) {
            auto& weights = instance.input_memory(2);
            args.insert({DNNL_ARG_BIAS, weights.get_onednn_memory(_pd.weights_desc(1))});
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

    static std::shared_ptr<dnnl::matmul::desc> get_gemm_descriptor(const kernel_impl_params& impl_params) {
        auto prim = impl_params.typed_desc<gemm>();
        auto gemm_with_bias = prim->dependencies().size() == 3;
        auto out_l = impl_params.get_output_layout();

        std::vector<layout> in_layouts { impl_params.get_input_layout(0), impl_params.get_input_layout(1) };
        if (gemm_with_bias) {
            in_layouts.emplace_back(impl_params.get_input_layout(2));
        }

        in_layouts = gemm_inst::transform_input_layouts(prim, in_layouts, out_l);
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

        dnnl::memory::data_type in0_dt = onednn::convert_data_type(in0_l.data_type);
        dnnl::memory::data_type in1_dt = onednn::convert_data_type(in1_l.data_type);
        dnnl::memory::data_type out_dt = onednn::convert_data_type(out_l.data_type);

        dnnl::memory::dims in0_dims = onednn::convert_gemm_tensor(in0_l.get_tensor(), rank, batched_dims_can_be_removed);
        dnnl::memory::dims in1_dims = onednn::convert_gemm_tensor(in1_l.get_tensor(), rank, batched_dims_can_be_removed);
        dnnl::memory::dims out_dims = onednn::convert_gemm_tensor(out_l.get_tensor(), rank, batched_dims_can_be_removed);

        dnnl::memory::format_tag in0_fmt = onednn::convert_gemm_data_format(in0_dims);
        dnnl::memory::format_tag in1_fmt = onednn::convert_gemm_data_format(in1_dims);
        dnnl::memory::format_tag out_fmt = onednn::convert_gemm_data_format(out_dims);

        if (prim->transpose_input0) {
            in0_fmt = transpose_format(in0_fmt);
            std::swap(in0_dims[in0_dims.size() - 1], in0_dims[in0_dims.size() - 2]);
        }

        if (prim->transpose_input1) {
            in1_fmt = transpose_format(in1_fmt);
            std::swap(in1_dims[in1_dims.size() - 1], in1_dims[in1_dims.size() - 2]);
        }

        dnnl::memory::desc in0_md(in0_dims, in0_dt, in0_fmt);
        dnnl::memory::desc in1_md(in1_dims, in1_dt, in1_fmt);
        dnnl::memory::desc out_md(out_dims, out_dt, out_fmt);

        if (gemm_with_bias) {
            auto bias_l = impl_params.get_input_layout(2);
            auto bias_rank = cldnn::format::dimension(bias_l.format);
            dnnl::memory::data_type bias_dt = onednn::convert_data_type(bias_l.data_type);
            dnnl::memory::dims bias_dims = onednn::convert_gemm_tensor(bias_l.get_tensor(), bias_rank, batched_dims_can_be_removed);
            dnnl::memory::format_tag bias_fmt = onednn::convert_gemm_data_format(bias_dims);
            dnnl::memory::desc bias_md(bias_dims, bias_dt, bias_fmt);

            return std::make_shared<dnnl::matmul::desc>(
                in0_md,
                in1_md,
                bias_md,
                out_md);
        } else {
            return std::make_shared<dnnl::matmul::desc>(
                in0_md,
                in1_md,
                out_md);
        }
    }

public:
    void save(BinaryOutputBuffer& ob) const override {
        parent::save(ob);

        ob << make_data(&_desc->data, sizeof(dnnl_matmul_desc_t));

        std::vector<uint8_t> prim_cache;
        prim_cache = _prim.get_cache_blob();
        ob << prim_cache;
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);

        const char dummy_mem[sizeof(dnnl::matmul::desc)] = {};
        const dnnl::matmul::desc *dummy_opdesc
            = reinterpret_cast<const dnnl::matmul::desc *>(&dummy_mem[0]);
        _desc = std::make_shared<dnnl::matmul::desc>(std::move(*dummy_opdesc));
        ib >> make_data(&_desc->data, sizeof(dnnl_matmul_desc_t));

        std::vector<uint8_t> prim_cache;
        ib >> prim_cache;

        _pd = dnnl::primitive_desc(&_desc->data, _attrs.get(), ib.get_engine().get_onednn_engine(), nullptr);
        _prim = dnnl::primitive(_pd, prim_cache);
    }

    static std::unique_ptr<primitive_impl> create(const gemm_node& arg, const kernel_impl_params& impl_params) {
        auto& engine = impl_params.prog->get_engine();
        auto desc = get_gemm_descriptor(impl_params);
        auto attr = arg.get_onednn_primitive_attributes();
        dnnl::primitive_desc prim_desc{&desc->data, attr.get(), engine.get_onednn_engine(), nullptr};

        return cldnn::make_unique<gemm_onednn>(engine, desc, attr, prim_desc);
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
        format::bfzyx,
        format::bfwzyx,
    };
    implementation_map<gemm>::add(impl_types::onednn, gemm_onednn::create, dt, fmt);
}

}  // namespace detail
}  // namespace onednn
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::onednn::gemm_onednn)
