// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sdpa_onednn.hpp"

#include "impls/onednn/primitive_onednn_base.h"
#include "impls/onednn/utils.hpp"

#include "common/opdesc.hpp"
#include "common/sdpa_test_iface.hpp"
#include "common/sdpa_types.hpp"

#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_types.h>

#include <cmath>
#include <sstream>

namespace cldnn {
namespace onednn {

namespace {

dnnl::memory::dims get_static_dims(const layout& l) {
    const auto shape = l.get_partial_shape().to_shape();
    return dnnl::memory::dims(shape.begin(), shape.end());
}

dnnl::memory::desc make_plain_4d_desc(const layout& l) {
    return dnnl::memory::desc(get_static_dims(l), convert_data_type(l.data_type), dnnl::memory::format_tag::abcd);
}

dnnl::memory::desc make_key_desc(const layout& l) {
    auto dims = get_static_dims(l);
    OPENVINO_ASSERT(dims.size() == 4, "[GPU] oneDNN SDPA expects static rank-4 key layout");
    std::swap(dims[2], dims[3]);
    return dnnl::memory::desc(dims, convert_data_type(l.data_type), dnnl::memory::format_tag::abdc);
}

bool has_runtime_scale(const scaled_dot_product_attention& prim) {
    return prim.has_scale_input && !prim.scale_val.has_value();
}

float get_scale_value(const kernel_impl_params& impl_params) {
    const auto prim = impl_params.typed_desc<scaled_dot_product_attention>();
    if (prim->scale_val.has_value())
        return prim->scale_val.value();

    const auto q_dims = get_static_dims(impl_params.get_input_layout(ScaledDotProductAttentionInputIdx::QUERY));
    OPENVINO_ASSERT(q_dims.size() == 4 && q_dims[3] > 0, "[GPU] oneDNN SDPA expects static rank-4 query layout");
    return 1.0f / std::sqrt(static_cast<float>(q_dims[3]));
}

dnnl::memory::desc make_scale_desc(const kernel_impl_params& impl_params, bool use_host_scale) {
    if (use_host_scale)
        return dnnl::memory::desc::host_scalar(dnnl::memory::data_type::f32);

    const auto& scale_layout = impl_params.get_input_layout(ScaledDotProductAttentionInputIdx::SCALE);
    return dnnl::memory::desc({1}, convert_data_type(scale_layout.data_type), dnnl::memory::format_tag::a);
}

dnnl::primitive_desc create_sdpa_primitive_desc(const kernel_impl_params& impl_params,
                                               cldnn::engine& engine,
                                               const dnnl::primitive_attr& attr,
                                               bool use_host_scale) {
    const auto prim = impl_params.typed_desc<scaled_dot_product_attention>();

    const auto q_md = make_plain_4d_desc(impl_params.get_input_layout(ScaledDotProductAttentionInputIdx::QUERY));
    const auto k_md = make_key_desc(impl_params.get_input_layout(ScaledDotProductAttentionInputIdx::KEY));
    const auto v_md = make_plain_4d_desc(impl_params.get_input_layout(ScaledDotProductAttentionInputIdx::VALUE));
    const auto dst_md = make_plain_4d_desc(impl_params.get_output_layout(0));
    const auto scale_md = make_scale_desc(impl_params, use_host_scale);
    const dnnl::memory::desc mask_md;

    dnnl::primitive_attr qk_attr;
    dnnl::primitive_attr vs_attr;

    const auto mask_type = prim->is_causal ? dnnl::impl::attn_mask_type::top_left : dnnl::impl::attn_mask_type::undef;
    const auto kv_head_number = static_cast<dnnl_dim_t>(k_md.get_dims()[1]);

    dnnl_primitive_desc_t c_pd = nullptr;
    const auto status = sdpa_primitive_desc_create(&c_pd,
                                                   engine.get_onednn_engine().get(),
                                                   q_md.get(),
                                                   k_md.get(),
                                                   v_md.get(),
                                                   dst_md.get(),
                                                   mask_md.get(),
                                                   scale_md.get(),
                                                   false,
                                                   kv_head_number,
                                                   static_cast<int>(mask_type),
                                                   dnnl_softmax_accurate,
                                                   dnnl_forward_inference,
                                                   attr.get(),
                                                   qk_attr.get(),
                                                   vs_attr.get());

    if (status != dnnl_success || c_pd == nullptr) {
        std::ostringstream failure_msg;
        failure_msg << "[GPU] Failed to create oneDNN SDPA primitive descriptor"
                    << " status=" << static_cast<int>(status)
                    << " (" << dnnl_status_to_string(status) << ")"
                    << " c_pd=" << c_pd
                    << " q layout=" << impl_params.get_input_layout(ScaledDotProductAttentionInputIdx::QUERY).to_short_string()
                    << " md=" << memory_desc_to_string(q_md)
                    << " k layout=" << impl_params.get_input_layout(ScaledDotProductAttentionInputIdx::KEY).to_short_string()
                    << " md=" << memory_desc_to_string(k_md)
                    << " v layout=" << impl_params.get_input_layout(ScaledDotProductAttentionInputIdx::VALUE).to_short_string()
                    << " md=" << memory_desc_to_string(v_md)
                    << " dst layout=" << impl_params.get_output_layout(0).to_short_string()
                    << " md=" << memory_desc_to_string(dst_md)
                    << " scale md=" << memory_desc_to_string(scale_md)
                    << " mask_type=" << static_cast<int>(mask_type);
        OPENVINO_ASSERT(false, failure_msg.str());
    }

    return dnnl::primitive_desc(c_pd);
}

}  // namespace

struct sdpa_onednn : typed_primitive_onednn_impl<scaled_dot_product_attention> {
    using parent = typed_primitive_onednn_impl<scaled_dot_product_attention>;
    using parent::parent;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::onednn::sdpa_onednn)

    bool _use_host_scale = true;
    float _scale_value = 1.0f;
    dnnl::memory::desc _scale_md;

protected:
    std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<sdpa_onednn>(*this);
    }

    std::unordered_map<int, dnnl::memory> get_arguments(scaled_dot_product_attention_inst& instance) const override {
        std::unordered_map<int, dnnl::memory> args;

        auto bind_input = [&](int dnnl_arg, size_t input_idx) {
            const auto md = _pd.query_md(dnnl::query::exec_arg_md, dnnl_arg);
            const auto offset = onednn::get_offset(instance.get_input_layout(input_idx),
                                                   _pd.query_md(dnnl::query::exec_arg_md, dnnl_arg));
            args[dnnl_arg] = instance.input_memory(input_idx).get_onednn_memory(md, offset);
        };

        bind_input(DNNL_ARG_QUERIES, ScaledDotProductAttentionInputIdx::QUERY);
        bind_input(DNNL_ARG_KEYS, ScaledDotProductAttentionInputIdx::KEY);
        bind_input(DNNL_ARG_VALUES, ScaledDotProductAttentionInputIdx::VALUE);

        if (_use_host_scale) {
            args[DNNL_ARG_SCALE] = dnnl::memory(_scale_md, _scale_value);
        } else {
            args[DNNL_ARG_SCALE] = instance.input_memory(ScaledDotProductAttentionInputIdx::SCALE).get_onednn_memory(_scale_md, 0);
        }

        const auto dst_md = _pd.query_md(dnnl::query::exec_arg_md, DNNL_ARG_DST);
        const auto output_offset = onednn::get_offset(instance.get_output_layout(),
                                                      _pd.query_md(dnnl::query::exec_arg_md, DNNL_ARG_DST));
        args[DNNL_ARG_DST] = instance.output_memory().get_onednn_memory(dst_md, output_offset);

        if (_scratchpad_md.get_size() != 0) {
            const auto& intermediates = instance.get_intermediates_memories();
            OPENVINO_ASSERT(!intermediates.empty(),
                            "[GPU] oneDNN SDPA primitive ", instance.id(), " requires scratchpad of size ",
                            _scratchpad_md.get_size(), " bytes, but intermediates memory is missing");
            args[DNNL_ARG_SCRATCHPAD] = intermediates[0]->get_onednn_memory(_scratchpad_md, 0);
        }

        return args;
    }

    void set_arguments_impl(scaled_dot_product_attention_inst& instance) override {
        if (instance.can_be_optimized())
            return;

        _args[instance.get_network().get_id()] = get_arguments(instance);
    }

public:
    void save(BinaryOutputBuffer& ob) const override {
#ifdef ONEDNN_PRIMITIVE_SERIALIZATION
        parent::save(ob);
        ob << _use_host_scale;
        ob << _scale_value;

        std::vector<uint8_t> prim_cache = _prim.get_cache_blob();
        ob << prim_cache;
#endif
    }

    void load(BinaryInputBuffer& ib) override {
#ifdef ONEDNN_PRIMITIVE_SERIALIZATION
        parent::load(ib);

        ib >> _use_host_scale;
        ib >> _scale_value;

        const auto* impl_params = reinterpret_cast<const kernel_impl_params*>(ib.getKernelImplParams());
        _scale_md = make_scale_desc(*impl_params, _use_host_scale);
        _pd = create_sdpa_primitive_desc(*impl_params, ib.get_engine(), *_attrs, _use_host_scale);

        std::vector<uint8_t> prim_cache;
        ib >> prim_cache;

        _scratchpad_md = _pd.scratchpad_desc();
        _prim = dnnl::primitive(_pd, prim_cache);
#endif
    }

    static std::unique_ptr<primitive_impl> create(const scaled_dot_product_attention_node&,
                                                  const kernel_impl_params& impl_params) {
        auto& engine = impl_params.prog->get_engine();
        auto& config = impl_params.prog->get_config();
        auto attr = impl_params.attrs_onednn;
        attr->set_scratchpad_mode(dnnl::scratchpad_mode::user);

        const auto prim = impl_params.typed_desc<scaled_dot_product_attention>();
        const auto use_host_scale = !has_runtime_scale(*prim);
        auto prim_desc = create_sdpa_primitive_desc(impl_params, engine, *attr, use_host_scale);

        auto impl = std::make_unique<sdpa_onednn>(engine, config, attr, prim_desc);
        impl->_use_host_scale = use_host_scale;
        impl->_scale_value = use_host_scale ? get_scale_value(impl_params) : 1.0f;
        impl->_scale_md = make_scale_desc(impl_params, use_host_scale);
        return impl;
    }
};

std::unique_ptr<primitive_impl> SDPAImplementationManager::create_impl(const program_node& node,
                                                                       const kernel_impl_params& params) const {
    assert(node.is_type<scaled_dot_product_attention>());
    return sdpa_onednn::create(static_cast<const scaled_dot_product_attention_node&>(node), params);
}

}  // namespace onednn
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::onednn::sdpa_onednn)