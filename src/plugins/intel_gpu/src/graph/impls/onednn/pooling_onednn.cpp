// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pooling_onednn.hpp"
#include "pooling_inst.h"
#include "primitive_onednn_base.h"
#include "registry/implementation_manager.hpp"
#include "max_pool_shape_inference.hpp"

#include <oneapi/dnnl/dnnl.hpp>

#include <algorithm>
#include <functional>
#include <memory>
#include <utility>
namespace cldnn {
namespace onednn {

namespace {

struct PoolingGeometry {
    ov::Shape kernel;
    ov::Strides stride;
    ov::Strides dilation;

    explicit PoolingGeometry(const pooling& prim) : kernel(prim.size), stride(prim.stride), dilation(prim.dilation) {
        if (dilation.empty()) {
            dilation.resize(stride.size(), 1);
        }
        kernel.resize(std::max<size_t>(2, kernel.size()), 1);
        stride.resize(std::max<size_t>(2, stride.size()), 1);
        dilation.resize(std::max<size_t>(2, dilation.size()), 0);
    }
};

std::pair<dnnl::memory::dims, dnnl::memory::dims> get_pooling_padding(const pooling& prim,
                                                                   const layout& input_layout,
                                                                   const dnnl::memory::dims& input_dims,
                                                                   const dnnl::memory::dims& output_dims,
                                                                   const PoolingGeometry& geometry) {
    const auto& [kernel, stride, dilation] = geometry;
    dnnl::memory::dims pad_l(prim.pads_begin.begin(), prim.pads_begin.end());
    dnnl::memory::dims pad_r(prim.pads_end.begin(), prim.pads_end.end());
    pad_l.resize(kernel.size(), 0);
    pad_r.resize(kernel.size(), 0);

    // VALID ignores explicit padding attributes, just as shape inference does.
    if (prim.auto_pad == ov::op::PadType::VALID) {
        std::fill(pad_l.begin(), pad_l.end(), 0);
        std::fill(pad_r.begin(), pad_r.end(), 0);
    }

    if (prim.auto_pad == ov::op::PadType::SAME_UPPER || prim.auto_pad == ov::op::PadType::SAME_LOWER) {
        ov::op::v8::MaxPool op;
        op.set_strides(stride);
        op.set_kernel(kernel);
        op.set_auto_pad(prim.auto_pad);
        ov::op::pooling::apply_padding(&op, input_layout.get_partial_shape(), dilation, pad_l, pad_r);
    } else {
        for (size_t i = 0; i < kernel.size(); i++) {
            pad_r[i] = (output_dims[2 + i] - 1) * static_cast<int64_t>(stride[i]) - input_dims[2 + i] +
                       (static_cast<int64_t>(kernel[i]) - 1) * static_cast<int64_t>(dilation[i]) - pad_l[i] + 1;
        }
    }

    return {std::move(pad_l), std::move(pad_r)};
}

}  // namespace

struct pooling_onednn : typed_primitive_onednn_impl<pooling> {
    using parent = typed_primitive_onednn_impl<pooling>;
    using parent::parent;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::onednn::pooling_onednn)

protected:
    std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<pooling_onednn>(*this);
    }

    static std::shared_ptr<dnnl::pooling_forward::primitive_desc> get_pooling_primitive_descriptor(const kernel_impl_params& impl_params,
                                                                                                   const dnnl::primitive_attr& attr = dnnl::primitive_attr()) {
        auto& engine = impl_params.prog->get_engine();
        auto prim = impl_params.typed_desc<pooling>();

        auto input_layout = impl_params.get_input_layout(0);
        auto output_layout = impl_params.get_output_layout();
        const PoolingGeometry geometry(*prim);
        dnnl::memory::dims stride(geometry.stride.begin(), geometry.stride.end());
        dnnl::memory::dims kernel(geometry.kernel.begin(), geometry.kernel.end());
        dnnl::memory::dims dilation(geometry.dilation.begin(), geometry.dilation.end());

        auto input_md = onednn::layout_to_memory_desc(input_layout, false, format::is_blocked(input_layout.format));
        auto output_md = onednn::layout_to_memory_desc(output_layout, false, format::is_blocked(output_layout.format));

        const auto [pad_l, pad_r] = get_pooling_padding(*prim, input_layout, input_md.get_dims(), output_md.get_dims(), geometry);

        dnnl::algorithm alg;
        switch (prim->mode) {
            case pooling_mode::average: alg = dnnl::algorithm::pooling_avg_include_padding; break;
            case pooling_mode::max: alg = dnnl::algorithm::pooling_max; break;
            case pooling_mode::average_no_padding: alg = dnnl::algorithm::pooling_avg_exclude_padding; break;
            default: throw std::runtime_error("unsupported pool mode");
        }
        // Adjust dilation to match oneDNN expectations
        // Dilation in oneDNN is defined as the number of skipped elements between kernel elements.
        std::transform(dilation.begin(), dilation.end(), dilation.begin(), [](dnnl::memory::dim v) { return v > 0 ? v - 1 : 0; });

        return std::make_shared<dnnl::pooling_forward::primitive_desc>(
            engine.get_onednn_engine(),
            dnnl::prop_kind::forward_inference,
            alg,
            input_md,
            output_md,
            stride,
            kernel,
            dilation,
            pad_l,
            pad_r,
            attr);
    }

public:
    void save(BinaryOutputBuffer& ob) const override {
#ifdef ONEDNN_PRIMITIVE_SERIALIZATION
        parent::save(ob);

        const dnnl::pooling_forward::primitive_desc *typed_pd
            = reinterpret_cast<const dnnl::pooling_forward::primitive_desc *>(&_pd);

        dnnl::algorithm alg = typed_pd->get_algorithm();
        ob << make_data(&alg, sizeof(dnnl::algorithm));
        ob << typed_pd->get_strides();
        ob << typed_pd->get_kernel();
        ob << typed_pd->get_dilations();
        ob << typed_pd->get_padding_l();
        ob << typed_pd->get_padding_r();

        ob << get_cache_blob();
#endif
    }

    void load(BinaryInputBuffer& ib) override {
#ifdef ONEDNN_PRIMITIVE_SERIALIZATION
        parent::load(ib);

        const kernel_impl_params* impl_params = reinterpret_cast<kernel_impl_params*>(ib.getKernelImplParams());

        dnnl::algorithm alg;
        ib >> make_data(&alg, sizeof(dnnl::algorithm));

        auto input_md = onednn::layout_to_memory_desc(impl_params->get_input_layout(0));
        auto output_md = onednn::layout_to_memory_desc(impl_params->get_output_layout());

        dnnl::memory::dims stride;
        dnnl::memory::dims kernel;
        dnnl::memory::dims dilation;
        dnnl::memory::dims pad_l;
        dnnl::memory::dims pad_r;
        ib >> stride;
        ib >> kernel;
        ib >> dilation;
        ib >> pad_l;
        ib >> pad_r;

        auto prim_desc = std::make_shared<dnnl::pooling_forward::primitive_desc>(ib.get_engine().get_onednn_engine(),
                                                                                 dnnl::prop_kind::forward_inference,
                                                                                 alg,
                                                                                 input_md,
                                                                                 output_md,
                                                                                 stride,
                                                                                 kernel,
                                                                                 dilation,
                                                                                 pad_l,
                                                                                 pad_r,
                                                                                 *_attrs);
        _pd = *prim_desc;

        std::vector<uint8_t> prim_cache;
        ib >> prim_cache;

        _scratchpad_md = _pd.scratchpad_desc();

        _prim = make_primitive_from_blob(prim_cache);
#endif
    }

    static std::unique_ptr<primitive_impl> create(const pooling_node& arg, const kernel_impl_params& impl_params) {
        auto& engine = impl_params.prog->get_engine();
        const auto& config = impl_params.prog->get_config();
        auto attr = impl_params.attrs_onednn;
        auto prim_desc = get_pooling_primitive_descriptor(impl_params, *attr);

        return std::make_unique<pooling_onednn>(engine, config, attr, *prim_desc);
    }
};

#ifdef OV_GPU_VENDOR_NVIDIA
bool PoolingImplementationManager::has_unsupported_average_padding(const pooling& prim, const layout& in_layout, const layout& out_layout) {
    if (in_layout.is_dynamic() || out_layout.is_dynamic()) {
        return true;
    }
    const PoolingGeometry geometry(prim);
    const auto input_md = onednn::layout_to_memory_desc(in_layout, false);
    const auto output_md = onednn::layout_to_memory_desc(out_layout, false);
    const auto [pad_l, pad_r] = get_pooling_padding(prim, in_layout, input_md.get_dims(), output_md.get_dims(), geometry);
    // cuDNN average pooling that includes padding rejects larger trailing padding.
    return !std::equal(pad_l.begin(), pad_l.end(), pad_r.begin(), pad_r.end(), std::greater_equal<>{});
}
#endif

std::unique_ptr<primitive_impl> PoolingImplementationManager::create_impl(const program_node& node, const kernel_impl_params& params) const {
    assert(node.is_type<pooling>());
    return onednn::pooling_onednn::create(static_cast<const pooling_node&>(node), params);
}

}  // namespace onednn
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::onednn::pooling_onednn)
