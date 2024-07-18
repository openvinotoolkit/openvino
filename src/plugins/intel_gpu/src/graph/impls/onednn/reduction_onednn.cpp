// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reduce_inst.h"
#include "primitive_onednn_base.h"
#include "impls/registry/implementation_map.hpp"

#include <oneapi/dnnl/dnnl.hpp>

#include <algorithm>
#include <memory>
namespace cldnn {
namespace onednn {

// Return true if one of blocked axes (b or f) is reduced and one of spatial axes is NOT reduced
static bool is_reduce_blocked_axes(reduce_node const& node) {
    auto prim = node.get_primitive();
    auto reduce_axes = prim->axes;
    auto input_layout = node.get_input_layout();
    auto num_spatial = format::spatial_num(node.get_output_layout().format);
    auto dims = node.get_output_layout().format.dimension();

    // Check if it reduces all spatial axes
    bool feature_axis_is_only_remaining = true;
    for (size_t idx_spatial = (dims - num_spatial); idx_spatial < dims; idx_spatial++) {
        if (count(reduce_axes.begin(), reduce_axes.end(), idx_spatial) == 0) {
            feature_axis_is_only_remaining = false;
            break;
        }
    }

    if (input_layout.is_static() &&
        (count(reduce_axes.begin(), reduce_axes.end(), 1) > 0 ||
        (count(reduce_axes.begin(), reduce_axes.end(), 0) > 0))) {
        if (!feature_axis_is_only_remaining)
            return true;
    }

    return false;
}


static void reorder_unreduced_axis_no_fusion(const cldnn::layout& input_layout, cldnn::layout& output_layout, std::vector<int64_t> axes) {
    auto in_dims = input_layout.get_tensor().sizes();
    auto num_dims = input_layout.format.dimension();
    auto num_spatial = format::spatial_num(input_layout.format);
    size_t num_others = num_dims - num_spatial;

    for (size_t idx = 0; idx < axes.size(); idx++) {
        if (axes[idx] < static_cast<int64_t>(num_others))
            in_dims[axes[idx]] = 1;
        else
            in_dims[(num_dims - axes[idx] - 1 + num_others)] = 1;
    }

    auto output_tensor = output_layout.get_tensor();
    for (size_t idx = 0; idx < output_layout.get_rank(); idx++) {
        output_tensor.raw[idx] = in_dims[idx];
    }

    output_layout.set_tensor(output_tensor);
}

struct reduction_onednn : typed_primitive_onednn_impl<reduce> {
    using parent = typed_primitive_onednn_impl<reduce>;
    using parent::parent;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::onednn::reduction_onednn)

protected:
    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<reduction_onednn>(*this);
    }

    static std::shared_ptr<dnnl::reduction::primitive_desc> get_reduction_primitive_descriptor(const kernel_impl_params& impl_params,
                                                                                               const dnnl::primitive_attr& attr = dnnl::primitive_attr()) {
        auto& engine = impl_params.prog->get_engine();
        auto prim = impl_params.typed_desc<reduce>();
        auto input_layout = impl_params.get_input_layout(0);
        auto output_layout = impl_params.get_output_layout();

        // A clDNN Reduce reorders un-reduced axes of its output tensor to b-f and spatial order when keep_dims is false.
        // oneDNN reduction does not allow this. So this function reverts it.
        reorder_unreduced_axis_no_fusion(input_layout, output_layout, prim->axes);

        auto input_md = onednn::layout_to_memory_desc(input_layout);
        auto output_md = onednn::layout_to_memory_desc(output_layout);

        float p = 0.f;
        float eps = 0.f;
        dnnl::algorithm alg;
        switch (prim->mode) {
            case reduce_mode::mean: alg = dnnl::algorithm::reduction_mean; break;
            case reduce_mode::max: alg = dnnl::algorithm::reduction_max; break;
            case reduce_mode::min: alg = dnnl::algorithm::reduction_min; break;
            case reduce_mode::sum: alg = dnnl::algorithm::reduction_sum; break;
            case reduce_mode::prod: alg = dnnl::algorithm::reduction_mul; break;
            case reduce_mode::sum_square:
                alg = dnnl::algorithm::reduction_norm_lp_power_p_sum;
                p = 2.0f;
                break;
            case reduce_mode::l1:
                alg = dnnl::algorithm::reduction_norm_lp_sum;
                p = 1.0f;
                break;
            case reduce_mode::l2:
                alg = dnnl::algorithm::reduction_norm_lp_sum;
                p = 2.0f;
                break;
            default: throw std::runtime_error("unsupported reduce mode");
        }

        return std::make_shared<dnnl::reduction::primitive_desc>(
            engine.get_onednn_engine(),
            alg,
            input_md,
            output_md,
            p,
            eps,
            attr);
    }

public:
    void save(BinaryOutputBuffer& ob) const override {
#ifdef ONEDNN_PRIMITIVE_SERIALIZATION
        parent::save(ob);

        const dnnl::reduction::primitive_desc *typed_pd
            = reinterpret_cast<const dnnl::reduction::primitive_desc *>(&_pd);

        dnnl::algorithm alg = typed_pd->get_algorithm();
        ob << make_data(&alg, sizeof(dnnl::algorithm));
        ob << typed_pd->get_p();
        ob << typed_pd->get_epsilon();

        std::vector<uint8_t> prim_cache;
        prim_cache = _prim.get_cache_blob();
        ob << prim_cache;
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

        float p, eps;
        ib >> p >> eps;

        auto prim_desc = std::make_shared<dnnl::reduction::primitive_desc>(
            ib.get_engine().get_onednn_engine(),
            alg,
            input_md,
            output_md,
            p,
            eps,
            *_attrs.get());
        _pd = *prim_desc;

        std::vector<uint8_t> prim_cache;
        ib >> prim_cache;

        _scratchpad_md = _pd.scratchpad_desc();

        _prim = dnnl::primitive(_pd, prim_cache);
#endif
    }

    static bool validate(const reduce_node& node) {
        auto preferred_format = node.get_preferred_input_fmt(0);

        auto reduce_prim = node.get_primitive();
        const auto& input_layout = node.get_input_layout(0);
        const auto& output_layout = node.get_output_layout(0);
        auto in_dt = input_layout.data_type;
        auto out_dt = output_layout.data_type;

        if (in_dt == data_types::f32 && out_dt == data_types::f32)
            return false;

        // oneDNN reduction currently does not support logical_and, logical_or, log_sum and log_sum_exp.
        switch (reduce_prim->mode) {
            case reduce_mode::mean:
            case reduce_mode::max:
            case reduce_mode::min:
            case reduce_mode::sum:
            case reduce_mode::prod:
                break;
            case reduce_mode::sum_square:
            case reduce_mode::l1:
            case reduce_mode::l2:
                // modes have a limitation of data type
                if (one_of(in_dt, {data_types::f16, data_types::f32}))
                    break;
            default:
                return false;
        }

        // redundant reduce is not acceptable on oneDNN reduction
        if (output_layout == input_layout) {
            return false;
        }

        // oneDNN reduction selects ref kernel for simple formats(bfyx..) which has perf regression with a decent tensor size.
        if (format::is_simple_data_format(preferred_format))
            return false;

        // Onednn reduction does NOT support reordering of unreduced-axes.
        // Currently, an Onednn reduce layer which contains reduction of blocked axes(b-f) is expected to select planar format.
        if (reduce_prim->keep_dims == false && is_reduce_blocked_axes(node))
            return false;

        return true;
    }

    static std::unique_ptr<primitive_impl> create(const reduce_node& arg, const kernel_impl_params& impl_params) {
        auto& engine = impl_params.prog->get_engine();
        auto& config = impl_params.prog->get_config();
        auto attr = impl_params.attrs_onednn;
        auto prim_desc = get_reduction_primitive_descriptor(impl_params, *attr);

        return cldnn::make_unique<reduction_onednn>(engine, config, attr, *prim_desc);
    }
};

struct reduce_factory : public cldnn::implementation_factory<reduce> {
    std::unique_ptr<primitive_impl> create(const program_node& node, const kernel_impl_params& params) const override {
        OPENVINO_ASSERT(node.is_type<reduce>());
        return onednn::reduction_onednn::create(static_cast<const reduce_node&>(node), params);
    }

    bool validate(const program_node& node) const override {
        OPENVINO_ASSERT(node.is_type<reduce>());
        return onednn::reduction_onednn::validate(static_cast<const reduce_node&>(node));
    }

    std::pair<std::vector<format>, std::vector<format>> query_formats(const program_node& node) const override {
        OPENVINO_NOT_IMPLEMENTED;
    }
};

namespace detail {

attach_reduction_onednn::attach_reduction_onednn() {
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
        format::b_fs_yx_fsv16,
        format::b_fs_yx_fsv32,
        format::b_fs_zyx_fsv32,
        format::bs_fs_yx_bsv16_fsv16,
        format::bs_fs_yx_bsv16_fsv32,
        format::bs_fs_yx_bsv32_fsv16,
        format::bs_fs_yx_bsv32_fsv32,
        format::bs_fs_zyx_bsv16_fsv16,
        format::bs_fs_zyx_bsv16_fsv32,
        format::bs_fs_zyx_bsv32_fsv16,
        format::bs_fs_zyx_bsv32_fsv32,
    };

    implementation_map<reduce>::add(impl_types::onednn, cldnn::make_unique<reduce_factory>(), dt, fmt);
}

}  // namespace detail
}  // namespace onednn
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::onednn::reduction_onednn)
