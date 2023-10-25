// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fully_connected_inst.h"
#include "primitive_onednn_base.h"
#include "implementation_map.hpp"

#include "impls/ocl/kernel_selector_helper.h"

#include <oneapi/dnnl/dnnl.hpp>

#include <algorithm>
#include <memory>
namespace cldnn {
namespace onednn {

struct fully_connected_onednn : typed_primitive_onednn_impl<fully_connected> {
    using parent = typed_primitive_onednn_impl<fully_connected>;
    using parent::parent;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::onednn::fully_connected_onednn)

private:
    static std::vector<int64_t> reshape_to_2d(const ov::PartialShape& shape, int64_t feature) {
        auto staticShape = shape.to_shape();
        size_t total =
            std::accumulate(staticShape.begin(), staticShape.end(), static_cast<size_t>(1), std::multiplies<size_t>());
        std::vector<int64_t> reshapeSize = { static_cast<int64_t>(total) / feature, feature };
        return reshapeSize;
    }

protected:
    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<fully_connected_onednn>(*this);
    }

    std::unordered_map<int, dnnl::memory> get_arguments(fully_connected_inst& instance) const override {
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

        return args;
    }

    static std::shared_ptr<WeightsReorderParams> get_weights_reorder(const kernel_impl_params& impl_params, const dnnl::primitive_desc& pd) {
        auto input_layout = impl_params.get_input_layout(0);
        auto weights_layout = impl_params.get_input_layout(1);
        auto cldnn_prim = impl_params.typed_desc<fully_connected>();

        auto input_pshape = input_layout.get_partial_shape();
        auto weights_pshape = weights_layout.get_partial_shape();
        int64_t feature = input_pshape[std::min(cldnn_prim->input_size, static_cast<size_t>(4)) - 1].get_length();
        if (cldnn_prim->input_size == 3) {
            feature = std::max({input_layout.spatial(0), input_layout.spatial(1), input_layout.spatial(2)});
        }
        if (weights_pshape.size() != 2) {
            weights_layout.set_partial_shape(reshape_to_2d(weights_pshape, feature));
        }

        format out_fmt = onednn::find_format(pd.weights_desc(0));

        auto output_weights_layout = weights_layout;
        output_weights_layout.format = out_fmt;

        return std::make_shared<WeightsReorderParams>(weights_layout, output_weights_layout, false);
    }

    static std::shared_ptr<dnnl::inner_product_forward::primitive_desc> get_fully_connected_primitive_descriptor(const kernel_impl_params& impl_params,
                                                                                                cldnn::engine& engine, size_t prim_input_size, bool has_bias,
                                                                                                const dnnl::primitive_attr& attr = dnnl::primitive_attr()) {
        auto input_layout = impl_params.get_input_layout(0);
        auto weights_layout = impl_params.get_input_layout(1);
        auto output_layout = impl_params.get_output_layout();

        auto input_pshape = input_layout.get_partial_shape();
        auto weights_pshape = weights_layout.get_partial_shape();

        size_t input_size = (prim_input_size > input_pshape.size()) ? input_pshape.size() : prim_input_size;
        int64_t feature = input_pshape[std::min(input_size, static_cast<size_t>(4)) - 1].get_length();
        if (input_size == 3) {
            feature = std::max({input_layout.spatial(0), input_layout.spatial(1), input_layout.spatial(2)});
        }

        if (input_size > 3) {
           input_layout.set_partial_shape(reshape_to_2d(input_pshape, feature));
        }
        if (weights_pshape.size() != 2) {
            weights_layout.set_partial_shape(reshape_to_2d(weights_pshape, feature));
        }
        if (input_size == 3) {
            output_layout.set_partial_shape({ input_layout.batch(), input_layout.feature(), weights_layout.batch(), 1 });
        } else {
            output_layout.set_partial_shape({ input_layout.batch(), weights_layout.batch() });
        }

        if (input_size == 3) {
            combine_bf_with_first_spatial_dim(input_layout);
            combine_bf_with_first_spatial_dim(output_layout);
        }

        auto input_md = onednn::layout_to_memory_desc(input_layout, dnnl::memory::format_tag::undef, false);
        auto weights_md = onednn::layout_to_memory_desc(weights_layout, dnnl::memory::format_tag::any);
        auto output_md = onednn::layout_to_memory_desc(output_layout, dnnl::memory::format_tag::ab, false);

        if (has_bias) {
            auto bias_md = onednn::layout_to_memory_desc(impl_params.get_input_layout(2), dnnl::memory::format_tag::any, true);
            return std::make_shared<dnnl::inner_product_forward::primitive_desc>(
                engine.get_onednn_engine(),
                dnnl::prop_kind::forward_inference,
                input_md,
                weights_md,
                bias_md,
                output_md,
                attr);
        } else {
            return std::make_shared<dnnl::inner_product_forward::primitive_desc>(
                engine.get_onednn_engine(),
                dnnl::prop_kind::forward_inference,
                input_md,
                weights_md,
                output_md,
                attr);
        }
    }

public:
    void save(BinaryOutputBuffer& ob) const override {
#ifdef ONEDNN_PRIMITIVE_SERIALIZATION
        parent::save(ob);

        const kernel_impl_params* impl_params = reinterpret_cast<kernel_impl_params*>(ob.getKernelImplParams());
        auto prim = impl_params->typed_desc<fully_connected>();
        size_t input_size = prim->input_size;
        bool has_bias = !prim->bias.empty();
        ob << input_size;
        ob << has_bias;

        std::vector<uint8_t> prim_cache;
        prim_cache = _prim.get_cache_blob();
        ob << prim_cache;
#endif
    }

    void load(BinaryInputBuffer& ib) override {
#ifdef ONEDNN_PRIMITIVE_SERIALIZATION
        parent::load(ib);

        size_t input_size = 2;
        bool has_bias = false;
        ib >> input_size;
        ib >> has_bias;

        const kernel_impl_params* impl_params = reinterpret_cast<kernel_impl_params*>(ib.getKernelImplParams());
        auto prim_desc = get_fully_connected_primitive_descriptor(*impl_params, ib.get_engine(), input_size, has_bias, *_attrs);
        _pd = *prim_desc;

        std::vector<uint8_t> prim_cache;
        ib >> prim_cache;

        _scratchpad_md = _pd.scratchpad_desc();

        _prim = dnnl::primitive(_pd, prim_cache);
#endif
    }

    static std::unique_ptr<primitive_impl> create(const fully_connected_node& arg, const kernel_impl_params& impl_params) {
        auto& engine = impl_params.prog->get_engine();
        auto& config = impl_params.prog->get_config();
        auto attr = arg.get_onednn_primitive_attributes();
        auto prim = impl_params.typed_desc<fully_connected>();
        auto prim_desc = get_fully_connected_primitive_descriptor(impl_params, impl_params.prog->get_engine(),
                                                                  prim->input_size, !prim->bias.empty(), *attr);

        return cldnn::make_unique<fully_connected_onednn>(engine, config, attr, *prim_desc, get_weights_reorder(impl_params, *prim_desc));
    }
};

namespace detail {

attach_fully_connected_onednn::attach_fully_connected_onednn() {
    std::vector<data_types> dt = {
        data_types::f32,
        data_types::f16,
        data_types::u8,
        data_types::i8,
    };
    std::vector<format::type> fmt = {
        format::bfyx,
    };
    implementation_map<fully_connected>::add(impl_types::onednn, fully_connected_onednn::create, dt, fmt);
}

}  // namespace detail
}  // namespace onednn
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::onednn::fully_connected_onednn)
