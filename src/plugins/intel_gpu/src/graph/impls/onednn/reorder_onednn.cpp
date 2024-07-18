// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "impls/onednn/utils.hpp"
#include "reorder_inst.h"
#include "primitive_onednn_base.h"
#include "impls/registry/implementation_map.hpp"

#include <oneapi/dnnl/dnnl.hpp>

#include <algorithm>
#include <memory>
namespace cldnn {
namespace onednn {

struct reorder_onednn : typed_primitive_onednn_impl<reorder, dnnl::reorder::primitive_desc, dnnl::reorder> {
    using parent = typed_primitive_onednn_impl<reorder, dnnl::reorder::primitive_desc, dnnl::reorder>;
    using parent::parent;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::onednn::reorder_onednn)

protected:
    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<reorder_onednn>(*this);
    }

    std::unordered_map<int, dnnl::memory> get_arguments(reorder_inst& instance) const override {
        std::unordered_map<int, dnnl::memory> args;

        int input_idx = DNNL_ARG_FROM;
        for (size_t i = 0; i < instance.inputs_memory_count(); i++) {
            auto& input = instance.input_memory(i);
            auto offset = onednn::get_offset(instance.get_input_layout(i),
                                             _pd.dnnl::primitive_desc_base::src_desc(static_cast<int>(i)));
            args.insert({input_idx++, input.get_onednn_memory(_pd.dnnl::primitive_desc_base::src_desc(static_cast<int>(i)), offset)});
        }

        {
            auto& output = instance.output_memory();
            auto offset = onednn::get_offset(instance.get_output_layout(), _pd.dnnl::primitive_desc_base::dst_desc(0));
            args.insert({DNNL_ARG_DST, output.get_onednn_memory(_pd.dnnl::primitive_desc_base::dst_desc(0), offset)});
        }

        return args;
    }

    static std::shared_ptr<dnnl::reorder::primitive_desc> get_reorder_primitive_descriptor(const kernel_impl_params& impl_params,
                                                                                           const dnnl::primitive_attr& attr) {
        auto& engine = impl_params.prog->get_engine();
        auto prim = impl_params.typed_desc<reorder>();

        auto input_layout = impl_params.get_input_layout(0);
        auto output_layout = impl_params.get_output_layout();

        auto input_md = onednn::layout_to_memory_desc(input_layout);
        auto output_md = onednn::layout_to_memory_desc(output_layout);

        OPENVINO_ASSERT(input_md.get_format_kind() != dnnl::memory::format_kind::any,
                        "[GPU] The format kind of the input memory descriptor of onednn reorder cannot be 'any'.");
        OPENVINO_ASSERT(output_md.get_format_kind() != dnnl::memory::format_kind::any,
                        "[GPU] The format kind of the output memory descriptor of onednn reorder cannot be 'any'.");

        return std::make_shared<dnnl::reorder::primitive_desc>(
            engine.get_onednn_engine(),
            input_md,
            engine.get_onednn_engine(),
            output_md,
            attr);
    }

public:
    void save(BinaryOutputBuffer& ob) const override {
#ifdef ONEDNN_PRIMITIVE_SERIALIZATION
        parent::save(ob);

        std::vector<uint8_t> prim_cache;
        prim_cache = _prim.get_cache_blob();
        ob << prim_cache;
#endif
    }

    void load(BinaryInputBuffer& ib) override {
#ifdef ONEDNN_PRIMITIVE_SERIALIZATION
        parent::load(ib);

        const kernel_impl_params* impl_params = reinterpret_cast<kernel_impl_params*>(ib.getKernelImplParams());

        auto input_md = onednn::layout_to_memory_desc(impl_params->get_input_layout(0));
        auto output_md = onednn::layout_to_memory_desc(impl_params->get_output_layout());

        auto prim_desc = std::make_shared<dnnl::reorder::primitive_desc>(
            ib.get_engine().get_onednn_engine(),
            input_md,
            ib.get_engine().get_onednn_engine(),
            output_md,
            *_attrs.get());
        _pd = *prim_desc;

        std::vector<uint8_t> prim_cache;
        ib >> prim_cache;

        _scratchpad_md = _pd.scratchpad_desc();
        if (prim_cache.size() > 0)
            _prim = dnnl::reorder(_pd, prim_cache);
        else
            _prim = dnnl::reorder(_pd);
#endif
    }

    static bool validate(const reorder_node& node) {
        std::vector<format> onednn_optimized_fmt = {
            format::bfyx,
            format::byxf,
            format::b_fs_zyx_fsv16,
            format::b_fs_yx_fsv16,
            format::b_fs_yx_fsv32,
            format::bs_fs_zyx_bsv8_fsv4,
            format::bs_fs_yx_bsv8_fsv4,
            format::bs_fs_yx_bsv16_fsv4,
            format::bs_fs_zyx_bsv16_fsv4,
            format::bs_fs_yx_bsv16_fsv2,
            format::bs_fs_zyx_bsv16_fsv2,
            format::bs_fs_zyx_bsv8_fsv2,
            format::bs_fs_yx_bsv8_fsv2,
            format::bs_fs_zyx_bsv16_fsv16,
            format::bs_fs_yx_bsv16_fsv16,
            format::bs_fs_yx_bsv16_fsv32,
            format::bs_fs_zyx_bsv32_fsv16,
            format::bs_fs_yx_bsv32_fsv16,
            format::bs_fs_zyx_bsv32_fsv32,
            format::bs_fs_yx_bsv32_fsv32,
        };

        const auto& input_layout = node.get_input_layout(0);
        const auto& output_layout = node.get_output_layout(0);

        auto input_fmt = input_layout.format;
        auto output_fmt = output_layout.format;

        auto in_dt = input_layout.data_type;
        auto out_dt = output_layout.data_type;

        if (output_fmt == format::custom)
            return true;

        if (std::find(onednn_optimized_fmt.begin(), onednn_optimized_fmt.end(), input_fmt) == onednn_optimized_fmt.end() ||
            std::find(onednn_optimized_fmt.begin(), onednn_optimized_fmt.end(), output_fmt) == onednn_optimized_fmt.end()) {
            return false;
        }

        // onednn doesn't support paddings
        if (input_layout.data_padding || output_layout.data_padding)
            return false;

        // Native impl works faster for this type of reorder
        if (input_fmt == format::bfyx && output_fmt == format::bfyx)
            return false;

        // onednn reorder doesn't support different number of dimensions in input and output layouts
        if (input_fmt.dimension() != output_fmt.dimension())
            return false;

        if (in_dt == data_types::i64 || out_dt == data_types::i64)
            return false;

        // For mixed precision case, oneDNN is slower than clDNN
        if (input_fmt == format::b_fs_yx_fsv16 && data_type_traits::is_i8_u8(in_dt))
            return false;
        if (output_fmt == format::b_fs_yx_fsv16 && data_type_traits::is_i8_u8(in_dt))
            return false;
        if (output_fmt == format::bfyx && out_dt == data_types::f32)
            return false;

        return true;
    }

    static std::unique_ptr<primitive_impl> create(const reorder_node& arg, const kernel_impl_params& impl_params) {
        bool is_reorder_weights = format::is_weights_format(impl_params.get_input_layout().format) ||
                                  format::is_weights_format(impl_params.get_output_layout().format);
        if (is_reorder_weights) {
            return create_reorder_weights(impl_params);
        } else {
            auto& engine = impl_params.prog->get_engine();
            auto& config = impl_params.prog->get_config();
            auto attr = impl_params.attrs_onednn;
            auto prim_desc = get_reorder_primitive_descriptor(impl_params, *attr);
            return cldnn::make_unique<reorder_onednn>(engine, config, attr, *prim_desc);
        }
    }

    static std::unique_ptr<primitive_impl> create_reorder_weights(const kernel_impl_params& impl_param) {
        auto& engine = impl_param.prog->get_engine();
        const auto& prim = impl_param.typed_desc<reorder>();
        const auto& weights_params = prim->weights_reorder_params;

        auto onednn_weights_params = std::dynamic_pointer_cast<WeightsReorderParamsOneDNN>(weights_params);

        OPENVINO_ASSERT(impl_param.get_input_layout().bytes_count() == weights_params->get_input_layout().bytes_count(),
                        "[GPU] Input layout doesn't match required reorder weights layout");

        auto input_md = onednn_weights_params ? onednn_weights_params->_in_desc : onednn::layout_to_memory_desc(weights_params->get_input_layout());
        auto output_md = onednn_weights_params ? onednn_weights_params->_out_desc : onednn::layout_to_memory_desc(weights_params->get_output_layout());

        auto attr = std::make_shared<dnnl::primitive_attr>();
        auto reorder_prim = std::make_shared<dnnl::reorder::primitive_desc>(
            engine.get_onednn_engine(),
            input_md,
            engine.get_onednn_engine(),
            output_md,
            *attr);

        return cldnn::make_unique<reorder_onednn>(engine, impl_param.prog->get_config(), attr, *reorder_prim);
    }
};

struct reorder_factory : public cldnn::implementation_factory<reorder> {
    std::unique_ptr<primitive_impl> create(const program_node& node, const kernel_impl_params& params) const override {
        OPENVINO_ASSERT(node.is_type<reorder>());
        return onednn::reorder_onednn::create(static_cast<const reorder_node&>(node), params);
    }

    bool validate(const program_node& node) const override {
        OPENVINO_ASSERT(node.is_type<reorder>());
        return onednn::reorder_onednn::validate(static_cast<const reorder_node&>(node));
    }

    in_out_fmts_t query_formats(const program_node& node) const override {
        OPENVINO_NOT_IMPLEMENTED;
    }
};


namespace detail {

attach_reorder_onednn::attach_reorder_onednn() {
    implementation_map<reorder>::add(impl_types::onednn, cldnn::make_unique<reorder_factory>(), {});
    WeightsReordersFactory::add(cldnn::impl_types::onednn, shape_types::static_shape, reorder_onednn::create_reorder_weights);
}

}  // namespace detail
}  // namespace onednn
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::onednn::reorder_onednn)
