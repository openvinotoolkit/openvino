// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fully_connected_onednn.hpp"
#include "fully_connected_inst.h"
#include "intel_gpu/primitives/fully_connected.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "primitive_onednn_base.h"
#include "registry/implementation_manager.hpp"

#include <oneapi/dnnl/dnnl.hpp>

#include <algorithm>
#include <memory>
#include <cmath>
namespace cldnn {
namespace onednn {

struct fully_connected_onednn : typed_primitive_onednn_impl<fully_connected> {
    using parent = typed_primitive_onednn_impl<fully_connected>;
    using parent::parent;
    static constexpr int COMMON = 0;
    static constexpr int PER_OC = 2;
    static constexpr int PER_TENSOR = 7;
    static constexpr int GROUPED = 3;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::onednn::fully_connected_onednn)

private:
    int _ds_group_size;
    dnnl::memory::data_type _ds_data_type;
    dnnl::memory::data_type _dzp_data_type;

    static std::vector<int64_t> reshape_to_2d(const ov::PartialShape& shape, int64_t feature) {
        auto staticShape = shape.to_shape();
        size_t total =
            std::accumulate(staticShape.begin(), staticShape.end(), static_cast<size_t>(1), std::multiplies<size_t>());
        std::vector<int64_t> reshapeSize = { static_cast<int64_t>(total) / feature, feature };
        return reshapeSize;
    }

protected:
    std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<fully_connected_onednn>(*this);
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

        const auto& prim = instance.get_impl_params()->typed_desc<fully_connected>();
        if (prim->compressed_weights) {
            const auto weights_dt = instance.get_input_layout(1).data_type;
            auto weight_bitwidth = ov::element::Type(weights_dt).bitwidth();
            OPENVINO_ASSERT(weight_bitwidth == 8 || weight_bitwidth == 4, "[GPU] oneDNN supports only 4bit/8bit compressed weights");
            int idx = prim->bias.is_valid() ? 3 : 2;

            if (prim->decompression_scale.is_valid()) {
                auto decompression_scale_idx = idx++;
                auto scale_mem = instance.dep_memory_ptr(decompression_scale_idx);
                dnnl::memory::desc desc = onednn::layout_to_memory_desc(scale_mem->get_layout(), dnnl::memory::format_tag::a, onednn::mem_flags::flatten);
                args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, scale_mem->get_onednn_memory(desc)});
            }

            if (prim->decompression_zero_point.is_valid()) {
                auto decompression_zp_idx = idx++;
                auto zp_mem = instance.dep_memory_ptr(decompression_zp_idx);
                dnnl::memory::desc desc = onednn::layout_to_memory_desc(zp_mem->get_layout(), dnnl::memory::format_tag::a, onednn::mem_flags::flatten);
                args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, zp_mem->get_onednn_memory(desc)});
            }
            bool is_dyn_quan_input = instance.get_input_layout(0).data_type == data_types::i8 || instance.get_input_layout(0).data_type == data_types::u8;

            if (is_dyn_quan_input && prim->activation_scale.is_valid()) {
                auto activation_scale_idx = idx++;
                auto act_scale_mem = instance.dep_memory_ptr(activation_scale_idx);
                dnnl::memory::desc desc = onednn::layout_to_memory_desc(act_scale_mem->get_layout(), dnnl::memory::format_tag::ab, onednn::mem_flags::flatten);
                args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_0, act_scale_mem->get_onednn_memory(desc)});
            }

            if (is_dyn_quan_input && prim->activation_zero_point.is_valid()) {
                auto activation_zp_idx = idx++;
                auto act_zp_mem = instance.dep_memory_ptr(activation_zp_idx);
                dnnl::memory::desc desc = onednn::layout_to_memory_desc(act_zp_mem->get_layout(), dnnl::memory::format_tag::ab, onednn::mem_flags::flatten);
                args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC_0, act_zp_mem->get_onednn_memory(desc)});
            }
        }

        return args;
    }

    static void transform_layouts(layout& input_layout, layout& weights_layout, layout& output_layout, size_t prim_input_size) {
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
    }

    static std::shared_ptr<dnnl::matmul::primitive_desc>
        get_matmul_primitive_descriptor(const kernel_impl_params& impl_params,
                                        cldnn::engine& engine,
                                        size_t prim_input_size,
                                        size_t prim_weights_rank,
                                        bool has_bias,
                                        const dnnl::primitive_attr& attr = dnnl::primitive_attr()) {
        auto input_layout = impl_params.get_input_layout(0);
        auto weights_layout = impl_params.get_input_layout(1);
        auto output_layout = impl_params.get_output_layout();

        dnnl::memory::format_tag target_fmt;
        dnnl::memory::format_tag weights_fmt;

        if (prim_input_size == 3) {
            target_fmt = dnnl::memory::format_tag::abc;
            weights_fmt = dnnl::memory::format_tag::acb;
        } else if (prim_input_size == 4) {
            target_fmt = dnnl::memory::format_tag::abcd;
            weights_fmt = dnnl::memory::format_tag::abdc;
        } else if (prim_input_size == 5) {
            target_fmt = dnnl::memory::format_tag::abcde;
            weights_fmt = dnnl::memory::format_tag::abced;
        } else if (prim_input_size == 6) {
            target_fmt = dnnl::memory::format_tag::abcdef;
            weights_fmt = dnnl::memory::format_tag::abcdfe;
        } else {
            target_fmt = dnnl::memory::format_tag::ab;
            weights_fmt = dnnl::memory::format_tag::ba;
        }

        if (prim_input_size < 4) {
            auto output_pshape = output_layout.get_partial_shape();
            if (output_pshape.size() > prim_input_size) {
                output_pshape.resize(prim_input_size);
                output_layout.set_partial_shape(output_pshape);
            }
        }

        // Transform weights_layout according to input layout
        {
            ov::PartialShape new_weights_pshape;
            std::vector<ov::Dimension::value_type> lower_sizes;
            std::vector<ov::Dimension::value_type> upper_sizes;

            for (size_t i = 0; i < (prim_input_size - prim_weights_rank); i++) {
                new_weights_pshape.push_back(1);
                lower_sizes.push_back(0);
                upper_sizes.push_back(0);
            }

            for (size_t i = 0; i < prim_weights_rank; i++) {
                new_weights_pshape.push_back(weights_layout.get_partial_shape()[i]);
                lower_sizes.push_back(weights_layout.data_padding._lower_size[i]);
                upper_sizes.push_back(weights_layout.data_padding._upper_size[i]);
            }

            weights_layout.set_partial_shape(new_weights_pshape);
            weights_layout.data_padding = cldnn::padding(lower_sizes, upper_sizes);
            weights_layout.format = input_layout.format;
        }

        auto use_strides_for_weight_md = (weights_layout.data_padding
                                         && format::is_default_format(weights_layout.format)
                                         && (weights_layout.data_type == data_types::i4 || weights_layout.data_type == data_types::u4)) ?
                                         onednn::mem_flags::use_strides : onednn::mem_flags::None;

        dnnl::memory::desc input_md = onednn::layout_to_memory_desc(input_layout, target_fmt);
        dnnl::memory::desc weights_md = onednn::layout_to_memory_desc(weights_layout, weights_fmt, use_strides_for_weight_md);
        dnnl::memory::desc output_md = onednn::layout_to_memory_desc(output_layout, target_fmt);

        if (has_bias) {
            auto bias_l = impl_params.get_input_layout(2);
            auto bias_b_size = (bias_l.get_partial_shape().size() == 1) ? 1 : bias_l.batch();
            auto bias_f_size = static_cast<int32_t>(bias_l.get_tensor().count()) / bias_b_size;

            if (prim_input_size == 3) {
                bias_l.set_partial_shape({ 1, bias_b_size, bias_f_size });
            } else if (prim_input_size == 4) {
                bias_l.set_partial_shape({ 1, 1, bias_b_size, bias_f_size });
            } else if (prim_input_size == 5) {
                bias_l.set_partial_shape({ 1, 1, 1, bias_b_size, bias_f_size });
            } else if (prim_input_size == 6) {
                bias_l.set_partial_shape({ 1, 1, 1, 1, bias_b_size, bias_f_size });
            } else {
                bias_l.set_partial_shape({ bias_b_size, bias_f_size });
            }

            auto bias_md = onednn::layout_to_memory_desc(bias_l, target_fmt);
            return std::make_shared<dnnl::matmul::primitive_desc>(
                engine.get_onednn_engine(),
                input_md,
                weights_md,
                bias_md,
                output_md,
                attr);
        } else {
            return std::make_shared<dnnl::matmul::primitive_desc>(
                engine.get_onednn_engine(),
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
        size_t weights_rank = prim->weights_rank;
        bool has_bias = prim->bias.is_valid();
        bool is_compressed = prim->compressed_weights;
        ob << input_size;
        ob << weights_rank;
        ob << has_bias;
        ob << is_compressed;
        ob << prim->dynamic_quantized_activation;
        ob << prim->dynamic_quantized_activation_zp;

        bool has_decompression_scale = prim->decompression_scale.is_valid();
        if (has_decompression_scale) {
            ob << _ds_group_size;
            ob << make_data(&_ds_data_type, sizeof(dnnl::memory::data_type));
        }

        bool has_decompression_zp = prim->decompression_zero_point.is_valid() || prim->decompression_zero_point_scalar.has_value();
        if (has_decompression_zp) {
            ob << make_data(&_dzp_data_type, sizeof(dnnl::memory::data_type));
        }

        std::vector<uint8_t> prim_cache;
        prim_cache = _prim.get_cache_blob();
        ob << prim_cache;
#endif
    }

    void load(BinaryInputBuffer& ib) override {
#ifdef ONEDNN_PRIMITIVE_SERIALIZATION
        parent::load(ib);

        size_t input_size = 2;
        size_t weights_rank = 2;
        bool has_bias = false;
        bool is_compressed = false;
        bool dynamic_quantized_activation;
        bool dynamic_quantized_activation_zp;
        ib >> input_size;
        ib >> weights_rank;
        ib >> has_bias;
        ib >> is_compressed;
        ib >> dynamic_quantized_activation;
        ib >> dynamic_quantized_activation_zp;

        const kernel_impl_params* impl_params = reinterpret_cast<kernel_impl_params*>(ib.getKernelImplParams());
        auto prim = impl_params->typed_desc<fully_connected>();
        auto weights_layout = impl_params->get_input_layout(1);
        auto shift_size = std::max<size_t>(prim->input_size - 2, 0);
        auto& arg = impl_params->get_program().get_node(impl_params->desc->id).as<fully_connected>();
        int idx = !arg.bias_term() ? 1 : 2;
        int per_oc = PER_OC << shift_size;
        int grouped = GROUPED | (1 << (prim->input_size - 1));

        bool has_decompression_scale = prim->decompression_scale.is_valid();
        if (has_decompression_scale) {
            ib >> _ds_group_size;
            ib >> make_data(&_ds_data_type, sizeof(dnnl::memory::data_type));

            auto decompression_scale_idx = ++idx;
            auto scale_layout = arg.get_dependency(decompression_scale_idx).get_output_layout();
            auto ngroups = scale_layout.get_dim(1);
            if (scale_layout.count() == 1) {
                _attrs->set_scales(DNNL_ARG_WEIGHTS, COMMON, dnnl::memory::dims{}, _ds_data_type);
            } else if (ngroups == 1) {
                _attrs->set_scales(DNNL_ARG_WEIGHTS, per_oc, dnnl::memory::dims{}, _ds_data_type);
            } else {
                _attrs->set_scales(DNNL_ARG_WEIGHTS, grouped, {_ds_group_size, 1}, _ds_data_type);
            }
        }

        bool has_decompression_zp = prim->decompression_zero_point.is_valid() || prim->decompression_zero_point_scalar.has_value();
        if (has_decompression_zp) {
            ib >> make_data(&_dzp_data_type, sizeof(dnnl::memory::data_type));
            auto decompression_zp_idx = ++idx;
            auto dzp_layout = arg.get_dependency(decompression_zp_idx).get_output_layout();

            if (dzp_layout.count() == 1) {
                _attrs->set_zero_points(DNNL_ARG_WEIGHTS, COMMON, dnnl::memory::dims{}, _dzp_data_type);
            } else {
                auto ngroups = dzp_layout.get_dim(1);
                if (ngroups == 1) {
                    _attrs->set_zero_points(DNNL_ARG_WEIGHTS, per_oc, dnnl::memory::dims{}, _dzp_data_type);
                } else {
                    _attrs->set_zero_points(DNNL_ARG_WEIGHTS, grouped, {_ds_group_size, 1}, _dzp_data_type);
                }
            }
        }

        bool is_dyn_quan_input = impl_params->get_input_layout(0).data_type == data_types::i8 || impl_params->get_input_layout(0).data_type == data_types::u8;
        if (is_dyn_quan_input && dynamic_quantized_activation) {
            auto src_scale_idx = ++idx;
            auto partial_shape = impl_params->get_input_layout(0).get_partial_shape();
            auto innermost_len = partial_shape[partial_shape.size() - 1].get_length();
            auto& src_scale_shape = impl_params->input_layouts[src_scale_idx].get_partial_shape();
            int src_scale_ngroups = src_scale_shape[src_scale_shape.size() - 1].get_length();
            int src_group_size = innermost_len / src_scale_ngroups;

            auto act_scale_data_type = convert_data_type(impl_params->get_input_layout(src_scale_idx).data_type);
            _attrs->set_scales(DNNL_ARG_SRC, grouped, dnnl::memory::dims{1, src_group_size}, act_scale_data_type);
            if (dynamic_quantized_activation_zp)
                _attrs->set_zero_points(DNNL_ARG_SRC, grouped, dnnl::memory::dims{1, src_group_size}, dnnl::memory::data_type::u8);
        }

        auto prim_desc = get_matmul_primitive_descriptor(*impl_params, ib.get_engine(), input_size, weights_rank, has_bias, *_attrs);
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
        auto attr = impl_params.attrs_onednn;
        auto prim = impl_params.typed_desc<fully_connected>();
        int group_size = 0;
        dnnl::memory::data_type ds_data_type = dnnl::memory::data_type::undef;
        dnnl::memory::data_type dzp_data_type = dnnl::memory::data_type::undef;
        int idx = !arg.bias_term() ? 1 : 2;

        if (prim->compressed_weights) {
            bool is_dyn_quan_input = impl_params.get_input_layout(0).data_type == data_types::i8 || impl_params.get_input_layout(0).data_type == data_types::u8;
            if (is_dyn_quan_input) {
                OPENVINO_ASSERT(prim->input_size <= 3, "[GPU] Dynamic quantization for 4D matmul is not implemented");
            } else {
                attr->set_fpmath_mode(dnnl::fpmath_mode::f16, true);
            }

            auto weights_layout = impl_params.get_input_layout(1);
            auto weight_shape = weights_layout.get_partial_shape();
            auto weight_rank = std::count_if(weight_shape.begin(), weight_shape.end(), [](ov::Dimension d) { return d.get_length() > 1; });
            weight_rank = std::max(static_cast<int64_t>(2), weight_rank);
            OPENVINO_ASSERT(weight_rank <= 3, "Currently only weights with equal to or less than 3D is supported");
            auto shift_size = std::max<size_t>(prim->input_size - 2, 0);
            int per_oc = PER_OC << shift_size;
            int grouped = GROUPED | (1 << (prim->input_size - 1));

            if (prim->decompression_scale.is_valid()) {
                auto decompression_scale_idx = ++idx;
                auto scale_layout = arg.get_dependency(decompression_scale_idx).get_output_layout();
                ds_data_type = convert_data_type(scale_layout.data_type);
                auto ifm = arg.get_dependency(1).get_output_layout().get_dim(weight_rank - 1);
                auto ngroups = scale_layout.get_dim(weight_rank - 1);
                group_size = ifm / ngroups;
                OPENVINO_ASSERT((group_size == 1 || ngroups == 1 || group_size % 32 == 0),
                    "[GPU] group_size should be aligned to 32 if it is not a single scale group or the group_size is not one.");
                if (scale_layout.count() == 1) {
                    attr->set_scales(DNNL_ARG_WEIGHTS, COMMON, dnnl::memory::dims{}, ds_data_type);
                } else if (ngroups == 1 && weight_rank <= 2) {
                    attr->set_scales(DNNL_ARG_WEIGHTS, per_oc, dnnl::memory::dims{}, ds_data_type);
                } else {
                    // should use {K, 1} for the group size + per tensor mask for 3d
                    // Example:
                    // input[32, 6, 2088], W_t[32, 5760, 2088], scale[32, 1, 5760]
                    // set scale group as [32, 2088, 1]
                    attr->set_scales(DNNL_ARG_WEIGHTS, grouped, {group_size, 1}, ds_data_type);
                }
            }

            if (prim->decompression_zero_point.is_valid()) {
                auto decompression_zp_idx = ++idx;
                auto dzp_layout = arg.get_dependency(decompression_zp_idx).get_output_layout();
                dzp_data_type = convert_data_type(dzp_layout.data_type);

                if (dzp_layout.count() == 1) {
                    attr->set_zero_points(DNNL_ARG_WEIGHTS, COMMON, dnnl::memory::dims{}, dzp_data_type);
                } else {
                    size_t rank = dzp_layout.get_partial_shape().size();
                    auto ngroups = dzp_layout.get_dim(rank - 1);
                    if (ngroups == 1 && rank <= 2) {
                        attr->set_zero_points(DNNL_ARG_WEIGHTS, per_oc, dnnl::memory::dims{}, dzp_data_type);
                    } else {
                        // should use {K, 1} for the group size + per tensor mask for 3d
                        attr->set_zero_points(DNNL_ARG_WEIGHTS, grouped, {group_size, 1}, dzp_data_type);
                    }
                }
            }

            if (is_dyn_quan_input && prim->dynamic_quantized_activation) {
                auto src_scale_idx = ++idx;
                auto& partial_shape = impl_params.input_layouts[0].get_partial_shape();
                auto innermost_len = partial_shape[partial_shape.size() - 1].get_length();
                auto& src_scale_shape = impl_params.input_layouts[src_scale_idx].get_partial_shape();
                int src_scale_ngroups = src_scale_shape[src_scale_shape.size() - 1].get_length();
                int src_group_size = innermost_len / src_scale_ngroups;

                auto act_scale_data_type = convert_data_type(impl_params.input_layouts[src_scale_idx].data_type);
                attr->set_scales(DNNL_ARG_SRC, grouped, dnnl::memory::dims{1, src_group_size}, act_scale_data_type);

                if (prim->activation_zero_point.is_valid())
                    attr->set_zero_points(DNNL_ARG_SRC, grouped, dnnl::memory::dims{1, src_group_size}, dnnl::memory::data_type::u8);
            }


            auto prim_desc = get_matmul_primitive_descriptor(impl_params, impl_params.prog->get_engine(),
                                                             prim->input_size, prim->weights_rank, prim->bias.is_valid(), *attr);

            auto prim_onednn = std::make_unique<fully_connected_onednn>(engine, config, attr, *prim_desc);
            prim_onednn->_ds_group_size = group_size;
            prim_onednn->_ds_data_type = ds_data_type;
            prim_onednn->_dzp_data_type = dzp_data_type;
            return prim_onednn;
        } else {
            auto prim_desc = get_matmul_primitive_descriptor(impl_params, impl_params.prog->get_engine(),
                                                             prim->input_size, prim->weights_rank, prim->bias.is_valid(), *attr);

            return std::make_unique<fully_connected_onednn>(engine, config, attr, *prim_desc);
        }
    }
};

std::unique_ptr<primitive_impl> FullyConnectedImplementationManager::create_impl(const program_node& node, const kernel_impl_params& params) const {
    assert(node.is_type<fully_connected>());
    return onednn::fully_connected_onednn::create(static_cast<const fully_connected_node&>(node), params);
}

}  // namespace onednn
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::onednn::fully_connected_onednn)
