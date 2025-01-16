// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fully_connected_onednn.hpp"
#include "fully_connected_inst.h"
#include "intel_gpu/primitives/fully_connected.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "primitive_onednn_base.h"
#include "impls/registry/implementation_manager.hpp"

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

        const auto& prim = instance.get_impl_params()->typed_desc<fully_connected>();
        if (prim->compressed_weights) {
            const auto weights_dt = instance.get_input_layout(1).data_type;
            auto weight_bitwidth = ov::element::Type(weights_dt).bitwidth();
            OPENVINO_ASSERT(weight_bitwidth == 8 || weight_bitwidth == 4, "[GPU] oneDNN supports only 4bit/8bit compressed weights");
            int idx = prim->bias.empty() ? 2 : 3;

            if (!prim->decompression_scale.empty()) {
                auto decompression_scale_idx = idx++;
                auto scale_mem = instance.dep_memory_ptr(decompression_scale_idx);
                dnnl::memory::desc desc = onednn::layout_to_memory_desc(scale_mem->get_layout(), dnnl::memory::format_tag::a, true);
                args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, scale_mem->get_onednn_memory(desc)});
            }

            if (!prim->decompression_zero_point.empty()) {
                auto decompression_zp_idx = idx++;
                auto zp_mem = instance.dep_memory_ptr(decompression_zp_idx);
                dnnl::memory::desc desc = onednn::layout_to_memory_desc(zp_mem->get_layout(), dnnl::memory::format_tag::a, true);
                args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, zp_mem->get_onednn_memory(desc)});
            }

            if (prim->activation_scale.is_valid()) {
                auto activation_scale_idx = idx++;
                auto act_scale_mem = instance.dep_memory_ptr(activation_scale_idx);
                // TODO: handle group_size here
                dnnl::memory::desc desc = onednn::layout_to_memory_desc(act_scale_mem->get_layout(), dnnl::memory::format_tag::a, true);
                args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_0, act_scale_mem->get_onednn_memory(desc)});
            }
        }

        return args;
    }

    static std::shared_ptr<WeightsReorderParams> get_weights_reorder(const kernel_impl_params& impl_params, const dnnl::primitive_desc& pd) {
        auto input_layout = impl_params.get_input_layout(0);
        auto source_weights_layout = impl_params.get_input_layout(1);
        auto cldnn_prim = impl_params.typed_desc<fully_connected>();

        auto input_pshape = input_layout.get_partial_shape();
        auto weights_pshape = source_weights_layout.get_partial_shape();

        int64_t feature = input_pshape[std::min(cldnn_prim->input_size, static_cast<size_t>(4)) - 1].get_length();
        if (cldnn_prim->input_size == 3) {
            feature = std::max({input_layout.spatial(0), input_layout.spatial(1), input_layout.spatial(2)});
        }
        auto target_weights_layout = source_weights_layout;
        if (weights_pshape.size() != 2) {
            target_weights_layout.set_partial_shape(reshape_to_2d(weights_pshape, feature));
        }

        auto target_weights_desc = pd.weights_desc(0);

        auto shape_consistent = onednn::keep_weights_reorder_shape_consistent(source_weights_layout, target_weights_desc);
        OPENVINO_ASSERT(shape_consistent, "[GPU] Input shape and output shape of weight reorder should be same.");

        auto source_weights_desc = onednn::layout_to_memory_desc(source_weights_layout);

        const bool weights_format = true;
        const bool grouped = false;

        auto traits = convert_memory_desc_to_traits(target_weights_desc, weights_format, grouped);

        target_weights_layout.format = format(traits);

        return std::make_shared<WeightsReorderParamsOneDNN>(source_weights_layout,
                                                            target_weights_layout,
                                                            source_weights_desc,
                                                            target_weights_desc,
                                                            false);
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

    static std::shared_ptr<dnnl::inner_product_forward::primitive_desc>
        get_inner_product_primitive_descriptor(const kernel_impl_params& impl_params,
                                               cldnn::engine& engine,
                                               size_t prim_input_size,
                                               bool has_bias,
                                               const dnnl::primitive_attr& attr = dnnl::primitive_attr()) {
        auto input_layout = impl_params.get_input_layout(0);
        auto weights_layout = impl_params.get_input_layout(1);
        auto output_layout = impl_params.get_output_layout();

        transform_layouts(input_layout, weights_layout, output_layout, prim_input_size);

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

    static std::shared_ptr<dnnl::matmul::primitive_desc>
        get_matmul_primitive_descriptor(const kernel_impl_params& impl_params,
                                        cldnn::engine& engine,
                                        size_t prim_input_size,
                                        bool has_bias,
                                        const dnnl::primitive_attr& attr = dnnl::primitive_attr()) {
        auto input_layout = impl_params.get_input_layout(0);
        auto weights_layout = impl_params.get_input_layout(1);
        auto output_layout = impl_params.get_output_layout();

        transform_layouts(input_layout, weights_layout, output_layout, prim_input_size);

        auto input_md = onednn::layout_to_memory_desc(input_layout, dnnl::memory::format_tag::ab, false);
        // TODO: should change format to any. May need a reorder.
        auto weights_md = onednn::layout_to_memory_desc(weights_layout, dnnl::memory::format_tag::ba);
        auto output_md = onednn::layout_to_memory_desc(output_layout, dnnl::memory::format_tag::ab, false);

        if (has_bias) {
            auto bias_md = onednn::layout_to_memory_desc(impl_params.get_input_layout(2), dnnl::memory::format_tag::ab, false);
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
        bool has_bias = !prim->bias.empty();
        bool is_compressed = prim->compressed_weights;
        ob << input_size;
        ob << has_bias;
        ob << is_compressed;
        ob << prim->dynamic_quantized_activation;

        bool has_decompression_scale = !prim->decompression_scale.empty();
        if (has_decompression_scale) {
            ob << _ds_group_size;
            ob << make_data(&_ds_data_type, sizeof(dnnl::memory::data_type));
        }

        bool has_decompression_zp = !prim->decompression_zero_point.empty() || prim->decompression_zero_point_scalar.has_value();
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
        bool has_bias = false;
        bool is_compressed = false;
        bool dynamic_quantized_activation;
        ib >> input_size;
        ib >> has_bias;
        ib >> is_compressed;
        ib >> dynamic_quantized_activation;

        const kernel_impl_params* impl_params = reinterpret_cast<kernel_impl_params*>(ib.getKernelImplParams());
        auto prim = impl_params->typed_desc<fully_connected>();
        auto weights_layout = impl_params->get_input_layout(1);
        bool is_four_bit_weight = weights_layout.data_type == data_types::u4 || weights_layout.data_type == data_types::i4;

        bool has_decompression_scale = !prim->decompression_scale.empty();
        if (has_decompression_scale) {
            ib >> _ds_group_size;
            ib >> make_data(&_ds_data_type, sizeof(dnnl::memory::data_type));
            if (!is_four_bit_weight)
                _attrs->set_scales(DNNL_ARG_WEIGHTS, PER_OC, dnnl::memory::dims{}, _ds_data_type);
            else
                _attrs->set_scales(DNNL_ARG_WEIGHTS, GROUPED, {_ds_group_size, 1}, _ds_data_type);
        }

        bool has_decompression_zp = !prim->decompression_zero_point.empty() || prim->decompression_zero_point_scalar.has_value();
        auto& arg = impl_params->get_program().get_node(impl_params->desc->id).as<fully_connected>();
        int idx = !arg.bias_term() ? 3 : 4;

        if (has_decompression_zp) {
            ib >> make_data(&_dzp_data_type, sizeof(dnnl::memory::data_type));
            auto dzp_layout = arg.get_dependency(idx++).get_output_layout();

            if (dzp_layout.count() == 1) {
                _attrs->set_zero_points(DNNL_ARG_WEIGHTS, COMMON, dnnl::memory::dims{}, _dzp_data_type);
            } else {
                auto ngroups = dzp_layout.get_dim(1);
                if (ngroups == 1) {
                    _attrs->set_zero_points(DNNL_ARG_WEIGHTS, PER_OC, dnnl::memory::dims{}, _dzp_data_type);
                } else {
                    _attrs->set_zero_points(DNNL_ARG_WEIGHTS, GROUPED, {_ds_group_size, 1}, _dzp_data_type);
                }
            }
        }

        if (dynamic_quantized_activation) {
            // TODO: it supports per-token activation scale only
            auto partial_shape = impl_params->get_input_layout(0).get_partial_shape();
            auto innermost_len = partial_shape[partial_shape.size() - 1].get_length();

            auto act_scale_data_type = convert_data_type(impl_params->get_input_layout(idx).data_type);
            _attrs->set_scales(DNNL_ARG_SRC, GROUPED, dnnl::memory::dims{1, innermost_len}, act_scale_data_type);
        }

        if (is_compressed) {
            auto prim_desc = get_matmul_primitive_descriptor(*impl_params, ib.get_engine(), input_size, has_bias, *_attrs);
            _pd = *prim_desc;
        } else {
            auto prim_desc = get_inner_product_primitive_descriptor(*impl_params, ib.get_engine(), input_size, has_bias, *_attrs);
            _pd = *prim_desc;
        }

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
        bool is_four_bit_weight = false;
        int idx = !arg.bias_term() ? 1 : 2;

        // There may be a performance difference between InnerProduct and MatMul primitives in oneDNN,
        // so use MatMul only for weights compression and IP for all other cases.
        if (prim->compressed_weights) {
            attr->set_fpmath_mode(dnnl::fpmath_mode::f16, true);
            auto weights_layout = impl_params.get_input_layout(1);
            is_four_bit_weight = weights_layout.data_type == data_types::u4 || weights_layout.data_type == data_types::i4;
            if (!prim->decompression_scale.empty()) {
                auto decompression_scale_idx = ++idx;
                ds_data_type = convert_data_type(arg.get_dependency(decompression_scale_idx).get_output_layout().data_type);
                auto ifm = arg.get_dependency(1).get_output_layout().get_dim(1);
                auto ngroups = arg.get_dependency(decompression_scale_idx).get_output_layout().get_dim(1);
                group_size = ifm / ngroups;
                if (!is_four_bit_weight) {
                    // 8-bit quantized weight
                    attr->set_scales(DNNL_ARG_WEIGHTS, PER_OC, dnnl::memory::dims{}, ds_data_type);
                } else {
                    // OneDNN does not support scalar zero-point for s4 and u8 type. Need to broadcast it.
                    attr->set_scales(DNNL_ARG_WEIGHTS, GROUPED, {group_size, 1}, ds_data_type);
                }
            }

            if (!prim->decompression_zero_point.empty()) {
                auto decompression_zp_idx = ++idx;
                auto dzp_layout = arg.get_dependency(decompression_zp_idx).get_output_layout();
                dzp_data_type = convert_data_type(dzp_layout.data_type);

                if (dzp_layout.count() == 1) {
                    attr->set_zero_points(DNNL_ARG_WEIGHTS, COMMON, dnnl::memory::dims{}, dzp_data_type);
                } else {
                    auto ngroups = dzp_layout.get_dim(1);
                    if (ngroups == 1) {
                        attr->set_zero_points(DNNL_ARG_WEIGHTS, PER_OC, dnnl::memory::dims{}, dzp_data_type);
                    } else {
                        attr->set_zero_points(DNNL_ARG_WEIGHTS, GROUPED, {group_size, 1}, dzp_data_type);
                    }
                }
            }

            if (prim->dynamic_quantized_activation) {
                // Note: it supports per-token activation scale only
                ++idx;
                auto partial_shape = impl_params.input_layouts[0].get_partial_shape();
                auto innermost_len = partial_shape[partial_shape.size() - 1].get_length();

                auto act_scale_data_type = convert_data_type(impl_params.input_layouts[idx].data_type);
                attr->set_scales(DNNL_ARG_SRC, GROUPED, dnnl::memory::dims{1, innermost_len}, act_scale_data_type);
            }

            auto prim_desc = get_matmul_primitive_descriptor(impl_params, impl_params.prog->get_engine(),
                                                             prim->input_size, !prim->bias.empty(), *attr);

            auto prim_onednn = cldnn::make_unique<fully_connected_onednn>(engine, config, attr, *prim_desc);
            prim_onednn->_ds_group_size = group_size;
            prim_onednn->_ds_data_type = ds_data_type;
            prim_onednn->_dzp_data_type = dzp_data_type;
            return prim_onednn;
        } else {
            auto prim_desc = get_inner_product_primitive_descriptor(impl_params, impl_params.prog->get_engine(),
                                                                    prim->input_size, !prim->bias.empty(), *attr);

            return cldnn::make_unique<fully_connected_onednn>(engine, config, attr, *prim_desc, get_weights_reorder(impl_params, *prim_desc));
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
