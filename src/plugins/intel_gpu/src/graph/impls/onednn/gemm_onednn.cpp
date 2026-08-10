// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gemm_onednn.hpp"
#include "gemm_inst.h"
#include "intel_gpu/runtime/utils.hpp"
#include "primitive_onednn_base.h"

#include <oneapi/dnnl/dnnl.hpp>

#include <algorithm>
#include <memory>
namespace cldnn {
namespace onednn {

struct gemm_onednn : typed_primitive_onednn_impl<gemm> {
    using parent = typed_primitive_onednn_impl<gemm>;
    using parent::parent;
    // (no per-instance cached memory descs needed at the moment)

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::onednn::gemm_onednn)

protected:
    std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<gemm_onednn>(*this);
    }

    std::unordered_map<int, dnnl::memory> get_arguments(gemm_inst& instance) const override {
        std::unordered_map<int, dnnl::memory> args = parent::get_arguments(instance);

        const bool is_dynamic = instance.get_node().is_dynamic();
        const bool gemm_with_bias = instance.inputs_memory_count() == 3;

        dnnl::memory::desc in0_md = _pd.src_desc();
        dnnl::memory::desc in1_md = _pd.weights_desc(0);
        dnnl::memory::desc out_md = _pd.dst_desc();
        dnnl::memory::desc bias_md;

        if (is_dynamic) {
            auto impl_params = instance.get_impl_params();
            OPENVINO_ASSERT(impl_params != nullptr, "[GPU] Missing impl params for dynamic oneDNN GEMM");

            dnnl::memory::data_type in0_dt;
            dnnl::memory::data_type in1_dt;
            dnnl::memory::data_type out_dt;
            dnnl::memory::data_type bias_dt = dnnl::memory::data_type::undef;

            dnnl::memory::dims in0_dims;
            dnnl::memory::dims in1_dims;
            dnnl::memory::dims out_dims;
            dnnl::memory::dims bias_dims;

            dnnl::memory::dims in0_strides;
            dnnl::memory::dims in1_strides;

            dnnl::memory::format_tag in0_fmt;
            dnnl::memory::format_tag in1_fmt;
            dnnl::memory::format_tag out_fmt;
            dnnl::memory::format_tag bias_fmt = dnnl::memory::format_tag::undef;

            const bool md_ok = get_gemm_primitive_md(*impl_params,
                                                     in0_dt,
                                                     in1_dt,
                                                     out_dt,
                                                     in0_dims,
                                                     in1_dims,
                                                     in0_strides,
                                                     in1_strides,
                                                     out_dims,
                                                     in0_fmt,
                                                     in1_fmt,
                                                     out_fmt,
                                                     gemm_with_bias,
                                                     bias_dt,
                                                     bias_dims,
                                                     bias_fmt,
                                                     /*is_update=*/true);
            OPENVINO_ASSERT(md_ok, "[GPU] oneDNN GEMM runtime memory descriptor creation failed");

            in0_md = get_input_memory_desc(in0_dims, in0_dt, in0_fmt, in0_strides);
            in1_md = get_input_memory_desc(in1_dims, in1_dt, in1_fmt, in1_strides);
            out_md = dnnl::memory::desc(out_dims, out_dt, out_fmt);
            if (gemm_with_bias) {
                bias_md = dnnl::memory::desc(bias_dims, bias_dt, bias_fmt);
            }
        }

        {
            dnnl::memory input0_mem;
            auto& input0 = instance.input_memory(0);
            auto offset0 = onednn::get_offset(instance.get_input_layout(0), in0_md);
            if (instance.get_input_layout(0).count() != 0) {
                input0_mem = input0.get_onednn_memory(in0_md, offset0);
            }
            args.insert_or_assign(DNNL_ARG_SRC, input0_mem);
        }
        {
            dnnl::memory input1_mem;
            auto& weights = instance.input_memory(1);
            auto offset = onednn::get_offset(instance.get_input_layout(1), in1_md);
            if (instance.get_input_layout(1).count() != 0) {
                input1_mem = weights.get_onednn_memory(in1_md, offset);
            }
            args.insert_or_assign(DNNL_ARG_WEIGHTS, input1_mem);
        }
        {
            auto& output = instance.output_memory();
            auto offset = onednn::get_offset(instance.get_output_layout(), out_md);
            args.insert_or_assign(DNNL_ARG_DST, output.get_onednn_memory(out_md, offset));
        }

        if (gemm_with_bias) {
            auto& weights = instance.input_memory(2);
            auto selected_bias_md = is_dynamic ? bias_md : _pd.weights_desc(1);
            auto offset = onednn::get_offset(instance.get_input_layout(2), selected_bias_md);
            args.insert_or_assign(DNNL_ARG_BIAS, weights.get_onednn_memory(selected_bias_md, offset));
        }

        return args;
    }

    void set_arguments_impl(gemm_inst& instance) override {
        if (instance.can_be_optimized())
            return;

        if (instance.get_input_layout(0).count() == 0 ||
            instance.get_input_layout(1).count() == 0) {
            return;
        }

        uint32_t net_id = instance.get_network().get_id();
        _args[net_id] = get_arguments(instance);
    }

    static dnnl::memory::format_tag transpose_format(dnnl::memory::format_tag fmt) {
        switch (fmt) {
            case dnnl::memory::format_tag::ab: return dnnl::memory::format_tag::ba;
            case dnnl::memory::format_tag::abc: return dnnl::memory::format_tag::acb;
            case dnnl::memory::format_tag::abcd: return dnnl::memory::format_tag::abdc;
            case dnnl::memory::format_tag::abcde: return dnnl::memory::format_tag::abced;
            case dnnl::memory::format_tag::abcdef: return dnnl::memory::format_tag::abcdfe;
            // Whitelist format from transpose-gemm optimizing out
            case dnnl::memory::format_tag::acbd: return dnnl::memory::format_tag::acdb;
            case dnnl::memory::format_tag::adbc: return dnnl::memory::format_tag::adcb;
            default: throw std::runtime_error("Unsupported fmt in transpose_format gemm function");
        }
    }

    // Returns false if transpose order is not supported by oneDNN
    static bool get_gemm_primitive_md(const kernel_impl_params& impl_params,
                                      dnnl::memory::data_type& in0_dt,
                                      dnnl::memory::data_type& in1_dt,
                                      dnnl::memory::data_type& out_dt,
                                      dnnl::memory::dims& in0_dims,
                                      dnnl::memory::dims& in1_dims,
                                      dnnl::memory::dims& in0_strides,
                                      dnnl::memory::dims& in1_strides,
                                      dnnl::memory::dims& out_dims,
                                      dnnl::memory::format_tag& in0_fmt,
                                      dnnl::memory::format_tag& in1_fmt,
                                      dnnl::memory::format_tag& out_fmt,
                                      bool gemm_with_bias,
                                      dnnl::memory::data_type& bias_dt,
                                      dnnl::memory::dims& bias_dims,
                                      dnnl::memory::format_tag& bias_fmt,
                                      bool is_update) {
        auto prim = impl_params.typed_desc<gemm>();
        auto out_l = impl_params.get_output_layout();

        std::vector<layout> in_layouts { impl_params.get_input_layout(0), impl_params.get_input_layout(1) };
        if (gemm_with_bias) {
            in_layouts.emplace_back(impl_params.get_input_layout(2));
        }
        in_layouts = gemm_inst::transform_input_layouts(prim, in_layouts, impl_params.get_program().is_new_shape_infer());
        out_l = gemm_inst::transform_output_layout(prim, in_layouts, out_l);

        const auto& in0_l = in_layouts[0];
        const auto& in1_l = in_layouts[1];

        bool is_dynamic_shape = in0_l.is_dynamic() || in1_l.is_dynamic() || out_l.is_dynamic();
        if (gemm_with_bias) {
            is_dynamic_shape = is_dynamic_shape || in_layouts[2].is_dynamic();
        }

        bool batched_dims_can_be_removed = !is_update;
        size_t rank = cldnn::format::dimension(out_l.format);

        in0_dt = onednn::convert_data_type(in0_l.data_type);
        in1_dt = onednn::convert_data_type(in1_l.data_type);
        out_dt = onednn::convert_data_type(out_l.data_type);

        in0_dims = onednn::convert_layout(in0_l, rank);
        in1_dims = onednn::convert_layout(in1_l, rank);
        out_dims = onednn::convert_layout(out_l, rank);

        in0_fmt = onednn::convert_gemm_data_format(in0_dims, in0_l.format);
        in1_fmt = onednn::convert_gemm_data_format(in1_dims, in1_l.format);
        out_fmt = onednn::convert_gemm_data_format(out_dims, out_l.format);

        if (in0_l.data_padding) {
            in0_strides = onednn::get_strides(in0_l.get_padded_dims());
            if (prim->transpose_input0) {
                std::swap(in0_strides[in0_strides.size() - 1], in0_strides[in0_strides.size() - 2]);
            }
        }

        if (in1_l.data_padding) {
            in1_strides = onednn::get_strides(in1_l.get_padded_dims());
            if (prim->transpose_input1)
                std::swap(in1_strides[in1_strides.size() - 1], in1_strides[in1_strides.size() - 2]);
        }

        // Check whether transpose_order increase sequential or not.
        // Return true when transpose_order is not 0, 1, 2, 3.
        auto has_transpose_order = [](std::vector<int64_t> transpose_order) {
            for (size_t i = 0; i < transpose_order.size(); i++) {
                if (i != static_cast<size_t>(transpose_order[i])) {
                    return true;
                }
            }
            return false;
        };

        // Check whether transpose_order has transpose only to the last two elements.
        // Return true when transpose_order is 0, 1, 3, 2.
        auto has_transpose_order_xy_only = [](std::vector<int64_t> transpose_order) {
            for (size_t i = 0; i < transpose_order.size() - 2; i++) {
                if (i != static_cast<size_t>(transpose_order[i])) {
                    return false;
                }
            }
            size_t last_idx = transpose_order.size() - 1;
            if (static_cast<size_t>(transpose_order[last_idx]) != last_idx - 1)
                return false;
            if (static_cast<size_t>(transpose_order[last_idx - 1]) != last_idx)
                return false;
            return true;
        };

        auto transpose_dims_and_format_tag = [&has_transpose_order_xy_only](std::vector<int64_t> transpose_order,
                                                dnnl::memory::dims& dims,
                                                dnnl::memory::format_tag& tag,
                                                bool is_input = true) -> bool {
            std::vector<size_t> order(std::begin(transpose_order), std::end(transpose_order));
            if (dims.size() > order.size()) {
                size_t orders_to_add = dims.size() - order.size();
                for (size_t i = 0; i < orders_to_add; ++i)
                    order.insert(order.begin(), i);
                for (size_t i = orders_to_add; i < order.size(); ++i)
                    order[i] = order[i] + orders_to_add;
            }

            // Convert order to int64_t vector for has_transpose_order_xy_only check
            std::vector<int64_t> order_int64(order.begin(), order.end());

            // Check if the expanded order represents xy-only transpose
            if (has_transpose_order_xy_only(order_int64)) {
                // Helper lambda to transpose format tag
                auto transpose_fmt = [](dnnl::memory::format_tag fmt) -> dnnl::memory::format_tag {
                    using tag = dnnl::memory::format_tag;
                    switch (fmt) {
                        case tag::ab: return tag::ba;
                        case tag::ba: return tag::ab;
                        case tag::abc: return tag::acb;
                        case tag::acb: return tag::abc;
                        case tag::abcd: return tag::abdc;
                        case tag::abdc: return tag::abcd;
                        case tag::abcde: return tag::abced;
                        case tag::abced: return tag::abcde;
                        case tag::abcdef: return tag::abcdfe;
                        case tag::abcdfe: return tag::abcdef;
                        default: return fmt;
                    }
                };
                tag = transpose_fmt(tag);
                std::swap(dims[dims.size() - 1], dims[dims.size() - 2]);
                return true;
            }

            bool ret = false;
            format transposed_format = format::bfyx;
            if (is_input) {
                ret = gemm_inst::is_fusable_permute_input_order_onednn(order, transposed_format);
            } else {
                ret = gemm_inst::is_fusable_permute_output_order_onednn(order, transposed_format);
            }

            if (ret) {
                tag = convert_data_format(transposed_format);
                dnnl::memory::dims original_dims = dims;
                if (is_input) {
                    for (size_t i = 0; i < original_dims.size(); ++i) {
                        dims[i] = original_dims[order[i]];
                    }
                } else {
                    // Get non-transposed dims for output dims
                    for (size_t i = 0; i < original_dims.size(); ++i) {
                        dims[order[i]] = original_dims[i];
                    }
                }
                return true;
            } else {
                // oneDNN GEMM doesn't support this transpose order - return false to skip this implementation
                return false;
            }
        };

        if (has_transpose_order(prim->input0_transpose_order)) {
            if (has_transpose_order_xy_only(prim->input0_transpose_order)) {
                in0_fmt = transpose_format(in0_fmt);
                std::swap(in0_dims[in0_dims.size() - 1], in0_dims[in0_dims.size() - 2]);
            } else {
                if (!transpose_dims_and_format_tag(prim->input0_transpose_order, in0_dims, in0_fmt)) {
                    return false;  // Unsupported transpose order
                }
            }
        }
        if (has_transpose_order(prim->input1_transpose_order)) {
            if (has_transpose_order_xy_only(prim->input1_transpose_order)) {
                in1_fmt = transpose_format(in1_fmt);
                std::swap(in1_dims[in1_dims.size() - 1], in1_dims[in1_dims.size() - 2]);
            } else {
                if (!transpose_dims_and_format_tag(prim->input1_transpose_order, in1_dims, in1_fmt)) {
                    return false;  // Unsupported transpose order
                }
            }
        }
        if (has_transpose_order(prim->output_transpose_order)) {
            if (has_transpose_order_xy_only(prim->output_transpose_order)) {
                out_fmt = transpose_format(out_fmt);
                std::swap(out_dims[out_dims.size() - 1], out_dims[out_dims.size() - 2]);
            } else {
                if (!transpose_dims_and_format_tag(prim->output_transpose_order, out_dims, out_fmt, false)) {
                    return false;  // Unsupported transpose order
                }
            }
        }

        if (gemm_with_bias) {
            // Use the already-transformed layout (in_layouts[2]), not the raw impl_params layout.
            // bias_dims is already set correctly in both the dynamic path (convert_gemm_shape)
            // and the static path (convert_gemm_tensor) above; do not overwrite it here.
            const auto& bias_l = in_layouts[2];
            bias_dt = onednn::convert_data_type(bias_l.data_type);
            bias_fmt = onednn::convert_gemm_data_format(bias_dims, bias_l.format);
        }

        return true;  // Success
    }

    static dnnl::memory::desc get_input_memory_desc(const dnnl::memory::dims& dims,
                                                    dnnl::memory::data_type dt,
                                                    dnnl::memory::format_tag fmt,
                                                    const dnnl::memory::dims& strides) {
        dnnl::memory::desc res;
        if (strides.empty()) {
            res = dnnl::memory::desc(dims, dt, fmt);
        } else {
            res = dnnl::memory::desc(dims, dt, strides);
        }
        return res;
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

        dnnl::memory::dims in0_strides;
        dnnl::memory::dims in1_strides;

        dnnl::memory::format_tag in0_fmt;
        dnnl::memory::format_tag in1_fmt;
        dnnl::memory::format_tag out_fmt;
        dnnl::memory::format_tag bias_fmt;

        if (!get_gemm_primitive_md(impl_params, in0_dt, in1_dt, out_dt, in0_dims, in1_dims, in0_strides, in1_strides,
                                   out_dims, in0_fmt, in1_fmt, out_fmt, gemm_with_bias, bias_dt, bias_dims, bias_fmt, false)) {
            // Transpose order not supported by oneDNN - return nullptr to skip this implementation
            return nullptr;
        }

        dnnl::memory::desc in0_md = get_input_memory_desc(in0_dims, in0_dt, in0_fmt, in0_strides);
        dnnl::memory::desc in1_md = get_input_memory_desc(in1_dims, in1_dt, in1_fmt, in1_strides);
        dnnl::memory::desc out_md(out_dims, out_dt, out_fmt);

        // Verify binary post-op rank matches the (potentially folded) matmul output rank.
        {
            auto post_ops = attr.get_post_ops();
            for (int i = 0; i < post_ops.len(); i++) {
                if (post_ops.kind(i) == dnnl::primitive::kind::binary) {
                    dnnl::algorithm alg;
                    dnnl::memory::desc src1_desc;
                    post_ops.get_params_binary(i, alg, src1_desc);
                    if (src1_desc.get_ndims() != out_md.get_ndims())
                        return nullptr;
                }
            }
        }

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

        dnnl::memory::dims in0_strides;
        dnnl::memory::dims in1_strides;

        dnnl::memory::format_tag in0_fmt;
        dnnl::memory::format_tag in1_fmt;
        dnnl::memory::format_tag out_fmt;
        dnnl::memory::format_tag bias_fmt;

        if (!get_gemm_primitive_md(*impl_params, in0_dt, in1_dt, out_dt, in0_dims, in1_dims, in0_strides, in1_strides,
                                   out_dims, in0_fmt, in1_fmt, out_fmt, gemm_with_bias, bias_dt, bias_dims, bias_fmt, false)) {
            OPENVINO_ASSERT(false, "[GPU] oneDNN GEMM save failed: transpose order not supported");
        }

        ob << make_data(&in0_dt, sizeof(dnnl::memory::data_type));
        ob << make_data(&in1_dt, sizeof(dnnl::memory::data_type));
        ob << make_data(&out_dt, sizeof(dnnl::memory::data_type));

        ob << in0_dims;
        ob << in1_dims;
        ob << out_dims;

        ob << in0_strides;
        ob << in1_strides;

        ob << make_data(&in0_fmt, sizeof(dnnl::memory::format_tag));
        ob << make_data(&in1_fmt, sizeof(dnnl::memory::format_tag));
        ob << make_data(&out_fmt, sizeof(dnnl::memory::format_tag));

        if (gemm_with_bias) {
            ob << make_data(&bias_dt, sizeof(dnnl::memory::data_type));
            ob << bias_dims;
            ob << make_data(&bias_fmt, sizeof(dnnl::memory::format_tag));
        }

        std::vector<uint8_t> prim_cache;
        try {
            prim_cache = _prim.get_cache_blob();
        } catch (const dnnl::error&) {
            prim_cache.clear();
        }
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

        dnnl::memory::dims in0_strides;
        dnnl::memory::dims in1_strides;

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

        ib >> in0_strides;
        ib >> in1_strides;

        ib >> make_data(&in0_fmt, sizeof(dnnl::memory::format_tag));
        ib >> make_data(&in1_fmt, sizeof(dnnl::memory::format_tag));
        ib >> make_data(&out_fmt, sizeof(dnnl::memory::format_tag));

        if (gemm_with_bias) {
            ib >> make_data(&bias_dt, sizeof(dnnl::memory::data_type));
            ib >> bias_dims;
            ib >> make_data(&bias_fmt, sizeof(dnnl::memory::format_tag));
        }

        dnnl::memory::desc in0_md = get_input_memory_desc(in0_dims, in0_dt, in0_fmt, in0_strides);
        dnnl::memory::desc in1_md = get_input_memory_desc(in1_dims, in1_dt, in1_fmt, in1_strides);
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

        if (prim_cache.empty()) {
            _prim = dnnl::primitive(_pd);
        } else {
            _prim = dnnl::primitive(_pd, prim_cache);
        }
#endif
    }

    static std::unique_ptr<primitive_impl> create(const gemm_node& arg, const kernel_impl_params& impl_params) {
        auto& engine = impl_params.prog->get_engine();
        auto& config = impl_params.prog->get_config();
        auto attr = impl_params.attrs_onednn;

        if (impl_params.get_input_layout(0).count() == 0 ||
            impl_params.get_input_layout(1).count() == 0) {
            return std::make_unique<gemm_onednn>(engine);
        }

        auto prim_desc = get_gemm_primitive_descriptor(impl_params, *attr);
        OPENVINO_ASSERT(prim_desc != nullptr,
                        "[GPU] oneDNN GEMM primitive descriptor creation failed. "
                        "This configuration must be rejected in validate_impl.");

        return std::make_unique<gemm_onednn>(engine, config, attr, *prim_desc);
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, typed_primitive_inst<gemm>& instance) override {
        if (instance.get_input_layout(0).count() == 0 ||
            instance.get_input_layout(1).count() == 0) {
            stream& stream = instance.get_network().get_stream();
            stream.enqueue_barrier();
            return instance.output_memory_ptr()->fill(stream, {}, false);
        }

        return parent::execute_impl(events, instance);
    }
};

std::unique_ptr<primitive_impl> GemmImplementationManager::create_impl(const program_node& node, const kernel_impl_params& params) const  {
    assert(node.is_type<gemm>());
    return onednn::gemm_onednn::create(static_cast<const gemm_node&>(node), params);
}

}  // namespace onednn
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::onednn::gemm_onednn)
