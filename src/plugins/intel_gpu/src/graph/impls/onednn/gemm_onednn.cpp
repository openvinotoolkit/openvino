// Copyright (C) 2018-2025 Intel Corporation
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

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::onednn::gemm_onednn)

protected:
    std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<gemm_onednn>(*this);
    }

    std::unordered_map<int, dnnl::memory> get_arguments(gemm_inst& instance) const override {
        std::unordered_map<int, dnnl::memory> args = parent::get_arguments(instance);
        auto& engine = instance.get_network().get_engine();
        auto dnnl_engine = engine.get_onednn_engine();

        {
            dnnl::memory input1_mem;
            auto& weights = instance.input_memory(1);
            auto offset = onednn::get_offset(instance.get_input_layout(1), _pd.dnnl::primitive_desc_base::weights_desc(0));
            if (instance.get_input_layout(1).count() != 0) {
                input1_mem = weights.get_onednn_memory(_pd.weights_desc(0), offset);
            }
            args.insert({DNNL_ARG_WEIGHTS, input1_mem});
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
            // Whitelist format from transpose-gemm optimizing out
            case dnnl::memory::format_tag::acbd: return dnnl::memory::format_tag::acdb;
            case dnnl::memory::format_tag::adbc: return dnnl::memory::format_tag::adcb;
            default: throw std::runtime_error("Unsupported fmt in transpose_format gemm function");
        }
    }

    static void get_gemm_primitive_md(const kernel_impl_params& impl_params,
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
                                      dnnl::memory::format_tag& bias_fmt) {
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

        bool batched_dims_can_be_removed = false;

        if (in0_l.count() != 0 && in1_l.count() != 0) {
            size_t in0_batched_size = in0_l.count() / (in0_l.spatial(0) * in0_l.spatial(1));
            size_t in1_batched_size = in1_l.count() / (in1_l.spatial(0) * in1_l.spatial(1));
            size_t out_batched_size = out_l.count() / (out_l.spatial(0) * out_l.spatial(1));

            batched_dims_can_be_removed = in0_batched_size == 1 && in1_batched_size == 1 && out_batched_size == 1;
        }

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

        if (in0_l.data_padding) {
            dnnl::memory::dims in0_padded_dims = onednn::convert_gemm_dims(in0_l.get_padded_dims(), rank, batched_dims_can_be_removed);
            in0_strides = onednn::get_strides(in0_padded_dims);
            if (prim->transpose_input0) {
                std::swap(in0_strides[in0_strides.size() - 1], in0_strides[in0_strides.size() - 2]);
            }
        }

        if (in1_l.data_padding) {
            dnnl::memory::dims in1_padded_dims = onednn::convert_gemm_dims(in1_l.get_padded_dims(), rank, batched_dims_can_be_removed);
            in1_strides = onednn::get_strides(in1_padded_dims);
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

        auto transpose_dims_and_format_tag = [](std::vector<int64_t> transpose_order,
                                                dnnl::memory::dims& dims,
                                                dnnl::memory::format_tag& tag,
                                                bool is_input = true) {
            std::vector<size_t> order(std::begin(transpose_order), std::end(transpose_order));
            if (dims.size() > order.size()) {
                size_t orders_to_add = dims.size() - order.size();
                for (size_t i = 0; i < orders_to_add; ++i)
                    order.insert(order.begin(), i);
                for (size_t i = orders_to_add; i < order.size(); ++i)
                    order[i] = order[i] + orders_to_add;
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
            } else {
                std::ostringstream ostream;
                std::copy(order.begin(), order.end(), std::ostream_iterator<size_t>(ostream, ", "));
                OPENVINO_ASSERT(false, "[GPU] Can't find fusable transpose format: ", ostream.str());
            }
        };

        if (has_transpose_order(prim->input0_transpose_order)) {
            if (has_transpose_order_xy_only(prim->input0_transpose_order)) {
                in0_fmt = transpose_format(in0_fmt);
                std::swap(in0_dims[in0_dims.size() - 1], in0_dims[in0_dims.size() - 2]);
            } else {
                transpose_dims_and_format_tag(prim->input0_transpose_order, in0_dims, in0_fmt);
            }
        }
        if (has_transpose_order(prim->input1_transpose_order)) {
            if (has_transpose_order_xy_only(prim->input1_transpose_order)) {
                in1_fmt = transpose_format(in1_fmt);
                std::swap(in1_dims[in1_dims.size() - 1], in1_dims[in1_dims.size() - 2]);
            } else {
                transpose_dims_and_format_tag(prim->input1_transpose_order, in1_dims, in1_fmt);
            }
        }
        if (has_transpose_order(prim->output_transpose_order)) {
            if (has_transpose_order_xy_only(prim->output_transpose_order)) {
                out_fmt = transpose_format(out_fmt);
                std::swap(out_dims[out_dims.size() - 1], out_dims[out_dims.size() - 2]);
            } else {
                transpose_dims_and_format_tag(prim->output_transpose_order, out_dims, out_fmt, false);
            }
        }

        if (gemm_with_bias) {
            auto bias_l = impl_params.get_input_layout(2);
            auto bias_rank = cldnn::format::dimension(bias_l.format);
            bias_dt = onednn::convert_data_type(bias_l.data_type);
            bias_dims = onednn::convert_gemm_tensor(bias_l.get_tensor(), bias_rank, batched_dims_can_be_removed);
            bias_fmt = onednn::convert_gemm_data_format(bias_dims, bias_l.format);
        }
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

        get_gemm_primitive_md(impl_params, in0_dt, in1_dt, out_dt, in0_dims, in1_dims, in0_strides, in1_strides,
                              out_dims, in0_fmt, in1_fmt, out_fmt, gemm_with_bias, bias_dt, bias_dims, bias_fmt);

        dnnl::memory::desc in0_md = get_input_memory_desc(in0_dims, in0_dt, in0_fmt, in0_strides);
        dnnl::memory::desc in1_md = get_input_memory_desc(in1_dims, in1_dt, in1_fmt, in1_strides);
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

        dnnl::memory::dims in0_strides;
        dnnl::memory::dims in1_strides;

        dnnl::memory::format_tag in0_fmt;
        dnnl::memory::format_tag in1_fmt;
        dnnl::memory::format_tag out_fmt;
        dnnl::memory::format_tag bias_fmt;

        get_gemm_primitive_md(*impl_params, in0_dt, in1_dt, out_dt, in0_dims, in1_dims, in0_strides, in1_strides,
                              out_dims, in0_fmt, in1_fmt, out_fmt, gemm_with_bias, bias_dt, bias_dims, bias_fmt);

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

        _prim = dnnl::primitive(_pd, prim_cache);
#endif
    }

    static std::unique_ptr<primitive_impl> create(const gemm_node& arg, const kernel_impl_params& impl_params) {
        auto& engine = impl_params.prog->get_engine();
        auto& config = impl_params.prog->get_config();
        auto attr = impl_params.attrs_onednn;
        auto prim_desc = get_gemm_primitive_descriptor(impl_params, *attr);

        return std::make_unique<gemm_onednn>(engine, config, attr, *prim_desc);
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, typed_primitive_inst<gemm>& instance) override {
        if (instance.get_input_layout(0).count() == 0 ||
            instance.get_input_layout(1).count() == 0) {
            stream& stream = instance.get_network().get_stream();
            stream.enqueue_barrier();
            return instance.output_memory_ptr()->fill(stream, false);
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
