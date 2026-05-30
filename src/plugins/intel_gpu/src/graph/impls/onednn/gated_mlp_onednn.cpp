// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gated_mlp_onednn.hpp"

#include "primitive_onednn_base.h"
#include "utils.hpp"
#include "common/gated_mlp_iface.hpp"

#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_types.h>
#include <openvino/core/shape.hpp>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cstdlib>
#include <filesystem>

namespace {

#ifndef DNNL_ARG_WEIGHTS_GATE
#define DNNL_ARG_WEIGHTS_GATE DNNL_ARG_WEIGHTS_0
#endif
#ifndef DNNL_ARG_WEIGHTS_UP
#define DNNL_ARG_WEIGHTS_UP DNNL_ARG_WEIGHTS_1
#endif
#ifndef DNNL_ARG_WEIGHTS_DOWN
#define DNNL_ARG_WEIGHTS_DOWN DNNL_ARG_WEIGHTS_2
#endif

static dnnl_alg_kind_t to_dnnl_alg(ov::op::internal::GLU::GluType type) {
    switch (type) {
        case ov::op::internal::GLU::GluType::Swish: return dnnl_eltwise_swish;
        case ov::op::internal::GLU::GluType::Gelu: return dnnl_eltwise_gelu_erf;
        case ov::op::internal::GLU::GluType::Gelu_Tanh: return dnnl_eltwise_gelu_tanh;
        default: return dnnl_eltwise_swish;
    }
}

}  // namespace

namespace cldnn {
namespace onednn {

struct gated_mlp_onednn : typed_primitive_onednn_impl<gated_mlp> {
    using parent = typed_primitive_onednn_impl<gated_mlp>;
    using parent::parent;
    static constexpr int COMMON = 0;
    static constexpr int PER_OC = 2;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::onednn::gated_mlp_onednn)

protected:
    std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<gated_mlp_onednn>(*this);
    }

    std::unordered_map<int, dnnl::memory> get_arguments(gated_mlp_inst& instance) const override {
        std::unordered_map<int, dnnl::memory> args = parent::get_arguments(instance);

        auto src_mem = instance.dep_memory_ptr(0);
        auto wg_mem = instance.dep_memory_ptr(1);
        auto wu_mem = instance.dep_memory_ptr(2);
        auto wd_mem = instance.dep_memory_ptr(3);

        auto src_md = _pd.query_md(dnnl::query::exec_arg_md, DNNL_ARG_SRC);
        auto wg_md = _pd.query_md(dnnl::query::exec_arg_md, DNNL_ARG_WEIGHTS_GATE);
        auto wu_md = _pd.query_md(dnnl::query::exec_arg_md, DNNL_ARG_WEIGHTS_UP);
        auto wd_md = _pd.query_md(dnnl::query::exec_arg_md, DNNL_ARG_WEIGHTS_DOWN);

        auto src_off = onednn::get_offset(instance.get_input_layout(0), _pd.query_md(dnnl::query::exec_arg_md, DNNL_ARG_SRC));
        auto wg_off = onednn::get_offset(instance.get_input_layout(1), _pd.query_md(dnnl::query::exec_arg_md, DNNL_ARG_WEIGHTS_GATE));
        auto wu_off = onednn::get_offset(instance.get_input_layout(2), _pd.query_md(dnnl::query::exec_arg_md, DNNL_ARG_WEIGHTS_UP));
        auto wd_off = onednn::get_offset(instance.get_input_layout(3), _pd.query_md(dnnl::query::exec_arg_md, DNNL_ARG_WEIGHTS_DOWN));

        args[DNNL_ARG_SRC] = src_mem->get_onednn_memory(src_md, src_off);
        args[DNNL_ARG_WEIGHTS_GATE] = wg_mem->get_onednn_memory(wg_md, wg_off);
        args[DNNL_ARG_WEIGHTS_UP] = wu_mem->get_onednn_memory(wu_md, wu_off);
        args[DNNL_ARG_WEIGHTS_DOWN] = wd_mem->get_onednn_memory(wd_md, wd_off);

        const auto& prim = instance.get_impl_params()->typed_desc<gated_mlp>();
        if (prim->compressed_weights) {
            int idx = 4;

            auto scale_gate_mem = instance.dep_memory_ptr(idx++);
            auto scale_up_mem = instance.dep_memory_ptr(idx++);
            auto scale_down_mem = instance.dep_memory_ptr(idx++);

            auto make_scale_zp_desc = [](const layout& l, const char* name) {
                const auto& ps = l.get_partial_shape();
                GPU_DEBUG_TRACE << "GMLP_SCALE_DBG " << name << " shape=" << ps << " dt=" << l.data_type << std::endl;
                return onednn::layout_to_memory_desc_flatten(l, dnnl::memory::format_tag::a);
            };

            auto scale_desc_gate = make_scale_zp_desc(scale_gate_mem->get_layout(), "scale_gate");
            auto scale_desc_up = make_scale_zp_desc(scale_up_mem->get_layout(), "scale_up");
            auto scale_desc_down = make_scale_zp_desc(scale_down_mem->get_layout(), "scale_down");

            args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS_GATE] = scale_gate_mem->get_onednn_memory(scale_desc_gate);
            args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS_UP] = scale_up_mem->get_onednn_memory(scale_desc_up);
            args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS_DOWN] = scale_down_mem->get_onednn_memory(scale_desc_down);

            if (prim->decompression_zero_point_gate.is_valid()) {
                auto zp_gate_mem = instance.dep_memory_ptr(idx++);
                auto zp_up_mem = instance.dep_memory_ptr(idx++);
                auto zp_down_mem = instance.dep_memory_ptr(idx++);

                auto zp_desc_gate = make_scale_zp_desc(zp_gate_mem->get_layout(), "zp_gate");
                auto zp_desc_up = make_scale_zp_desc(zp_up_mem->get_layout(), "zp_up");
                auto zp_desc_down = make_scale_zp_desc(zp_down_mem->get_layout(), "zp_down");

                args[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS_GATE] = zp_gate_mem->get_onednn_memory(zp_desc_gate);
                args[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS_UP] = zp_up_mem->get_onednn_memory(zp_desc_up);
                args[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS_DOWN] = zp_down_mem->get_onednn_memory(zp_desc_down);
            }

            if (prim->dynamic_quantized_activation) {
                if (prim->activation_scale.is_valid()) {
                    auto act_scale_mem = instance.dep_memory_ptr(idx++);
                    dnnl::memory::desc act_scale_desc = onednn::layout_to_memory_desc_flatten(
                        act_scale_mem->get_layout(), dnnl::memory::format_tag::ab);
                    args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC] = act_scale_mem->get_onednn_memory(act_scale_desc);
                }
                if (prim->activation_zero_point.is_valid()) {
                    auto act_zp_mem = instance.dep_memory_ptr(idx++);
                    dnnl::memory::desc act_zp_desc = onednn::layout_to_memory_desc_flatten(
                        act_zp_mem->get_layout(), dnnl::memory::format_tag::ab);
                    args[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC] = act_zp_mem->get_onednn_memory(act_zp_desc);
                }
                if (prim->activation_precomputed_reduction.is_valid()) {
                    auto act_red_mem = instance.dep_memory_ptr(idx++);
                    dnnl::memory::desc act_red_desc = onednn::layout_to_memory_desc_flatten(
                        act_red_mem->get_layout(), dnnl::memory::format_tag::ab);
                    args[DNNL_ARG_ATTR_PRECOMPUTED_REDUCTIONS | DNNL_ARG_SRC] = act_red_mem->get_onednn_memory(act_red_desc);
                }
            }
        }
#if 0
        // DO NOT MERGE - Dump all gated_mlp inputs to /tmp/gmlp_dump/ when OV_GPU_GMLP_DUMP is set
        static int dump_call_count = 0;
        const char* dump_env = std::getenv("OV_GPU_GMLP_DUMP");
        if (dump_env && dump_call_count == 0) {
            const std::string dump_dir(dump_env);
            std::filesystem::create_directories(dump_dir);

            auto dump_mem = [&](memory::ptr mem, const std::string& name) {
                auto& stream = instance.get_network().get_stream();
                const auto& layout = mem->get_layout();
                const size_t bytes = layout.bytes_count();
                std::vector<uint8_t> buf(bytes);
                mem->copy_to(stream, buf.data(), false);
                stream.finish();

                std::ofstream f(dump_dir + "/" + name + ".bin", std::ios::binary);
                f.write(reinterpret_cast<const char*>(buf.data()), bytes);
                f.close();

                std::cerr << "GMLP_DUMP: " << name << " shape=" << layout.get_partial_shape()
                          << " dt=" << layout.data_type << " bytes=" << bytes << std::endl;
            };

            dump_mem(src_mem, "src");
            dump_mem(wg_mem, "w_gate");
            dump_mem(wu_mem, "w_up");
            dump_mem(wd_mem, "w_down");

            if (prim->compressed_weights) {
                int didx = 4;
                dump_mem(instance.dep_memory_ptr(didx++), "scale_gate");
                dump_mem(instance.dep_memory_ptr(didx++), "scale_up");
                dump_mem(instance.dep_memory_ptr(didx++), "scale_down");
                if (prim->decompression_zero_point_gate.is_valid()) {
                    dump_mem(instance.dep_memory_ptr(didx++), "zp_gate");
                    dump_mem(instance.dep_memory_ptr(didx++), "zp_up");
                    dump_mem(instance.dep_memory_ptr(didx++), "zp_down");
                }
            }

            // Also dump output after execution for reference comparison
            std::cerr << "GMLP_DUMP: dumped layer " << dump_call_count << " inputs to " << dump_dir << std::endl;
        }
        dump_call_count++;
#endif
        return args;
    }

    public:
    static std::unique_ptr<primitive_impl> create(const gated_mlp_node& arg, const kernel_impl_params& impl_params) {
        auto& engine = impl_params.prog->get_engine();
        auto& config = impl_params.prog->get_config();
        auto attr = impl_params.attrs_onednn;
        auto prim = impl_params.typed_desc<gated_mlp>();

        auto to_2d_md = [](const layout& l, const char* tensor_name) {
            const auto& ps = l.get_partial_shape();
            OPENVINO_ASSERT(ps.rank().is_static() && ps.rank().get_length() >= 2,
                            "[GPU] gated_mlp expects rank >= 2 for ", tensor_name);
            OPENVINO_ASSERT(ps.is_static(),
                            "[GPU] gated_mlp expects static shape for ", tensor_name,
                            " at oneDNN primitive descriptor creation");

            const auto shape = ps.to_shape();
            const auto inner = static_cast<dnnl::memory::dim>(shape.back());
            const auto outer = static_cast<dnnl::memory::dim>(ov::shape_size(shape) / shape.back());
            return dnnl::memory::desc({outer, inner}, onednn::convert_data_type(l.data_type), dnnl::memory::format_tag::ab);
        };

        // Weights are [OC, IC] in OpenVINO but oneDNN expects [IC, OC] in ba format.
        // {inner, outer} with ba gives the same physical layout as {outer, inner} with ab.
        auto to_2d_weight_md = [](const layout& l, const char* tensor_name) {
            const auto& ps = l.get_partial_shape();
            OPENVINO_ASSERT(ps.rank().is_static() && ps.rank().get_length() >= 2,
                            "[GPU] gated_mlp expects rank >= 2 for ", tensor_name);
            OPENVINO_ASSERT(ps.is_static(),
                            "[GPU] gated_mlp expects static shape for ", tensor_name,
                            " at oneDNN primitive descriptor creation");

            const auto shape = ps.to_shape();
            const auto inner = static_cast<dnnl::memory::dim>(shape.back());
            const auto outer = static_cast<dnnl::memory::dim>(ov::shape_size(shape) / shape.back());
            return dnnl::memory::desc({inner, outer}, onednn::convert_data_type(l.data_type), dnnl::memory::format_tag::ba);
        };

        auto src_md = to_2d_md(impl_params.get_input_layout(0), "src");
        auto wg_md = to_2d_weight_md(impl_params.get_input_layout(1), "weights_gate");
        auto wu_md = to_2d_weight_md(impl_params.get_input_layout(2), "weights_up");
        auto wd_md = to_2d_weight_md(impl_params.get_input_layout(3), "weights_down");
        auto dst_md = to_2d_md(impl_params.get_output_layout(0), "dst");

        if (prim->compressed_weights) {
            static constexpr int GROUPED = 3;

            attr->set_fpmath_mode(dnnl::fpmath_mode::any, true);

            auto set_scales = [&](int arg, const layout& weight_layout, const layout& scale_layout) {
                const auto scale_dt = onednn::convert_data_type(scale_layout.data_type);
                if (scale_layout.count() == 1) {
                    attr->set_scales(arg, COMMON, dnnl::memory::dims{}, scale_dt);
                    return;
                }

                // Weight is [OC, IC], scale is [ngroups, OC] after transpose in fuse_gated_mlp
                const auto ifm = weight_layout.get_dim(1);
                const auto ngroups = scale_layout.get_dim(0);
                OPENVINO_ASSERT(ngroups > 0, "[GPU] Invalid grouped scale layout for gated_mlp: ngroups is zero");
                OPENVINO_ASSERT(ifm % ngroups == 0,
                                "[GPU] Invalid grouped scale layout for gated_mlp: ifm ", ifm,
                                " is not divisible by ngroups ", ngroups);
                const auto group_size = ifm / ngroups;

                if (ngroups == 1) {
                    attr->set_scales(arg, PER_OC, dnnl::memory::dims{}, scale_dt);
                } else {
                    attr->set_scales(arg, GROUPED, dnnl::memory::dims{group_size, 1}, scale_dt);
                }
            };

            set_scales(DNNL_ARG_WEIGHTS_GATE, impl_params.get_input_layout(1), impl_params.get_input_layout(4));
            set_scales(DNNL_ARG_WEIGHTS_UP, impl_params.get_input_layout(2), impl_params.get_input_layout(5));
            set_scales(DNNL_ARG_WEIGHTS_DOWN, impl_params.get_input_layout(3), impl_params.get_input_layout(6));

            if (prim->decompression_zero_point_gate.is_valid()) {
                auto set_zero_points = [&](int arg, const layout& weight_layout, const layout& zp_layout) {
                    const auto zp_dt = onednn::convert_data_type(zp_layout.data_type);
                    if (zp_layout.count() == 1) {
                        attr->set_zero_points(arg, COMMON, dnnl::memory::dims{}, zp_dt);
                        return;
                    }

                    // Weight is [OC, IC], zp is [ngroups, OC] after transpose in fuse_gated_mlp
                    const auto ifm = weight_layout.get_dim(1);
                    const auto ngroups = zp_layout.get_dim(0);
                    OPENVINO_ASSERT(ngroups > 0, "[GPU] Invalid grouped zero-point layout for gated_mlp: ngroups is zero");
                    OPENVINO_ASSERT(ifm % ngroups == 0,
                                    "[GPU] Invalid grouped zero-point layout for gated_mlp: ifm ", ifm,
                                    " is not divisible by ngroups ", ngroups);
                    const auto group_size = ifm / ngroups;

                    if (ngroups == 1) {
                        attr->set_zero_points(arg, PER_OC, dnnl::memory::dims{}, zp_dt);
                    } else {
                        attr->set_zero_points(arg, GROUPED, dnnl::memory::dims{group_size, 1}, zp_dt);
                    }
                };

                set_zero_points(DNNL_ARG_WEIGHTS_GATE, impl_params.get_input_layout(1), impl_params.get_input_layout(7));
                set_zero_points(DNNL_ARG_WEIGHTS_UP, impl_params.get_input_layout(2), impl_params.get_input_layout(8));
                set_zero_points(DNNL_ARG_WEIGHTS_DOWN, impl_params.get_input_layout(3), impl_params.get_input_layout(9));
            }

            // Dynamic quantized activation (src scales/zp/precomputed_reduction)
            if (prim->dynamic_quantized_activation && prim->activation_scale.is_valid()) {
                int act_idx = 10;  // activation_scale is at index 10
                const auto& act_scale_layout = impl_params.get_input_layout(act_idx);
                const auto act_scale_dt = onednn::convert_data_type(act_scale_layout.data_type);
                const auto src_innermost = impl_params.get_input_layout(0).get_dim(impl_params.get_input_layout(0).get_partial_shape().size() - 1);
                const auto src_scale_ngroups = act_scale_layout.get_dim(act_scale_layout.get_partial_shape().size() - 1);
                const int64_t src_group_size = src_innermost / src_scale_ngroups;

                attr->set_scales(DNNL_ARG_SRC, GROUPED, dnnl::memory::dims{1, src_group_size}, act_scale_dt);

                if (prim->activation_zero_point.is_valid()) {
                    const auto& act_zp_layout = impl_params.get_input_layout(act_idx + 1);
                    attr->set_zero_points(DNNL_ARG_SRC, GROUPED, dnnl::memory::dims{1, src_group_size},
                                          onednn::convert_data_type(act_zp_layout.data_type));
                }

                if (prim->activation_precomputed_reduction.is_valid()) {
                    const auto& act_red_layout = impl_params.get_input_layout(act_idx + 2);
                    attr->set_precomputed_reductions(DNNL_ARG_SRC, GROUPED, dnnl::memory::dims{1, src_group_size},
                                                    onednn::convert_data_type(act_red_layout.data_type));
                }
            }
        }

        dnnl_primitive_desc_t c_pd = nullptr;
        auto activation = to_dnnl_alg(prim->activation);
        auto status = dnnl_gated_mlp_primitive_desc_create(&c_pd,
                                                           engine.get_onednn_engine().get(),
                                                           src_md.get(),
                                                           wg_md.get(),
                                                           wu_md.get(),
                                                           wd_md.get(),
                                                           dst_md.get(),
                                                           activation,
                                                           attr->get());

        if (status != dnnl_success || c_pd == nullptr) {
            std::ostringstream failure_msg;
            failure_msg << "[GPU] Failed to create oneDNN gated_mlp primitive descriptor"
                << " status=" << static_cast<int>(status)
                << " (" << onednn::dnnl_status_to_string(status) << ")"
                << " c_pd=" << c_pd
                << " activation=" << static_cast<int>(activation)
                << " compressed_weights=" << prim->compressed_weights
                << " has_decompression_zero_points=" << prim->decompression_zero_point_gate.is_valid()
                << " src layout=" << impl_params.get_input_layout(0).to_short_string()
                << " md=" << onednn::memory_desc_to_string(src_md)
                << " wg layout=" << impl_params.get_input_layout(1).to_short_string()
                << " md=" << onednn::memory_desc_to_string(wg_md)
                << " wu layout=" << impl_params.get_input_layout(2).to_short_string()
                << " md=" << onednn::memory_desc_to_string(wu_md)
                << " wd layout=" << impl_params.get_input_layout(3).to_short_string()
                << " md=" << onednn::memory_desc_to_string(wd_md)
                << " dst layout=" << impl_params.get_output_layout(0).to_short_string()
                << " md=" << onednn::memory_desc_to_string(dst_md);
            OPENVINO_ASSERT(false, failure_msg.str());
        }

        dnnl::primitive_desc prim_desc(c_pd);

        return std::make_unique<gated_mlp_onednn>(engine, config, attr, prim_desc);
    }
};

std::unique_ptr<primitive_impl> GatedMLPImplementationManager::create_impl(const program_node& node, const kernel_impl_params& params) const {
    assert(node.is_type<gated_mlp>());
    return gated_mlp_onednn::create(static_cast<const gated_mlp_node&>(node), params);
}

}  // namespace onednn
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::onednn::gated_mlp_onednn)
