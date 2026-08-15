// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vk_program.hpp"

#include "runtime/except.hpp"

#include <cstring>
#include <numeric>
#include <ranges>
#include <string_view>

namespace ov::core {
namespace vulkan {
namespace cross_platform {

namespace {

// Builtin kernel id for every op that has a native kernel.
std::string_view ir_op_kernel_name(const ir_node& node) {
    switch (node.op) {
        case ir_op::relu: return "relu_f32";
        case ir_op::add: return "eltwise_add_f32";
        case ir_op::max_pool: return "maxpool_f32";
        case ir_op::avg_pool: return "avgpool_f32";
        case ir_op::convolution: return "conv2d_f32";
        case ir_op::matmul:
            return node.matmul_transpose_b ? "matmul_transpose_b_f32" : "matmul_f32";
        default: return {};
    }
}

// Base id used to build unique per-node kernel ids.
std::string_view ir_op_base_name(ir_op op) {
    switch (op) {
        case ir_op::relu: return "relu";
        case ir_op::add: return "eltwise_add";
        case ir_op::max_pool: return "maxpool";
        case ir_op::avg_pool: return "avgpool";
        case ir_op::convolution: return "conv2d";
        case ir_op::matmul: return "matmul";
        default: return {};
    }
}

size_t element_count(std::span<const size_t> shape) {
    return std::ranges::fold_left(shape, size_t{1}, std::multiplies<size_t>{});
}

// Push-constant scalars are the pod argument values of the native kernels,
// in clspv reflection order (relu/add: total; maxpool: ih..pw,total_out;
// conv: ih..och,total_out; matmul: M,N,K,total_out).
scalar_desc make_u32_scalar(uint32_t value) {
    scalar_desc s;
    s.t = scalar_t::UINT32;
    s.v.u32 = value;
    return s;
}

}  // namespace

vk_program_builder::vk_program_builder(vk_engine& engine, const ExecutionConfig& config)
    : _engine(engine)
    , _stream(engine.create_stream(config))
    , _kern_builder(engine.create_kernel_builder()) {}

vk_memory_ptr vk_program_builder::make_memory(const std::vector<size_t>& shape, const std::string& id, bool host_visible) {
    const layout l(shape, 4);  // f32 buffers
    const auto alloc_type = host_visible ? allocation_type::usm_host : _engine.get_default_allocation_type();
    auto mem = _engine.allocate_memory(l, alloc_type, true);
    _engine.get_service_stream().flush();
    return mem;
}

// Raw-byte buffer (element size 1): used for quantized weight payloads, which
// are uploaded verbatim and unpacked by the kernel.
vk_memory_ptr vk_program_builder::make_memory_bytes(size_t bytes, const std::string& id, bool host_visible) {
    const layout l({bytes}, 1);
    const auto alloc_type = host_visible ? allocation_type::usm_host : _engine.get_default_allocation_type();
    auto mem = _engine.allocate_memory(l, alloc_type, true);
    _engine.get_service_stream().flush();
    return mem;
}

vk_kernel_ptr vk_program_builder::get_build_kernel(std::string_view kernel_id) {
    const std::string key(kernel_id);
    auto it = _kernels.find(key);
    if (it != _kernels.end())
        return it->second;

    std::vector<vk_kernel_ptr> out;
    _kern_builder->build_native_kernel(key, out);
    OPENVINO_ASSERT(out.size() == 1, "[GPU] vk_program: expected 1 kernel from native store, got ", out.size());
    _kernels[key] = out[0];
    return out[0];
}

std::shared_ptr<vk_program> vk_program_builder::build(const ir_graph& graph) {
    auto prog = std::make_shared<vk_program>();
    auto unique_id = [count = 0](const std::string& base) mutable {
        return base + "_" + std::to_string(count++);
    };
    auto shape_of = [&graph](const std::string& id) -> const std::vector<size_t>& {
        auto it = graph.tensor_shapes.find(id);
        OPENVINO_ASSERT(it != graph.tensor_shapes.end(), "[GPU] vk_program: unknown tensor ", id);
        return it->second;
    };

    for (const auto& node : graph.nodes) {
        switch (node.op) {
            case ir_op::parameter: {
                prog->memories[node.id] = make_memory(shape_of(node.id), node.id, true);
                continue;
            }
            case ir_op::constant: {
                // Quantized weights are uploaded as raw block bytes and unpacked
                // in-shader; f32 constants go through the plain lock() path.
                auto quant_it = graph.quant_constants.find(node.id);
                if (quant_it != graph.quant_constants.end()) {
                    auto mem = make_memory_bytes(quant_it->second.bytes.size(), node.id, true);
                    void* ptr = mem->lock();
                    std::memcpy(ptr, quant_it->second.bytes.data(), quant_it->second.bytes.size());
                    mem->unlock();
                    prog->memories[node.id] = mem;
                    continue;
                }
                auto data_it = graph.constant_data.find(node.id);
                OPENVINO_ASSERT(data_it != graph.constant_data.end(),
                                "[GPU] vk_program: constant ", node.id, " has no data");
                // Host-visible so the constant can be written via lock() at
                // build time (device-local buffers cannot be locked yet).
                auto mem = make_memory(shape_of(node.id), node.id, true);
                void* ptr = mem->lock();
                const size_t bytes = data_it->second.size() * sizeof(float);
                std::memcpy(ptr, data_it->second.data(), bytes);
                mem->unlock();
                prog->memories[node.id] = mem;
                continue;
            }
            case ir_op::result:
                continue;
            default:
                break;
        }

        std::string_view kernel_id = ir_op_kernel_name(node);
        OPENVINO_ASSERT(!kernel_id.empty(), "[GPU] vk_program: unsupported op ", node.id);

        std::shared_ptr<vk_program_node> node_op;
        switch (node.op) {
            case ir_op::relu: {
                const auto& out_shape = shape_of(node.id);
                node_op = std::make_shared<vk_program_node>();
                node_op->input_ids = {node.inputs.at(0)};
                node_op->scalars = {make_u32_scalar(static_cast<uint32_t>(element_count(out_shape)))};
                break;
            }
            case ir_op::add: {
                const auto& out_shape = shape_of(node.id);
                node_op = std::make_shared<vk_program_node>();
                node_op->input_ids = {node.inputs.at(0), node.inputs.at(1)};
                node_op->scalars = {make_u32_scalar(static_cast<uint32_t>(element_count(out_shape)))};
                break;
            }
            case ir_op::max_pool: {
                const auto& in_shape = shape_of(node.inputs.at(0));
                const auto& out_shape = shape_of(node.id);
                OPENVINO_ASSERT(node.pool.kernel.size() == 2, "[GPU] vk_program: MaxPool only supports 2D kernels");
                OPENVINO_ASSERT(node.pool.strides.size() == 2 && node.pool.pads_begin.size() == 2,
                                "[GPU] vk_program: MaxPool expects 2D kernel/strides/pads");
                OPENVINO_ASSERT(in_shape.size() == 4 && out_shape.size() == 4, "[GPU] vk_program: MaxPool expects NCHW");
                node_op = std::make_shared<vk_program_node>();
                node_op->input_ids = {node.inputs.at(0)};
                node_op->scalars = {make_u32_scalar(static_cast<uint32_t>(in_shape[2])),
                                    make_u32_scalar(static_cast<uint32_t>(in_shape[3])),
                                    make_u32_scalar(static_cast<uint32_t>(out_shape[2])),
                                    make_u32_scalar(static_cast<uint32_t>(out_shape[3])),
                                    make_u32_scalar(static_cast<uint32_t>(node.pool.kernel[0])),
                                    make_u32_scalar(static_cast<uint32_t>(node.pool.kernel[1])),
                                    make_u32_scalar(static_cast<uint32_t>(node.pool.strides[0])),
                                    make_u32_scalar(static_cast<uint32_t>(node.pool.strides[1])),
                                    make_u32_scalar(static_cast<uint32_t>(node.pool.pads_begin[0])),
                                    make_u32_scalar(static_cast<uint32_t>(node.pool.pads_begin[1])),
                                    make_u32_scalar(static_cast<uint32_t>(element_count(out_shape))),
                                    make_u32_scalar(static_cast<uint32_t>(in_shape[0]))};
                break;
            }
            case ir_op::avg_pool: {
                const auto& in_shape = shape_of(node.inputs.at(0));
                const auto& out_shape = shape_of(node.id);
                OPENVINO_ASSERT(node.pool.kernel.size() == 2, "[GPU] vk_program: AvgPool only supports 2D kernels");
                OPENVINO_ASSERT(node.pool.strides.size() == 2 && node.pool.pads_begin.size() == 2,
                                "[GPU] vk_program: AvgPool expects 2D kernel/strides/pads");
                OPENVINO_ASSERT(in_shape.size() == 4 && out_shape.size() == 4, "[GPU] vk_program: AvgPool expects NCHW");
                node_op = std::make_shared<vk_program_node>();
                node_op->input_ids = {node.inputs.at(0)};
                node_op->scalars = {make_u32_scalar(static_cast<uint32_t>(in_shape[2])),
                                    make_u32_scalar(static_cast<uint32_t>(in_shape[3])),
                                    make_u32_scalar(static_cast<uint32_t>(out_shape[2])),
                                    make_u32_scalar(static_cast<uint32_t>(out_shape[3])),
                                    make_u32_scalar(static_cast<uint32_t>(node.pool.kernel[0])),
                                    make_u32_scalar(static_cast<uint32_t>(node.pool.kernel[1])),
                                    make_u32_scalar(static_cast<uint32_t>(node.pool.strides[0])),
                                    make_u32_scalar(static_cast<uint32_t>(node.pool.strides[1])),
                                    make_u32_scalar(static_cast<uint32_t>(node.pool.pads_begin[0])),
                                    make_u32_scalar(static_cast<uint32_t>(node.pool.pads_begin[1])),
                                    make_u32_scalar(static_cast<uint32_t>(element_count(out_shape))),
                                    make_u32_scalar(static_cast<uint32_t>(in_shape[0]))};
                break;
            }
            case ir_op::convolution: {
                const auto& in_shape = shape_of(node.inputs.at(0));
                const auto& w_shape = shape_of(node.inputs.at(1));
                const auto& out_shape = shape_of(node.id);
                OPENVINO_ASSERT(in_shape.size() == 4 && w_shape.size() == 4 && out_shape.size() == 4,
                                "[GPU] vk_program: Convolution expects NCHW");
                OPENVINO_ASSERT(node.pool.pads_begin.size() == 2, "[GPU] vk_program: Convolution expects 2D pads");
                OPENVINO_ASSERT(node.pool.strides.size() == 2, "[GPU] vk_program: Convolution expects 2D strides");
                node_op = std::make_shared<vk_program_node>();
                node_op->input_ids = {node.inputs.at(0)};
                node_op->scalars = {make_u32_scalar(static_cast<uint32_t>(in_shape[2])),
                                    make_u32_scalar(static_cast<uint32_t>(in_shape[3])),
                                    make_u32_scalar(static_cast<uint32_t>(out_shape[2])),
                                    make_u32_scalar(static_cast<uint32_t>(out_shape[3])),
                                    make_u32_scalar(static_cast<uint32_t>(in_shape[1])),
                                    make_u32_scalar(static_cast<uint32_t>(w_shape[2])),
                                    make_u32_scalar(static_cast<uint32_t>(w_shape[3])),
                                    make_u32_scalar(static_cast<uint32_t>(w_shape[0])),
                                    make_u32_scalar(static_cast<uint32_t>(node.pool.pads_begin[0])),
                                    make_u32_scalar(static_cast<uint32_t>(node.pool.pads_begin[1])),
                                    make_u32_scalar(static_cast<uint32_t>(node.pool.strides[0])),
                                    make_u32_scalar(static_cast<uint32_t>(node.pool.strides[1])),
                                    make_u32_scalar(static_cast<uint32_t>(element_count(out_shape))),
                                    make_u32_scalar(static_cast<uint32_t>(in_shape[0]))};
                break;
            }
            case ir_op::matmul: {
                const auto& a_shape = shape_of(node.inputs.at(0));
                const auto& b_shape = shape_of(node.inputs.at(1));
                const auto& out_shape = shape_of(node.id);
                OPENVINO_ASSERT(a_shape.size() == 2 && b_shape.size() == 2 && out_shape.size() == 2,
                                "[GPU] vk_program: MatMul supports 2D activations only");
                // With transpose_b=false, B is stored [K,N]; with transpose_b=true it is
                // stored [N,K], so the inner dimension is b_shape[1] in that case.
                const size_t b_inner = node.matmul_transpose_b ? b_shape[1] : b_shape[0];
                OPENVINO_ASSERT(a_shape[1] == b_inner, "[GPU] vk_program: MatMul shapes mismatch (",
                                a_shape[0], "x", a_shape[1], " vs ", b_shape[0], "x", b_shape[1], ")");
                node_op = std::make_shared<vk_program_node>();
                node_op->input_ids = {node.inputs.at(0), node.inputs.at(1)};
                node_op->scalars = {make_u32_scalar(static_cast<uint32_t>(a_shape[0])),
                                    make_u32_scalar(static_cast<uint32_t>(out_shape[1])),
                                    make_u32_scalar(static_cast<uint32_t>(a_shape[1])),
                                    make_u32_scalar(static_cast<uint32_t>(element_count(out_shape)))};
                // Quantized second input: switch to the native in-shader dequant
                // kernel and pass the block type as an extra push scalar.
                auto qit = graph.quant_constants.find(node.inputs.at(1));
                if (qit != graph.quant_constants.end()) {
                    kernel_id = "matmul_q_f32";
                    node_op->scalars.push_back(make_u32_scalar(qit->second.type));
                }
                break;
            }
            default:
                OPENVINO_THROW("[GPU] vk_program: unsupported op ", node.id);
        }

        node_op->id = unique_id(std::string(ir_op_base_name(node.op)) + "_" + node.id);
        node_op->k = get_build_kernel(kernel_id);

        const auto& out_shape = shape_of(node.id);
        node_op->output_id = node_op->id + "_out";
        node_op->wgs.global = {element_count(out_shape), 1, 1};
        const auto out_mem = make_memory(out_shape, node_op->output_id, false);
        prog->memories[node_op->output_id] = out_mem;
        prog->memories[node.id] = out_mem;
        prog->producer[node.id] = node_op;
        node_op->outputs.push_back(out_mem);

        for (const auto& in_id : node_op->input_ids) {
            auto it = prog->memories.find(in_id);
            OPENVINO_ASSERT(it != prog->memories.end(),
                            "[GPU] vk_program: input ", in_id, " of node ", node_op->id, " has no buffer yet");
            node_op->inputs.push_back(it->second);
        }

        if (node.op == ir_op::convolution) {
            const auto& w_id = node.inputs.at(1);
            auto w_it = prog->memories.find(w_id);
            OPENVINO_ASSERT(w_it != prog->memories.end(), "[GPU] vk_program: conv weights buffer ", w_id, " not found");
            node_op->weights = w_it->second;
            if (node.inputs.size() > 2) {
                const auto& b_id = node.inputs.at(2);
                auto b_it = prog->memories.find(b_id);
                OPENVINO_ASSERT(b_it != prog->memories.end(), "[GPU] vk_program: conv bias buffer ", b_id, " not found");
                node_op->bias = b_it->second;
            }
        }

        prog->nodes.push_back(node_op);
    }

    for (size_t i = 0; i < graph.inputs.size(); ++i) {
        const auto& id = graph.inputs[i];
        OPENVINO_ASSERT(prog->memories.count(id) != 0, "[GPU] vk_program: model input ", id, " has no buffer");
        prog->input_port_to_id[i] = id;
    }
    for (size_t i = 0; i < graph.outputs.size(); ++i) {
        const auto& id = graph.outputs[i];
        auto it = prog->producer.find(id);
        OPENVINO_ASSERT(it != prog->producer.end(), "[GPU] vk_program: no producer for model output ", id);
        // Map the model output to the producer's own output buffer id, so that
        // set_output_memory() (keyed by this id) matches the node->output_id
        // used in vk_network::execute() for the override.
        prog->output_port_to_id[i] = it->second->output_id;
    }
    OPENVINO_ASSERT(!prog->nodes.empty(), "[GPU] vk_program: model produced no executable nodes");

    return prog;
}

}  // namespace cross_platform
}  // namespace vulkan
}  // namespace ov
