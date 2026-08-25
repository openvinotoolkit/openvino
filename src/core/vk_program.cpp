// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vk_program.hpp"

#include "runtime/except.hpp"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <ranges>
#include <string_view>

namespace ov::core {
namespace vulkan {
namespace cross_platform {

namespace {

// Builtin kernel id for every op that has a native kernel.
// |matmul_a_rank| / |matmul_b_rank| disambiguate the MatMul kernels:
// 2Dx2D, batched-shared 3Dx2D, pairwise-batched 3Dx3D.
std::string_view ir_op_kernel_name(const ir_node& node, size_t matmul_a_rank = 2, size_t matmul_b_rank = 2) {
    switch (node.op) {
        case ir_op::relu: return "relu_f32";
        case ir_op::add: return "eltwise_add_f32";
        case ir_op::mul: return "eltwise_mul_f32";
        case ir_op::sub: return "eltwise_sub_f32";
        case ir_op::div: return "eltwise_div_f32";
        case ir_op::sigmoid: return "sigmoid_f32";
        case ir_op::tanh: return "tanh_f32";
        case ir_op::leaky_relu: return "leaky_relu_f32";
        case ir_op::gelu: return "gelu_f32";
        case ir_op::swiglu: return "swiglu_f32";
        case ir_op::quick_gelu: return "quick_gelu_f32";
        case ir_op::rms_norm: return "rms_norm_f32";
        case ir_op::causal_softmax: return "causal_softmax_f32";
        case ir_op::rope: return "rope_f32";
        case ir_op::cache_write: return "cache_write_f32";
        case ir_op::argmax: return "argmax_f32";
        case ir_op::transpose: return "transpose_f32";
        case ir_op::softmax: return "softmax_f32";
        // Concat is wired manually from slab copies inside build(); the id
        // only satisfies the generic lookup before its case continues.
        case ir_op::concat: return "slab_copy_f32";
        case ir_op::pad: return "pad_f32";
        case ir_op::crop: return "crop_f32";
        case ir_op::reduce_mean:
        case ir_op::reduce_sum:
        case ir_op::reduce_max: return "reduce_f32";
        case ir_op::max_pool: return "maxpool_f32";
        case ir_op::avg_pool: return "avgpool_f32";
        case ir_op::convolution: return "conv2d_f32";
        case ir_op::matmul:
            if (matmul_a_rank == 3 && matmul_b_rank == 3)
                return "matmul_bb_f32";
            if (matmul_a_rank == 3)
                return node.matmul_transpose_b ? "matmul_batched_transpose_b_f32" : "matmul_batched_f32";
            return node.matmul_transpose_b ? "matmul_transpose_b_f32" : "matmul_f32";
        default: return {};
    }
}

// Base id used to build unique per-node kernel ids.
std::string_view ir_op_base_name(ir_op op) {
    switch (op) {
        case ir_op::relu: return "relu";
        case ir_op::add: return "eltwise_add";
        case ir_op::mul: return "eltwise_mul";
        case ir_op::sub: return "eltwise_sub";
        case ir_op::div: return "eltwise_div";
        case ir_op::sigmoid: return "sigmoid";
        case ir_op::tanh: return "tanh";
        case ir_op::leaky_relu: return "leaky_relu";
        case ir_op::gelu: return "gelu";
        case ir_op::swiglu: return "swiglu";
        case ir_op::quick_gelu: return "quick_gelu";
        case ir_op::causal_softmax: return "causal_softmax";
        case ir_op::rope: return "rope";
        case ir_op::cache_write: return "cache_write";
        case ir_op::argmax: return "argmax";
        case ir_op::transpose: return "transpose";
        case ir_op::concat: return "concat";
        case ir_op::softmax: return "softmax";
        case ir_op::reshape: return "reshape";
        case ir_op::pad: return "pad";
        case ir_op::crop: return "crop";
        case ir_op::reduce_mean: return "reduce_mean";
        case ir_op::reduce_sum: return "reduce_sum";
        case ir_op::reduce_max: return "reduce_max";
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
// in clspv reflection order.
scalar_desc make_u32_scalar(uint32_t value) {
    scalar_desc s;
    s.t = scalar_t::UINT32;
    s.v.u32 = value;
    return s;
}

scalar_desc make_f32_scalar(float value) {
    scalar_desc s;
    s.t = scalar_t::FLOAT32;
    s.v.f32 = value;
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
            case ir_op::reshape: {
                // Flat f32 buffers make reshape a pure metadata op: the
                // output aliases the input buffer and no kernel is dispatched.
                const auto& in_id = node.inputs.at(0);
                auto it = prog->memories.find(in_id);
                OPENVINO_ASSERT(it != prog->memories.end(),
                                "[GPU] vk_program: reshape input ", in_id, " of ", node.id, " has no buffer");
                const size_t in_total = element_count(shape_of(in_id));
                const size_t out_total = element_count(shape_of(node.id));
                OPENVINO_ASSERT(in_total == out_total,
                                "[GPU] vk_program: reshape ", in_total, " -> ", out_total, " elements in ", node.id);
                prog->memories[node.id] = it->second;
                if (std::find(graph.outputs.begin(), graph.outputs.end(), node.id) != graph.outputs.end()) {
                    // Model outputs resolve through a producer; forward the
                    // input's producer so a reshaped buffer can be an output.
                    auto pit = prog->producer.find(in_id);
                    OPENVINO_ASSERT(pit != prog->producer.end(),
                                    "[GPU] vk_program: reshape of non-produced tensor ", in_id,
                                    " cannot be a model output yet");
                    prog->producer[node.id] = pit->second;
                }
                continue;
            }
            default:
                break;
        }

        std::string_view kernel_id = ir_op_kernel_name(
            node,
            node.inputs.empty() ? 2u : shape_of(node.inputs.at(0)).size(),
            node.inputs.size() < 2 ? 2u : shape_of(node.inputs.at(1)).size());
        OPENVINO_ASSERT(!kernel_id.empty(), "[GPU] vk_program: unsupported op ", node.id);

        std::shared_ptr<vk_program_node> node_op;
        switch (node.op) {
            case ir_op::relu:
            case ir_op::sigmoid:
            case ir_op::tanh:
            case ir_op::gelu:
            case ir_op::quick_gelu: {
                const auto& out_shape = shape_of(node.id);
                node_op = std::make_shared<vk_program_node>();
                node_op->input_ids = {node.inputs.at(0)};
                node_op->scalars = {make_u32_scalar(static_cast<uint32_t>(element_count(out_shape)))};
                break;
            }
            case ir_op::leaky_relu: {
                const auto& out_shape = shape_of(node.id);
                node_op = std::make_shared<vk_program_node>();
                node_op->input_ids = {node.inputs.at(0)};
                node_op->scalars = {make_u32_scalar(static_cast<uint32_t>(element_count(out_shape))),
                                    make_f32_scalar(node.alpha)};
                break;
            }
            case ir_op::add:
            case ir_op::mul:
            case ir_op::sub:
            case ir_op::div:
            case ir_op::swiglu: {
                const auto& out_shape = shape_of(node.id);
                const size_t out_total = element_count(out_shape);
                node_op = std::make_shared<vk_program_node>();
                // Broadcast inputs are materialized here when they are
                // constants (the fc/matmul+bias pattern); dynamic inputs of a
                // mismatching shape are rejected loudly instead of silently
                // producing wrong numbers.
                for (size_t i = 0; i < 2; ++i) {
                    const auto& in_id = node.inputs.at(i);
                    const size_t in_total = element_count(shape_of(in_id));
                    if (in_total == out_total) {
                        node_op->input_ids.push_back(in_id);
                        continue;
                    }
                    auto cit = graph.constant_data.find(in_id);
                    OPENVINO_ASSERT(cit != graph.constant_data.end(),
                                    "[GPU] vk_program: broadcast input ", in_id, " of ", node.id,
                                    " must be a constant (materialize dynamic inputs upstream)");
                    OPENVINO_ASSERT(in_total > 0 && out_total % in_total == 0,
                                    "[GPU] vk_program: cannot broadcast ", in_total, " elements to ",
                                    out_total, " in ", node.id);
                    const auto& src = cit->second;
                    std::vector<float> expanded(out_total);
                    for (size_t j = 0; j < out_total; ++j)
                        expanded[j] = src[j % in_total];
                    const std::string exp_id = in_id + "#bcast#" + node.id;
                    auto mem = make_memory(out_shape, exp_id, true);
                    void* p = mem->lock();
                    std::memcpy(p, expanded.data(), expanded.size() * sizeof(float));
                    mem->unlock();
                    prog->memories[exp_id] = mem;
                    node_op->input_ids.push_back(exp_id);
                }
                node_op->scalars = {make_u32_scalar(static_cast<uint32_t>(out_total))};
                break;
            }
            case ir_op::transpose: {
                const auto& in_shape = shape_of(node.inputs.at(0));
                const auto& out_shape = shape_of(node.id);
                const auto& perm = node.transpose_order;
                OPENVINO_ASSERT(!in_shape.empty() && in_shape.size() <= 8 && perm.size() == in_shape.size() &&
                                    out_shape.size() == in_shape.size(),
                                "[GPU] vk_program: Transpose expects 1..8D input with a full permutation in ",
                                node.id);
                std::vector<bool> seen(in_shape.size(), false);
                for (size_t d = 0; d < perm.size(); ++d) {
                    const size_t src = perm[d];
                    OPENVINO_ASSERT(src < in_shape.size() && !seen[src],
                                    "[GPU] vk_program: bad transpose order in ", node.id);
                    seen[src] = true;
                    OPENVINO_ASSERT(out_shape[d] == in_shape[src],
                                    "[GPU] vk_program: transpose output shape mismatch in ", node.id);
                }
                uint32_t in_strides[8] = {1, 1, 1, 1, 1, 1, 1, 1};
                uint32_t out_dims[8] = {1, 1, 1, 1, 1, 1, 1, 1};
                uint32_t perm8[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                uint64_t stride = 1;
                for (size_t i = in_shape.size(); i-- > 0;) {
                    in_strides[i] = static_cast<uint32_t>(stride);
                    stride *= in_shape[i];
                }
                for (size_t d = 0; d < in_shape.size(); ++d) {
                    out_dims[d] = static_cast<uint32_t>(out_shape[d]);
                    perm8[d] = static_cast<uint32_t>(perm[d]);
                }
                node_op = std::make_shared<vk_program_node>();
                node_op->input_ids = {node.inputs.at(0)};
                node_op->scalars = {make_u32_scalar(static_cast<uint32_t>(element_count(out_shape))),
                                    make_u32_scalar(static_cast<uint32_t>(in_shape.size()))};
                for (int d = 0; d < 8; ++d)
                    node_op->scalars.push_back(make_u32_scalar(out_dims[d]));
                for (int d = 0; d < 8; ++d)
                    node_op->scalars.push_back(make_u32_scalar(in_strides[d]));
                for (int d = 0; d < 8; ++d)
                    node_op->scalars.push_back(make_u32_scalar(perm8[d]));
                break;
            }
            case ir_op::softmax: {
                const auto& in_shape = shape_of(node.inputs.at(0));
                const auto& out_shape = shape_of(node.id);
                OPENVINO_ASSERT(!in_shape.empty() && node.axis < in_shape.size(),
                                "[GPU] vk_program: Softmax axis out of range in ", node.id);
                OPENVINO_ASSERT(element_count(out_shape) == element_count(in_shape),
                                "[GPU] vk_program: Softmax preserves shape in ", node.id);
                const size_t len = in_shape[node.axis];
                size_t inner = 1;
                for (size_t d = node.axis + 1; d < in_shape.size(); ++d)
                    inner *= in_shape[d];
                const size_t lines = element_count(in_shape) / len;  // outer*inner
                node_op = std::make_shared<vk_program_node>();
                node_op->input_ids = {node.inputs.at(0)};
                node_op->scalars = {make_u32_scalar(static_cast<uint32_t>(lines)),
                                    make_u32_scalar(static_cast<uint32_t>(len)),
                                    make_u32_scalar(static_cast<uint32_t>(inner))};
                break;
            }
            case ir_op::causal_softmax: {
                const auto& in_shape = shape_of(node.inputs.at(0));
                const auto& out_shape = shape_of(node.id);
                OPENVINO_ASSERT(in_shape.size() >= 2 && element_count(out_shape) == element_count(in_shape),
                                "[GPU] vk_program: causal Softmax expects [...,L,L] in ", node.id);
                const size_t len = in_shape.back();
                node_op = std::make_shared<vk_program_node>();
                node_op->input_ids = {node.inputs.at(0)};
                node_op->scalars = {make_u32_scalar(static_cast<uint32_t>(element_count(in_shape) / len)),
                                    make_u32_scalar(static_cast<uint32_t>(len))};
                break;
            }
            case ir_op::rms_norm: {
                const auto& in_shape = shape_of(node.inputs.at(0));
                const auto& w_shape = shape_of(node.inputs.at(1));
                OPENVINO_ASSERT(!in_shape.empty() && node.axis + 1 == in_shape.size(),
                                "[GPU] vk_program: RMSNorm supports the last axis only in ", node.id);
                OPENVINO_ASSERT(w_shape.size() == 1 && w_shape[0] == in_shape.back(),
                                "[GPU] vk_program: RMSNorm weight must be [axis_size] in ", node.id);
                const size_t len = in_shape.back();
                const size_t lines = element_count(in_shape) / len;
                node_op = std::make_shared<vk_program_node>();
                node_op->input_ids = {node.inputs.at(0), node.inputs.at(1)};
                node_op->scalars = {make_u32_scalar(static_cast<uint32_t>(lines)),
                                    make_u32_scalar(static_cast<uint32_t>(len)),
                                    make_f32_scalar(node.alpha)};
                break;
            }
            case ir_op::reduce_mean:
            case ir_op::reduce_sum:
            case ir_op::reduce_max: {
                const auto& in_shape = shape_of(node.inputs.at(0));
                const auto& out_shape = shape_of(node.id);
                OPENVINO_ASSERT(!in_shape.empty() && node.axis < in_shape.size(),
                                "[GPU] vk_program: Reduce axis out of range in ", node.id);
                const size_t len = in_shape[node.axis];
                size_t inner = 1;
                for (size_t d = node.axis + 1; d < in_shape.size(); ++d)
                    inner *= in_shape[d];
                const size_t outer = element_count(in_shape) / (len * inner);
                OPENVINO_ASSERT(element_count(out_shape) == outer * inner,
                                "[GPU] vk_program: Reduce output must drop the axis in ", node.id);
                const uint32_t mode = node.op == ir_op::reduce_sum ? 0u : (node.op == ir_op::reduce_mean ? 1u : 2u);
                node_op = std::make_shared<vk_program_node>();
                node_op->input_ids = {node.inputs.at(0)};
                node_op->scalars = {make_u32_scalar(static_cast<uint32_t>(outer * inner)),
                                    make_u32_scalar(static_cast<uint32_t>(len)),
                                    make_u32_scalar(static_cast<uint32_t>(inner)),
                                    make_u32_scalar(mode)};
                break;
            }
            case ir_op::pad: {
                const auto& in_shape = shape_of(node.inputs.at(0));
                const auto& out_shape = shape_of(node.id);
                const auto& pb = node.pool.pads_begin;
                const auto& pe = node.pool.pads_end;
                OPENVINO_ASSERT(!in_shape.empty() && in_shape.size() <= 8 && out_shape.size() == in_shape.size(),
                                "[GPU] vk_program: Pad expects matching 1..8D ranks in ", node.id);
                OPENVINO_ASSERT(pb.size() == in_shape.size() && pe.size() == in_shape.size(),
                                "[GPU] vk_program: Pad needs per-dim pads_begin/pads_end in ", node.id);
                uint32_t od8[8] = {1, 1, 1, 1, 1, 1, 1, 1};
                uint32_t pb8[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                uint32_t pe8[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                for (size_t d = 0; d < in_shape.size(); ++d) {
                    od8[d] = static_cast<uint32_t>(out_shape[d]);
                    pb8[d] = static_cast<uint32_t>(pb[d]);
                    pe8[d] = static_cast<uint32_t>(pe[d]);
                    OPENVINO_ASSERT(out_shape[d] == in_shape[d] + pb[d] + pe[d],
                                    "[GPU] vk_program: Pad output dim mismatch on axis ", d, " in ", node.id);
                }
                node_op = std::make_shared<vk_program_node>();
                node_op->input_ids = {node.inputs.at(0)};
                // in_dims are derived in-shader (out - begin - end) to keep the
                // push block within the 128-byte budget at rank 8.
                node_op->scalars = {make_u32_scalar(static_cast<uint32_t>(element_count(out_shape))),
                                    make_u32_scalar(static_cast<uint32_t>(in_shape.size()))};
                for (int d = 0; d < 8; ++d)
                    node_op->scalars.push_back(make_u32_scalar(od8[d]));
                for (int d = 0; d < 8; ++d)
                    node_op->scalars.push_back(make_u32_scalar(pb8[d]));
                for (int d = 0; d < 8; ++d)
                    node_op->scalars.push_back(make_u32_scalar(pe8[d]));
                node_op->scalars.push_back(make_f32_scalar(node.alpha));  // fill value
                break;
            }
            case ir_op::crop: {
                const auto& in_shape = shape_of(node.inputs.at(0));
                const auto& out_shape = shape_of(node.id);
                const auto& begin = node.pool.pads_begin;  // per-dim begin offsets
                OPENVINO_ASSERT(!in_shape.empty() && in_shape.size() <= 8 && out_shape.size() == in_shape.size(),
                                "[GPU] vk_program: Crop expects matching 1..8D ranks in ", node.id);
                OPENVINO_ASSERT(begin.size() == in_shape.size(),
                                "[GPU] vk_program: Crop needs per-dim begin offsets (pads_begin) in ", node.id);
                uint32_t od8[8] = {1, 1, 1, 1, 1, 1, 1, 1};
                uint32_t is8[8] = {1, 1, 1, 1, 1, 1, 1, 1};
                uint32_t bg8[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                uint64_t stride = 1;
                for (size_t i = in_shape.size(); i-- > 0;) {
                    is8[i] = static_cast<uint32_t>(stride);
                    stride *= in_shape[i];
                }
                for (size_t d = 0; d < in_shape.size(); ++d) {
                    od8[d] = static_cast<uint32_t>(out_shape[d]);
                    bg8[d] = static_cast<uint32_t>(begin[d]);
                    OPENVINO_ASSERT(begin[d] + out_shape[d] <= in_shape[d],
                                    "[GPU] vk_program: Crop window exceeds input on axis ", d, " in ", node.id);
                }
                node_op = std::make_shared<vk_program_node>();
                node_op->input_ids = {node.inputs.at(0)};
                node_op->scalars = {make_u32_scalar(static_cast<uint32_t>(element_count(out_shape))),
                                    make_u32_scalar(static_cast<uint32_t>(in_shape.size()))};
                for (int d = 0; d < 8; ++d)
                    node_op->scalars.push_back(make_u32_scalar(od8[d]));
                for (int d = 0; d < 8; ++d)
                    node_op->scalars.push_back(make_u32_scalar(is8[d]));
                for (int d = 0; d < 8; ++d)
                    node_op->scalars.push_back(make_u32_scalar(bg8[d]));
                break;
            }
            case ir_op::rope: {
                const auto& x_shape = shape_of(node.inputs.at(0));
                const auto& c_shape = shape_of(node.inputs.at(1));
                const auto& s_shape = shape_of(node.inputs.at(2));
                OPENVINO_ASSERT(x_shape.size() >= 3 && x_shape.back() % 2 == 0,
                                "[GPU] vk_program: RoPE x must be [...,H,D] with even D in ", node.id);
                const size_t half = x_shape.back() / 2;
                // bl in-shader spans all leading dims except the head axis.
                std::vector<size_t> expect_cs(x_shape.begin(), x_shape.end() - 2);
                expect_cs.push_back(half);  // [...(no H), half]
                OPENVINO_ASSERT(c_shape == expect_cs && s_shape == expect_cs,
                                "[GPU] vk_program: RoPE cos/sin must be x_dims[:-2]+[D/2] in ", node.id);
                node_op = std::make_shared<vk_program_node>();
                node_op->input_ids = {node.inputs.at(0), node.inputs.at(1), node.inputs.at(2)};
                node_op->scalars = {make_u32_scalar(static_cast<uint32_t>(element_count(x_shape) / 2)),
                                    make_u32_scalar(static_cast<uint32_t>(half)),
                                    make_u32_scalar(static_cast<uint32_t>(x_shape[x_shape.size() - 2]))};
                break;
            }
            case ir_op::argmax: {
                const auto& in_shape = shape_of(node.inputs.at(0));
                OPENVINO_ASSERT(in_shape.size() >= 1,
                                "[GPU] vk_program: ArgMax expects at least 1D input in ", node.id);
                const size_t len = in_shape.back();
                node_op = std::make_shared<vk_program_node>();
                node_op->input_ids = {node.inputs.at(0)};
                node_op->scalars = {make_u32_scalar(static_cast<uint32_t>(element_count(in_shape) / len)),
                                    make_u32_scalar(static_cast<uint32_t>(len))};
                break;
            }
            case ir_op::cache_write: {
                // Functional KV-cache append: out [B,S,D] = cache with rows
                // [pos, pos+L) replaced by new [B,L,D]; pos = node.axis.
                const auto& n_shape = shape_of(node.inputs.at(0));
                const auto& c_shape = shape_of(node.inputs.at(1));
                OPENVINO_ASSERT(n_shape.size() == 3 && c_shape.size() == 3 &&
                                    n_shape[0] == c_shape[0] && n_shape[2] == c_shape[2],
                                "[GPU] vk_program: cache_write expects new [B,L,D], cache [B,S,D] in ",
                                node.id);
                OPENVINO_ASSERT(node.axis + n_shape[1] <= c_shape[1],
                                "[GPU] vk_program: cache_write overflows the cache sequence in ", node.id);
                node_op = std::make_shared<vk_program_node>();
                node_op->input_ids = {node.inputs.at(0), node.inputs.at(1)};
                node_op->scalars = {make_u32_scalar(static_cast<uint32_t>(c_shape[2])),
                                    make_u32_scalar(static_cast<uint32_t>(c_shape[1])),
                                    make_u32_scalar(static_cast<uint32_t>(n_shape[1])),
                                    make_u32_scalar(static_cast<uint32_t>(element_count(c_shape))),
                                    make_u32_scalar(static_cast<uint32_t>(node.axis))};
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
                OPENVINO_ASSERT(node.inputs.size() == 3,
                                "[GPU] vk_program: Convolution requires exactly 3 inputs "
                                "(data, weights, bias); materialize a zero bias upstream in ", node.id);
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
                if (a_shape.size() == 3) {
                    // Batched A [B,M,K]; B is a shared 2D matrix [K,N|N,K],
                    // per-batch [B,K,N], or GQA-shared [1,K,N].
                    const bool pairwise = b_shape.size() == 3;
                    OPENVINO_ASSERT(pairwise || b_shape.size() == 2,
                                    "[GPU] vk_program: batched MatMul B must be [K,N] or [B,K,N] in ", node.id);
                    size_t b_batch = 0;  // 0 = not pairwise
                    if (pairwise) {
                        OPENVINO_ASSERT(!node.matmul_transpose_b,
                                        "[GPU] vk_program: pairwise-batched MatMul does not support "
                                        "transpose_b in ", node.id);
                        // GQA: b_shape[0]==1 shares one matrix across the batch.
                        OPENVINO_ASSERT(b_shape[0] == a_shape[0] || b_shape[0] == 1,
                                        "[GPU] vk_program: pairwise MatMul batch mismatch (GQA allows "
                                        "b_batch=1) in ", node.id);
                        b_batch = b_shape[0];
                        OPENVINO_ASSERT(b_shape[1] == a_shape[2] &&
                                            out_shape.size() == 3 && b_shape[2] == out_shape[2],
                                        "[GPU] vk_program: pairwise-batched MatMul shapes mismatch in ",
                                        node.id);
                    }
                    const size_t K = pairwise ? b_shape[1]
                                              : (node.matmul_transpose_b ? b_shape[1] : b_shape[0]);
                    OPENVINO_ASSERT(a_shape[2] == K && out_shape.size() == 3 &&
                                        a_shape[0] == out_shape[0] && a_shape[1] == out_shape[1],
                                    "[GPU] vk_program: batched MatMul shapes mismatch in ", node.id);
                    node_op = std::make_shared<vk_program_node>();
                    node_op->input_ids = {node.inputs.at(0), node.inputs.at(1)};
                    node_op->scalars = {make_u32_scalar(static_cast<uint32_t>(out_shape[1])),
                                        make_u32_scalar(static_cast<uint32_t>(out_shape[2])),
                                        make_u32_scalar(static_cast<uint32_t>(a_shape[2])),
                                        make_u32_scalar(static_cast<uint32_t>(element_count(out_shape))),
                                        make_u32_scalar(static_cast<uint32_t>(out_shape[0])),
                                        make_u32_scalar(static_cast<uint32_t>(b_batch))};
                    // Quantized weights ride the shared-matrix path only.
                    auto qit = graph.quant_constants.find(node.inputs.at(1));
                    if (qit != graph.quant_constants.end()) {
                        OPENVINO_ASSERT(!pairwise && !node.matmul_transpose_b,
                                        "[GPU] vk_program: quantized batched MatMul requires a shared "
                                        "non-transposed matrix in ", node.id);
                        kernel_id = "matmul_q_batched_f32";
                        node_op->scalars.push_back(make_u32_scalar(qit->second.type));
                    } else if (!node.matmul_transpose_b && out_shape[1] >= 16 && a_shape[2] >= 16) {
                        // Large f32 GEMMs go through the tiled kernels.
                        kernel_id = pairwise ? "matmul_bb_tiled_f32" : "matmul_batched_tiled_f32";
                    }
                    break;
                }
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
                } else if (!node.matmul_transpose_b && a_shape[0] >= 16 && out_shape[1] >= 16 &&
                           a_shape[1] >= 16) {
                    // Large f32 GEMM: shared-memory tiled kernel.
                    kernel_id = "matmul_tiled_f32";
                }
                break;
            }
            case ir_op::concat: {
                // Concat is assembled from slab copies into one shared output
                // buffer (the stream binds a fixed descriptor per program
                // node, so a variable-arity gather kernel is not an option).
                // A final full-size identity copy is the registered producer,
                // which also makes host-side output overrides capture the
                // complete buffer.
                const auto& out_shape = shape_of(node.id);
                const size_t rank = out_shape.size();
                const size_t axis = node.axis;
                const size_t k = node.inputs.size();
                OPENVINO_ASSERT(rank >= 1 && axis < rank && k >= 2,
                                "[GPU] vk_program: Concat needs axis inside rank and >=2 inputs in ", node.id);
                size_t inner = 1, outer = 1;
                for (size_t d = 0; d < axis; ++d)
                    outer *= out_shape[d];
                for (size_t d = axis + 1; d < rank; ++d)
                    inner *= out_shape[d];
                std::vector<uint32_t> prefix(k + 1, 0);
                for (size_t i = 0; i < k; ++i) {
                    const auto& s = shape_of(node.inputs[i]);
                    OPENVINO_ASSERT(s.size() == rank, "[GPU] vk_program: Concat rank mismatch in ", node.id);
                    for (size_t d = 0; d < rank; ++d) {
                        if (d == axis)
                            continue;
                        OPENVINO_ASSERT(s[d] == out_shape[d],
                                        "[GPU] vk_program: Concat non-axis dims mismatch in ", node.id);
                    }
                    prefix[i + 1] = prefix[i] + static_cast<uint32_t>(s[axis]);
                }
                const uint32_t total_axis = prefix[k];
                OPENVINO_ASSERT(total_axis == out_shape[axis],
                                "[GPU] vk_program: Concat axis sizes do not sum to the output in ", node.id);

                const std::string buf_id = node.id + "_cat_out";
                auto out_mem = make_memory(out_shape, buf_id, false);
                prog->memories[buf_id] = out_mem;
                prog->memories[node.id] = out_mem;
                const auto kernel = get_build_kernel("slab_copy_f32");

                for (size_t i = 0; i < k; ++i) {
                    auto n = std::make_shared<vk_program_node>();
                    n->id = unique_id(std::string("concat_") + node.id);
                    n->input_ids = {node.inputs[i]};
                    n->inputs = {prog->memories.at(node.inputs[i])};
                    const uint32_t k_i = static_cast<uint32_t>(shape_of(node.inputs[i])[axis]);
                    n->scalars = {make_u32_scalar(static_cast<uint32_t>(outer)),
                                  make_u32_scalar(k_i),
                                  make_u32_scalar(total_axis),
                                  make_u32_scalar(prefix[i]),
                                  make_u32_scalar(k_i),
                                  make_u32_scalar(static_cast<uint32_t>(inner)),
                                  make_u32_scalar(static_cast<uint32_t>(outer * k_i * inner))};
                    n->output_id = n->id + "_out";  // private: never overridden
                    n->wgs.global = {static_cast<size_t>(outer * k_i * inner), 1, 1};
                    n->k = kernel;
                    n->outputs.push_back(out_mem);
                    prog->nodes.push_back(n);
                }

                // Full-buffer identity copy: reads and writes every cell of
                // the shared buffer once (same value), so overrides on its
                // public output id capture the complete result.
                auto fin = std::make_shared<vk_program_node>();
                fin->id = unique_id(std::string("concat_full_") + node.id);
                fin->input_ids = {node.id};
                fin->inputs = {out_mem};
                fin->scalars = {make_u32_scalar(static_cast<uint32_t>(outer)),
                                make_u32_scalar(total_axis),
                                make_u32_scalar(total_axis),
                                make_u32_scalar(0),
                                make_u32_scalar(total_axis),
                                make_u32_scalar(static_cast<uint32_t>(inner)),
                                make_u32_scalar(static_cast<uint32_t>(element_count(out_shape)))};
                fin->output_id = fin->id + "_out";
                fin->wgs.global = {element_count(out_shape), 1, 1};
                fin->k = kernel;
                fin->outputs.push_back(out_mem);
                prog->memories[fin->output_id] = out_mem;
                prog->nodes.push_back(fin);
                prog->producer[node.id] = fin;
                continue;
            }
            default:
                OPENVINO_THROW("[GPU] vk_program: unsupported op ", node.id);
        }

        node_op->id = unique_id(std::string(ir_op_base_name(node.op)) + "_" + node.id);
        node_op->k = get_build_kernel(kernel_id);

        const auto& out_shape = shape_of(node.id);
        node_op->output_id = node_op->id + "_out";
        node_op->wgs.global = {element_count(out_shape), 1, 1};
        if (kernel_id == "matmul_tiled_f32" || kernel_id == "matmul_batched_tiled_f32" ||
            kernel_id == "matmul_bb_tiled_f32") {
            // Tiled GEMMs use a 2D grid over the output matrix (x = N,
            // y = M, z = batch); the local size comes from shader reflection.
            const bool batched = out_shape.size() == 3;
            const size_t rows = out_shape[batched ? 1 : 0];
            const size_t batches = batched ? out_shape[0] : 1;
            node_op->wgs.global = {out_shape.back(), rows, batches};
        }
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
