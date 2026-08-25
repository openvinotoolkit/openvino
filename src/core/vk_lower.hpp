// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// vk_lower: lowers a frontend-neutral bridge graph into the core IR.
//
// Frontends (pytorch / tensorflow / tflite / jax) each walk their own graph
// and emit bridge::node records (canonical type string + edges + attributes).
// The lowering maps types onto ir_op, validates contracts eagerly and
// aggregates EVERY unsupported type into one error report, so a frontend
// author sees the full gap list per run instead of whack-a-mole.
//
// Header-only, zero dependencies beyond vk_ir.hpp: fully unit-testable on
// CPU without any runtime.

#pragma once

#include "vk_ir.hpp"

#include <algorithm>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace ov::core {
namespace vulkan {
namespace cross_platform {
namespace bridge {

struct node {
    // Canonical type. Matching is case-insensitive; an optional "aten::"/
    // "prim::"/"tf." prefix is stripped ("aten::relu_" == "Relu_" == "relu").
    std::string type;
    std::string id;
    std::vector<std::string> inputs;
    std::vector<size_t> shape;          // output shape
    std::vector<float> constant;        // for constant nodes
    // Optional attributes consumed by specific ops.
    bool matmul_transpose_b = false;
    float alpha = 0.0f;                 // leaky slope / rms eps / pad fill
    uint32_t axis = 0;
    std::vector<size_t> order;          // transpose permutation
    ir_pool_params pool;                // pools/conv/pad/crop
};

struct graph {
    std::vector<node> nodes;            // topologically sorted by the caller
    std::vector<std::string> outputs;
};

}  // namespace bridge

namespace detail {

inline std::string lower_type(std::string t) {
    // strip known prefixes
    for (const char* p : {"aten::", "prim::", "tf.", "tfl.", "jax."}) {
        if (t.rfind(p, 0) == 0)
            t.erase(0, std::strlen(p));
    }
    std::transform(t.begin(), t.end(), t.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    while (!t.empty() && (t.back() == '_' || t.back() == '.'))
        t.pop_back();  // relu_/add./in-place suffixes
    return t;
}

inline const ir_pool_params& pool_of(const bridge::node& n) {
    return n.pool;
}

}  // namespace detail

inline ir_graph lower(const bridge::graph& bg) {
    std::set<std::string> unsupported;
    ir_graph g;

    auto op_of = [&](const bridge::node& n) -> ir_op {
        const std::string t = detail::lower_type(n.type);
        if (t == "parameter" || t == "input") return ir_op::parameter;
        if (t == "constant") return ir_op::constant;
        if (t == "result" || t == "output") return ir_op::result;
        if (t == "relu") return ir_op::relu;
        if (t == "add") return ir_op::add;
        if (t == "mul") return ir_op::mul;
        if (t == "sub") return ir_op::sub;
        if (t == "div") return ir_op::div;
        if (t == "sigmoid") return ir_op::sigmoid;
        if (t == "tanh") return ir_op::tanh;
        if (t == "leakyrelu" || t == "leaky_relu") return ir_op::leaky_relu;
        if (t == "gelu") return ir_op::gelu;
        if (t == "swiglu" || t == "silu_and_mul") return ir_op::swiglu;
        if (t == "quickgelu") return ir_op::quick_gelu;
        if (t == "rmsnorm" || t == "rms_norm") return ir_op::rms_norm;
        if (t == "transpose" || t == "permute") return ir_op::transpose;
        if (t == "reshape" || t == "view" || t == "flatten") return ir_op::reshape;
        if (t == "cat" || t == "concat" || t == "concatv2" || t == "concatenate") return ir_op::concat;
        if (t == "softmax") return ir_op::softmax;
        if (t == "causalsoftmax") return ir_op::causal_softmax;
        if (t == "rope" || t == "rotaryembedding" || t == "apply_rotary_pos_emb") return ir_op::rope;
        if (t == "cachewrite") return ir_op::cache_write;
        if (t == "argmax") return ir_op::argmax;
        if (t == "pad") return ir_op::pad;
        if (t == "crop" || t == "narrow" || t == "slice") return ir_op::crop;
        if (t == "maxpool2d" || t == "maxpool") return ir_op::max_pool;
        if (t == "avgpool2d" || t == "avgpool" || t == "avg_pool2d") return ir_op::avg_pool;
        if (t == "conv2d" || t == "convolution") return ir_op::convolution;
        if (t == "matmul" || t == "mm" || t == "bmm" || t == "linear") return ir_op::matmul;
        if (t == "reducemean" || t == "mean") return ir_op::reduce_mean;
        if (t == "reducesum" || t == "sum") return ir_op::reduce_sum;
        if (t == "reducemax" || t == "max" || t == "amax") return ir_op::reduce_max;
        unsupported.insert(n.type);  // original spelling for the report
        return ir_op::parameter;     // placeholder; graph is rejected below
    };

    for (const auto& n : bg.nodes) {
        const ir_op op = op_of(n);
        if (!unsupported.empty())
            continue;  // collect the FULL gap list before failing

        ir_node out;
        out.id = n.id;
        out.op = op;
        out.inputs = n.inputs;
        out.alpha = n.alpha;
        out.axis = n.axis;
        out.matmul_transpose_b = n.matmul_transpose_b;
        out.transpose_order = n.order;
        out.pool = detail::pool_of(n);

        // Arity guard: elementwise ops indexing inputs[0]/[1] must not see
        // empty vectors (that would be silent UB downstream).
        const size_t want_ins = (op == ir_op::relu || op == ir_op::sigmoid || op == ir_op::tanh ||
                                 op == ir_op::gelu || op == ir_op::quick_gelu || op == ir_op::leaky_relu ||
                                 op == ir_op::transpose || op == ir_op::reshape || op == ir_op::softmax ||
                                 op == ir_op::causal_softmax || op == ir_op::rms_norm || op == ir_op::pad ||
                                 op == ir_op::crop || op == ir_op::argmax ||
                                 op == ir_op::reduce_mean || op == ir_op::reduce_sum || op == ir_op::reduce_max)
                                    ? 1
                                    : ((op == ir_op::add || op == ir_op::mul || op == ir_op::sub ||
                                        op == ir_op::div || op == ir_op::swiglu || op == ir_op::matmul ||
                                        op == ir_op::cache_write)
                                           ? 2
                                           : (op == ir_op::rope ? 3 : 0));
        if (want_ins != 0 && n.inputs.size() < want_ins)
            throw std::runtime_error("[lower] op '" + n.type + "' (" + n.id + ") expects " +
                                     std::to_string(want_ins) + " input(s), got " +
                                     std::to_string(n.inputs.size()));

        switch (op) {
            case ir_op::constant: {
                if (n.constant.empty()) throw std::runtime_error("[lower] constant '" + n.id + "' has no payload");
                g.constant_data[n.id] = n.constant;
                break;
            }
            case ir_op::convolution: {
                if (n.inputs.size() != 3)
                    throw std::runtime_error("[lower] conv '" + n.id + "' needs data+weights+bias; materialize a zero bias upstream");
                break;
            }
            case ir_op::matmul: {
                if (n.type.find("linear") != std::string::npos)
                    out.matmul_transpose_b = true;  // torch.Linear stores W [N,K]
                break;
            }
            default:
                break;
        }

        g.nodes.push_back(std::move(out));
        g.tensor_shapes[n.id] = n.shape;
    }

    if (!unsupported.empty()) {
        std::string list;
        for (const auto& u : unsupported)
            list += " " + u;
        throw std::runtime_error("[lower] unsupported op(s):" + list +
                                 " вЂ” extend vk_pass/kernels or map it onto existing ops");
    }

    g.outputs = bg.outputs;
    g.inputs.clear();
    for (const auto& n : g.nodes)
        if (n.op == ir_op::parameter)
            g.inputs.push_back(n.id);
    return g;
}

}  // namespace cross_platform
}  // namespace vulkan
}  // namespace ov


