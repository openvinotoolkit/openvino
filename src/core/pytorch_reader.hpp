// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// pt_r: reader for the PyTorch export bundle produced by
// src/frontends/pytorch/tools/torch_export.py:
//   <model>.graph.vktorch  — line-based graph (OP/CONST/OUT records)
//   <model>.weights.safetensors — every parameter/buffer as F32
//
// The reader feeds bridge::graph and lowers it with vk_lower, so all
// validation (arity, aggregated unsupported report) happens in one place.

#pragma once

#include "safetensors_reader.hpp"
#include "vk_ir.hpp"
#include "vk_lower.hpp"

#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace ov::core {
namespace vulkan {
namespace cross_platform {
namespace pt_r {

inline std::vector<size_t> parse_dims(const std::string& s) {
    std::vector<size_t> out;
    if (s == "scalar" || s == "-")
        return out;
    size_t pos = 0;
    while (pos < s.size()) {
        const size_t x = s.find('x', pos);
        if (x == std::string::npos) {
            out.push_back(static_cast<size_t>(std::stoull(s.substr(pos))));
            break;
        }
        out.push_back(static_cast<size_t>(std::stoull(s.substr(pos, x - pos))));
        pos = x + 1;
    }
    return out;
}

inline std::vector<std::string> parse_list(const std::string& s) {
    std::vector<std::string> out;
    if (s == "-")
        return out;
    size_t pos = 0;
    while (pos < s.size()) {
        const size_t c = s.find(',', pos);
        if (c == std::string::npos) {
            out.push_back(s.substr(pos));
            break;
        }
        out.push_back(s.substr(pos, c - pos));
        pos = c + 1;
    }
    return out;
}

// Loads the bundle and lowers it into the core IR.
inline ir_graph load_export(const std::string& graph_path, const std::string& weights_path) {
    const auto weights = st_r::load_safetensors(weights_path);

    std::ifstream f(graph_path);
    if (!f)
        throw std::runtime_error("[pt_r] cannot open graph: " + graph_path);

    bridge::graph bg;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#')
            continue;
        std::vector<std::string> tok;
        size_t pos = 0;
        while (pos < line.size()) {
            const size_t sp = line.find(' ', pos);
            if (sp == std::string::npos) {
                tok.push_back(line.substr(pos));
                break;
            }
            tok.push_back(line.substr(pos, sp - pos));
            pos = sp + 1;
        }
        if (tok[0] == "OP" && tok.size() >= 5) {
            bridge::node n;
            n.id = tok[1];
            n.type = tok[2];
            n.shape = parse_dims(tok[3]);
            n.inputs = parse_list(tok[4]);
            bg.nodes.push_back(std::move(n));
        } else if (tok[0] == "CONST" && tok.size() >= 4) {
            bridge::node n;
            n.id = tok[1];
            n.type = "constant";
            const std::string key = tok[3];
            auto it = weights.find(key);
            if (it == weights.end())
                throw std::runtime_error("[pt_r] CONST '" + tok[1] + "': key '" + key +
                                         "' missing from weights");
            n.shape = it->second.shape;
            n.constant = it->second.data;
            bg.nodes.push_back(std::move(n));
        } else if (tok[0] == "OUT" && tok.size() >= 2) {
            bg.outputs.push_back(tok[1]);
        }
    }
    if (bg.outputs.empty())
        throw std::runtime_error("[pt_r] graph has no OUT records: " + graph_path);
    return lower(bg);
}

}  // namespace pt_r
}  // namespace cross_platform
}  // namespace vulkan
}  // namespace ov
