// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fuse_hadamard_fc.hpp"

#include <algorithm>
#include <bitset>
#include <cmath>
#include <unordered_map>

#include "intel_gpu/op/fully_connected.hpp"
#include "intel_gpu/op/fully_connected_compressed.hpp"
#include "intel_gpu/primitives/fully_connected.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/transpose.hpp"

namespace ov::intel_gpu {

namespace {

std::shared_ptr<ov::op::v0::Constant> constant_behind(const ov::Output<ov::Node>& out) {
    auto node = out.get_node_shared_ptr();
    // ConvertMatMulToFullyConnected leaves Convert(Transpose(H)); H is symmetric.
    for (int i = 0; i < 3; ++i) {
        if (ov::as_type_ptr<ov::op::v0::Convert>(node) || ov::as_type_ptr<ov::op::v1::Transpose>(node)) {
            node = node->get_input_node_shared_ptr(0);
        }
    }
    return ov::as_type_ptr<ov::op::v0::Constant>(node);
}

// The normalised Sylvester H_B (B a supported power of two): H[i][j] =
// (-1)^popcount(i & j) / sqrt(B). Checking a stride of rows is enough to tell it
// from any other BxB constant. Returns B, or 0.
size_t hadamard_block_of(const std::shared_ptr<ov::op::v0::Constant>& c) {
    if (!c) {
        return 0;
    }
    const auto& s = c->get_shape();
    if (s.size() != 2 || s[0] != s[1] || !cldnn::fully_connected::is_supported_hadamard_block(s[0])) {
        return 0;
    }
    const size_t b = s[0];
    const auto v = c->cast_vector<float>();
    const float want = 1.0F / std::sqrt(static_cast<float>(b));
    for (size_t i = 0; i < b; i += 37) {
        for (size_t j = 0; j < b; j += 41) {
            const float sign = (std::bitset<32>(i & j).count() & 1) ? -1.0F : 1.0F;
            if (std::fabs(v[(i * b) + j] - (sign * want)) > 1e-4F) {
                return 0;
            }
        }
    }
    return b;
}

// True if Reshape `r` only splits ([..., K] -> [..., K/B, B]) or merges ([..., K/B, B] -> [..., K]) the trailing
// dims: its target is a constant whose trailing entries multiply to K and whose leading entries each keep the input
// dim (0 with special_zero or the same static value). A single -1 is accepted only when the reshaped input dims are
// static and multiply to K; otherwise the -1 could absorb part of them (e.g. [2, 128] -> [-1, 1, 256]).
bool reshapes_trailing_dims_only(const std::shared_ptr<ov::op::v1::Reshape>& r, size_t K, bool split) {
    const auto target = ov::as_type_ptr<ov::op::v0::Constant>(r->get_input_node_shared_ptr(1));
    const auto& in = r->get_input_partial_shape(0);
    if (!target || in.rank().is_dynamic() || in.rank().get_length() < (split ? 1 : 2)) {
        return false;
    }
    const auto t = target->cast_vector<int64_t>();
    const auto in_rank = static_cast<size_t>(in.rank().get_length());
    if (t.size() != (split ? in_rank + 1 : in_rank - 1)) {
        return false;
    }
    const size_t lead = std::min(in_rank, t.size()) - 1;
    int64_t trailing = 1;
    for (size_t i = lead; i < t.size(); ++i) {
        if (t[i] <= 0) {
            return false;
        }
        trailing *= t[i];
    }
    if (trailing != static_cast<int64_t>(K)) {
        return false;
    }
    size_t inferred = 0;
    for (size_t i = 0; i < lead; ++i) {
        if (t[i] == 0 && r->get_special_zero()) {
            continue;
        }
        if (t[i] == -1) {
            ++inferred;
            continue;
        }
        if (!in[i].is_static() || in[i].get_length() != t[i]) {
            return false;
        }
    }
    if (inferred == 0) {
        return true;
    }
    if (inferred > 1) {
        return false;
    }
    int64_t reshaped = 1;
    for (size_t i = lead; i < in_rank; ++i) {
        if (!in[i].is_static()) {
            return false;
        }
        reshaped *= in[i].get_length();
    }
    return reshaped == static_cast<int64_t>(K);
}

}  // namespace

bool FuseHadamardIntoFC::run_on_model(const std::shared_ptr<ov::Model>& model) {
    size_t fused = 0;
    // The converter shares one H constant across the projections; check each one once (H_4096 is 16M values).
    std::unordered_map<const ov::Node*, size_t> blocks;
    for (const auto& node : model->get_ordered_ops()) {
        auto fc = ov::as_type_ptr<op::FullyConnectedCompressed>(node);
        // K = wshape[1] below only holds for NK weights.
        if (!fc || fc->get_weights_layout() != op::FcWeightsLayout::NK) {
            continue;
        }
        const auto& wshape = fc->get_input_partial_shape(1);
        if (wshape.rank().is_dynamic() || wshape.rank().get_length() < 2 || !wshape[1].is_static()) {
            continue;
        }
        const size_t K = static_cast<size_t>(wshape[1].get_length());

        // Reshape back to [..., K], absent when K == B
        auto back = ov::as_type_ptr<ov::op::v1::Reshape>(fc->get_input_node_shared_ptr(0));
        auto mm = back ? back->get_input_node_shared_ptr(0) : fc->get_input_node_shared_ptr(0);
        if (auto matmul = ov::as_type_ptr<ov::op::v0::MatMul>(mm); matmul && matmul->get_transpose_a()) {
            continue;
        }
        // MatMul / FullyConnected with the H constant
        std::shared_ptr<ov::op::v0::Constant> h;
        if (ov::as_type_ptr<ov::op::v0::MatMul>(mm) || ov::as_type_ptr<op::FullyConnected>(mm)) {
            h = constant_behind(mm->input_value(1));
        }
        size_t block = 0;
        if (h) {
            auto it = blocks.find(h.get());
            if (it == blocks.end()) {
                it = blocks.emplace(h.get(), hadamard_block_of(h)).first;
            }
            block = it->second;
        }
        if (block == 0 || K % block != 0) {
            GPU_DEBUG_TRACE_DETAIL << "[hadamard-fc] " << fc->get_friendly_name() << ": no Hadamard constant dividing K=" << K << " behind "
                                   << mm->get_type_name() << std::endl;
            continue;
        }
        // Reshape to [..., K/B, B]. The split and the merge come as a pair (both absent when K == B), and they
        // may only touch the trailing dims: otherwise bypassing them would change the FC input shape, e.g. an
        // unrelated flatten [2, 3, K] -> [6, K] in front of the MatMul.
        ov::Output<ov::Node> x = mm->input_value(0);
        auto fwd = ov::as_type_ptr<ov::op::v1::Reshape>(x.get_node_shared_ptr());
        if (static_cast<bool>(fwd) != static_cast<bool>(back) ||
            (fwd && (!reshapes_trailing_dims_only(fwd, K, true) || !reshapes_trailing_dims_only(back, K, false)))) {
            GPU_DEBUG_TRACE_DETAIL << "[hadamard-fc] " << fc->get_friendly_name() << ": the Reshapes around the Hadamard MatMul change more than the last dim"
                                   << std::endl;
            continue;
        }
        if (fwd) {
            x = fwd->input_value(0);
        }
        std::vector<int8_t> signs;
        if (auto mul = ov::as_type_ptr<ov::op::v1::Multiply>(x.get_node_shared_ptr())) {
            std::shared_ptr<ov::op::v0::Constant> sc;
            size_t data_port = 0;
            for (size_t p = 0; p < 2 && !sc; ++p) {
                sc = constant_behind(mul->input_value(p));
                data_port = 1 - p;
            }
            // Only a [1, ..., 1, K] sign vector applied to data that is provably K wide, so the Multiply does not
            // broadcast. Otherwise (e.g. a dynamic last dim, which could be 1) the Multiply stays in the graph and
            // only the rotation is fused.
            const auto& data_shape = mul->get_input_partial_shape(data_port);
            const bool data_is_k_wide = data_shape.rank().is_static() && data_shape.rank().get_length() > 0 &&
                                        data_shape[data_shape.rank().get_length() - 1].is_static() &&
                                        data_shape[data_shape.rank().get_length() - 1].get_length() == static_cast<int64_t>(K);
            if (sc && data_is_k_wide && ov::shape_size(sc->get_shape()) == K && !sc->get_shape().empty() && sc->get_shape().back() == K &&
                sc->get_shape().size() <= static_cast<size_t>(data_shape.rank().get_length())) {
                const auto vals = sc->cast_vector<float>();
                bool ok = true;
                signs.resize(K);
                for (size_t i = 0; i < K && ok; ++i) {
                    ok = std::fabs(std::fabs(vals[i]) - 1.0F) < 1e-6F;
                    signs[i] = vals[i] < 0 ? -1 : 1;
                }
                if (ok) {
                    x = mul->input_value(data_port);
                } else {
                    signs.clear();
                }
            }
        }
        const auto& xs = x.get_partial_shape();
        const auto& fc_in = fc->get_input_partial_shape(0);
        if (xs.rank().is_dynamic() || !xs.compatible(fc_in)) {
            GPU_DEBUG_TRACE_DETAIL << "[hadamard-fc] " << fc->get_friendly_name() << ": source shape " << xs << " does not match the FC input shape " << fc_in
                                   << std::endl;
            continue;
        }
        fc->input(0).replace_source_output(x);
        auto& rt = fc->get_rt_info();
        rt[int2_hadamard_block_key] = static_cast<int64_t>(block);
        rt[int2_hadamard_signs_key] = signs;
        ++fused;
        GPU_DEBUG_TRACE_DETAIL << "[hadamard-fc] fused into " << fc->get_friendly_name() << " K=" << K << " block=" << block
                               << " signs=" << (signs.empty() ? "folded" : "explicit") << std::endl;
    }
    GPU_DEBUG_TRACE << "[hadamard-fc] fused " << fused << " input rotations" << std::endl;
    return fused != 0;
}

}  // namespace ov::intel_gpu
