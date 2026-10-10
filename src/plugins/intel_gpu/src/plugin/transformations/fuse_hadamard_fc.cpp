// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fuse_hadamard_fc.hpp"

#include <bitset>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include "intel_gpu/op/fully_connected.hpp"
#include "intel_gpu/op/fully_connected_compressed.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/util/env_util.hpp"

namespace ov::intel_gpu {

namespace {

constexpr size_t kBlock = 1024;

std::shared_ptr<ov::op::v0::Constant> constant_behind(const ov::Output<ov::Node>& out) {
    auto node = out.get_node_shared_ptr();
    // ConvertMatMulToFullyConnected leaves Convert(Transpose(H)); H is symmetric.
    for (int i = 0; i < 3; ++i) {
        if (ov::as_type_ptr<ov::op::v0::Convert>(node) || ov::as_type_ptr<ov::op::v1::Transpose>(node))
            node = node->get_input_node_shared_ptr(0);
    }
    return ov::as_type_ptr<ov::op::v0::Constant>(node);
}

// The normalised Sylvester H_1024: +1/32 everywhere in row/column 0, and
// H[i][j] = (-1)^popcount(i & j) / 32. Checking a stride of rows is enough to
// tell it from any other 1024x1024 constant.
bool is_hadamard_1024(const std::shared_ptr<ov::op::v0::Constant>& c) {
    if (!c)
        return false;
    const auto& s = c->get_shape();
    if (s.size() != 2 || s[0] != kBlock || s[1] != kBlock)
        return false;
    const auto v = c->cast_vector<float>();
    const float want = 1.0f / 32.0f;
    for (size_t i = 0; i < kBlock; i += 37) {
        for (size_t j = 0; j < kBlock; j += 41) {
            const float sign = (std::bitset<32>(i & j).count() & 1) ? -1.0f : 1.0f;
            if (std::fabs(v[i * kBlock + j] - sign * want) > 1e-4f)
                return false;
        }
    }
    return true;
}

}  // namespace

bool FuseHadamardIntoFC::run_on_model(const std::shared_ptr<ov::Model>& model) {
    const bool trace = ov::util::getenv_bool("OV_TERNOCL_HADAMARD_DEBUG");
    size_t fused = 0;
    for (const auto& node : model->get_ordered_ops()) {
        auto fc = ov::as_type_ptr<op::FullyConnectedCompressed>(node);
        if (!fc)
            continue;
        const auto& wshape = fc->get_input_partial_shape(1);
        if (wshape.rank().is_dynamic() || wshape.rank().get_length() < 2 || !wshape[1].is_static())
            continue;
        const size_t K = static_cast<size_t>(wshape[1].get_length());
        if (K % kBlock != 0)
            continue;

        // Reshape back to [..., K]
        auto back = ov::as_type_ptr<ov::op::v1::Reshape>(fc->get_input_node_shared_ptr(0));
        if (!back) {
            if (trace)
                std::cerr << "[hadamard-fc] " << fc->get_friendly_name() << ": input is " << fc->get_input_node_shared_ptr(0)->get_type_name() << std::endl;
            continue;
        }
        // MatMul / FullyConnected with the H constant
        auto mm = back->get_input_node_shared_ptr(0);
        std::shared_ptr<ov::op::v0::Constant> h;
        if (ov::as_type_ptr<ov::op::v0::MatMul>(mm) || ov::as_type_ptr<op::FullyConnected>(mm))
            h = constant_behind(mm->input_value(1));
        if (!is_hadamard_1024(h)) {
            if (trace) {
                auto w = mm->get_input_node_shared_ptr(1);
                std::cerr << "[hadamard-fc] " << fc->get_friendly_name() << ": no H behind " << mm->get_type_name() << " inputs=" << mm->get_input_size()
                          << " w=" << w->get_type_name() << " " << w->get_output_element_type(0) << " " << w->get_output_partial_shape(0);
                if (w->get_input_size() > 0) {
                    std::cerr << " <- " << w->get_input_node_shared_ptr(0)->get_type_name() << " "
                              << w->get_input_node_shared_ptr(0)->get_output_partial_shape(0);
                }
                if (h) {
                    const auto v = h->cast_vector<float>();
                    std::cerr << " const[0,0..3]=" << v[0] << "," << v[1] << "," << v[2] << "," << v[3] << " [1,1]=" << v[kBlock + 1]
                              << " [1,3]=" << v[kBlock + 3];
                }
                std::cerr << std::endl;
            }
            continue;
        }
        // Reshape to [..., K/1024, 1024]
        auto fwd = ov::as_type_ptr<ov::op::v1::Reshape>(mm->get_input_node_shared_ptr(0));
        if (!fwd)
            continue;
        ov::Output<ov::Node> x = fwd->input_value(0);
        std::vector<int8_t> signs;
        if (auto mul = ov::as_type_ptr<ov::op::v1::Multiply>(x.get_node_shared_ptr())) {
            std::shared_ptr<ov::op::v0::Constant> sc;
            size_t data_port = 0;
            for (size_t p = 0; p < 2 && !sc; ++p) {
                sc = constant_behind(mul->input_value(p));
                data_port = 1 - p;
            }
            if (sc && ov::shape_size(sc->get_shape()) == K) {
                const auto vals = sc->cast_vector<float>();
                bool ok = true;
                signs.resize(K);
                for (size_t i = 0; i < K && ok; ++i) {
                    ok = std::fabs(std::fabs(vals[i]) - 1.0f) < 1e-6f;
                    signs[i] = vals[i] < 0 ? -1 : 1;
                }
                if (ok) {
                    x = mul->input_value(data_port);
                } else {
                    signs.clear();
                }
            }
        }
        // The Reshape to [..., K/1024, 1024] already pins the source's last dim
        // to K; it only needs to be checked when the graph states it.
        const auto& xs = x.get_partial_shape();
        if (xs.rank().is_dynamic() || (xs[xs.rank().get_length() - 1].is_static() && static_cast<size_t>(xs[xs.rank().get_length() - 1].get_length()) != K)) {
            if (trace)
                std::cerr << "[hadamard-fc] " << fc->get_friendly_name() << ": source shape " << xs << " does not end in K=" << K << std::endl;
            continue;
        }
        fc->input(0).replace_source_output(x);
        auto& rt = fc->get_rt_info();
        rt[int2_hadamard_block_key] = static_cast<int64_t>(kBlock);
        rt[int2_hadamard_signs_key] = signs;
        ++fused;
        if (trace) {
            std::cerr << "[hadamard-fc] fused into " << fc->get_friendly_name() << " K=" << K << " signs=" << (signs.empty() ? "folded" : "explicit")
                      << std::endl;
        }
    }
    if (trace || (fused && ov::util::getenv_bool("OV_TERNOCL_INT2_DEBUG")))
        std::cerr << "[hadamard-fc] fused " << fused << " input rotations" << std::endl;
    return fused != 0;
}

}  // namespace ov::intel_gpu
