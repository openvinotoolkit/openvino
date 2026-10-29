// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_matmul_to_fc_gguf.hpp"

#include <cstring>
#include <memory>

#include "intel_gpu/op/fully_connected_gguf.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gguf_dequantize.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/util/pp.hpp"

namespace ov::intel_gpu {

namespace {

using ov::op::internal::GGUFDequantize;

// Q6_K block: ql[128] | qh[64] | scales[16] | d (f16). Regroup the bytes of all blocks into four planes so that the
// kernels read aligned vectors.
std::shared_ptr<ov::op::v0::Constant> make_q6k_planes(const ov::op::v0::Constant& blocks) {
    const size_t block_bytes = GGUFDequantize::block_bytes(GGUFDequantize::Format::Q6_K);
    const auto& shape = blocks.get_shape();
    const size_t nblk = ov::shape_size(shape) / block_bytes;
    ov::Tensor planes(ov::element::u8, shape);
    const auto* src = blocks.get_data_ptr<uint8_t>();
    auto* dst = static_cast<uint8_t*>(planes.data());
    for (size_t i = 0; i < nblk; ++i) {
        const uint8_t* b = src + i * block_bytes;
        std::memcpy(dst + i * 128, b, 128);
        std::memcpy(dst + nblk * 128 + i * 64, b + 128, 64);
        std::memcpy(dst + nblk * 192 + i * 16, b + 192, 16);
        std::memcpy(dst + nblk * 208 + i * 2, b + 208, 2);
    }
    return std::make_shared<ov::op::v0::Constant>(planes);
}

}  // namespace

ConvertMatMulToFullyConnectedGGUF::ConvertMatMulToFullyConnectedGGUF() {
    using namespace ov::pass::pattern;

    auto activations_m = any_input(rank_more_than(1));
    auto blocks_m = wrap_type<ov::op::v0::Constant>(type_matches(ov::element::u8) && rank_equals(2));
    auto dequantize_m = wrap_type<GGUFDequantize>({blocks_m}, [](const ov::Output<ov::Node>& output) {
        const auto dequantize = ov::as_type_ptr<GGUFDequantize>(output.get_node_shared_ptr());
        return dequantize && output.get_target_inputs().size() == 1 && op::FullyConnectedGGUF::is_supported(dequantize->get_format());
    });
    auto convert_m = optional<ov::op::v0::Convert>({dequantize_m}, consumers_count(1));
    auto matmul_m = wrap_type<ov::op::v0::MatMul>({activations_m, convert_m}, {{"transpose_a", false}, {"transpose_b", true}});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        if (transformation_callback(m.get_match_root())) {
            return false;
        }
        const auto& pattern_map = m.get_pattern_value_map();
        const auto matmul = pattern_map.at(matmul_m).get_node_shared_ptr();
        const auto dequantize = ov::as_type_ptr<GGUFDequantize>(pattern_map.at(dequantize_m).get_node_shared_ptr());
        const auto blocks = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(blocks_m).get_node_shared_ptr());

        std::shared_ptr<ov::op::v0::Constant> weights = blocks;
        if (dequantize->get_format() == GGUFDequantize::Format::Q6_K) {
            weights = make_q6k_planes(*blocks);
            weights->set_friendly_name(blocks->get_friendly_name());
            ov::copy_runtime_info(blocks, weights);
        }

        auto fc = std::make_shared<op::FullyConnectedGGUF>(pattern_map.at(activations_m), weights, dequantize->get_format());
        fc->set_friendly_name(matmul->get_friendly_name());
        ov::NodeVector from{matmul, dequantize};
        if (pattern_map.count(convert_m)) {
            from.push_back(pattern_map.at(convert_m).get_node_shared_ptr());
        }
        ov::copy_runtime_info(from, fc);
        ov::replace_node(matmul, fc);
        return true;
    };

    auto m = std::make_shared<Matcher>(matmul_m, "ConvertMatMulToFullyConnectedGGUF");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
