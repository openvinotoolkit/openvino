// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "scaled_attn.hpp"

#include <cstddef>
#include <functional>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cpu_memory.h"
#include "cpu_types.h"
#include "openvino/core/except.hpp"
#include "openvino/core/type.hpp"
#include "shape_inference/shape_inference.hpp"
#include "shape_inference/shape_inference_cpu.hpp"
#include "shape_inference/shape_inference_status.hpp"
#include "transformations/cpu_opset/common/op/sdpa.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu::node {

class SDPAShapeInfer : public ShapeInferEmptyPads {
public:
    SDPAShapeInfer(ScaledDotProductAttentionWithKVCache::Config config) : m_config(std::move(config)) {}

    IShapeInfer::Result infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                              [[maybe_unused]] const std::unordered_map<size_t, MemoryPtr>& data_dependency) override {
        const auto inputs_size = input_shapes.size();
        OPENVINO_ASSERT(inputs_size >= 3, "SDPAShapeInfer: expected at least 3 inputs, got ", inputs_size);
        const auto& query_dims = input_shapes.front().get();
        VectorDims present_v_dims = input_shapes.back().get();
        const auto& beam_idx_dims = input_shapes[inputs_size - 3].get();
        const auto& permute_axes_origin = m_config.permute_axes;
        static const std::vector<size_t> direct_axis_order = {0, 1, 2, 3};
        const auto& permute_axes = (permute_axes_origin.empty()) ? direct_axis_order : permute_axes_origin;
        // permute_axes[0,1,2,3] gives axis indices of B,H,L,S for query & present_kv
        const size_t batch_index = permute_axes[0];
        const size_t length_index = permute_axes[2];
        present_v_dims[batch_index] = beam_idx_dims[0];
        present_v_dims[length_index] += query_dims[length_index];

        auto n_dims = query_dims.size();
        VectorDims output_dims(n_dims);
        for (size_t i = 0; i < n_dims; i++) {
            output_dims[i] = query_dims[permute_axes[i]];
        }
        if (inputs_size == 7 && !m_config.is_causal) {
            const auto& attn_mask_dims = input_shapes[3].get();
            bool attn_mask_ok = true;
            auto attn_mask_dims_size = attn_mask_dims.size();
            auto weight_dims = output_dims;
            auto weight_dims_size = weight_dims.size();
            if (attn_mask_dims_size >= 2 && attn_mask_dims_size <= weight_dims_size) {
                auto check_broadcast = [](const size_t& target, const size_t& to) -> bool {
                    return any_of(target, to, 1U);
                };
                weight_dims[3] = present_v_dims[length_index];
                auto offset = weight_dims_size - attn_mask_dims_size;
                for (int i = attn_mask_dims_size - 1; i >= 0; i--) {
                    attn_mask_ok = attn_mask_ok && check_broadcast(attn_mask_dims[i], weight_dims[i + offset]);
                }
            } else {
                attn_mask_ok = false;
            }
            if (!attn_mask_ok) {
                const auto& cur_k_dims = input_shapes[1].get();
                const auto& cur_v_dims = input_shapes[2].get();
                const auto& cache_k_dims = input_shapes[5].get();
                const auto& cache_v_dims = input_shapes[6].get();
                OPENVINO_THROW("attention_mask do not match q and k,",
                               " query_dims:",
                               ov::intel_cpu::vec2str(query_dims),
                               " cur_k_dims:",
                               ov::intel_cpu::vec2str(cur_k_dims),
                               " cur_v_dims:",
                               ov::intel_cpu::vec2str(cur_v_dims),
                               " attn_mask_dims:",
                               ov::intel_cpu::vec2str(attn_mask_dims),
                               " beam_idx_dims:",
                               ov::intel_cpu::vec2str(beam_idx_dims),
                               " cache_k_dims:",
                               ov::intel_cpu::vec2str(cache_k_dims),
                               " cache_v_dims:",
                               ov::intel_cpu::vec2str(cache_v_dims));
            }
        }

        // normal and fast path
        if (present_v_dims[3] == query_dims[3]) {
            return {{output_dims, present_v_dims, present_v_dims}, ShapeInferStatus::success};
        }

        // diff kv feature size
        output_dims[3] = present_v_dims[3];
        auto present_k_dims = present_v_dims;
        present_k_dims[3] = query_dims[3];
        return {{output_dims, present_k_dims, present_v_dims}, ShapeInferStatus::success};
    }

    [[nodiscard]] port_mask_t get_port_mask() const override {
        return EMPTY_PORT_MASK;
    }

private:
    ScaledDotProductAttentionWithKVCache::Config m_config;
};

ShapeInferPtr SDPAShapeInferFactory::makeShapeInfer() const {
    if (auto sdpa = ov::as_type_ptr<const ScaledDotProductAttentionWithKVCache>(m_op)) {
        const auto& config = sdpa->get_config();
        if (!config.output_BLHxS) {
            return std::make_shared<SDPAShapeInfer>(config);
        }
    }
    // fallback to ngraph shape infer on non-perf-critical case
    return make_shape_inference(m_op);
}

}  // namespace ov::intel_cpu::node
