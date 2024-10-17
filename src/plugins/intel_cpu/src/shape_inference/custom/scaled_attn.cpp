// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "scaled_attn.hpp"

#include "shape_inference/shape_inference_cpu.hpp"
#include "shape_inference/shape_inference_ngraph.hpp"
#include "transformations/cpu_opset/common/op/sdpa.hpp"
#include "utils.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

class SDPAShapeInfer : public ShapeInferEmptyPads {
public:
    SDPAShapeInfer(const ScaledDotProductAttentionWithKVCache::Config& config) : m_config(config) {}

    IShapeInfer::Result infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                              const std::unordered_map<size_t, MemoryPtr>& data_dependency) override {
        const auto& query_dims = input_shapes.front().get();
        VectorDims present_v_dims = input_shapes.back().get();
        const auto& beam_idx_dims = input_shapes.end()[-3].get();
        const auto& permute_axes = m_config.permute_axes;

        if (permute_axes.empty()) {
            // [B, H, L, S]
            present_v_dims[0] = beam_idx_dims[0];
            present_v_dims[2] += query_dims[2];
            // normal and fast path
            if (present_v_dims[3] == query_dims[3])
                return {{query_dims, present_v_dims, present_v_dims}, ShapeInferStatus::success};

            // diff kv feature size
            auto output_dims = query_dims;
            output_dims[3] = present_v_dims[3];
            auto present_k_dims = present_v_dims;
            present_k_dims[3] = query_dims[3];
            return {{output_dims, present_k_dims, present_v_dims}, ShapeInferStatus::success};
        }

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

        // normal and fast path
        if (present_v_dims[3] == query_dims[3])
            return {{output_dims, present_v_dims, present_v_dims}, ShapeInferStatus::success};

        // diff kv feature size
        output_dims[3] = present_v_dims[3];
        auto present_k_dims = present_v_dims;
        present_k_dims[3] = query_dims[3];
        return {{output_dims, present_k_dims, present_v_dims}, ShapeInferStatus::success};
    }

    port_mask_t get_port_mask() const override {
        return EMPTY_PORT_MASK;
    }

private:
    ScaledDotProductAttentionWithKVCache::Config m_config;
};

ShapeInferPtr SDPAShapeInferFactory::makeShapeInfer() const {
    if (auto sdpa = std::dynamic_pointer_cast<const ScaledDotProductAttentionWithKVCache>(m_op)) {
        const auto& config = sdpa->get_config();
        if (config.output_BLHxS == false)
            return std::make_shared<SDPAShapeInfer>(config);
    }
    // fallback to ngraph shape infer on non-perf-critical case
    return std::make_shared<NgraphShapeInfer>(make_shape_inference(m_op), EMPTY_PORT_MASK);
}

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
