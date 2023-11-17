// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/node.hpp>
#include <ngraph/op/op.hpp>

namespace ov {
namespace intel_cpu {

/**
 * The operation performs rotary positional embedding operation described in:
 *   ROFORMER: ENHANCED TRANSFORMER WITH ROTARY POSITION EMBEDDING by Jianlin Su
 *
 *  the core computation is application of 2x2 rotation matrix on basis
 *  of pair of input states x[i0] & x[i1] to get the rotary embedded pair of output
 *  states y[i0] and y[i1]:
 *
 *  suppose dimension of hidden states (of each attention head) is N and 2*d of which
 *  are to be embedded (2*d <= N), non-embedded parts are copied into output.
 *
 *  for i in 0...d/2
 *      if (is_interleaved) {
 *          // interleaving style of indexing
 *          i0 = i*2
 *          i1 = i*2 + 1
 *      } else {
 *          // rotate-half style of indexing
 *          i0 = i
 *          i1 = i + d/2
 *      }
 *      y[i0] = x[i0]*cos(m * xita[i]) - x[i1]*sin(m * xita[i])
 *      y[i1] = x[i1]*cos(m * xita[i]) + x[i0]*sin(m * xita[i])
 *  Note: m is token position of current input
 *
 *  based on configuration, additional preprocessing steps maybe performed as well:
 *      - slicing last dimension of input tensor
 *          (when q/k/v are merged and only q or k part is to be extracted & embedded)
 *      - transpose input tensor
 *          (when q/k comes from fullyconnect has layout [batch, seq_len, head_cnt, head_dim]
 *           but output of RoPE is required to be of layout [batch, head_cnt, seq_length, head_dims])
 *      - gather sin/cos from input tensor 2&3 using position index tensor passed through input 4
 *
 * Inputs:
 *     1. Input hidden states tensor of type T1 - shape:
 *           [batch, seq_length, head_cnt, head_dims] when input_trans0213 == false OR
 *           [batch, head_cnt, seq_length, head_dims] when input_trans0213 == true
 *     2. pre-calculated cos(m*xita[n]) tensor of type T2 - shape [1, 1, max_position_embeddings, d].
 *     3. pre-calculated sin(m*xita[n]) tensor of type T2 - shape [1, 1, max_position_embeddings, d].
 *        input 3 is combined with 2 when is_interleaved is true.
 *     4. postion index tensor of type T3 - shape [batch, 1, seq_length, 1 or d] OR [batch, seq_length] optional
 * Outputs:
 *     1. New embedding tensor of type T1 and of shape [batch, head_cnt, seq_length, head_dims]
 * Types:
 *     T1 - FP32 or BF16
 *     T2 - FP32
 *     T3 - I32
 */
class RoPENode : public ngraph::op::Op {
public:
    OPENVINO_OP("RoPE", "cpu_plugin_opset");

    RoPENode() = default;

    struct Config {
        size_t slice_start = 0;  // slice inner-most dimensions of input
        size_t slice_stop = 0;
        bool input_trans0213 = false;  // transpose input dim 1&2
        bool is_interleaved = false;   // interleaved mode, implies trans0213 happens after RoPE
        size_t rotary_ndims = 0;       // dimensions to be embedded (the 2*d in the description)
        int gather_position_arg_id =
            0;  // arg id of position tensor, ==3 when gather from sin/cos inputs according to position is required
    };

    RoPENode(const OutputVector& args, const Config& cfg);

    bool visit_attributes(ngraph::AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override;

    const Config& get_config() const {
        return m_config;
    }

    Config& get_config() {
        return m_config;
    }

private:
    Config m_config;
};

}  // namespace intel_cpu
}  // namespace ov
