// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace op {
namespace internal {

///
/// \brief Rotary Positional Embeddings operation
/// Internal operation which may change in the future
class TRANSFORMATIONS_API RoPE : public Op {
public:
    OPENVINO_OP("RoPE", "ie_internal_opset", Op);

    RoPE() = default;

    struct Config {
        size_t slice_start = 0;  // slice inner-most dimensions of input
        size_t slice_stop = 0;
        bool input_trans0213 = false;  // transpose input dim 1&2
        bool is_interleaved = false;   // interleaved mode, implies trans0213 happens after RoPE
        size_t rotary_ndims = 0;       // dimensions to be embedded (d in the description)
        bool is_chatglm = false;       // chatglm is special which overrides other setting
        bool support_2d_rope = false;  // 2d rope mode, Support 2 dimentional rope which is independant of batch and
                                       // each head. change input order to [batch, head_cnt, 4608] to support 2d rope
        bool is_qwen = false;          // Qwen is special which overrides other setting
        size_t head_cnt = 0;
        size_t head_size = 0;
        int gather_position_arg_id =
            0;  // arg id of position tensor, ==3 when gather from sin/cos inputs according to position is required
    };

    RoPE(const OutputVector& args, const Config& config);

    const Config& get_config() const;
    void set_config(const Config& config);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

private:
    Config m_config{};
};

}  // namespace internal
}  // namespace op
}  // namespace ov
