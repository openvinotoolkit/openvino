// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace internal {

/// \brief Compressed linear (MatMul) operation with IQ3_XXS quantized weights.
///
/// Performs Y = X @ W^T where W is stored in IQ3_XXS compressed format.
/// The compressed weight blob is passed as an opaque u8 Constant input.
/// The plugin is responsible for fused on-the-fly dequantization during compute.
///
/// Inputs:
///   0: activation [M, K] - float16/float32
///   1: compressed_weights [compressed_bytes] - u8 opaque blob
///
/// Attributes:
///   weight_shape: logical shape of the weight matrix [N, K]
///   block_size: number of weights per super-block (256 for IQ3_XXS)
///   bytes_per_block: bytes per super-block (98 for IQ3_XXS)
///
/// Output:
///   0: result [M, N] - same element type as activation
///
class OPENVINO_API IQ3XXSLinear : public ov::op::Op {
public:
    OPENVINO_OP("IQ3XXSLinear");

    IQ3XXSLinear() = default;

    /// \brief Constructs an IQ3XXSLinear operation.
    ///
    /// \param activation Input activation tensor [M, K]
    /// \param compressed_weights Opaque u8 blob containing IQ3_XXS encoded weights
    /// \param weight_shape Logical weight shape [N, K]
    /// \param block_size Number of weights per block (default 256)
    /// \param bytes_per_block Bytes per block (default 98)
    IQ3XXSLinear(const Output<Node>& activation,
                 const Output<Node>& compressed_weights,
                 const ov::Shape& weight_shape,
                 int64_t block_size = 256,
                 int64_t bytes_per_block = 98);

    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    const ov::Shape& get_weight_shape() const { return m_weight_shape; }
    int64_t get_block_size() const { return m_block_size; }
    int64_t get_bytes_per_block() const { return m_bytes_per_block; }

private:
    ov::Shape m_weight_shape;
    int64_t m_block_size{256};
    int64_t m_bytes_per_block{98};
};

}  // namespace internal
}  // namespace op
}  // namespace ov
