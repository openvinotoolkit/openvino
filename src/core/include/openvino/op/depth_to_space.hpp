// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "openvino/op/util/attr_types.hpp"

namespace ov {
namespace op {
namespace v0 {
/// \brief DepthToSpace permutes data from the depth dimension of the input blob into
///        spatial dimensions.
///
/// \note  Values from the depth dimension (assuming NCHW layout) are moved in
///        spatial blocks to the height and width dimensions.
///
///        Output node produces a tensor with shape:
///        [N, C/(blocksize * blocksize), H * blocksize, W * blocksize]
/// \ingroup ov_ops_cpp_api
class OPENVINO_API DepthToSpace : public Op {
public:
    OPENVINO_OP("DepthToSpace", "opset1");
    BWDCMP_RTTI_DECLARATION;

    enum class DepthToSpaceMode {
        // The input depth is divided to [block_size, ..., block_size, new_depth]
        BLOCKS_FIRST,
        // The input depth is divided to [new_depth, block_size, ..., block_size]
        DEPTH_FIRST
    };

    DepthToSpace() = default;
    /// \brief Constructs a DepthToSpace operation.
    ///
    /// \param data Node producing the input tensor
    /// \param mode Specifies how the input depth dimension is split to block
    /// coordinates
    /// \param block_size The size of the block of values to be moved
    DepthToSpace(const Output<Node>& data, const DepthToSpaceMode& mode, std::size_t block_size = 1);

    DepthToSpace(const Output<Node>& data, const std::string& mode, std::size_t block_size = 1);
    bool visit_attributes(AttributeVisitor& visitor) override;

    const std::size_t& get_block_size() const {
        return m_blocksize;
    }
    DepthToSpaceMode get_mode() const {
        return m_mode;
    }
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    void validate_and_infer_types() override;
    OPENVINO_SUPPRESS_DEPRECATED_START
    bool evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const override;
    OPENVINO_SUPPRESS_DEPRECATED_END
    bool has_evaluate() const override;

protected:
    std::size_t m_blocksize;
    DepthToSpaceMode m_mode;
};
}  // namespace v0
}  // namespace op

OPENVINO_API
std::ostream& operator<<(std::ostream& s, const op::v0::DepthToSpace::DepthToSpaceMode& type);

template <>
class OPENVINO_API AttributeAdapter<op::v0::DepthToSpace::DepthToSpaceMode>
    : public EnumAttributeAdapterBase<op::v0::DepthToSpace::DepthToSpaceMode> {
public:
    AttributeAdapter(op::v0::DepthToSpace::DepthToSpaceMode& value)
        : EnumAttributeAdapterBase<op::v0::DepthToSpace::DepthToSpaceMode>(value) {}

    OPENVINO_RTTI("AttributeAdapter<ov::op::v0::DepthToSpace::DepthToSpaceMode>");
    BWDCMP_RTTI_DECLARATION;
};

}  // namespace ov
