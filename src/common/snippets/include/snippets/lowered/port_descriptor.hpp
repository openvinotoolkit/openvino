// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "snippets/shape_types.hpp"
#include "snippets/emitter.hpp"


namespace ov {
namespace snippets {
namespace lowered {

class LinearIRBuilder;
class PortDescriptor;
using PortDescriptorPtr = std::shared_ptr<PortDescriptor>;
class PortDescriptor {
    friend class LinearIRBuilder;
public:
    explicit PortDescriptor(const ov::Input<ov::Node>& node,
                            VectorDims subtensor_shape = {},
                            std::vector<size_t> layout = {});
    explicit PortDescriptor(const ov::Input<const ov::Node>& node,
                            VectorDims subtensor_shape = {},
                            std::vector<size_t> layout = {});
    explicit PortDescriptor(const ov::Output<ov::Node>& node,
                            VectorDims subtensor_shape = {},
                            std::vector<size_t> layout = {});
    explicit PortDescriptor(const ov::Output<const ov::Node>& node,
                            VectorDims subtensor_shape = {},
                            std::vector<size_t> layout = {});
    PortDescriptor(VectorDims shape, VectorDims subtensor_shape, std::vector<size_t> layout = {}, Reg reg = {});
    PortDescriptor(VectorDimsPtr shape, VectorDims subtensor_shape, std::vector<size_t> layout = {}, Reg reg = {});
    PortDescriptor();

    const VectorDims& get_shape() const;
    const VectorDims& get_subtensor() const {return m_subtensor_shape;}
    const std::vector<size_t>& get_layout() const {return m_layout;}
    const Reg& get_reg() const { return m_reg; }

    void set_shape(const VectorDims& tensor);
    void set_layout(const std::vector<size_t>& layout);
    void set_subtensor(const VectorDims& subtensor);
    void set_reg(Reg reg) { m_reg = std::move(reg); }
    void set_reg_type(RegType type) { m_reg.type = type; }
    void set_reg_idx(size_t idx) { m_reg.idx = idx; }

    // Indexing starts from the end (rbegin() + idx)
    void set_subtensor_dim(size_t idx, VectorDims::value_type value);

    std::string serialize() const;
    bool empty() const { return m_layout.empty() && m_subtensor_shape.empty();}
    PortDescriptorPtr clone() const;

    friend bool operator==(const PortDescriptor& lhs, const PortDescriptor& rhs);
    friend bool operator!=(const PortDescriptor& lhs, const PortDescriptor& rhs) {return !(lhs == rhs);}

private:
    void validate_arguments();
    /// \brief Original tensor shape
    VectorDimsPtr m_tensor_shape = nullptr;
    /// \brief Order of dimensions: NCHW == {0, 1, 2, 3}, NHWC == {0, 2, 3, 1}, NCHW16c == {0, 1, 2, 3, 1}
    std::vector<size_t> m_layout{};
    /// \brief Minimal tensor size that could be processed in one call
    VectorDims m_subtensor_shape{};
    /// \brief The corresponding abstract/physical register
    Reg m_reg { RegType::gpr, 0 };

    /// Notes:
    ///   - `m_tensor_shape` is dense shape which is controlled by expression outputs.
    ///     It means that the result of data writing of expression outputs should be read using this shape by the next expression inputs.
    ///   - `m_layout` is the order of data reading or writing by MemoryAccess ops. Note that only MemoryAccess ops may have `m_layout`.
    ///     For other expressions this order parameter is simply ignored for now.
    ///     if it's input port of MemoryAccess expression:
    ///      - `m_layout` shows how the data should be read (by which strides) using m_tensor_shape.
    ///     If it's output port of MemoryAccess expression:
    ///      - `m_layout` shows how the data should be written (by which strides) to get m_tensor_shape.
};

class PortDescriptorUtils {
public:
    static void set_port_descriptor_ptr(const ov::Input<ov::Node>& in, const PortDescriptorPtr& desc);
    static void set_port_descriptor_ptr(const ov::Output<ov::Node>& out, const PortDescriptorPtr& desc);
    static void set_port_descriptor(const ov::Input<ov::Node>& in, std::vector<size_t> subtensor, std::vector<size_t> layout = {});
    static void set_port_descriptor(const ov::Output<ov::Node>& out, std::vector<size_t> subtensor, std::vector<size_t> layout = {});

    static PortDescriptorPtr get_port_descriptor_ptr(const ov::Input<ov::Node>& in);
    static PortDescriptorPtr get_port_descriptor_ptr(const ov::Input<const ov::Node>& in);
    static PortDescriptorPtr get_port_descriptor_ptr(const ov::Output<ov::Node>& out);
    static PortDescriptorPtr get_port_descriptor_ptr(const ov::Output<const ov::Node>& out);

    static void clean(const std::shared_ptr<ov::Node>& node);

private:
    static void init_default(std::vector<PortDescriptorPtr>& in_descs, std::vector<PortDescriptorPtr>& out_descs, const std::shared_ptr<ov::Node>& node);
};

// PortDescriptorVectorAttribute is not copyable attribute!
// It's needed to avoid incorrect copies of rt info between different nodes in call copy_runtime_info() (for example, in transformations)
// The attribute must be manually copied if needed
class PortDescriptorVectorAttribute : public ov::RuntimeAttribute {
public:
    OPENVINO_RTTI("PortDescriptorVectorAttribute", "", ov::RuntimeAttribute);

    PortDescriptorVectorAttribute() = default;
    explicit PortDescriptorVectorAttribute(std::vector<PortDescriptorPtr> in_descs = {}, std::vector<PortDescriptorPtr> out_descs = {})
            : inputs(std::move(in_descs)), outputs(std::move(out_descs)) {}

    bool is_copyable() const override { return false; }

    std::vector<PortDescriptorPtr> inputs{};
    std::vector<PortDescriptorPtr> outputs{};
};

} // namespace lowered
} // namespace snippets
} // namespace ov
