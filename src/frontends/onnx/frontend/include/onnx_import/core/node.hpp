// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <string>

#include "ngraph/except.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/constant.hpp"
#include "onnx_import/onnx_importer_visibility.hpp"

namespace ONNX_NAMESPACE {
// forward declaration
class NodeProto;
}  // namespace ONNX_NAMESPACE

namespace ngraph {
namespace onnx_import {
namespace error {
namespace node {
struct UnknownAttribute : ngraph_error {
    explicit UnknownAttribute(const std::string& node, const std::string& name)
        : ngraph_error{"Node (" + node + "): unknown attribute \'" + name + "\'"} {}
};

}  // namespace node

}  // namespace error

// forward declaration
class Graph;
class Subgraph;
class Tensor;
class SparseTensor;
class Attribute;

class ONNX_IMPORTER_API Node {
public:
    Node() = delete;
    // TODO: hide this ctor since it uses protobufs generated structures
    Node(const ONNX_NAMESPACE::NodeProto& node_proto, const Graph& graph);

    Node(Node&&) noexcept;
    Node(const Node&);

    Node& operator=(Node&&) noexcept = delete;
    Node& operator=(const Node&) = delete;

    OutputVector get_ng_inputs() const;
    const std::string& domain() const;
    const std::string& op_type() const;
    const std::string& get_name() const;
    std::vector<std::string> get_attribute_names() const;
    const Attribute& get_attribute(const std::string& name) const;

    /// \brief Describe the ONNX Node to make debugging graphs easier
    /// Function will return the Node's name if it has one, or the names of its outputs.
    /// \return Description of Node
    const std::string& get_description() const;

    const std::string& input(int index) const;
    std::size_t get_inputs_size() const;

    const std::vector<std::reference_wrapper<const std::string>>& get_output_names() const;
    const std::string& output(int index) const;
    std::size_t get_outputs_size() const;

    bool has_attribute(const std::string& name) const;

    bool has_subgraphs() const;
    const std::unordered_map<std::string, std::shared_ptr<Subgraph>>& get_subgraphs() const;

    template <typename T>
    T get_attribute_value(const std::string& name, T default_value) const;

    template <typename T>
    T get_attribute_value(const std::string& name) const;

    template <typename T>
    std::shared_ptr<ov::op::v0::Constant> get_attribute_as_constant(const std::string& name) const;

    template <typename T>
    std::shared_ptr<ov::op::v0::Constant> get_attribute_as_constant(const std::string& name, element::Type type) const;

    template <typename T>
    std::shared_ptr<ov::op::v0::Constant> get_attribute_as_constant(const std::string& name, T default_value) const;

    template <typename T>
    std::shared_ptr<ov::op::v0::Constant> get_attribute_as_constant(const std::string& name,
                                                                    T default_value,
                                                                    element::Type type) const;

private:
    class Impl;
    // In this case we need custom deleter, because Impl is an incomplete
    // type. Node's are elements of std::vector. Without custom deleter
    // compilation fails; the compiler is unable to parameterize an allocator's
    // default deleter due to incomple type.
    std::unique_ptr<Impl, void (*)(Impl*)> m_pimpl;
};

template <>
ONNX_IMPORTER_API float Node::get_attribute_value(const std::string& name, float default_value) const;

template <>
ONNX_IMPORTER_API double Node::get_attribute_value(const std::string& name, double default_value) const;

template <>
ONNX_IMPORTER_API std::int64_t Node::get_attribute_value(const std::string& name, std::int64_t default_value) const;

template <>
ONNX_IMPORTER_API std::string Node::get_attribute_value(const std::string& name, std::string default_value) const;

template <>
ONNX_IMPORTER_API Tensor Node::get_attribute_value(const std::string& name, Tensor default_value) const;

template <>
ONNX_IMPORTER_API SparseTensor Node::get_attribute_value(const std::string& name, SparseTensor default_value) const;

template <>
ONNX_IMPORTER_API Graph Node::get_attribute_value(const std::string& name, Graph default_value) const;

template <>
ONNX_IMPORTER_API std::vector<float> Node::get_attribute_value(const std::string& name,
                                                               std::vector<float> default_value) const;

template <>
ONNX_IMPORTER_API std::vector<double> Node::get_attribute_value(const std::string& name,
                                                                std::vector<double> default_value) const;

template <>
ONNX_IMPORTER_API std::vector<std::int64_t> Node::get_attribute_value(const std::string& name,
                                                                      std::vector<std::int64_t> default_value) const;

template <>
ONNX_IMPORTER_API std::vector<std::size_t> Node::get_attribute_value(const std::string& name,
                                                                     std::vector<std::size_t> default_value) const;

template <>
ONNX_IMPORTER_API std::vector<std::string> Node::get_attribute_value(const std::string& name,
                                                                     std::vector<std::string> default_value) const;

template <>
ONNX_IMPORTER_API std::vector<Tensor> Node::get_attribute_value(const std::string& name,
                                                                std::vector<Tensor> default_value) const;

template <>
ONNX_IMPORTER_API std::vector<SparseTensor> Node::get_attribute_value(const std::string& name,
                                                                      std::vector<SparseTensor> default_value) const;

template <>
ONNX_IMPORTER_API std::vector<Graph> Node::get_attribute_value(const std::string& name,
                                                               std::vector<Graph> default_value) const;

template <>
ONNX_IMPORTER_API float Node::get_attribute_value(const std::string& name) const;

template <>
ONNX_IMPORTER_API double Node::get_attribute_value(const std::string& name) const;

template <>
ONNX_IMPORTER_API std::int64_t Node::get_attribute_value(const std::string& name) const;

template <>
ONNX_IMPORTER_API std::size_t Node::get_attribute_value(const std::string& name) const;

template <>
ONNX_IMPORTER_API std::string Node::get_attribute_value(const std::string& name) const;

template <>
ONNX_IMPORTER_API Tensor Node::get_attribute_value(const std::string& name) const;

template <>
ONNX_IMPORTER_API SparseTensor Node::get_attribute_value(const std::string& name) const;

template <>
ONNX_IMPORTER_API Subgraph Node::get_attribute_value(const std::string& name) const;

template <>
ONNX_IMPORTER_API std::vector<float> Node::get_attribute_value(const std::string& name) const;

template <>
ONNX_IMPORTER_API std::vector<double> Node::get_attribute_value(const std::string& name) const;

template <>
ONNX_IMPORTER_API std::vector<std::int64_t> Node::get_attribute_value(const std::string& name) const;

template <>
ONNX_IMPORTER_API std::vector<std::size_t> Node::get_attribute_value(const std::string& name) const;

template <>
ONNX_IMPORTER_API std::vector<std::string> Node::get_attribute_value(const std::string& name) const;

template <>
ONNX_IMPORTER_API std::vector<Tensor> Node::get_attribute_value(const std::string& name) const;

template <>
ONNX_IMPORTER_API std::vector<SparseTensor> Node::get_attribute_value(const std::string& name) const;

template <>
ONNX_IMPORTER_API std::vector<Graph> Node::get_attribute_value(const std::string& name) const;

template <>
ONNX_IMPORTER_API std::shared_ptr<ov::op::v0::Constant> Node::get_attribute_as_constant<std::vector<int64_t>>(
    const std::string& name) const;

template <>
ONNX_IMPORTER_API std::shared_ptr<ov::op::v0::Constant> Node::get_attribute_as_constant<std::vector<int64_t>>(
    const std::string& name,
    element::Type type) const;

template <>
ONNX_IMPORTER_API std::shared_ptr<ov::op::v0::Constant> Node::get_attribute_as_constant(
    const std::string& name,
    std::vector<int64_t> default_value) const;

template <>
ONNX_IMPORTER_API std::shared_ptr<ov::op::v0::Constant>
Node::get_attribute_as_constant(const std::string& name, std::vector<int64_t> default_value, element::Type type) const;

template <>
ONNX_IMPORTER_API std::shared_ptr<ov::op::v0::Constant> Node::get_attribute_as_constant<float>(
    const std::string& name) const;

template <>
ONNX_IMPORTER_API std::shared_ptr<ov::op::v0::Constant> Node::get_attribute_as_constant(const std::string& name,
                                                                                        float default_value) const;

template <>
ONNX_IMPORTER_API std::shared_ptr<ov::op::v0::Constant> Node::get_attribute_as_constant(const std::string& name,
                                                                                        float default_value,
                                                                                        element::Type type) const;

template <>
ONNX_IMPORTER_API std::shared_ptr<ov::op::v0::Constant> Node::get_attribute_as_constant<float>(
    const std::string& name,
    element::Type type) const;

template <>
ONNX_IMPORTER_API std::shared_ptr<ov::op::v0::Constant> Node::get_attribute_as_constant<double>(
    const std::string& name) const;

template <>
ONNX_IMPORTER_API std::shared_ptr<ov::op::v0::Constant> Node::get_attribute_as_constant(const std::string& name,
                                                                                        double default_value) const;

template <>
ONNX_IMPORTER_API std::shared_ptr<ov::op::v0::Constant> Node::get_attribute_as_constant<double>(
    const std::string& name,
    element::Type type) const;

template <>
ONNX_IMPORTER_API std::shared_ptr<ov::op::v0::Constant> Node::get_attribute_as_constant(const std::string& name,
                                                                                        double default_value,
                                                                                        element::Type type) const;

template <>
ONNX_IMPORTER_API std::shared_ptr<ov::op::v0::Constant> Node::get_attribute_as_constant<int64_t>(
    const std::string& name) const;

template <>
ONNX_IMPORTER_API std::shared_ptr<ov::op::v0::Constant> Node::get_attribute_as_constant<int64_t>(
    const std::string& name,
    element::Type type) const;

template <>
ONNX_IMPORTER_API std::shared_ptr<ov::op::v0::Constant> Node::get_attribute_as_constant(const std::string& name,
                                                                                        int64_t default_value) const;

template <>
ONNX_IMPORTER_API std::shared_ptr<ov::op::v0::Constant> Node::get_attribute_as_constant<int64_t>(
    const std::string& name,
    element::Type type) const;

template <>
ONNX_IMPORTER_API std::shared_ptr<ov::op::v0::Constant> Node::get_attribute_as_constant(const std::string& name,
                                                                                        int64_t default_value,
                                                                                        element::Type type) const;

inline std::ostream& operator<<(std::ostream& outs, const Node& node) {
    return (outs << "<Node(" << node.op_type() << "): " << node.get_description() << ">");
}

}  // namespace onnx_import

}  // namespace ngraph
