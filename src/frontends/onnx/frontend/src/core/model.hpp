// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <onnx/onnx_pb.h>

#include <ostream>
#include <string>
#include <unordered_map>

#include "onnx_import/core/operator_set.hpp"

namespace ngraph {
namespace onnx_import {
/// \brief      Type of container which stores opset version and domain in ONNX format
using OpsetImports = ::google::protobuf::RepeatedPtrField<ONNX_NAMESPACE::OperatorSetIdProto>;

std::string get_node_domain(const ONNX_NAMESPACE::NodeProto& node_proto);

std::int64_t get_opset_version(const ONNX_NAMESPACE::ModelProto& model_proto, const std::string& domain);

class OperatorsBridge;

class Model {
public:
    // a container with OperatorSets covering all domains used in a given model
    // built based on the opset imports in the ModelProto object
    using ModelOpSet = std::unordered_map<std::string, OperatorSet>;

    explicit Model(std::shared_ptr<ONNX_NAMESPACE::ModelProto> model_proto, ModelOpSet&& model_opset);

    Model(const Model&) = delete;
    Model(Model&&) = delete;

    Model& operator=(const Model&) = delete;
    Model& operator=(Model&&) = delete;

    const std::string& get_producer_name() const {
        return m_model_proto->producer_name();
    }
    const ONNX_NAMESPACE::GraphProto& get_graph() const {
        return m_model_proto->graph();
    }
    std::int64_t get_model_version() const {
        return m_model_proto->model_version();
    }
    const OpsetImports& get_opset_imports() const {
        return m_model_proto->opset_import();
    }

    const std::string& get_producer_version() const {
        return m_model_proto->producer_version();
    }

    std::map<std::string, std::string> get_metadata() const {
        std::map<std::string, std::string> metadata;

        const auto& model_metadata = m_model_proto->metadata_props();
        for (const auto& prop : model_metadata) {
            metadata.emplace(prop.key(), prop.value());
        }
        return metadata;
    }

    /// \brief Access an operator object by its type name and domain name
    /// The function will return the operator object if it exists, or report an error
    /// in case of domain or operator absence.
    /// \param name       type name of the operator object,
    /// \param domain     domain name of the operator object.
    /// \return Reference to the operator object.
    /// \throw error::UnknownDomain    there is no operator set defined for the given
    ///                                domain,
    /// \throw error::UnknownOperator  the given operator type name does not exist in
    ///                                operator set.
    const Operator& get_operator(const std::string& name, const std::string& domain) const;

    /// \brief Check availability of operator base on NodeProto.
    /// \param name       Type name of the operator.
    /// \param domain     Domain name of the operator.
    /// \return `true` if the operator is available, otherwise it returns `false`.
    bool is_operator_available(const std::string& name, const std::string& domain) const;

    /// \brief      Enable operators from provided domain to use by this model.
    ///
    /// \note       This function makes visible all currently registered in provided domain
    ///             operators for use in this model.
    ///
    /// \param[in]  domain  The domain name.
    ///
    void enable_opset_domain(const std::string& domain, const OperatorsBridge& ops_bridge);

private:
    const std::shared_ptr<ONNX_NAMESPACE::ModelProto> m_model_proto;
    ModelOpSet m_opset;
};

inline std::ostream& operator<<(std::ostream& outs, const Model& model) {
    return (outs << "<Model: " << model.get_producer_name() << ">");
}

}  // namespace onnx_import

}  // namespace ngraph
