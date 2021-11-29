// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <onnx/onnx_pb.h>
#include <ostream>
#include <string>
#include <unordered_map>

#include "onnx_import/core/operator_set.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        /// \brief      Type of container which stores opset version and domain in ONNX format
        using OpsetImports =
            ::google::protobuf::RepeatedPtrField<ONNX_NAMESPACE::OperatorSetIdProto>;

        std::string get_node_domain(const ONNX_NAMESPACE::NodeProto& node_proto);

        std::int64_t get_opset_version(const ONNX_NAMESPACE::ModelProto& model_proto,
                                       const std::string& domain);

        class Model
        {
        public:
            Model() = delete;
            explicit Model(std::unique_ptr<ONNX_NAMESPACE::ModelProto>&& model_proto);

            Model(const Model&) = delete;
            Model(Model&&) = delete;

            Model& operator=(const Model&) = delete;
            Model& operator=(Model&&) = delete;

            const std::string& get_producer_name() const { return m_model_proto->producer_name(); }
            const ONNX_NAMESPACE::GraphProto& get_graph() const { return m_model_proto->graph(); }
            std::int64_t get_model_version() const { return m_model_proto->model_version(); }
            const OpsetImports& get_opset_imports() const;
            const std::string& get_producer_version() const
            {
                return m_model_proto->producer_version();
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
            /// \return `true` if the operator is available, otherwise it returns `false`.
            bool is_operator_available(const ONNX_NAMESPACE::NodeProto& node_proto) const;

            /// \brief      Enable operators from provided domain to use by this model.
            ///
            /// \note       This function makes visible all currently registered in provided domain
            ///             operators for use in this model.
            ///
            /// \param[in]  domain  The domain name.
            ///
            void enable_opset_domain(const std::string& domain);

        private:
            const std::unique_ptr<ONNX_NAMESPACE::ModelProto> m_model_proto;
            std::unordered_map<std::string, OperatorSet> m_opset;
        };

        inline std::ostream& operator<<(std::ostream& outs, const Model& model)
        {
            return (outs << "<Model: " << model.get_producer_name() << ">");
        }

    } // namespace onnx_import

} // namespace ngraph
