//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <onnx/onnx_pb.h>

#include "model.hpp"
#include "ngraph/log.hpp"
#include "onnx_import/ops_bridge.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        std::string get_node_domain(const ONNX_NAMESPACE::NodeProto& node_proto)
        {
            return node_proto.has_domain() ? node_proto.domain() : "";
        }

        std::int64_t get_opset_version(const ONNX_NAMESPACE::ModelProto& model_proto,
                                       const std::string& domain)
        {
            for (const auto& opset_import : model_proto.opset_import())
            {
                if (domain == opset_import.domain())
                {
                    return opset_import.version();
                }
            }

            throw ngraph_error("Couldn't find operator set's version for domain: " + domain + ".");
        }

        Model::Model(const ONNX_NAMESPACE::ModelProto& model_proto)
            : m_model_proto{&model_proto}
        {
            // Walk through the elements of opset_import field and register operator sets
            // for each domain. An exception UnknownDomain() will raise if the domain is
            // unknown or invalid.
            for (const auto& id : m_model_proto->opset_import())
            {
                auto domain = id.has_domain() ? id.domain() : "";
                m_opset.emplace(domain, OperatorsBridge::get_operator_set(domain, id.version()));
            }
            // onnx.proto(.3): the empty string ("") for domain or absence of opset_import field
            // implies the operator set that is defined as part of the ONNX specification.
            const auto dm = m_opset.find("");
            if (dm == std::end(m_opset))
            {
                m_opset.emplace("", OperatorsBridge::get_operator_set("", ONNX_OPSET_VERSION));
            }
        }

        const Operator& Model::get_operator(const std::string& name,
                                            const std::string& domain) const
        {
            const auto dm = m_opset.find(domain);
            if (dm == std::end(m_opset))
            {
                throw error::UnknownDomain{domain};
            }
            const auto op = dm->second.find(name);
            if (op == std::end(dm->second))
            {
                throw error::UnknownOperator{name, domain};
            }
            return op->second;
        }

        bool Model::is_operator_available(const ONNX_NAMESPACE::NodeProto& node_proto) const
        {
            const auto dm = m_opset.find(get_node_domain(node_proto));
            if (dm == std::end(m_opset))
            {
                return false;
            }
            const auto op = dm->second.find(node_proto.op_type());
            return (op != std::end(dm->second));
        }

        void Model::enable_opset_domain(const std::string& domain)
        {
            // There is no need to 'update' already enabled domain.
            // Since this function may be called only during model import,
            // (maybe multiple times) the registered domain opset won't differ
            // between subsequent calls.
            if (m_opset.find(domain) == std::end(m_opset))
            {
                OperatorSet opset{OperatorsBridge::get_operator_set(domain)};
                if (opset.empty())
                {
                    NGRAPH_WARN << "Couldn't enable domain: " << domain
                                << " since it hasn't any registered operators.";

                    return;
                }
                m_opset.emplace(domain, opset);
            }
        }

    } // namespace onnx_import

} // namespace ngraph
