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

#pragma once

#include <onnx/onnx_pb.h>
#include <memory>
#include <map>
#include <string>

#include "default_opset.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/constant.hpp"
#include "tensor.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        class GraphCache
        {
            public:
                GraphCache(const ONNX_NAMESPACE::GraphProto& graph_proto);

                void set_node(const std::string& name, std::shared_ptr<ngraph::Node>&& node);
                
                virtual std::shared_ptr<ngraph::Node> get_node(const std::string& name) const;
                virtual bool contains(const std::string& node_name) const;

                const std::map<std::string, Tensor>& initializers() const;

            protected:
                std::map<std::string, std::shared_ptr<ngraph::Node>> m_graph_cache_map;

            private:
                void add_provenance_tag_to_initializer(
                    const Tensor& initializer, std::shared_ptr<default_opset::Constant> node) const;

                std::map<std::string, Tensor> m_initializers;
        };

        class SubgraphCache : public GraphCache
        {
            public:
                SubgraphCache(const ONNX_NAMESPACE::GraphProto& graph_proto, const std::shared_ptr<GraphCache> parent_graph_cache);
                
                std::shared_ptr<ngraph::Node> get_node(const std::string& name) const override;
                bool contains(const std::string& node_name) const override;

            private:
                const std::shared_ptr<GraphCache> m_parent_graph_cache;
        };
        
    } // namespace onnx_import
} // namespace ngraph