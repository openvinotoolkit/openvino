/*******************************************************************************
 * Copyright 2017-2020 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/
#pragma once

#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>
#include <tensorflow_frontend/place.hpp>
#include <tensorflow_frontend/utility.hpp>

namespace ngraph {
namespace frontend {
namespace tensorflow {

namespace detail {

class TFNodeDecoder;

/// Generic NodeContext that hides graph representation
/// It is base class for specific implementations for protobuf and run-time graph
class NodeContext {
    OutputVector m_ng_inputs;
    std::shared_ptr<detail::TFNodeDecoder> m_decoder;

    // If shape is overridden for a particular node, it exists in the following map
    std::map<std::string, ngraph::PartialShape> m_overridden_shapes;

    // For special kind inputs (args) there are shapes defined externally here:
    const std::vector<ngraph::PartialShape>& m_indexed_shapes;

public:
    NodeContext(const OutputVector& _ng_inputs,
                std::shared_ptr<detail::TFNodeDecoder> _decoder,
                const std::map<std::string, ngraph::PartialShape>& overridden_shapes,
                const std::vector<ngraph::PartialShape>& indexed_shapes = {});

    NodeContext(const OutputVector& _ng_inputs,
                std::shared_ptr<detail::TFNodeDecoder> _decoder,
                const std::vector<Place::Ptr>& _inputs);

    size_t get_ng_input_size() const;

    /// Returns a vector of already converted inputs for this node
    const OutputVector& get_ng_inputs() const;

    Output<Node> get_ng_input(size_t input_port) const;

    virtual std::string get_op_type() const;

    virtual std::vector<std::string> get_output_names() const;

    virtual std::vector<std::string> get_names() const;

    virtual std::string get_name() const;

    /// Temporary method for the transition period during migration to NodeContext
    // TODO: Remove this method and port all dependent code to the remaining methods
    const detail::TFNodeDecoder* _get_decoder() const;

    template <typename T>
    T get_attribute(const std::string& name) const;

    template <typename T>
    T get_attribute(const std::string& name, const T& default_value) const;

    // Meta-attributes like op type, domain, version -- some FW specific but common for all operations properties

    template <typename T>
    T get_meta_attribute(const std::string& name) const;

    template <typename T>
    T get_meta_attribute(const std::string& name, const T& default_value) const;

    const std::map<std::string, ngraph::PartialShape>& get_overridden_shapes() const;

    const std::vector<ngraph::PartialShape>& get_indexed_shapes() const;
};

}  // namespace detail
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ngraph
