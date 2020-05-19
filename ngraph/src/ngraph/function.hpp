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

#include <atomic>
#include <initializer_list>
#include <list>
#include <memory>
#include <string>
#include <vector>

#include "ngraph/lambda.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/result.hpp"

namespace ngraph
{
    /// A user-defined function.
    class NGRAPH_API Function : public Lambda
    {
    public:
        static constexpr DiscreteTypeInfo type_info{"Function", 0};
        const DiscreteTypeInfo& get_type_info() const { return type_info; }
        Function(const NodeVector& results,
                 const ParameterVector& parameters,
                 const std::string& name = "");

        Function(const OutputVector& results,
                 const ParameterVector& parameters,
                 const std::string& name = "");

        Function(const std::shared_ptr<Node>& result,
                 const ParameterVector& parameters,
                 const std::string& name = "");

        Function(const ResultVector& results,
                 const ParameterVector& parameters,
                 const std::string& name = "");

        void init();

        virtual ~Function() {}
    public:
        /// Return the number of outputs for this function.
        size_t get_output_size() const;

        /// Return the op that generates output i
        std::shared_ptr<Node> get_output_op(size_t i) const;

        Output<Node> output(size_t i) const;

        /// Return the element type of output i
        const element::Type& get_output_element_type(size_t i) const;

        /// Return the shape of element i
        const Shape& get_output_shape(size_t i) const;

        /// Return the partial shape of element i
        const PartialShape& get_output_partial_shape(size_t i) const;

        /// Check that there is a single result and return it.
        std::shared_ptr<Node> get_result() const;

        /// \brief Get the unique name of the function.
        /// \returns A const reference to the function's unique name.
        const std::string& get_name() const;

        /// \brief Sets a friendly name for a function. This does not overwrite the unique name
        ///        of the function and is retrieved via get_friendly_name(). Used mainly for
        ///        debugging.
        ///        The friendly name may be set exactly once.
        /// \param name is the friendly name to set
        void set_friendly_name(const std::string& name);

        /// \brief Gets the friendly name for a function. If no friendly name has been set via
        ///        set_friendly_name then the function's unique name is returned.
        /// \returns A const reference to the function's friendly name.
        const std::string& get_friendly_name() const;

        std::vector<std::shared_ptr<Node>> get_ops() const;
        std::vector<std::shared_ptr<Node>> get_ordered_ops() const;
        void map_unordered_ops(std::function<void(Node*)> f) const;

        friend std::ostream& operator<<(std::ostream&, const Function&);
        size_t get_instance_id() { return m_instance_id; }
        size_t get_temporary_pool_size();
        void set_temporary_pool_size(size_t);
        // updates graph and m_results list
        void replace_node(std::shared_ptr<Node> old, std::shared_ptr<Node> repl);

        void validate_nodes_and_infer_types();

        /// \brief Returns the sum of the size of all nodes in the graph plus the size of
        /// all constant data. This has little value beyond comparing the relative size of
        /// graphs and should not be considered the actual memory consumption of a graph.
        size_t get_graph_size() const;

        /// \brief Returns true if any of the op's defined in the function contains partial shape
        bool is_dynamic() const;

        /// \brief Replace the `parameter_index`th parameter of the function with `parameter`.
        ///
        /// All users of the `parameter_index`th parameter are redirected to `parameter`, and the
        /// `parameter_index`th entry in the function parameter list is replaced with `parameter`.
        ///
        /// \param parameter_index The index of the parameter to replace.
        /// \param parameter The parameter to substitute for the `parameter_index`th parameter.
        void replace_parameter(size_t parameter_index,
                               const std::shared_ptr<op::Parameter>& parameter);

        using topological_sort_t = std::function<std::vector<std::shared_ptr<Node>>(
            const std::vector<std::shared_ptr<Node>>& root_nodes)>;
        void set_topological_sort(topological_sort_t);

    protected:
        size_t m_temporary_pool_size;

    private:
        Function(const Function&) = delete;
        Function(const Function&&) = delete;
        Function& operator=(const Function&) = delete;

        static std::atomic<size_t> m_next_instance_id;
        size_t m_instance_id;
        std::string m_name;
        const std::string m_unique_name;
        size_t m_placement{0};
        topological_sort_t m_topological_sorter;
    };
}
