// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <initializer_list>
#include <list>
#include <memory>
#include <string>
#include <vector>

#include "ngraph/ngraph_visibility.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/assign.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/read_value.hpp"
#include "ngraph/op/result.hpp"
#include "ngraph/op/sink.hpp"
#include "ngraph/op/util/variable.hpp"

namespace ngraph
{
    /// A user-defined function.
    class NGRAPH_API Function
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

        Function(const ResultVector& results,
                 const SinkVector& sinks,
                 const ParameterVector& parameters,
                 const std::string& name = "");

        Function(const OutputVector& results,
                 const SinkVector& sinks,
                 const ParameterVector& parameters,
                 const std::string& name = "");

        Function(const ResultVector& results,
                 const SinkVector& sinks,
                 const ParameterVector& parameters,
                 const VariableVector& variables,
                 const std::string& name = "");

        Function(const OutputVector& results,
                 const SinkVector& sinks,
                 const ParameterVector& parameters,
                 const VariableVector& variables,
                 const std::string& name = "");

        Function(const ResultVector& results,
                 const ParameterVector& parameters,
                 const VariableVector& variables,
                 const std::string& name = "");

        Function(const OutputVector& results,
                 const ParameterVector& parameters,
                 const VariableVector& variables,
                 const std::string& name = "");

        /// Constructs a Function. Lists of parameters and variables will be generated automatically
        /// based on traversing the graph from the results.
        explicit Function(const OutputVector& results, const std::string& name = "");

        /// Constructs a Function. Lists of parameters and variables will be generated automatically
        /// based on traversing the graph from the results and the sinks.
        Function(const OutputVector& results,
                 const SinkVector& sinks,
                 const std::string& name = "");

        virtual ~Function() = default;
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
        // updates graph and m_results list
        void replace_node(std::shared_ptr<Node> old, std::shared_ptr<Node> repl);

        void validate_nodes_and_infer_types() const;

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

        virtual bool visit_attributes(AttributeVisitor& visitor);

        /// Return the function parameters
        const ParameterVector& get_parameters() const { return m_parameters; };
        /// Return a list of function's outputs
        const ResultVector& get_results() const { return m_results; };
        /// Index for parameter, or -1
        int64_t get_parameter_index(const std::shared_ptr<op::Parameter>& parameter) const;

        /// Index for value or result referencing it, or -1
        int64_t get_result_index(const Output<Node>& value) const;

        /// \brief Evaluate the function on inputs, putting results in outputs.
        /// \param output_tensors Tensors for the outputs to compute. One for each result
        /// \param input_tensors Tensors for the inputs. One for each inputs.
        /// \param evaluation_context Storage of additional settings and attributes that can be used
        /// when evaluating the function. This additional information can be shared across nodes.
        bool evaluate(const HostTensorVector& output_tensors,
                      const HostTensorVector& input_tensors,
                      EvaluationContext evaluation_context = EvaluationContext()) const;

        /// \brief Return a list of function's sinks.
        const SinkVector& get_sinks() const { return m_sinks; }
        /// \brief Add new sink nodes to the list. Method doesn't validate graph, it should be done
        /// manually after all changes.
        /// \param sinks new sink nodes
        void add_sinks(const SinkVector& sinks);

        /// \brief Delete sink node from the list of sinks. Method doesn't delete node from graph.
        /// \param sink Sink to delete
        void remove_sink(const std::shared_ptr<op::Sink>& sink);

        /// \brief Add new Result nodes to the list. Method doesn't validate graph, it should be
        /// done manually after all changes.
        /// \param results new Result nodes
        void add_results(const ResultVector& results);

        /// \brief Delete Result node from the list of results. Method will not delete node from
        /// graph.
        /// \param result Result node to delete
        void remove_result(const std::shared_ptr<op::Result>& result);

        /// \brief Add new Parameter nodes to the list.
        ///
        /// Method doesn't change or validate graph, it should be done manually.
        /// For example, if you want to replace `ReadValue` node by `Parameter`, you should do the
        /// following steps:
        /// * replace node `ReadValue` by `Parameter` in graph
        /// * call add_parameter() to add new input to the list
        /// * call graph validation to check correctness of changes
        ///
        /// \param params new Parameter nodes
        void add_parameters(const ParameterVector& params);

        /// \brief Delete Parameter node from the list of parameters. Method will not delete node
        /// from graph. You need to replace Parameter with other operation manually.
        /// Attention: Indexing of parameters can be changed.
        ///
        /// Possible use of method is to replace input by variable. For it the following steps
        /// should be done:
        /// * `Parameter` node should be replaced by `ReadValue`
        /// * call remove_parameter(param) to remove input from the list
        /// * check if any parameter indexes are saved/used somewhere, update it for all inputs
        /// because indexes can be changed
        /// * call graph validation to check all changes
        ///
        /// \param param Parameter node to delete
        void remove_parameter(const std::shared_ptr<op::Parameter>& param);

        /// \brief Add new variables to the list. Method doesn't validate graph, it should be done
        /// manually after all changes.
        /// \param variables new variables to add
        void add_variables(const VariableVector& variables);

        /// \brief Delete variable from the list of variables.
        /// Method doesn't delete nodes that used this variable from the graph.
        /// \param variable Variable to delete
        void remove_variable(const VariablePtr& variable);

        /// \brief Return a list of function's variables.
        const VariableVector& get_variables() const { return m_variables; }

        /// \brief Return a variable by specified variable_id.
        VariablePtr get_variable_by_id(const std::string& variable_id) const;

    private:
        Function(const Function&) = delete;
        Function(const Function&&) = delete;
        Function& operator=(const Function&) = delete;

        /// \brief Depending on the options selected,
        /// checks all the Parameter/Variables are registered in the list of Function
        /// parameters/variables or finds all Parameters/Variables in a function and registers them.
        /// \param detect_variables If this flag is true, then it finds all Variables in a function
        /// and registers them, otherwise checks all the Variables are registered.
        /// \param detect_parameters If this flag is true, then it finds all Parameters in a
        /// function and registers them, otherwise checks all the Parameters are registered.
        void prerequirements(bool detect_variables, bool detect_parameters);

        static std::atomic<size_t> m_next_instance_id;
        std::string m_name;
        const std::string m_unique_name;
        size_t m_placement{0};
        topological_sort_t m_topological_sorter;

        ResultVector m_results;
        // List of the nodes with side effect in graph.
        // These nodes are not outputs of graph but should not be removed even if have no children.
        SinkVector m_sinks;
        ParameterVector m_parameters;
        VariableVector m_variables;
    };

    template <>
    class NGRAPH_API AttributeAdapter<std::shared_ptr<Function>>
        : public DirectValueAccessor<std::shared_ptr<Function>>
    {
    public:
        AttributeAdapter(std::shared_ptr<Function>& value)
            : DirectValueAccessor<std::shared_ptr<Function>>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<std::shared_ptr<Function>>",
                                                    0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    };
} // namespace ngraph
