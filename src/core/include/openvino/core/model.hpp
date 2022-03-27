// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <initializer_list>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "openvino/core/core_visibility.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/rtti.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/sink.hpp"
#include "openvino/op/util/variable.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
class Model;

OPENVINO_API
std::shared_ptr<Model> clone_model(const Model& func, std::unordered_map<Node*, std::shared_ptr<Node>>& node_map);

namespace frontend {
class FrontEnd;
}

class ModelAccessor;

/**
 * @brief A user-defined model
 * @ingroup ov_model_cpp_api
 */
class OPENVINO_API Model : public std::enable_shared_from_this<Model> {
    friend class frontend::FrontEnd;
    friend OPENVINO_API std::shared_ptr<Model> clone_model(const Model& func,
                                                           std::unordered_map<Node*, std::shared_ptr<Node>>& node_map);
    std::shared_ptr<void> m_shared_object;  // Frontend plugin shared object handle.

public:
    _OPENVINO_HIDDEN_METHOD static const ::ov::DiscreteTypeInfo& get_type_info_static() {
        static const ::ov::DiscreteTypeInfo type_info_static{"Model", static_cast<uint64_t>(0)};
        return type_info_static;
    }
    const ::ov::DiscreteTypeInfo& get_type_info() const {
        return get_type_info_static();
    }
    OPENVINO_DEPRECATED("This member was deprecated. Please use ::get_type_info_static() instead.")
    static const ov::DiscreteTypeInfo type_info;

    Model(const ov::NodeVector& results, const ov::ParameterVector& parameters, const std::string& name = "");

    Model(const ov::OutputVector& results, const ov::ParameterVector& parameters, const std::string& name = "");

    Model(const std::shared_ptr<ov::Node>& result, const ov::ParameterVector& parameters, const std::string& name = "");

    Model(const ov::ResultVector& results, const ov::ParameterVector& parameters, const std::string& name = "");

    Model(const ov::ResultVector& results,
          const ov::SinkVector& sinks,
          const ov::ParameterVector& parameters,
          const std::string& name = "");

    Model(const ov::OutputVector& results,
          const ov::SinkVector& sinks,
          const ov::ParameterVector& parameters,
          const std::string& name = "");

    Model(const ov::ResultVector& results,
          const ov::SinkVector& sinks,
          const ov::ParameterVector& parameters,
          const ov::op::util::VariableVector& variables,
          const std::string& name = "");

    Model(const ov::OutputVector& results,
          const ov::SinkVector& sinks,
          const ov::ParameterVector& parameters,
          const ov::op::util::VariableVector& variables,
          const std::string& name = "");

    Model(const ov::ResultVector& results,
          const ov::ParameterVector& parameters,
          const ov::op::util::VariableVector& variables,
          const std::string& name = "");

    Model(const ov::OutputVector& results,
          const ov::ParameterVector& parameters,
          const ov::op::util::VariableVector& variables,
          const std::string& name = "");

    /// Constructs a Model. Lists of parameters and variables will be generated automatically
    /// based on traversing the graph from the results.
    explicit Model(const ov::OutputVector& results, const std::string& name = "");

    /// Constructs a Model. Lists of parameters and variables will be generated automatically
    /// based on traversing the graph from the results and the sinks.
    Model(const ov::OutputVector& results, const ov::SinkVector& sinks, const std::string& name = "");

    virtual ~Model() = default;
    /// Return the number of outputs for this Model.
    size_t get_output_size() const;

    /// Return the op that generates output i
    std::shared_ptr<ov::Node> get_output_op(size_t i) const;

    /// Model outputs
    std::vector<ov::Output<ov::Node>> outputs();
    ov::Output<ov::Node> output();
    ov::Output<ov::Node> output(size_t i);
    ov::Output<ov::Node> output(const std::string& tensor_name);
    std::vector<ov::Output<const ov::Node>> outputs() const;
    ov::Output<const ov::Node> output() const;
    ov::Output<const ov::Node> output(size_t i) const;
    ov::Output<const ov::Node> output(const std::string& tensor_name) const;
    /// Model inputs
    std::vector<ov::Output<ov::Node>> inputs();
    ov::Output<ov::Node> input();
    ov::Output<ov::Node> input(size_t i);
    ov::Output<ov::Node> input(const std::string& tensor_name);
    std::vector<ov::Output<const ov::Node>> inputs() const;
    ov::Output<const ov::Node> input() const;
    ov::Output<const ov::Node> input(size_t i) const;
    ov::Output<const ov::Node> input(const std::string& tensor_name) const;

    ov::Output<ov::Node> add_output(const std::string& tensor_name);
    ov::Output<ov::Node> add_output(const std::string& op_name, size_t output_idx);
    ov::Output<ov::Node> add_output(const ov::Output<ov::Node>& port);

    void reshape(const ov::PartialShape& partial_shape);
    void reshape(const std::map<size_t, ov::PartialShape>& partial_shapes);
    void reshape(const std::map<std::string, ov::PartialShape>& partial_shapes);
    void reshape(const std::map<ov::Output<ov::Node>, ov::PartialShape>& partial_shapes);

    /// Return the element type of output i
    const ov::element::Type& get_output_element_type(size_t i) const;

    /// Return the shape of element i
    const Shape& get_output_shape(size_t i) const;

    /// Return the partial shape of element i
    const PartialShape& get_output_partial_shape(size_t i) const;

    /// Check that there is a single result and return it.
    std::shared_ptr<ov::Node> get_result() const;

    /// \brief Get the unique name of the model.
    /// \returns A const reference to the model's unique name.
    const std::string& get_name() const;

    /// \brief Sets a friendly name for a model. This does not overwrite the unique name
    ///        of the model and is retrieved via get_friendly_name(). Used mainly for
    ///        debugging.
    /// \param name is the friendly name to set
    void set_friendly_name(const std::string& name);

    /// \brief Gets the friendly name for a model. If no friendly name has been set via
    ///        set_friendly_name then the model's unique name is returned.
    /// \returns A const reference to the model's friendly name.
    const std::string& get_friendly_name() const;

    std::vector<std::shared_ptr<ov::Node>> get_ops() const;
    std::vector<std::shared_ptr<ov::Node>> get_ordered_ops() const;
    void map_unordered_ops(std::function<void(ov::Node*)> f) const;

    // updates graph and m_results list
    void replace_node(std::shared_ptr<ov::Node> old, std::shared_ptr<ov::Node> repl);

    void validate_nodes_and_infer_types() const;

    /// \brief Returns the sum of the size of all nodes in the graph plus the size of
    /// all constant data. This has little value beyond comparing the relative size of
    /// graphs and should not be considered the actual memory consumption of a graph.
    size_t get_graph_size() const;

    /// \brief Returns true if any of the op's defined in the model contains partial shape
    bool is_dynamic() const;

    /// \brief Replace the `parameter_index`th parameter of the model with `parameter`.
    ///
    /// All users of the `parameter_index`th parameter are redirected to `parameter`, and the
    /// `parameter_index`th entry in the model parameter list is replaced with `parameter`.
    ///
    /// \param parameter_index The index of the parameter to replace.
    /// \param parameter The parameter to substitute for the `parameter_index`th parameter.
    void replace_parameter(size_t parameter_index, const std::shared_ptr<ov::op::v0::Parameter>& parameter);

    using topological_sort_t =
        std::function<std::vector<std::shared_ptr<ov::Node>>(const std::vector<std::shared_ptr<ov::Node>>& root_nodes)>;
    void set_topological_sort(topological_sort_t);

    virtual bool visit_attributes(ov::AttributeVisitor& visitor);

    /// Return the model parameters
    const ov::ParameterVector& get_parameters() const {
        return m_parameters;
    };
    /// Return a list of model's outputs
    const ov::ResultVector& get_results() const {
        return m_results;
    };
    /// Index for parameter, or -1
    int64_t get_parameter_index(const std::shared_ptr<ov::op::v0::Parameter>& parameter) const;

    /// \brief Return the index of this model's Result represented by the "value" Output object.
    /// This method returns -1 if an the passed output is not related to the Results of a model.
    /// \param value Output containing Node
    int64_t get_result_index(const ov::Output<ov::Node>& value) const;

    /// \brief Return the index of this model's Result represented by the "value" Output object.
    /// This method returns -1 if an the passed output is not related to the Results of a model.
    /// \param value Output containing Node
    int64_t get_result_index(const ov::Output<const ov::Node>& value) const;

    /// \deprecated Use evaluate with ov::Tensor instead
    /// \brief Evaluate the model on inputs, putting results in outputs.
    /// \param output_tensors Tensors for the outputs to compute. One for each result
    /// \param input_tensors Tensors for the inputs. One for each inputs.
    /// \param evaluation_context Storage of additional settings and attributes that can be used
    /// when evaluating the model. This additional information can be shared across nodes.
    OPENVINO_DEPRECATED(
        "This method is deprecated and will be removed soon. Please use evaluate with ov::Tensor instead.")
    bool evaluate(const ov::HostTensorVector& output_tensors,
                  const ov::HostTensorVector& input_tensors,
                  ov::EvaluationContext evaluation_context = ov::EvaluationContext()) const;

    /// \brief Evaluate the model on inputs, putting results in outputs.
    /// \param output_tensors Tensors for the outputs to compute. One for each result
    /// \param input_tensors Tensors for the inputs. One for each inputs.
    /// \param evaluation_context Storage of additional settings and attributes that can be used
    /// when evaluating the model. This additional information can be shared across nodes.
    bool evaluate(ov::TensorVector& output_tensors,
                  const ov::TensorVector& input_tensors,
                  ov::EvaluationContext evaluation_context = ov::EvaluationContext()) const;

    /// \brief Return a list of model's sinks.
    const ov::SinkVector& get_sinks() const {
        return m_sinks;
    }
    /// \brief Add new sink nodes to the list. Method doesn't validate graph, it should be done
    /// manually after all changes.
    /// \param sinks new sink nodes
    void add_sinks(const ov::SinkVector& sinks);

    /// \brief Delete sink node from the list of sinks. Method doesn't delete node from graph.
    /// \param sink Sink to delete
    void remove_sink(const std::shared_ptr<ov::op::Sink>& sink);

    /// \brief Add new Result nodes to the list. Method doesn't validate graph, it should be
    /// done manually after all changes.
    /// \param results new Result nodes
    void add_results(const ov::ResultVector& results);

    /// \brief Delete Result node from the list of results. Method will not delete node from
    /// graph.
    /// \param result Result node to delete
    void remove_result(const std::shared_ptr<ov::op::v0::Result>& result);

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
    void add_parameters(const ov::ParameterVector& params);

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
    void remove_parameter(const std::shared_ptr<ov::op::v0::Parameter>& param);

    /// \brief Add new variables to the list. Method doesn't validate graph, it should be done
    /// manually after all changes.
    /// \param variables new variables to add
    void add_variables(const ov::op::util::VariableVector& variables);

    /// \brief Delete variable from the list of variables.
    /// Method doesn't delete nodes that used this variable from the graph.
    /// \param variable Variable to delete
    void remove_variable(const ov::op::util::Variable::Ptr& variable);

    /// \brief Return a list of model's variables.
    const ov::op::util::VariableVector& get_variables() const {
        return m_variables;
    }

    /// \brief Return a variable by specified variable_id.
    ov::op::util::Variable::Ptr get_variable_by_id(const std::string& variable_id) const;
    RTMap& get_rt_info() {
        return m_rt_info;
    }
    const RTMap& get_rt_info() const {
        return m_rt_info;
    }

    Model(const Model&) = delete;
    Model(Model&&) = delete;
    Model& operator=(const Model&) = delete;
    Model& operator=(Model&&) = delete;

private:
    friend class ov::ModelAccessor;

    /// \brief Depending on the options selected,
    /// checks all the Parameter/Variables are registered in the list of Model
    /// parameters/variables or finds all Parameters/Variables in a model and registers them.
    /// \param detect_variables If this flag is true, then it finds all Variables in a model
    /// and registers them, otherwise checks all the Variables are registered.
    /// \param detect_parameters If this flag is true, then it finds all Parameters in a
    /// model and registers them, otherwise checks all the Parameters are registered.
    void prerequirements(bool detect_variables, bool detect_parameters);

    static std::atomic<size_t> m_next_instance_id;
    std::string m_name;
    const std::string m_unique_name;
    size_t m_placement{0};
    topological_sort_t m_topological_sorter;

    ov::ResultVector m_results;
    // List of the nodes with side effect in graph.
    // These nodes are not outputs of graph but should not be removed even if have no children.
    ov::SinkVector m_sinks;
    ov::ParameterVector m_parameters;
    ov::op::util::VariableVector m_variables;
    RTMap m_rt_info;

    // Cache of topologically sorted nodes which is stored as a vector
    // of weak_ptr not to increase node ref counter to prevent the situation when
    // node has no consumers but still exists in a graph.
    mutable std::vector<std::weak_ptr<Node>> m_cached_ordered_ops;

    mutable std::unordered_map<std::string, Output<Node>> m_cached_output_names;
    mutable std::unordered_map<std::string, std::weak_ptr<Node>> m_cached_op_names;

    // Private runtime info which is shared across nodes and used only
    // for internal purposes.
    std::shared_ptr<SharedRTInfo> m_shared_rt_info;

    mutable std::mutex m_topological_sort_mutex;
};

OPENVINO_API
std::ostream& operator<<(std::ostream&, const Model&);

template <>
class OPENVINO_API AttributeAdapter<std::shared_ptr<ov::Model>>
    : public DirectValueAccessor<std::shared_ptr<ov::Model>> {
public:
    AttributeAdapter(std::shared_ptr<ov::Model>& value) : DirectValueAccessor<std::shared_ptr<ov::Model>>(value) {}

    OPENVINO_RTTI("AttributeAdapter<std::shared_ptr<Model>");
    BWDCMP_RTTI_DECLARATION;
};

/// \brief Helper method to get associated batch size for a Model
/// \details Checks layout of each parameter in a Model and extracts value for N (B) dimension. All values are then
/// merged and returned
///
/// \throws ::ov::AssertFailure with details in case of error. Possible errors are:
/// * There is no parameter with layout set. Model shall have at least one parameter with layout with 'N' dimension.
/// Recommended fix is to use `Parameter::set_layout` API, e.g.
/// `model->get_parameters()[some_index]->set_layout("NCHW");`
/// * Several parameters have conflicting N dimension, e.g. param1 NCHW{1,3,224,224} and param2 NCHW{2,3,224,224}. This
/// is ambiguous, most probably first dimension is incorrectly marked as 'batch' (N) in some layout. User shall
///// fix it before using of 'get_batch' (in example above correct layout for param2 from 'NCHW' to 'CHWN')
///
/// \param f Model where to look for a batch_size value
/// \return Dimension representing current batch size. Can represent a number or be a dynamic
OPENVINO_API ov::Dimension get_batch(const std::shared_ptr<const ov::Model>& f);

/// \brief Helper method to set batch size to a Model
///
/// \details Checks layout of each parameter in a Model and sets value for N (B) dimension. Then performs validation
/// and type propagation
///
/// \throws ::ov::AssertFailure with details in case of error. Possible errors are:
/// * There is no parameter with N dimension in layout. Model shall have at least one parameter with layout with 'N'
/// dimension. Recommended fix is to use `Parameter::set_layout` API, e.g.
/// `model->get_parameters()[some_index]->set_layout("NCHW");`
/// * Several parameters have conflicting N dimension, e.g. param1 NCHW{1,3,224,224} and param2 NCHW{3,224,224,1}. This
/// is ambiguous (1 != 3), most probably some dimension is incorrectly marked as 'batch' (N) in some layout. User shall
/// fix it before using of 'set_batch' (in example above correct layout for param2 from 'NCHW' to 'CHWN')
/// * Validation fails after setting batch_size. Model becomes in inconsistent state after new batch size value is
/// applied. Possible reason could be that layout was not set for some parameters, or batch size can't be applied to
/// model at all
///
/// \param model model where to set batch_size value
/// \param batch_size Batch size value. For dynamic batch size, Dimension::dynamic() can be passed.
OPENVINO_API void set_batch(const std::shared_ptr<ov::Model>& model, ov::Dimension batch_size);

}  // namespace ov
