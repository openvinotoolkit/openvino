// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <vector>
#include "frontend_manager_defs.hpp"
#include "ngraph/partial_shape.hpp"
#include "ngraph/type/element_type.hpp"
#include "place.hpp"

namespace ngraph
{
    namespace frontend
    {
        /// \brief InputModel class represents an original, not yet converted model graph in a
        /// framework format given services to find places of interest in a graph or specialize/edit
        /// the model before conversion.
        ///
        /// \note Class methods are divided into several groups: searching for places, naming and
        /// annotation, topology editing, setting tensor properties.
        ///
        /// Editing requests may affect ability to convert the original model to nGraph function.
        /// Aim to provide these editing capabilities is to unlock conversion for models that
        /// are not natively supported "as-is" because of undefined shapes, types or operations.
        ///
        /// Specific front-end implementation is supposed to have a lazy implementation for
        /// all methods, not doing a complete load of a model without an explicit method call.
        /// For example, the list of all inputs are not pre-fetched by InputModel derived
        /// class instance creation, but only when get_inputs method is called. But it is not
        /// an obligation, the most convenient way should be chosen depending on the framework
        /// model representation.
        ///
        /// All editing requests affect the model representation that is held behind the scene
        /// successive method calls observe a new graph structure.
        class FRONTEND_API InputModel
        {
        public:
            typedef std::shared_ptr<InputModel> Ptr;

            virtual ~InputModel() = default;

            /////  Searching for places  /////

            /// \brief Returns all inputs for a model
            /// An input is a place in a graph where data is supposed to flow inside graph from
            /// outside. It can be a tensor, port, operation; which kind of place can be an output
            /// is FW dependent. Usually framework models have a dedicated artifact to code model
            /// input, it can be a tensor without producer, that writes to it in ONNX, or a special
            /// operation like Placeholder in TensorFlow.
            ///
            /// \return A vector of input place references
            virtual std::vector<Place::Ptr> get_inputs() const;

            /// \brief Returns all output for a model
            /// An output is a terminal place in a graph where data escapes the flow. It can be a
            /// tensor, port, operation; which kind of place can be an output is FW dependent. In
            /// comparison to a graph input, the output is less formally defined thing and
            /// determination of initial list of outputs may include some conventions defined by a
            /// frontend itself, not a framework. For example, all output ports without consumers
            /// may be considered as outputs.
            ///
            /// \return A vector of output place references
            virtual std::vector<Place::Ptr> get_outputs() const;

            /// \brief Returns a tensor place by a tensor name following framework conventions, or
            /// nullptr if a tensor with this name doesn't exist.
            /// \param tensorName Name of tensor
            /// \return Tensor place corresponding to specifed tensor name
            virtual Place::Ptr get_place_by_tensor_name(const std::string& tensorName) const;

            /// \brief Returns an operation place by an operation name following framework
            /// conventions, or nullptr if an operation with this name doesn't exist. \param
            /// operationName Name of operation \return Place representing operation
            virtual Place::Ptr get_place_by_operation_name(const std::string& operationName);

            /// \brief Returns an input port place by operation name and appropriate port index
            /// \param operationName Name of operation
            /// \param outputPortIndex Index of input port for this operation
            /// \return Place representing input port of operation
            virtual Place::Ptr
                get_place_by_operation_name_and_input_port(const std::string& operationName,
                                                           int inputPortIndex);

            /// \brief Returns an output port place by operation name and appropriate port index
            /// \param operationNameNname of operation
            /// \param outputPortIndex Index of output port for this operation
            /// \return Place representing output port of operation
            virtual Place::Ptr
                get_place_by_operation_name_and_output_port(const std::string& operationName,
                                                            int outputPortIndex);

            ///// Naming and annotation  /////

            /// \brief Sets name for tensor. Overwrites existing names of this place
            /// \param operation Tensor place
            /// \param newName New name for this tensor
            virtual void set_name_for_tensor(Place::Ptr tensor, const std::string& newName);

            /// \brief Adds new name for tensor
            /// \param operation Tensor place
            /// \param newName New name to be added to this place
            virtual void add_name_for_tensor(Place::Ptr tensor, const std::string& newName);

            /// \brief Sets name for operation. Overwrites existing names of this place
            /// \param operation Operation place
            /// \param newName New name for this operation
            virtual void set_name_for_operation(Place::Ptr operation, const std::string& newName);

            /// \brief Unassign specified name from tensor place(s)
            /// \param name Name of tensor
            virtual void free_name_for_tensor(const std::string& name);

            /// \brief Unassign specified name from operation place(s)
            /// \param name Name of operation
            virtual void free_name_for_operation(const std::string& name);

            /// \brief Set name for a particular dimension of a place (e.g. batch dimension)
            /// \param place Model's place
            /// \param shapeDimIndex Dimension index
            /// \param dimName Name to assign on this dimension
            virtual void set_name_for_dimension(Place::Ptr place,
                                                size_t shapeDimIndex,
                                                const std::string& dimName);

            ///// Topology Editing  /////

            /// \brief Cut immediately before this place and assign this place as new input; prune
            /// all nodes that don't contribute to any output.
            /// \param place New place to be assigned as input
            /// \param newNameOptional Optional new name assigned to this input place
            virtual void cut_and_add_new_input(Place::Ptr place,
                                               const std::string& newNameOptional = "");

            /// \brief Cut immediately after this place and assign this place as new output; prune
            /// all nodes that don't contribute to any output.
            /// \param place New place to be assigned as output
            /// \param newNameOptional Optional new name assigned to this output place
            virtual void cut_and_add_new_output(Place::Ptr place,
                                                const std::string& newNameOptional = "");

            /// \brief Assign this place as new output or add necessary nodes to represent a new
            /// output.
            ///
            /// \param place Anchor point to add an output
            /// \return new output place, may be the same as a given place
            virtual Place::Ptr add_output(Place::Ptr place);

            /// \brief Removes any sinks directly attached to this place with all inbound data flow
            /// if it is not required by any other output.
            /// \param place Model place
            virtual void remove_output(Place::Ptr place);

            /// \brief Replaces all existing outputs with new ones removing all data flow that is
            /// not required for new outputs.
            ///
            /// \param outputs Vector with places that will become new outputs; may intersect
            /// existing outputs.
            /// \param outputs Array of new output places
            virtual void override_all_outputs(const std::vector<Place::Ptr>& outputs);

            /// \brief Modifies the graph to use new inputs instead of existing ones. New inputs
            /// should completely satisfy all existing outputs.
            /// \param inputs Array of new input places
            virtual void override_all_inputs(const std::vector<Place::Ptr>& inputs);

            /// \brief Leaves only subgraph that are defined by new inputs and new outputs.
            /// \param inputs Array of new input places
            /// \param outputs Array of new output places
            virtual void extract_subgraph(const std::vector<Place::Ptr>& inputs,
                                          const std::vector<Place::Ptr>& outputs);

            ///// Setting tensor properties  /////

            /// \brief Defines all possible shape that may be used for this place; place should be
            /// uniquely refer to some data. This partial shape will be converted to corresponding
            /// shape of results ngraph nodes and will define shape inference when the model is
            /// converted to ngraph.
            /// \param place Model place
            /// \param shape Partial shape for this place
            virtual void set_partial_shape(Place::Ptr place, const ngraph::PartialShape& shape);

            /// \brief Returns current partial shape used for this place
            /// \param place Model place
            /// \return Partial shape for this place
            virtual ngraph::PartialShape get_partial_shape(Place::Ptr place) const;

            /// \brief Sets new element type for a place
            /// \param place Model place
            /// \param type New element type
            virtual void set_element_type(Place::Ptr place, const ngraph::element::Type& type);

            /// \brief Freezes a tensor with statically defined value or replace existing value for
            /// already constant node or tensor
            /// \param place Tensor place
            /// \param value Value for tensor place representing a memory buffer
            virtual void set_tensor_value(Place::Ptr place, const void* value);

            /// \brief Defines partial value (lower bound and upper bound) for a tensor place
            /// TODO: more details for minValue and maxValue format; who defines shape?
            /// \param place Tensor place
            /// \param minValue Lower bound of partial value for tensor place
            /// \param maxValue Upper bound of partial value for tensor place
            virtual void set_tensor_partial_value(Place::Ptr place,
                                                  const void* minValue,
                                                  const void* maxValue);
        };

    } // namespace frontend
} // namespace ngraph
