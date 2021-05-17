// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include "frontend_manager_defs.hpp"
#include "ngraph/function.hpp"

namespace ngraph
{
    namespace frontend
    {
        /// \brief An interface for identifying a place in a graph and iterate over it; can refer to
        /// an operation node, tensor, port etc.
        ///
        /// \note Each front end implementation provides specialization of this interface to
        /// represent a place in a model graph. Various methods in the front end classes accept and
        /// retrieve instances of Place to point to particular node part which should be modified or
        /// satisfies some criteria. For example, this class is used to report model inputs
        /// and outputs, for searching operations and tensors by name, for setting shape etc.
        ///
        /// Place can refer to Tensor, Input Edge, Input Port, Operation, Output Port, Output Edge
        ///
        ///                [Tensor A]
        ///                    |
        ///                    | [Input Edge]
        ///                    |
        ///                    V
        ///           -------------------
        ///           [  [Input Port 0] ]
        ///           [                 ]
        ///           [   Operation A   ]
        ///           [                 ]
        ///           [ [Output Port 0] ]
        ///           -------------------
        ///                    |
        ///                    | [Output Edge]
        ///                    |
        ///                    V
        ///                [Tensor B]
        ///                    |
        ///                    | [Input Edge]
        ///                    |
        ///                    V
        ///           -------------------
        ///           [  [Input Port 0] ]
        ///           [                 ]
        ///           [   Operation B   ]
        ///           [                 ]
        ///           [ [Output Port 0] ]
        ///           -------------------
        ///                    |
        ///                    | [Output Edge]
        ///                    |
        ///                    V
        ///                [Tensor C]
        ///
        class FRONTEND_API Place
        {
        public:
            typedef std::shared_ptr<Place> Ptr;

            virtual ~Place() = default;

            /// \brief All associated names (synonyms) that identify this place in the graph in a
            /// framework specific way
            ///
            /// \return A vector of strings each representing a name that identifies this place in
            /// the graph. Can be empty if there are no names associated with this place or name
            /// cannot be attached.
            virtual std::vector<std::string> get_names() const;

            /// \brief Returns references to all operation nodes that consume data from this place
            /// \note It can be called for any kind of graph place searching for the first consuming
            /// operations.
            ///
            /// \param outputPortIndex If place is an operational node it specifies which output
            /// port should be considered. It is optional if place has only one output port
            ///
            /// \return A vector with all operation node references that consumes data from this
            /// place
            virtual std::vector<Ptr> get_consuming_operations(int outputPortIndex = -1) const;

            /// \brief Returns a tensor place that gets data from this place; applicable for
            /// operations, output ports and output edges
            ///
            /// \param outputPortIndex Output port index if the current place is an operation node
            /// and has multiple output ports. It is optional if place has only one output port
            ///
            /// \return A tensor place which hold the resulting value for this place
            virtual Ptr get_target_tensor(int outputPortIndex = -1) const;

            /// \brief Returns a tensor place that supplies data for this place; applicable for
            /// operations, input ports and input edges
            ///
            /// \param inputPortIndex Input port index for operational nodes. It is optional if
            /// place has only one input port
            /// \return A tensor place which supplies data for this place
            virtual Ptr get_source_tensor(int inputPortIndex = -1) const;

            /// \brief Get an operation node place that immediately produces data for this place
            ///
            /// \param inputPortIndex If a given place is itself an operation node, this specifies a
            /// port index. It is optional if place has only one input port
            ///
            /// \return An operation place that produces data for this place
            virtual Ptr get_producing_operation(int inputPortIndex = -1) const;

            /// Returns a port that produces data for this place
            virtual Ptr get_producing_port() const;

            /// For operation node returns reference to an input port with specified index
            /// \param inputPortIndex Input port index. It is optional if place has only one input
            /// port
            virtual Ptr get_input_port(int inputPortIndex = -1) const;

            /// For operation node returns reference to an input port with specified name and index
            /// \param inputName Name of port group, each group can have multiple ports
            /// \param inputPortIndex Input port index. It is optional if port group has only one
            /// input port
            virtual Ptr get_input_port(const std::string& inputName, int inputPortIndex = -1) const;

            /// For operation node returns reference to an output port with specified index
            /// \param outputPortIndex Output port index. It is optional if place has only one
            /// output port
            virtual Ptr get_output_port(int outputPortIndex = -1) const;

            /// For operation node returns reference to an output port with specified name and index
            /// \param outputName Name of output port group, each group can have multiple ports
            /// \param outputPortIndex Output port index. It is optional if port group has only one
            /// output port
            virtual Ptr get_output_port(const std::string& outputName,
                                        int outputPortIndex = -1) const;

            /// Returns all input ports that consume data flows through this place
            virtual std::vector<Place::Ptr> get_consuming_ports() const;

            /// Returns true if this place is input for a model.
            virtual bool is_input() const;

            /// Returns true if this place is output for a model.
            virtual bool is_output() const;

            /// Returns true if another place is the same as this place.
            /// \param another Another place object
            virtual bool is_equal(Ptr another) const;

            /// \brief Returns true if another place points to the same data.
            /// \note The same data means all places on path: output port -> output edge -> tensor
            /// -> input edge -> input port.
            /// \param another Another place object
            virtual bool is_equal_data(Ptr another) const;
        };

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
                get_place_by_operation_and_input_port(const std::string& operationName,
                                                      int inputPortIndex);

            /// \brief Returns an output port place by operation name and appropriate port index
            /// \param operationNameNname of operation
            /// \param outputPortIndex Index of output port for this operation
            /// \return Place representing output port of operation
            virtual Place::Ptr
                get_place_by_operation_and_output_port(const std::string& operationName,
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

        /// \brief An interface for identifying a frontend for a particular framework.
        /// Provides an ability to load and convert of input model
        class FRONTEND_API FrontEnd
        {
        public:
            typedef std::shared_ptr<FrontEnd> Ptr;

            FrontEnd();

            virtual ~FrontEnd();

            /// \brief Loads an input model by specified model file path
            /// If model is stored in several files (e.g. model topology and model weights) -
            /// frontend implementation is responsible to handle this case, generally frontend may
            /// retrieve other file names from main file
            /// \param path Main model file path
            /// \return Loaded input model
            virtual InputModel::Ptr load_from_file(const std::string& path) const;

            /// \brief Loads an input model by specified number of model files
            /// This shall be used for cases when client knows all model files (model, weights, etc)
            /// \param paths Array of model files
            /// \return Loaded input model
            virtual InputModel::Ptr load_from_files(const std::vector<std::string>& paths) const;

            /// \brief Loads an input model by already loaded memory buffer
            /// Memory structure is frontend-defined and is not specified in generic API
            /// \param model Model memory buffer
            /// \return Loaded input model
            virtual InputModel::Ptr load_from_memory(const void* model) const;

            /// \brief Loads an input model from set of memory buffers
            /// Memory structure is frontend-defined and is not specified in generic API
            /// \param modelParts Array of model memory buffers
            /// \return Loaded input model
            virtual InputModel::Ptr
                load_from_memory_fragments(const std::vector<const void*>& modelParts) const;

            /// \brief Loads an input model by input stream representing main model file
            /// \param stream Input stream of main model
            /// \return Loaded input model
            virtual InputModel::Ptr load_from_stream(std::istream& stream) const;

            /// \brief Loads an input model by input streams representing all model files
            /// \param streams Array of input streams for model
            /// \return Loaded input model
            virtual InputModel::Ptr
                load_from_streams(const std::vector<std::istream*>& streams) const;

            /// \brief Completely convert and normalize entire function, throws if it is not
            /// possible
            /// \param model Input model
            /// \return fully converted nGraph function
            virtual std::shared_ptr<ngraph::Function> convert(InputModel::Ptr model) const;

            /// \brief Completely convert the remaining, not converted part of a function.
            /// \param partiallyConverted partially converted nGraph function
            /// \return fully converted nGraph function
            virtual std::shared_ptr<ngraph::Function>
                convert(std::shared_ptr<ngraph::Function> partiallyConverted) const;

            /// \brief Convert only those parts of the model that can be converted leaving others
            /// as-is. Converted parts are not normalized by additional transformations; normalize
            /// function or another form of convert function should be called to finalize the
            /// conversion process.
            /// \param model Input model
            /// \return partially converted nGraph function
            virtual std::shared_ptr<ngraph::Function>
                convert_partially(InputModel::Ptr model) const;

            /// \brief Convert operations with one-to-one mapping with decoding nodes.
            /// Each decoding node is an nGraph node representing a single FW operation node with
            /// all attributes represented in FW-independent way.
            /// \param model Input model
            /// \return nGraph function after decoding
            virtual std::shared_ptr<ngraph::Function> decode(InputModel::Ptr model) const;

            /// \brief Runs normalization passes on function that was loaded with partial conversion
            /// \param function partially converted nGraph function
            virtual void normalize(std::shared_ptr<ngraph::Function> function) const;
        };

        /// Capabilities for requested FrontEnd
        /// In general, frontend implementation may be divided into several libraries by capability
        /// level It will allow faster load of frontend when only limited usage is expected by
        /// client application as well as binary size can be minimized by removing not needed parts
        /// from application's package
        namespace FrontEndCapabilities
        {
            /// \brief Just reading and conversion, w/o any modifications; intended to be used in
            /// Reader
            static const int FEC_DEFAULT = 0;

            /// \brief Topology cutting capability
            static const int FEC_CUT = 1;

            /// \brief Query entities by names, renaming and adding new names for operations and
            /// tensors
            static const int FEC_NAMES = 2;

            /// \brief Partial model conversion and decoding capability
            static const int FEC_WILDCARDS = 4;
        }; // namespace FrontEndCapabilities

        // -------------- FrontEndManager -----------------
        using FrontEndCapFlags = int;
        using FrontEndFactory = std::function<FrontEnd::Ptr(FrontEndCapFlags fec)>;

        /// \brief Frontend management class, loads available frontend plugins on construction
        /// Allows load of frontends for particular framework, register new and list available
        /// frontends This is a main frontend entry point for client applications
        class FRONTEND_API FrontEndManager final
        {
        public:
            FrontEndManager();

            FrontEndManager(FrontEndManager&&);
            FrontEndManager& operator=(FrontEndManager&&);

            ~FrontEndManager();

            /// \brief Loads frontend by name of framework and capabilities
            /// \param framework Framework name. Throws exception if name is not in list of
            /// available frontends \param fec Frontend capabilities. It is recommended to use only
            /// those capabilities which are needed to minimize load time
            /// \return Frontend interface for further loading of models
            FrontEnd::Ptr
                load_by_framework(const std::string& framework,
                                  FrontEndCapFlags fec = FrontEndCapabilities::FEC_DEFAULT);

            /// \brief Loads frontend by model file path. Selects and loads appropriate frontend
            /// depending on model file extension and other file info (header) \param framework
            /// Framework name. Throws exception if name is not in list of available frontends
            /// \param fec Frontend capabilities. It is recommended to use only those capabilities
            /// which are needed to minimize load time
            /// \return Frontend interface for further loading of model
            FrontEnd::Ptr load_by_model(const std::string& path,
                                        FrontEndCapFlags fec = FrontEndCapabilities::FEC_DEFAULT);

            /// \brief Gets list of registered frontends
            std::vector<std::string> get_available_front_ends() const;

            /// \brief Register frontend with name and factory creation method
            void register_front_end(const std::string& name, FrontEndFactory creator);

        private:
            class Impl;

            std::unique_ptr<Impl> m_impl;
        };

        // --------- Plugin exporting information --------------

        /// \brief Each frontend plugin is responsible to export GetAPIVersion function returning
        /// version of frontend API used for this plugin
        /// If version is not matched with OV_FRONTEND_API_VERSION - plugin will not be loaded by
        /// FrontEndManager
        using FrontEndVersion = uint64_t;

        /// \brief Each frontend plugin is responsible to export GetFrontEndData function returning
        /// heap-allocated pointer to this structure. Will be used by FrontEndManager during loading
        /// of plugins
        struct FrontEndPluginInfo
        {
            std::string m_name;
            FrontEndFactory m_creator;
        };

    } // namespace frontend

} // namespace ngraph
