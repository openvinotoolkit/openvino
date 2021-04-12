//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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

#include <memory>
#include <string>
#include "ngraph/function.hpp"
#include "ngraph/visibility.hpp"

namespace ngraph
{
namespace frontend
{


/// \brief An interface for identifying a place in a graph and iterate over it; can refer to an operation node, tensor, port etc.
///
/// \note Each front end implementation provides specialization of this interface  to represent a place
///       in a model graph. Various methods in the front end classes accept and retrieve instances
///       of Place to point to particular node part which should be modified or satisfies some criteria.
///       For example, this class is used to report model inputs and outputs, for searching operations and tensors
///       by name, for setting shape etc.
///
///       Place can refer to Tensor, Input Edge, Input Port, Operation, Output Port, Output Edge
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
class NGRAPH_API Place
{
public:

    typedef std::shared_ptr<Place> Ptr;

    virtual ~Place() = default;

    /// \brief All associated names (synonyms) that identify this place in the graph in a framework specific way
    /// \return A vector of strings each representing a name that identifies this place in the graph.
    ///         Can be empty if there are no names associated with this place or name cannot be attached.
    virtual std::vector<std::string> getNames () const;

    /// \brief Returns references to all operation nodes that consume data from this place
    /// \note It can be called for any kind of graph place searching for the first consuming opertions.
    ///
    /// \param outputPortIndex If place is an operational node it specifies which output port should be considered
    /// \return A vector with all operation node references that consumes data from this place
    virtual std::vector<Ptr> getConsumingOperations (int outputPortIndex = -1) const;

    /// \brief Returns a tensor place that gets data from this place; applicable for operations, output ports and output edges
    ///
    /// \param outputPortIndex Output port index if the current place is an operation node and has multiple output ports
    /// \return A tensor place which hold the resulting value for this place
    virtual Ptr getTargetTensor (int outputPortIndex = -1) const;

    /// \brief Returns a tensor place that supplies data for this place; applicable for operations, input ports and input edges
    ///
    /// \param inputPortIndex Input port index for operational nodes
    /// \return A tensor place which supplies data for this place
    virtual Ptr getSourceTensor (int inputPortIndex = -1) const;

    /// \brief Get an operation node place that immediately produces data for this place
    ///
    /// \param inputPortIndex If a given place is itself an operation node, this specifies a port index
    /// \return An operation place that produces data for this place
    virtual Ptr getProducingOperation (int inputPortIndex = -1) const;

    /// Returns a port that produces data for this place
    virtual Ptr getProducingPort () const;

    /// For operation node returns reference to an input port with specified index
    virtual Ptr getInputPort (int inputPortIndex = -1) const;

    /// For operation node returns reference to an output port with specified index
    virtual Ptr getOutputPort (int outputPortIndex = -1) const;

    /// Returns all input ports that consume data flows through this place
    virtual std::vector<Place::Ptr> getConsumingPorts () const;

    /// Returns true if this place is input for a model.
    virtual bool isInput () const;

    /// Returns true if this place is output for a model.
    virtual bool isOutput () const;

    /// Returns true if another place is the same as this place.
    virtual bool isEqual (Ptr another) const;

    /// \brief Returns true if another place points to the same data.
    /// \note The same data means all places on path: output port -> output edge -> tensor -> input edge -> input port.
    virtual bool isEqualData (Ptr another) const;
};


/// \brief InputModel class represents an original, not yet converted model graph in a framework format given
/// services to find places of interest in a graph or specialize/edit the model before conversion.
///
/// \note Class methods are divided into several groups: searching for places, naming and annotation,
///       topology editing, setting tensor properties.
///
///       Editing requests may affect ability to convert the original model to nGraph function. Aim to provide
///       these editing capabilities is to unlock conversion for models that are not natively supported "as-is"
///       because of undefined shapes, types or operations.
///
///       Specific front-end implementation is supposed to have a lazy implementation for all methods, not doing
///       a complete load of a model without an explicit method call. For example, the list of all inputs
///       are not pre-fetched by InputModel derived class instance creation, but only when getInputs method is called.
///       But it is not an obligation, the most convenient way should be chosen depending on the framework model
///       representation.
///
///       All editing requests affect the model representation that is held behind the scene and successive method
///       calls observe a new graph structure.
class NGRAPH_API InputModel
{
public:

    typedef std::shared_ptr<InputModel> Ptr;

    virtual ~InputModel() = default;

    /////  Searching for places  /////


    /// \brief Returns all inputs for a model
    /// An input is a place in a graph where data is supposed to flow inside graph from outside.
    /// It can be a tensor, port, operation; which kind of place can be an output is FW dependent.
    /// Usually framework models have a dedicated artifact to code model input, it can be a tensor without producer,
    /// that writes to it in ONNX, or a special operation like Placeholder in TensorFlow.
    /// \return A vector of input place references
    virtual std::vector<Place::Ptr> getInputs () const;

    /// \brief Returns all output for a model
    /// An output is a terminal place in a graph where data escapes the flow. It can be a tensor, port, operation;
    /// which kind of place can be an output is FW dependent. In comparison to a graph input, the output is less
    /// formally defined thing and determination of initial list of outputs may include some conventions defined
    /// by a frontend itself, not a framework. For example, all output ports without consumers may be considered
    /// as outputs.
    /// \return A vector of output place references
    virtual std::vector<Place::Ptr> getOutputs () const;

    /// Returns a tensor place by a tensor name following framework conventions, or nullptr if a tensor with this name doesn't exist.
    virtual Place::Ptr getPlaceByTensorName (const std::string& tensorName);

    /// Returns an operation place by a tensor name following framework conventions, or nullptr if an operation with this name doesn't exist.
    virtual Place::Ptr getPlaceByOperationName (const std::string& operationName);

    /// Returns an input port.
    virtual Place::Ptr getPlaceByOperationAndInputPort (const std::string& operationName, int inputPortIndex);

    /// Returns an output port.
    virtual Place::Ptr getPlaceByOperationAndOutputPort (const std::string& operationName, int outputPortIndex);


    ///// Naming and annotation  /////


    virtual void setNameForTensor (Place::Ptr tensor, const std::string& newName);
    virtual void addNameForTensor (Place::Ptr tensor, const std::string& newName);
    virtual void setNameForOperation (Place::Ptr operation, const std::string& newName);
    virtual void freeNameForTensor (const std::string& name);
    virtual void freeNameForOperation (const std::string& name);

    virtual void setNameForDimension (Place::Ptr place, size_t shapeDimIndex, const std::string& dimName);


    ///// Topology Editing  /////

    /// Cut immediately before this place and assign this place as new input; prune all nodes that don't contribute to any output.
    virtual void cutAndAddNewInput (Place::Ptr place, const std::string& newNameOptional = "");

    /// Cut immediately after this place and assign this place as new output; prune all nodes that don't contribute to any output.
    virtual void cutAndAddNewOutput (Place::Ptr place, const std::string& newNameOptional = "");

    /// \brief Assign this place as new output or add necessary nodes to represent a new output.
    ///
    /// \param place Anchor point to add an output
    /// \return new output place, may be the same as a given place
    virtual Place::Ptr addOutput (Place::Ptr place);

    /// Removes any sinks directly attached to this place with all inbound data flow if it is not required by any other output.
    virtual void removeOutput (Place::Ptr place);

    /// Removes an input place and all data flow that depends on it.
    // TODO: remove it as something not practically useful in the API?
    virtual void removeInput (Place::Ptr place);

    /// Replaces all existing outputs with new ones removing all data flow that is not required for new outputs.
    ///
    /// \param outputs Vector with places that will become new outputs; may intersect existing outputs.
    virtual void overrideAllOutputs (const std::vector<Place::Ptr>& outputs);

    /// \brief Modifies the graph to use new inputs instead of existing ones. New inputs should completely satisfy all existing outputs.
    virtual void overrideAllInputs (const std::vector<Place::Ptr>& inputs);

    /// Leaves only subgraph that are defined by new inputs and new outputs.
    virtual void extractSubgraph (const std::vector<Place::Ptr>& inputs, const std::vector<Place::Ptr>& outputs);

    ///// Setting tensor properties  /////

    /// Sets shape that would be used by default for this place; place should be uniquely refer to some data.
    // TODO: define clearly which scenario requires it -- currently it should satisfy requirement to have statically defined shapes for tensors
    virtual void setDefaultShape (Place::Ptr place, const ngraph::Shape&);

    /// Defines all possible shape that may be used for this place; place should be uniquely refer to some data.
    /// This partial shape will be converted to corresponding shape of results ngraph nodes and will define shape inference
    /// when the model is converted to ngraph.
    virtual void setPartialShape (Place::Ptr place, const ngraph::PartialShape&);

    /// Sets new element type for a place.
    virtual void setElementType (Place::Ptr place, const ngraph::element::Type&);

    /// Freezes a tensor with statically defined value or replace existing value for already constant node or tensor.
    virtual void setTensorValue (Place::Ptr place, const void* value);

    /// Defines partial value (lower bound and upper bound) for a tensor place.
    // TODO: more details for minValue and maxValue format; who defines shape?
    virtual void setTensorPartialValue (Place::Ptr place, const void* minValue, const void* maxValue);

    // TODO: Document "inputs/output assymetry" in more details

    // Traversing
    // TODO: remove or add something; most likely will have only a single method that provides a list of operation nodes sorted topologically

    // Support querying
    // TODO: remove or add something; there are no candidates, all queries can be satisfied without any API extension here
};

class NGRAPH_API FrontEnd
{
public:
    typedef std::shared_ptr<FrontEnd> Ptr;
    virtual ~FrontEnd() = default;

    virtual InputModel::Ptr loadFromFile (const std::string& path) const;
    virtual InputModel::Ptr loadFromFiles (const std::vector<std::string>& paths) const;
    virtual InputModel::Ptr loadFromMemory (const void* model) const;
    virtual InputModel::Ptr loadFromMemoryFragments (const std::vector<const void*>& modelParts) const;
    virtual InputModel::Ptr loadFromStream (std::istream& path) const;
    virtual InputModel::Ptr loadFromStreams (const std::vector<std::istream*>& paths) const;

    // Extra ctors may be provided by FW-specialized data structure for graph representaion

    /// Completely convert and normalize entire function, throws if it is not possible
    virtual std::shared_ptr<ngraph::Function> convert (InputModel::Ptr model) const;

    /// Completely convert the remaining, not converted part of a function.
    virtual std::shared_ptr<ngraph::Function> convert (std::shared_ptr<ngraph::Function> partiallyConverted) const;

    /// Convert only those parts of the model that can be converted leaving others as-is.
    /// Converted parts are not normalized by additional transformations; normalize function
    /// or another form of convert function should be called to finalize the conversion process.
    virtual std::shared_ptr<ngraph::Function> convertPartially (InputModel::Ptr model) const;

    /// Convert operations with one-to-one mapping with decoding nodes.
    /// Each decoding node is an nGraph node representing a single FW operation node with all attributes
    /// represented in FW-independent way.
    virtual std::shared_ptr<ngraph::Function> decode (InputModel::Ptr model) const;

    /// Runs normalization passes on function that was loaded with partial conversion
    virtual void normalize (std::shared_ptr<ngraph::Function> function) const;
};

enum FrontEndCapabilities {
    FEC_DEFAULT   =  0,    // Just reading and conversion, w/o any modifications; intended to be used in Reader
    FEC_CUT       =  1,
    FEC_NAMES     =  2,
    FEC_REPLACE   =  4,
    FEC_TRAVERSE  =  8,
    FEC_WILDCARDS = 16,
};

class NGRAPH_API FrontEndManager
{
public:
    FrontEndManager();
    ~FrontEndManager();
    FrontEnd::Ptr loadByFramework(const std::string& framework, FrontEndCapabilities fec = FEC_DEFAULT);
    FrontEnd::Ptr loadByModel(const std::string& path, FrontEndCapabilities fec = FEC_DEFAULT);
    std::vector<std::string> availableFrontEnds() const;

    using FrontEndFactory = std::function<FrontEnd::Ptr(FrontEndCapabilities fec)>;
    void registerFrontEnd(const std::string& name, FrontEndFactory creator);
private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace frontend

} // namespace ngraph
