// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include "ngraph/function.hpp"
#include "ngraph/visibility.hpp"
#include <frontend_manager/ifrontend_manager.hpp>

namespace ngraph
{
namespace frontend
{

class Place;
class PlaceImpl;
class InputModel;
class InputModelImpl;
class FrontEnd;
class FrontEndImpl;
class IFrontEnd;
class FrontEndManagerImpl;

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
class FRONTEND_API Place final
{
public:

    Place();
    Place(const Place&);
    Place& operator=(const Place&);
    Place(Place&&);
    Place& operator=(Place&&);

    ~Place();

    /// \brief Checks of place represents a real place in the graph
    /// \return True if place represents a real place in the graph, false otherwise
    bool isValid() const;

    /// \brief All associated names (synonyms) that identify this place in the graph in a framework specific way
    /// \return A vector of strings each representing a name that identifies this place in the graph.
    ///         Can be empty if there are no names associated with this place or name cannot be attached.
    std::vector<std::string> getNames() const;

    /// \brief Returns references to all operation nodes that consume data from this place
    /// \note It can be called for any kind of graph place searching for the first consuming opertions.
    ///
    /// \param outputPortIndex If place is an operational node it specifies which output port should be considered
    /// \return A vector with all operation node references that consumes data from this place
    std::vector<Place> getConsumingOperations(int outputPortIndex = -1) const;

    /// \brief Returns a tensor place that gets data from this place; applicable for operations, output ports and output edges
    ///
    /// \param outputPortIndex Output port index if the current place is an operation node and has multiple output ports
    /// \return A tensor place which hold the resulting value for this place
    Place getTargetTensor(int outputPortIndex = -1) const;

    /// \brief Returns a tensor place that supplies data for this place; applicable for operations, input ports and input edges
    ///
    /// \param inputPortIndex Input port index for operational nodes
    /// \return A tensor place which supplies data for this place
    Place getSourceTensor(int inputPortIndex = -1) const;

    /// \brief Get an operation node place that immediately produces data for this place
    ///
    /// \param inputPortIndex If a given place is itself an operation node, this specifies a port index
    /// \return An operation place that produces data for this place
    Place getProducingOperation(int inputPortIndex = -1) const;

    /// Returns a port that produces data for this place
    Place getProducingPort() const;

    /// For operation node returns reference to an input port with specified index
    Place getInputPort(int inputPortIndex = -1) const;

    /// For operation node returns reference to an input port with specified name and index
    Place getInputPort(const std::string& inputName, int inputPortIndex = -1) const;

    /// For operation node returns reference to an output port with specified index
    Place getOutputPort(int outputPortIndex = -1) const;

    /// For operation node returns reference to an output port with specified name and index
    Place getOutputPort(const std::string& outputName, int outputPortIndex = -1) const;

    /// Returns all input ports that consume data flows through this place
    std::vector<Place>getConsumingPorts () const;

    /// Returns true if this place is input for a model.
    bool isInput() const;

    /// Returns true if this place is output for a model.
    bool isOutput() const;

    /// Returns true if another place is the same as this place.
    bool isEqual(const Place& another) const;

    /// \brief Returns true if another place points to the same data.
    /// \note The same data means all places on path: output port -> output edge -> tensor -> input edge -> input port.
    bool isEqualData(const Place& another) const;

private:
    /// \brief Internal constructor. Not for public usage.
    /// \param impl Internal implementation If place is an operational node it specifies which output port should be considered
    Place(const std::shared_ptr<PlaceImpl>& impl);
    friend class InputModelImpl;
    friend class PlaceImpl;

private:
    // Shared ptr to allow copy of place
    std::shared_ptr<PlaceImpl> m_impl;
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
class FRONTEND_API InputModel final
{
public:

    InputModel();
    InputModel(InputModel&& other);
    InputModel& operator=(InputModel&& other);

    ~InputModel();

    /////  Searching for places  /////

    /// \brief Returns all inputs for a model
    /// An input is a place in a graph where data is supposed to flow inside graph from outside.
    /// It can be a tensor, port, operation; which kind of place can be an output is FW dependent.
    /// Usually framework models have a dedicated artifact to code model input, it can be a tensor without producer,
    /// that writes to it in ONNX, or a special operation like Placeholder in TensorFlow.
    /// \return A vector of input place references
    std::vector<Place> getInputs () const;

    /// \brief Returns all output for a model
    /// An output is a terminal place in a graph where data escapes the flow. It can be a tensor, port, operation;
    /// which kind of place can be an output is FW dependent. In comparison to a graph input, the output is less
    /// formally defined thing and determination of initial list of outputs may include some conventions defined
    /// by a frontend itself, not a framework. For example, all output ports without consumers may be considered
    /// as outputs.
    /// \return A vector of output place references
    std::vector<Place> getOutputs () const;

    /// Returns a tensor place by a tensor name following framework conventions, or nullptr if a tensor with this name doesn't exist.
    Place getPlaceByTensorName (const std::string& tensorName) const;

    /// Returns an operation place by a tensor name following framework conventions, or nullptr if an operation with this name doesn't exist.
    Place getPlaceByOperationName (const std::string& operationName);

    /// Returns an input port.
    Place getPlaceByOperationAndInputPort (const std::string& operationName, int inputPortIndex);

    /// Returns an output port.
    Place getPlaceByOperationAndOutputPort (const std::string& operationName, int outputPortIndex);


    ///// Naming and annotation  /////

    void setNameForTensor (Place& tensor, const std::string& newName);
    void addNameForTensor (Place& tensor, const std::string& newName);
    void setNameForOperation (Place& operation, const std::string& newName);
    void freeNameForTensor (Place& operation, const std::string& name);
    void freeNameForOperation (Place& operation, const std::string& name);

    void setNameForDimension (Place& place, size_t shapeDimIndex, const std::string& dimName);

    ///// Topology Editing  /////

    /// Cut immediately before this place and assign this place as new input; prune all nodes that don't contribute to any output.
    void cutAndAddNewInput (Place& place, const std::string& newNameOptional = "");

    /// Cut immediately after this place and assign this place as new output; prune all nodes that don't contribute to any output.
    void cutAndAddNewOutput (Place& place, const std::string& newNameOptional = "");

    /// \brief Assign this place as new output or add necessary nodes to represent a new output.
    ///
    /// \param place Anchor point to add an output
    /// \return new output place, may be the same as a given place
    Place addOutput (Place& place);

    /// Removes any sinks directly attached to this place with all inbound data flow if it is not required by any other output.
    void removeOutput (Place& place);

    /// Removes an input place and all data flow that depends on it.
    // TODO: remove it as something not practically useful in the API?
    void removeInput (Place& place);

    /// Replaces all existing outputs with new ones removing all data flow that is not required for new outputs.
    ///
    /// \param outputs Vector with places that will become new outputs; may intersect existing outputs.
    void overrideAllOutputs (const std::vector<Place>& outputs);

    /// \brief Modifies the graph to use new inputs instead of existing ones. New inputs should completely satisfy all existing outputs.
    void overrideAllInputs (const std::vector<Place>& inputs);

    /// Leaves only subgraph that are defined by new inputs and new outputs.
    void extractSubgraph (const std::vector<Place>& inputs, const std::vector<Place>& outputs);

    ///// Setting tensor properties  /////

    /// Sets shape that would be used by default for this place; place should be uniquely refer to some data.
    // TODO: define clearly which scenario requires it -- currently it should satisfy requirement to have statically defined shapes for tensors
    void setDefaultShape (Place& place, const ngraph::Shape&);

    /// Defines all possible shape that may be used for this place; place should be uniquely refer to some data.
    /// This partial shape will be converted to corresponding shape of results ngraph nodes and will define shape inference
    /// when the model is converted to ngraph.
    void setPartialShape (Place& place, const ngraph::PartialShape&);

    /// Sets new element type for a place.
    void setElementType (Place& place, const ngraph::element::Type&);

    /// Freezes a tensor with statically defined value or replace existing value for already constant node or tensor.
    void setTensorValue (Place& place, const void* value);

    /// Defines partial value (lower bound and upper bound) for a tensor place.
    // TODO: more details for minValue and maxValue format; who defines shape?
    void setTensorPartialValue (Place& place, const void* minValue, const void* maxValue);

    // TODO: Document "inputs/output assymetry" in more details

    // Traversing
    // TODO: remove or add something; most likely will have only a single method that provides a list of operation nodes sorted topologically

    // Support querying
    // TODO: remove or add something; there are no candidates, all queries can be satisfied without any API extension here

private:
    friend class FrontEndImpl;
    InputModel(std::unique_ptr<InputModelImpl>&& impl);

private:
    std::unique_ptr<InputModelImpl> m_impl;
};

struct InputModelShared {
    InputModel inputModel;
    InputModelShared(InputModel&& m): inputModel(std::move(m)) {}
};

// Helper function for those who need shared pointers
inline std::shared_ptr<InputModelShared> to_shared(InputModel&& m) {
    return std::make_shared<InputModelShared>(std::move(m));
}

class FRONTEND_API FrontEnd final
{
public:
    FrontEnd();
    FrontEnd(FrontEnd&& other);
    FrontEnd& operator=(FrontEnd&& other);

    ~FrontEnd();

    InputModel loadFromFile (const std::string& path) const;
    InputModel loadFromFiles (const std::vector<std::string>& paths) const;
    InputModel loadFromMemory (const void* model) const;
    InputModel loadFromMemoryFragments (const std::vector<const void*>& modelParts) const;
    InputModel loadFromStream (std::istream& path) const;
    InputModel loadFromStreams (const std::vector<std::istream*>& paths) const;

    // Extra ctors may be provided by FW-specialized data structure for graph representaion

    /// Completely convert and normalize entire function, throws if it is not possible
    std::shared_ptr<ngraph::Function> convert (const InputModel& model) const;

    /// Completely convert the remaining, not converted part of a function.
    std::shared_ptr<ngraph::Function> convert (std::shared_ptr<ngraph::Function> partiallyConverted) const;

    /// Convert only those parts of the model that can be converted leaving others as-is.
    /// Converted parts are not normalized by additional transformations; normalize function
    /// or another form of convert function should be called to finalize the conversion process.
    std::shared_ptr<ngraph::Function> convertPartially (const InputModel& model) const;

    /// Convert operations with one-to-one mapping with decoding nodes.
    /// Each decoding node is an nGraph node representing a single FW operation node with all attributes
    /// represented in FW-independent way.
    std::shared_ptr<ngraph::Function> decode (const InputModel& model) const;

    /// Runs normalization passes on function that was loaded with partial conversion
    void normalize (std::shared_ptr<ngraph::Function> function) const;

protected:
    FrontEnd(std::unique_ptr<FrontEndImpl>&& impl);

private:
    std::unique_ptr<FrontEndImpl> m_impl;
    friend class FrontEndManagerImpl;
};

struct FrontEndShared {
    FrontEnd frontEnd;
    FrontEndShared(FrontEnd&& f): frontEnd(std::move(f)) {}
};

// Helper function for those who need shared pointers
inline std::shared_ptr<FrontEndShared> to_shared(FrontEnd&& f) {
    return std::make_shared<FrontEndShared>(std::move(f));
}

class FRONTEND_API FrontEndManager final
{
public:
    FrontEndManager();
    ~FrontEndManager();
    FrontEnd loadByFramework(const std::string& framework, FrontEndCapabilities fec = FrontEndCapabilities::FEC_DEFAULT);
    FrontEnd loadByModel(const std::string& path, FrontEndCapabilities fec = FrontEndCapabilities::FEC_DEFAULT);
    std::vector<std::string> availableFrontEnds() const;

    void registerFrontEnd(const std::string& name, FrontEndFactory creator);
private:
    std::unique_ptr<FrontEndManagerImpl> m_impl;
};

} // namespace frontend

} // namespace ngraph
