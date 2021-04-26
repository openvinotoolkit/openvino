// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/except.hpp>
#ifdef NGRAPH_ONNX_IMPORT_ENABLE
#include "onnx_import/onnx.hpp"
#endif

#ifdef NGRAPH_ONNX_EDITOR_ENABLE
#include "onnx_editor/editor.hpp"
#endif

#ifdef NGRAPH_TF_FRONTEND_ENABLE
#include "tensorflow_frontend/tensorflow.hpp"
#endif

#include "frontend_manager/ifrontend_manager.hpp"
#ifdef NGRAPH_PDPD_FRONTEND_ENABLE
#include "paddlepaddle_frontend/frontend.hpp"
#endif

namespace ngraph
{
    namespace frontend
    {

        #define FRONT_END_NOT_IMPLEMENTED(NAME) throw std::runtime_error(#NAME " is not implemented for this FrontEnd class");
        #define FRONT_END_ASSERT(EXPRESSION) \
            { if (!(EXPRESSION)) throw "AssertionFailed"; }

        //--------------- IInputModel -------------------
        IInputModel::IInputModel() = default;
        IInputModel::~IInputModel() = default;

        std::vector<IPlace::Ptr> IInputModel::getInputs () const
        {
            FRONT_END_NOT_IMPLEMENTED(getInputs);
        }

        std::vector<IPlace::Ptr> IInputModel::getOutputs () const
        {
            FRONT_END_NOT_IMPLEMENTED(getOutputs);
        }

        IPlace::Ptr IInputModel::getPlaceByTensorName (const std::string& tensorName) const
        {
            FRONT_END_NOT_IMPLEMENTED(getPlaceByTensorName);
        }

        IPlace::Ptr IInputModel::getPlaceByOperationName (const std::string& operationName)
        {
            FRONT_END_NOT_IMPLEMENTED(getPlaceByOperationName);
        }

        IPlace::Ptr IInputModel::getPlaceByOperationAndInputPort (const std::string& operationName, int inputPortIndex)
        {
            FRONT_END_NOT_IMPLEMENTED(getPlaceByOperationAndInputPort);
        }

        IPlace::Ptr IInputModel::getPlaceByOperationAndOutputPort (const std::string& operationName, int outputPortIndex)
        {
            FRONT_END_NOT_IMPLEMENTED(getPlaceByOperationAndOutputPort);
        }

        void IInputModel::setNameForTensor (IPlace::Ptr tensor, const std::string& newName)
        {
            FRONT_END_NOT_IMPLEMENTED(setNameForTensor);
        }

        void IInputModel::addNameForTensor (IPlace::Ptr tensor, const std::string& newName)
        {
            FRONT_END_NOT_IMPLEMENTED(addNameForTensor);
        }

        void IInputModel::setNameForOperation (IPlace::Ptr operation, const std::string& newName)
        {
            FRONT_END_NOT_IMPLEMENTED(setNameForOperation);
        }

        void IInputModel::freeNameForTensor (IPlace::Ptr tensor, const std::string& name)
        {
            FRONT_END_NOT_IMPLEMENTED(freeNameForTensor);
        }

        void IInputModel::freeNameForOperation (IPlace::Ptr operation, const std::string& name)
        {
            FRONT_END_NOT_IMPLEMENTED(freeNameForOperation);
        }

        void IInputModel::setNameForDimension (IPlace::Ptr IPlace, size_t shapeDimIndex, const std::string& dimName)
        {
            FRONT_END_NOT_IMPLEMENTED(setNameForDimension);
        }

        void IInputModel::cutAndAddNewInput (IPlace::Ptr IPlace, const std::string& newNameOptional)
        {
            FRONT_END_NOT_IMPLEMENTED(cutAndAddNewInput);
        }

        void IInputModel::cutAndAddNewOutput (IPlace::Ptr IPlace, const std::string& newNameOptional)
        {
            FRONT_END_NOT_IMPLEMENTED(cutAndAddNewOutput);
        }

        IPlace::Ptr IInputModel::addOutput (IPlace::Ptr IPlace)
        {
            FRONT_END_NOT_IMPLEMENTED(addOutput);
        }

        void IInputModel::removeOutput (IPlace::Ptr IPlace)
        {
            FRONT_END_NOT_IMPLEMENTED(removeOutput);
        }

        void IInputModel::removeInput (IPlace::Ptr IPlace)
        {
            FRONT_END_NOT_IMPLEMENTED(removeInput);
        }

        void IInputModel::overrideAllOutputs (const std::vector<IPlace::Ptr>& outputs)
        {
            FRONT_END_NOT_IMPLEMENTED(overrideAllOutputs);
        }

        void IInputModel::overrideAllInputs (const std::vector<IPlace::Ptr>& inputs)
        {
            FRONT_END_NOT_IMPLEMENTED(overrideAllInputs);
        }

        void IInputModel::extractSubgraph (const std::vector<IPlace::Ptr>& inputs, const std::vector<IPlace::Ptr>& outputs)
        {
            FRONT_END_NOT_IMPLEMENTED(extractSubgraph);
        }

        // Setting tensor properties
        void IInputModel::setDefaultShape (IPlace::Ptr IPlace, const ngraph::Shape&)
        {
            FRONT_END_NOT_IMPLEMENTED(setDefaultShape);
        }

        void IInputModel::setPartialShape (IPlace::Ptr IPlace, const ngraph::PartialShape&)
        {
            FRONT_END_NOT_IMPLEMENTED(setPartialShape);
        }

        void IInputModel::setElementType (IPlace::Ptr IPlace, const ngraph::element::Type&)
        {
            FRONT_END_NOT_IMPLEMENTED(setElementType);
        }

        void IInputModel::setTensorValue (IPlace::Ptr IPlace, const void*)
        {
            FRONT_END_NOT_IMPLEMENTED(setTensorValue);
        }

        void IInputModel::setTensorPartialValue (IPlace::Ptr IPlace, const void* minValue, const void* maxValue)
        {
            FRONT_END_NOT_IMPLEMENTED(setTensorPartialValue);
        }

        //--------------- IPlace -------------------
        IPlace::IPlace() = default;
        IPlace::~IPlace() = default;

        std::vector<std::string> IPlace::getNames () const
        {
            FRONT_END_NOT_IMPLEMENTED(getNames);
        }

        std::vector<IPlace::Ptr> IPlace::getConsumingOperations (int outputPortIndex) const
        {
            FRONT_END_NOT_IMPLEMENTED(getConsumingOperations);
        }

        IPlace::Ptr IPlace::getTargetTensor (int outputPortIndex) const
        {
            FRONT_END_NOT_IMPLEMENTED(getTargetTensor);
        }

        IPlace::Ptr IPlace::getProducingOperation (int inputPortIndex) const
        {
            FRONT_END_NOT_IMPLEMENTED(getProducingOperation);
        }

        IPlace::Ptr IPlace::getProducingPort () const
        {
            FRONT_END_NOT_IMPLEMENTED(getProducingPort);
        }

        IPlace::Ptr IPlace::getInputPort (int inputPortIndex) const
        {
            FRONT_END_NOT_IMPLEMENTED(getInputPort);
        }

        IPlace::Ptr IPlace::getInputPort (const std::string& intputName, int inputPortIndex) const
        {
            FRONT_END_NOT_IMPLEMENTED(getInputPort);
        }

        IPlace::Ptr IPlace::getOutputPort (int outputPortIndex) const
        {
            FRONT_END_NOT_IMPLEMENTED(getOutputPort);
        }

        IPlace::Ptr IPlace::getOutputPort (const std::string& outputName, int outputPortIndex) const
        {
            FRONT_END_NOT_IMPLEMENTED(getOutputPort);
        }

        std::vector<IPlace::Ptr> IPlace::getConsumingPorts () const
        {
            FRONT_END_NOT_IMPLEMENTED(getConsumingPorts);
        }

        bool IPlace::isInput () const
        {
            FRONT_END_NOT_IMPLEMENTED(isInput);
        }

        bool IPlace::isOutput () const
        {
            FRONT_END_NOT_IMPLEMENTED(isOutput);
        }

        bool IPlace::isEqual (Ptr another) const
        {
            FRONT_END_NOT_IMPLEMENTED(isEqual);
        }

        bool IPlace::isEqualData (Ptr another) const
        {
            FRONT_END_NOT_IMPLEMENTED(isEqualData);
        }

        IPlace::Ptr IPlace::getSourceTensor (int inputPortIndex) const
        {
            FRONT_END_NOT_IMPLEMENTED(getSourceTensor);
        }

        //--------------- IFrontEnd -------------------------
        IFrontEnd::IFrontEnd() = default;
        IFrontEnd::~IFrontEnd() = default;

        IInputModel::Ptr IFrontEnd::loadFromFile (const std::string& path) const
        {
            FRONT_END_NOT_IMPLEMENTED(loadFromFile);
        }

        IInputModel::Ptr IFrontEnd::loadFromFiles (const std::vector<std::string>& paths) const
        {
            FRONT_END_NOT_IMPLEMENTED(loadFromFiles);
        }

        IInputModel::Ptr IFrontEnd::loadFromMemory (const void* model) const
        {
            FRONT_END_NOT_IMPLEMENTED(loadFromMemory);
        }

        IInputModel::Ptr IFrontEnd::loadFromMemoryFragments (const std::vector<const void*>& modelParts) const
        {
            FRONT_END_NOT_IMPLEMENTED(loadFromMemoryFragments);
        }

        IInputModel::Ptr IFrontEnd::loadFromStream (std::istream& path) const
        {
            FRONT_END_NOT_IMPLEMENTED(loadFromStream);
        }

        IInputModel::Ptr IFrontEnd::loadFromStreams (const std::vector<std::istream*>& paths) const
        {
            FRONT_END_NOT_IMPLEMENTED(loadFromStreams);
        }

        std::shared_ptr<ngraph::Function> IFrontEnd::convert (IInputModel::Ptr model) const
        {
            FRONT_END_NOT_IMPLEMENTED(convert);
        }

        std::shared_ptr<ngraph::Function> IFrontEnd::convert (std::shared_ptr<ngraph::Function>) const
        {
            FRONT_END_NOT_IMPLEMENTED(convert);
        }

        std::shared_ptr<ngraph::Function> IFrontEnd::convertPartially (IInputModel::Ptr model) const
        {
            FRONT_END_NOT_IMPLEMENTED(convertPartially);
        }

        std::shared_ptr<ngraph::Function> IFrontEnd::decode (IInputModel::Ptr model) const
        {
            FRONT_END_NOT_IMPLEMENTED(convertDecodingOnly);
        }

        void IFrontEnd::normalize (std::shared_ptr<ngraph::Function> function) const
        {
            FRONT_END_NOT_IMPLEMENTED(normalize);
        }

    } // namespace frontend

} // namespace ngraph
