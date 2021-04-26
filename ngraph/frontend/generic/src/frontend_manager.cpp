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

#include "frontend_manager/frontend_manager.hpp"
#include "frontend_manager/ifrontend_manager.hpp"
#ifdef NGRAPH_PDPD_FRONTEND_ENABLE
#include "paddlepaddle_frontend/frontend.hpp"
#endif

namespace ngraph
{
    namespace frontend
    {

        #define FRONT_END_NOT_IMPLEMENTED(NAME) throw std::runtime_error(#NAME " is not implemented");

        #define FRONT_END_ASSERT(EXPRESSION) \
            { if (!(EXPRESSION)) throw "AssertionFailed"; }

        //-------- PlaceImpl -----------------------
        class PlaceImpl {
            IPlace::Ptr m_place;
            friend class InputModelImpl;
        public:
            PlaceImpl(const IPlace::Ptr& ptr): m_place(ptr) {}

            bool operator==(const PlaceImpl& other) const {
                return m_place == other.m_place;
            }

            std::vector<std::string> getNames () const
            {
                return m_place->getNames();
            }

            std::vector<Place> getConsumingOperations (int outputPortIndex) const
            {
                auto iplaces = m_place->getConsumingOperations(outputPortIndex);
                std::vector<Place> res;
                for (auto & iplace : iplaces) {
                    res.push_back({ std::make_shared<PlaceImpl>(iplace) });
                }
                return res;
            }

            Place getTargetTensor (int outputPortIndex) const
            {
                return { std::make_shared<PlaceImpl>(m_place->getTargetTensor(outputPortIndex)) };
            }

            Place getProducingOperation (int inputPortIndex) const
            {
                return { std::make_shared<PlaceImpl>(m_place->getProducingOperation(inputPortIndex)) };
            }

            Place getProducingPort () const
            {
                return { std::make_shared<PlaceImpl>(m_place->getProducingPort()) };
            }

            Place getInputPort (int inputPortIndex) const
            {
                return { std::make_shared<PlaceImpl>(m_place->getInputPort(inputPortIndex)) };
            }

            Place getInputPort (const std::string& intputName, int inputPortIndex) const
            {
                return { std::make_shared<PlaceImpl>(m_place->getInputPort(intputName, inputPortIndex)) };
            }

            Place getOutputPort (int outputPortIndex) const
            {
                return { std::make_shared<PlaceImpl>(m_place->getOutputPort(outputPortIndex)) };
            }

            Place getOutputPort (const std::string& outputName, int outputPortIndex) const
            {
                return { std::make_shared<PlaceImpl>(m_place->getOutputPort(outputName, outputPortIndex)) };
            }

            std::vector<Place> getConsumingPorts () const
            {
                auto iplaces = m_place->getConsumingPorts();
                std::vector<Place> res;
                for (auto & iplace : iplaces) {
                    res.push_back({ std::make_shared<PlaceImpl>(iplace) });
                }
                return res;
            }

            bool isInput () const
            {
                return m_place->isInput();
            }

            bool isOutput () const
            {
                return m_place->isOutput();
            }

            bool isEqual (const Place& another) const
            {
                return m_place->isEqual(another.m_impl->m_place);
            }

            bool isEqualData (const Place& another) const
            {
                return m_place->isEqualData(another.m_impl->m_place);
            }

            Place getSourceTensor (int inputPortIndex) const
            {
                return { std::make_shared<PlaceImpl>(m_place->getSourceTensor(inputPortIndex)) };
            }
        };

        //-------- Place -----------------------
        Place::Place(): m_impl(new PlaceImpl(std::make_shared<IPlace>())) {}
        Place::Place(const std::shared_ptr<PlaceImpl>& impl): m_impl(impl) {}
        Place::~Place() = default;

        std::vector<std::string> Place::getNames () const
        {
            return m_impl->getNames();
        }

        std::vector<Place> Place::getConsumingOperations (int outputPortIndex) const
        {
            return m_impl->getConsumingOperations(outputPortIndex);
        }

        Place Place::getTargetTensor (int outputPortIndex) const
        {
            return m_impl->getTargetTensor(outputPortIndex);
        }

        Place Place::getProducingOperation (int inputPortIndex) const
        {
            return m_impl->getProducingOperation(inputPortIndex);
        }

        Place Place::getProducingPort () const
        {
            return m_impl->getProducingPort();
        }

        Place Place::getInputPort (int inputPortIndex) const
        {
            return m_impl->getInputPort(inputPortIndex);
        }

        Place Place::getInputPort (const std::string& intputName, int inputPortIndex) const
        {
            return m_impl->getInputPort(intputName, inputPortIndex);
        }

        Place Place::getOutputPort (int outputPortIndex) const
        {
            return m_impl->getOutputPort( outputPortIndex);
        }

        Place Place::getOutputPort (const std::string& outputName, int outputPortIndex) const
        {
            return m_impl->getOutputPort(outputName, outputPortIndex);
        }

        std::vector<Place> Place::getConsumingPorts () const
        {
            return m_impl->getConsumingPorts();
        }

        bool Place::isInput () const
        {
            return m_impl->isInput();
        }

        bool Place::isOutput () const
        {
            return m_impl->isOutput();
        }

        bool Place::isEqual (const Place& another) const
        {
            return m_impl->isEqual(another);
        }

        bool Place::isEqualData (const Place& another) const
        {
            return m_impl->isEqualData(another);
        }

        Place Place::getSourceTensor (int inputPortIndex) const
        {
            return m_impl->getSourceTensor(inputPortIndex);
        }

        //-------- InputModelImpl -----------------------

        class InputModelImpl {
            IInputModel::Ptr m_inputModel;
            friend class FrontEndImpl;
        public:
            InputModelImpl(const IInputModel::Ptr& ptr): m_inputModel(ptr) {}

            std::vector<Place> getInputs () const {
                auto iplaces = m_inputModel->getInputs();
                std::vector<Place> res;
                for (auto & iplace : iplaces) {
                    res.push_back({ std::unique_ptr<PlaceImpl>(new PlaceImpl(iplace)) });
                }
                return res;
            }

            std::vector<Place> getOutputs () const
            {
                auto iplaces = m_inputModel->getOutputs();
                std::vector<Place> res;
                for (auto & iplace : iplaces) {
                    res.push_back({ std::make_shared<PlaceImpl>(iplace) });
                }
                return res;
            }

            Place getPlaceByTensorName (const std::string& tensorName) const
            {
                return { std::make_shared<PlaceImpl>(m_inputModel->getPlaceByTensorName(tensorName)) };
            }

            Place getPlaceByOperationName (const std::string& operationName) const
            {
                return { std::make_shared<PlaceImpl>(m_inputModel->getPlaceByOperationName(operationName)) };
            }

            Place getPlaceByOperationAndInputPort (const std::string& operationName, int inputPortIndex) const
            {
                return { std::make_shared<PlaceImpl>(m_inputModel->getPlaceByOperationAndInputPort(
                        operationName, inputPortIndex)) };
            }

            Place getPlaceByOperationAndOutputPort (const std::string& operationName, int outputPortIndex) const
            {
                return { std::make_shared<PlaceImpl>(m_inputModel->getPlaceByOperationAndOutputPort(
                        operationName, outputPortIndex)) };
            }

            void setNameForTensor (Place& tensor, const std::string& newName)
            {
                m_inputModel->setNameForTensor(tensor.m_impl->m_place, newName);
            }

            void addNameForTensor (Place& tensor, const std::string& newName)
            {
                m_inputModel->addNameForTensor(tensor.m_impl->m_place, newName);
            }

            void setNameForOperation (Place& operation, const std::string& newName)
            {
                m_inputModel->setNameForOperation(operation.m_impl->m_place, newName);
            }

            void freeNameForTensor (Place& tensor, const std::string& name)
            {
                m_inputModel->freeNameForTensor(tensor.m_impl->m_place, name);
            }

            void freeNameForOperation (Place& operation, const std::string& name)
            {
                m_inputModel->freeNameForTensor(operation.m_impl->m_place, name);
            }

            void setNameForDimension (Place& place, size_t shapeDimIndex, const std::string& dimName)
            {
                m_inputModel->setNameForDimension(place.m_impl->m_place, shapeDimIndex, dimName);
            }

            void cutAndAddNewInput (Place& place, const std::string& newNameOptional)
            {
                m_inputModel->cutAndAddNewInput(place.m_impl->m_place, newNameOptional);
            }

            void cutAndAddNewOutput (Place& place, const std::string& newNameOptional)
            {
                m_inputModel->cutAndAddNewOutput(place.m_impl->m_place, newNameOptional);
            }

            Place addOutput (Place& place)
            {
                return { std::make_shared<PlaceImpl>(m_inputModel->addOutput(place.m_impl->m_place)) };
            }

            void removeOutput (Place& place)
            {
                m_inputModel->removeOutput(place.m_impl->m_place);
            }

            void removeInput (Place& place)
            {
                m_inputModel->removeInput(place.m_impl->m_place);
            }

            void overrideAllOutputs (const std::vector<Place>& outputs)
            {
                std::vector<IPlace::Ptr> iplaces;
                for (const auto& place : outputs) {
                    iplaces.push_back(place.m_impl->m_place);
                }
                m_inputModel->overrideAllOutputs(iplaces);
            }

            void overrideAllInputs (const std::vector<Place>& inputs)
            {
                std::vector<IPlace::Ptr> iplaces;
                for (const auto& place : inputs) {
                    iplaces.push_back(place.m_impl->m_place);
                }
                m_inputModel->overrideAllInputs(iplaces);
            }

            void extractSubgraph (const std::vector<Place>& inputs, const std::vector<Place>& outputs)
            {
                std::vector<IPlace::Ptr> iinputs, ioutputs;
                for (const auto& place : inputs) {
                    iinputs.push_back(place.m_impl->m_place);
                }
                for (const auto& place : outputs) {
                    ioutputs.push_back(place.m_impl->m_place);
                }
                m_inputModel->extractSubgraph(iinputs, ioutputs);
            }

            // Setting tensor properties
            void setDefaultShape (Place& place, const ngraph::Shape& shape)
            {
                m_inputModel->setDefaultShape(place.m_impl->m_place, shape);
            }

            void setPartialShape (Place& place, const ngraph::PartialShape& shape)
            {
                m_inputModel->setPartialShape(place.m_impl->m_place, shape);
            }

            void setElementType (Place& place, const ngraph::element::Type& type)
            {
                m_inputModel->setElementType(place.m_impl->m_place, type);
            }

            void setTensorValue (Place& place, const void* value)
            {
                m_inputModel->setTensorValue(place.m_impl->m_place, value);
            }

            void setTensorPartialValue (Place& place, const void* minValue, const void* maxValue)
            {
                m_inputModel->setTensorPartialValue(place.m_impl->m_place, minValue, maxValue);
            }
        };

        //-------- InputModel -----------------------

        InputModel::InputModel(): m_impl(new InputModelImpl(std::make_shared<IInputModel>())) {}
        InputModel::InputModel(std::unique_ptr<InputModelImpl>&& impl): m_impl(std::move(impl)) {}
        InputModel::InputModel(InputModel &&other) = default;
        InputModel& InputModel::operator=(InputModel &&other) = default;
        InputModel::~InputModel() = default;

        std::vector<Place> InputModel::getInputs () const
        {
            return m_impl->getInputs();
        }

        std::vector<Place> InputModel::getOutputs () const
        {
            return m_impl->getOutputs();
        }

        Place InputModel::getPlaceByTensorName (const std::string& tensorName) const
        {
            return m_impl->getPlaceByTensorName(tensorName);
        }

        Place InputModel::getPlaceByOperationName (const std::string& operationName)
        {
            return m_impl->getPlaceByOperationName(operationName);
        }

        Place InputModel::getPlaceByOperationAndInputPort (const std::string& operationName, int inputPortIndex)
        {
            return m_impl->getPlaceByOperationAndInputPort(operationName, inputPortIndex);
        }

        Place InputModel::getPlaceByOperationAndOutputPort (const std::string& operationName, int outputPortIndex)
        {
            return m_impl->getPlaceByOperationAndOutputPort(operationName, outputPortIndex);
        }

        void InputModel::setNameForTensor (Place& tensor, const std::string& newName)
        {
            m_impl->setNameForTensor(tensor, newName);
        }

        void InputModel::addNameForTensor (Place& tensor, const std::string& newName)
        {
            m_impl->addNameForTensor(tensor, newName);
        }

        void InputModel::setNameForOperation (Place& operation, const std::string& newName)
        {
            m_impl->setNameForOperation(operation, newName);
        }

        void InputModel::freeNameForTensor (Place& tensor, const std::string& name)
        {
            m_impl->freeNameForTensor(tensor, name);
        }

        void InputModel::freeNameForOperation (Place& operation, const std::string& name)
        {
            m_impl->freeNameForOperation(operation, name);
        }

        void InputModel::setNameForDimension (Place& place, size_t shapeDimIndex, const std::string& dimName)
        {
            m_impl->setNameForDimension(place, shapeDimIndex, dimName);
        }

        void InputModel::cutAndAddNewInput (Place& place, const std::string& newNameOptional)
        {
            m_impl->cutAndAddNewInput(place, newNameOptional);
        }

        void InputModel::cutAndAddNewOutput (Place& place, const std::string& newNameOptional)
        {
            m_impl->cutAndAddNewOutput(place, newNameOptional);
        }

        Place InputModel::addOutput (Place& place)
        {
            return m_impl->addOutput(place);
        }

        void InputModel::removeOutput (Place& place)
        {
            m_impl->removeOutput(place);
        }

        void InputModel::removeInput (Place& place)
        {
            m_impl->removeInput(place);
        }

        void InputModel::overrideAllOutputs (const std::vector<Place>& outputs)
        {
            m_impl->overrideAllOutputs(outputs);
        }

        void InputModel::overrideAllInputs (const std::vector<Place>& inputs)
        {
            m_impl->overrideAllInputs(inputs);
        }

        void InputModel::extractSubgraph (const std::vector<Place>& inputs, const std::vector<Place>& outputs)
        {
            m_impl->extractSubgraph(inputs, outputs);
        }

        void InputModel::setDefaultShape (Place& place, const ngraph::Shape& shape)
        {
            m_impl->setDefaultShape(place, shape);
        }

        void InputModel::setPartialShape (Place& place, const ngraph::PartialShape& shape)
        {
            m_impl->setPartialShape(place, shape);
        }

        void InputModel::setElementType (Place& place, const ngraph::element::Type& type)
        {
            m_impl->setElementType(place, type);
        }

        void InputModel::setTensorValue (Place& place, const void* value)
        {
            m_impl->setTensorValue(place, value);
        }

        void InputModel::setTensorPartialValue (Place& place, const void* minValue, const void* maxValue)
        {
            m_impl->setTensorPartialValue(place, minValue, maxValue);
        }

        //-------- FrontEndImpl -----------------------

        class FrontEndImpl {
            IFrontEnd::Ptr m_frontEnd;
        public:
            FrontEndImpl(const IFrontEnd::Ptr& ptr): m_frontEnd(ptr) {}

            InputModel loadFromFile (const std::string& path) const
            {
                return { std::unique_ptr<InputModelImpl>(
                        new InputModelImpl(m_frontEnd->loadFromFile(path))) };
            }

            InputModel loadFromFiles (const std::vector<std::string>& paths) const
            {
                return { std::unique_ptr<InputModelImpl>(
                        new InputModelImpl(m_frontEnd->loadFromFiles(paths))) };
            }

            InputModel loadFromMemory (const void* model) const
            {
                return { std::unique_ptr<InputModelImpl>(
                        new InputModelImpl(m_frontEnd->loadFromMemory(model))) };
            }

            InputModel loadFromMemoryFragments (const std::vector<const void*>& modelParts) const
            {
                return { std::unique_ptr<InputModelImpl>(
                        new InputModelImpl(m_frontEnd->loadFromMemoryFragments(modelParts))) };
            }

            InputModel loadFromStream (std::istream& path) const
            {
                return { std::unique_ptr<InputModelImpl>(
                        new InputModelImpl(m_frontEnd->loadFromStream(path))) };
            }

            InputModel loadFromStreams (const std::vector<std::istream*>& paths) const
            {
                return { std::unique_ptr<InputModelImpl>(
                        new InputModelImpl(m_frontEnd->loadFromStreams(paths))) };
            }

            std::shared_ptr<ngraph::Function> convert (const InputModel& model) const
            {
                return m_frontEnd->convert(model.m_impl->m_inputModel);
            }

            std::shared_ptr<ngraph::Function> convert (std::shared_ptr<ngraph::Function> func) const
            {
                return m_frontEnd->convert(func);
            }

            std::shared_ptr<ngraph::Function> convertPartially (const InputModel& model) const
            {
                return m_frontEnd->convertPartially(model.m_impl->m_inputModel);
            }

            std::shared_ptr<ngraph::Function> decode (const InputModel& model) const
            {
                return m_frontEnd->decode(model.m_impl->m_inputModel);
            }

            void normalize (std::shared_ptr<ngraph::Function> function) const
            {
                m_frontEnd->normalize(function);
            }

        };

        //-------- FrontEnd -----------------------

        FrontEnd::FrontEnd(): m_impl(new FrontEndImpl(std::make_shared<IFrontEnd>())) {}
        FrontEnd::FrontEnd(std::unique_ptr<FrontEndImpl>&& impl): m_impl(std::move(impl)) {}
        FrontEnd::FrontEnd(FrontEnd&& other) = default;
        FrontEnd& FrontEnd::operator=(FrontEnd&& other) = default;

        FrontEnd::~FrontEnd() = default;

        InputModel FrontEnd::loadFromFile (const std::string& path) const
        {
            return m_impl->loadFromFile(path);
        }

        InputModel FrontEnd::loadFromFiles (const std::vector<std::string>& paths) const
        {
            return m_impl->loadFromFiles(paths);
        }

        InputModel FrontEnd::loadFromMemory (const void* model) const
        {
            return m_impl->loadFromMemory(model);
        }

        InputModel FrontEnd::loadFromMemoryFragments (const std::vector<const void*>& modelParts) const
        {
            return m_impl->loadFromMemoryFragments(modelParts);
        }

        InputModel FrontEnd::loadFromStream (std::istream& path) const
        {
            return m_impl->loadFromStream(path);
        }

        InputModel FrontEnd::loadFromStreams (const std::vector<std::istream*>& paths) const
        {
            return m_impl->loadFromStreams(paths);
        }

        std::shared_ptr<ngraph::Function> FrontEnd::convert (const InputModel& model) const
        {
            return m_impl->convert(model);
        }

        std::shared_ptr<ngraph::Function> FrontEnd::convert (std::shared_ptr<ngraph::Function> f) const
        {
            return m_impl->convert(f);
        }

        std::shared_ptr<ngraph::Function> FrontEnd::convertPartially (const InputModel& model) const
        {
            return m_impl->convertPartially(model);
        }

        std::shared_ptr<ngraph::Function> FrontEnd::decode (const InputModel& model) const
        {
            return m_impl->decode(model);
        }

        void FrontEnd::normalize (std::shared_ptr<ngraph::Function> function) const
        {
            m_impl->normalize(function);
        }

        //////////////////////////////////////////////////////////////
        class FrontEndManagerImpl
        {
            std::map<std::string, FrontEndManager::FrontEndFactory> m_factories;

            void registerDefault() {
#if defined(NGRAPH_ONNX_IMPORT_ENABLE) && defined(NGRAPH_ONNX_EDITOR_ENABLE)
                registerFrontEnd("onnx", [](FrontEndCapabilities){return std::make_shared<FrontEndONNX>();});
#endif
#ifdef NGRAPH_PDPD_FRONTEND_ENABLE
                registerFrontEnd("pdpd", [](FrontEndCapabilities){return std::make_shared<FrontEndPDPD>();});
#endif
#ifdef NGRAPH_TF_FRONTEND_ENABLE
                registerFrontEnd("tf", [](FrontEndCapabilities){return std::make_shared<FrontEndTensorflow>();});
#endif
            }
        public:
            FrontEndManagerImpl() {
                registerDefault();
            }
            ~FrontEndManagerImpl() = default;

            FrontEnd loadByFramework(const std::string& framework, FrontEndCapabilities fec) {
                FRONT_END_ASSERT(m_factories.count(framework))
                auto ife = m_factories[framework](fec);
                return { std::unique_ptr<FrontEndImpl>(new FrontEndImpl(ife)) };
            }

            std::vector<std::string> availableFrontEnds() const {
                std::vector<std::string> keys;

                std::transform(m_factories.begin(), m_factories.end(),
                               std::back_inserter(keys),
                               [](const std::pair<std::string, FrontEndManager::FrontEndFactory>& item) {
                                   return item.first;
                               });
                return keys;
            }

            FrontEnd loadByModel (const std::string& path, FrontEndCapabilities fec)
            {
                throw std::runtime_error("LoadByModel is not yet implemented");
            }

            void registerFrontEnd(const std::string& name, FrontEndManager::FrontEndFactory creator) {
                m_factories.insert({name, creator});
            }
        };

        FrontEndManager::FrontEndManager(): m_impl(new FrontEndManagerImpl()) {
        }

        FrontEndManager::~FrontEndManager() = default;

        FrontEnd FrontEndManager::loadByFramework(const std::string& framework, FrontEndCapabilities fec)
        {
            return m_impl->loadByFramework(framework, fec);
        }

        FrontEnd FrontEndManager::loadByModel(const std::string& path, FrontEndCapabilities fec)
        {
            return m_impl->loadByModel(path, fec);
        }

        std::vector<std::string> FrontEndManager::availableFrontEnds() const
        {
            return m_impl->availableFrontEnds();
        }

        void FrontEndManager::registerFrontEnd(const std::string& name, FrontEndFactory creator)
        {
            m_impl->registerFrontEnd(name, creator);
        }

    } // namespace frontend

} // namespace ngraph
