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
#include "paddlepaddle_frontend/frontend.hpp"


namespace ngraph
{
    namespace frontend
    {

        #define FRONT_END_NOT_IMPLEMENTED(NAME) throw #NAME " is not implemented for this FrontEnd class";
        #define FRONT_END_ASSERT(EXPRESSION) \
            { if (!(EXPRESSION)) throw "AssertionFailed"; }

        std::vector<Place::Ptr> InputModel::getInputs () const
        {
            FRONT_END_NOT_IMPLEMENTED(getInputs);
        }

        std::vector<Place::Ptr> InputModel::getOutputs () const
        {
            FRONT_END_NOT_IMPLEMENTED(getOutputs);
        }

        Place::Ptr InputModel::getPlaceByTensorName (const std::string& tensorName)
        {
            FRONT_END_NOT_IMPLEMENTED(getPlaceByTensorName);
        }

        Place::Ptr InputModel::getPlaceByOperationName (const std::string& operationName)
        {
            FRONT_END_NOT_IMPLEMENTED(getPlaceByOperationName);
        }

        Place::Ptr InputModel::getPlaceByOperationAndInputPort (const std::string& operationName, int inputPortIndex)
        {
            FRONT_END_NOT_IMPLEMENTED(getPlaceByOperationAndInputPort);
        }

        Place::Ptr InputModel::getPlaceByOperationAndOutputPort (const std::string& operationName, int outputPortIndex)
        {
            FRONT_END_NOT_IMPLEMENTED(getPlaceByOperationAndOutputPort);
        }

        void InputModel::setNameForTensor (Place::Ptr tensor, const std::string& newName)
        {
            FRONT_END_NOT_IMPLEMENTED(setNameForTensor);
        }

        void InputModel::addNameForTensor (Place::Ptr tensor, const std::string& newName)
        {
            FRONT_END_NOT_IMPLEMENTED(addNameForTensor);
        }

        void InputModel::setNameForOperation (Place::Ptr operation, const std::string& newName)
        {
            FRONT_END_NOT_IMPLEMENTED(setNameForOperation);
        }

        void InputModel::freeNameForTensor (const std::string& name)
        {
            FRONT_END_NOT_IMPLEMENTED(freeNameForTensor);
        }

        void InputModel::freeNameForOperation (const std::string& name)
        {
            FRONT_END_NOT_IMPLEMENTED(freeNameForOperation);
        }

        void InputModel::setNameForDimension (Place::Ptr place, size_t shapeDimIndex, const std::string& dimName)
        {
            FRONT_END_NOT_IMPLEMENTED(setNameForDimension);
        }

        void InputModel::cutAndAddNewInput (Place::Ptr place, const std::string& newNameOptional)
        {
            FRONT_END_NOT_IMPLEMENTED(cutAndAddNewInput);
        }

        void InputModel::cutAndAddNewOutput (Place::Ptr place, const std::string& newNameOptional)
        {
            FRONT_END_NOT_IMPLEMENTED(cutAndAddNewOutput);
        }

        Place::Ptr InputModel::addOutput (Place::Ptr place)
        {
            FRONT_END_NOT_IMPLEMENTED(addOutput);
        }

        void InputModel::removeOutput (Place::Ptr place)
        {
            FRONT_END_NOT_IMPLEMENTED(removeOutput);
        }

        void InputModel::removeInput (Place::Ptr place)
        {
            FRONT_END_NOT_IMPLEMENTED(removeInput);
        }

        void InputModel::overrideAllOutputs (const std::vector<Place::Ptr>& outputs)
        {
            FRONT_END_NOT_IMPLEMENTED(overrideAllOutputs);
        }

        void InputModel::overrideAllInputs (const std::vector<Place::Ptr>& inputs)
        {
            FRONT_END_NOT_IMPLEMENTED(overrideAllInputs);
        }

        void InputModel::extractSubgraph (const std::vector<Place::Ptr>& inputs, const std::vector<Place::Ptr>& outputs)
        {
            FRONT_END_NOT_IMPLEMENTED(extractSubgraph);
        }

        // Setting tensor properties
        void InputModel::setDefaultShape (Place::Ptr place, const ngraph::Shape&)
        {
            FRONT_END_NOT_IMPLEMENTED(setDefaultShape);
        }

        void InputModel::setPartialShape (Place::Ptr place, const ngraph::PartialShape&)
        {
            FRONT_END_NOT_IMPLEMENTED(setPartialShape);
        }

        void InputModel::setElementType (Place::Ptr place, const ngraph::element::Type&)
        {
            FRONT_END_NOT_IMPLEMENTED(setElementType);
        }

        void InputModel::setTensorValue (Place::Ptr place, const void*)
        {
            FRONT_END_NOT_IMPLEMENTED(setTensorValue);
        }

        void InputModel::setTensorPartialValue (Place::Ptr place, const void* minValue, const void* maxValue)
        {
            FRONT_END_NOT_IMPLEMENTED(setTensorPartialValue);
        }

        std::vector<std::string> Place::getNames () const
        {
            FRONT_END_NOT_IMPLEMENTED(getNames);
        }

        std::vector<Place::Ptr> Place::getConsumingOperations (int outputPortIndex) const
        {
            FRONT_END_NOT_IMPLEMENTED(getConsumingOperations);
        }

        Place::Ptr Place::getTargetTensor (int outputPortIndex) const
        {
            FRONT_END_NOT_IMPLEMENTED(getTargetTensor);
        }

        Place::Ptr Place::getProducingOperation (int inputPortIndex) const
        {
            FRONT_END_NOT_IMPLEMENTED(getProducingOperation);
        }

        Place::Ptr Place::getProducingPort () const
        {
            FRONT_END_NOT_IMPLEMENTED(getProducingPort);
        }

        Place::Ptr Place::getInputPort (int inputPortIndex) const
        {
            FRONT_END_NOT_IMPLEMENTED(getInputPort);
        }

        Place::Ptr Place::getOutputPort (int outputPortIndex) const
        {
            FRONT_END_NOT_IMPLEMENTED(getOutputPort);
        }

        std::vector<Place::Ptr> Place::getConsumingPorts () const
        {
            FRONT_END_NOT_IMPLEMENTED(getConsumingPorts);
        }

        bool Place::isInput () const
        {
            FRONT_END_NOT_IMPLEMENTED(isInput);
        }

        bool Place::isOutput () const
        {
            FRONT_END_NOT_IMPLEMENTED(isOutput);
        }

        bool Place::isEqual (Ptr another) const
        {
            FRONT_END_NOT_IMPLEMENTED(isEqual);
        }

        bool Place::isEqualData (Ptr another) const
        {
            FRONT_END_NOT_IMPLEMENTED(isEqualData);
        }

        Place::Ptr Place::getSourceTensor (int inputPortIndex) const
        {
            FRONT_END_NOT_IMPLEMENTED(getSourceTensor);
        }

#if defined(NGRAPH_ONNX_IMPORT_ENABLE) && defined(NGRAPH_ONNX_EDITOR_ENABLE)
        class PlaceInputEdgeONNX : public Place
        {
        public:

            onnx_editor::InputEdge edge;

            PlaceInputEdgeONNX (const std::string& _sourceTensorName, int _operationNodeIndex) :
                    edge(_operationNodeIndex, _sourceTensorName)
            {}
        };

        class PlaceOutputEdgeONNX : public Place
        {
        public:

            onnx_editor::OutputEdge edge;

            PlaceOutputEdgeONNX (int _operationNodeIndex, const std::string& _targetTensorName) :
                    edge(_operationNodeIndex, _targetTensorName)
            {}


        };

        class InputModelONNX;

        class PlaceTensorONNX : public Place
        {
            std::string tensorName;
            const InputModelONNX* model;

        public:

            PlaceTensorONNX (const std::string& _tensorName, const InputModelONNX* _model) : tensorName(_tensorName), model(_model){}

            virtual std::vector<std::string> getNames () const override
            {
                return std::vector<std::string>(1, tensorName);
            }

            virtual std::vector<Place::Ptr> getConsumingPorts () const override;
            virtual Place::Ptr getProducingPort () const override;
        };

        class InputModelONNX : public InputModel
        {

        public:
            // TODO: Move to private
            onnx_editor::ONNXModelEditor editor;

            InputModelONNX (const std::string& model_path) : editor(model_path) {}

            Place::Ptr getPlaceByTensorName (const std::string& tensorName) override
            {
                if(!editor.validate_tensor_name(tensorName)) {
                    std::cerr << " [ ERROR ] Node with name " << tensorName << " is not valid for a given model\n";
                    return nullptr;
                }
                return std::make_shared<PlaceTensorONNX>(tensorName, this);
            }

            std::vector<Place::Ptr> getInputs () const override {
                auto inputs = editor.model_inputs();
                std::vector<Place::Ptr> outputs;
                outputs.reserve(inputs.size());
                for(auto const& input: inputs)
                {
                    outputs.push_back(std::make_shared<PlaceTensorONNX>(input, this));
                }
                return outputs;
            }

            void setPartialShape (Place::Ptr place, const ngraph::PartialShape& shape) override
            {
                std::map<std::string, ngraph::PartialShape> m;
                m[place->getNames()[0]] = shape;
                editor.set_input_shapes(m);
            }

            void extractSubgraph (const std::vector<Place::Ptr>& inputs, const std::vector<Place::Ptr>& outputs)
            {
                std::cerr << "\nTTTTTTTTTTTTT\n";
                std::cerr << "inputs.size() = " << inputs.size() << "\n";
                // Current implementation is limited by tensor places only, each input tensor should be consumed by a single op only
                // TODO Extend to non tensor inputs/outputs and remove other limitations
                std::vector<onnx_editor::InputEdge> onnx_inputs;
                onnx_inputs.reserve(inputs.size());
                for(const auto& input: inputs)
                {
                    std::cerr << "[] = " << input.get() << "\n";
                    // TODO check if input is a tensor
                    auto inputPorts = input->getConsumingPorts();
                    std::cerr << "{1}\n";
                    NGRAPH_CHECK(inputPorts.size() == 1);
                    std::cerr << "{2}\n";
                    auto inputPort = inputPorts.front();
                    std::cerr << "{3}\n";
                    auto onnxInputEdge = std::dynamic_pointer_cast<PlaceInputEdgeONNX>(inputPort);
                    NGRAPH_CHECK(onnxInputEdge);
                    onnx_inputs.push_back(onnxInputEdge->edge);
                }
                std::cerr << "{4}\n";

                std::vector<onnx_editor::OutputEdge> onnx_outputs;
                onnx_outputs.reserve(outputs.size());
                for(const auto& output: outputs)
                {
                    // TODO check if output is a tensor
                    auto outputPort = output->getProducingPort();
                    auto onnxOutputEdge = std::dynamic_pointer_cast<PlaceOutputEdgeONNX>(outputPort);
                    NGRAPH_CHECK(onnxOutputEdge);
                    onnx_outputs.push_back(onnxOutputEdge->edge);
                }

                editor.cut_graph_fragment(onnx_inputs, onnx_outputs);
            }
        };

        std::vector<Place::Ptr> PlaceTensorONNX::getConsumingPorts () const
        {
            // ONNX specific code to find a node indices for all operations that consume a given tensor name
            std::vector<Place::Ptr> result;
            for(int i: model->editor.find_consumeing_node_idxs(tensorName))
            {
                result.push_back(std::make_shared<PlaceInputEdgeONNX>(tensorName, i));
            }
            return result;
        }

        Place::Ptr PlaceTensorONNX::getProducingPort () const
        {
            return std::make_shared<PlaceOutputEdgeONNX>(model->editor.find_producing_node_idx(tensorName), tensorName);
        }

        class FrontEndONNX : public FrontEnd
        {
        public:

            FrontEndONNX ()
            {
            }

            virtual InputModel::Ptr loadFromFile (const std::string& path) const override
            {
                return std::make_shared<InputModelONNX>(path);
            }

            virtual std::shared_ptr<ngraph::Function> convert (InputModel::Ptr model) const override
            {
                return onnx_import::import_onnx_model(std::dynamic_pointer_cast<InputModelONNX>(model)->editor.model_path());
            }
        };
#endif
        InputModel::Ptr FrontEnd::loadFromFile (const std::string& paths) const
        {
            FRONT_END_NOT_IMPLEMENTED(loadFromFile);
        }

        InputModel::Ptr FrontEnd::loadFromFiles (const std::vector<std::string>& paths) const
        {
            FRONT_END_NOT_IMPLEMENTED(loadFromFiles);
        }

        InputModel::Ptr FrontEnd::loadFromMemory (const void* model) const
        {
            FRONT_END_NOT_IMPLEMENTED(loadFromMemory);
        }

        InputModel::Ptr FrontEnd::loadFromMemoryFragments (const std::vector<const void*>& modelParts) const
        {
            FRONT_END_NOT_IMPLEMENTED(loadFromMemoryFragments);
        }

        InputModel::Ptr FrontEnd::loadFromStream (std::istream& path) const
        {
            FRONT_END_NOT_IMPLEMENTED(loadFromStream);
        }

        InputModel::Ptr FrontEnd::loadFromStreams (const std::vector<std::istream*>& paths) const
        {
            FRONT_END_NOT_IMPLEMENTED(loadFromStreams);
        }

        std::shared_ptr<ngraph::Function> FrontEnd::convert (InputModel::Ptr model) const
        {
            FRONT_END_NOT_IMPLEMENTED(convert);
        }

        std::shared_ptr<ngraph::Function> FrontEnd::convert (std::shared_ptr<ngraph::Function>) const
        {
            FRONT_END_NOT_IMPLEMENTED(convert);
        }

        std::shared_ptr<ngraph::Function> FrontEnd::convertPartially (InputModel::Ptr model) const
        {
            FRONT_END_NOT_IMPLEMENTED(convertPartially);
        }

        std::shared_ptr<ngraph::Function> FrontEnd::decode (InputModel::Ptr model) const
        {
            FRONT_END_NOT_IMPLEMENTED(convertDecodingOnly);
        }

        void FrontEnd::normalize (std::shared_ptr<ngraph::Function> function) const
        {
            FRONT_END_NOT_IMPLEMENTED(normalize);
        }

        //////////////////////////////////////////////////////////////
        class FrontEndManager::Impl
        {
            std::map<std::string, FrontEndFactory> m_factories;

            void registerDefault() {
#if defined(NGRAPH_ONNX_IMPORT_ENABLE) && defined(NGRAPH_ONNX_EDITOR_ENABLE)
                registerFrontEnd("onnx", [](FrontEndCapabilities){return std::make_shared<FrontEndONNX>();});
#endif
                registerFrontEnd("pdpd", [](FrontEndCapabilities){return std::make_shared<FrontEndPDPD>();});
#ifdef NGRAPH_TF_FRONTEND_ENABLE
                registerFrontEnd("tf", [](FrontEndCapabilities){return std::make_shared<FrontEndTensorflow>();});
#endif
            }
        public:
            Impl() {
                registerDefault();
            }
            ~Impl() = default;
            FrontEnd::Ptr loadByFramework(const std::string& framework, FrontEndCapabilities fec) {
                FRONT_END_ASSERT(m_factories.count(framework))
                return m_factories[framework](fec);
            }

            std::vector<std::string> availableFrontEnds() const {
                std::vector<std::string> keys;

                std::transform(m_factories.begin(), m_factories.end(),
                               std::back_inserter(keys),
                               [](const std::pair<std::string, FrontEndFactory>& item) {
                                   return item.first;
                               });
                return keys;
            }

            FrontEnd::Ptr loadByModel (const std::string& path, FrontEndCapabilities fec)
            {
                FRONT_END_NOT_IMPLEMENTED(loadByModel);
            }

            void registerFrontEnd(const std::string& name, FrontEndFactory creator) {
                m_factories.insert({name, creator});
            }
        };

        FrontEndManager::FrontEndManager(): m_impl(new Impl()) {
        }
        FrontEndManager::~FrontEndManager() = default;

        FrontEnd::Ptr FrontEndManager::loadByFramework(const std::string& framework, FrontEndCapabilities fec)
        {
            return m_impl->loadByFramework(framework, fec);
        }

        FrontEnd::Ptr FrontEndManager::loadByModel(const std::string& path, FrontEndCapabilities fec)
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
