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
#include "onnx_import/onnx.hpp"
#include "onnx_import/editor/editor.hpp"
#include "frontend_manager/frontend_manager.hpp"


namespace ngraph
{
    namespace frontend
    {

        #define FRONT_END_NOT_IMPLEMENTED(NAME) throw #NAME " is not implemented for this FrontEnd class";

        std::vector<Place::Ptr> InputModel::InputModel::getInputs () const
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

        void InputModel::setNameForDimension (Place::Ptr place, const std::string& dimName)
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

        void InputModel::addOutput (Place::Ptr place)
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

        void InputModel::setSourceTensorDimSpecialization (Place::Ptr place, unsigned int dimIndex, const std::string& specialization)
        {
            FRONT_END_NOT_IMPLEMENTED(setSourceTensorDimSpecialization);
        }

        void InputModel::setTargetTensorDimSpecialization (Place::Ptr place, unsigned int dimIndex, const std::string& specialization)
        {
            FRONT_END_NOT_IMPLEMENTED(setTargetTensorDimSpecialization);
        }

        // All associated names that uniquely identify this place in the graph
        // from the FW perspective
        std::vector<std::string> Place::getNames () const
        {
            FRONT_END_NOT_IMPLEMENTED(getNames);
        }

        // -1 means port 0 is selected if it is exists and exception otherwise
        Place::Ptr Place::getConsumingOperation (int outputPortIndex) const
        {
            FRONT_END_NOT_IMPLEMENTED(getConsumingOperation);
        }

        Place::Ptr Place::getConsumingTensor (int outputPortIndex) const
        {
            FRONT_END_NOT_IMPLEMENTED(getConsumingTensor);
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

        class PlaceInputEdgeONNX : public Place
        {
        public:

            onnx_import::InputEdge edge;

            PlaceInputEdgeONNX (const std::string& _sourceTensorName, int _operationNodeIndex) :
                    edge(_operationNodeIndex, _sourceTensorName)
            {}
        };

        class PlaceOutputEdgeONNX : public Place
        {
        public:

            onnx_import::OutputEdge edge;

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
            onnx_import::ONNXModelEditor editor;

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
                std::vector<onnx_import::InputEdge> onnx_inputs;
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

                std::vector<onnx_import::OutputEdge> onnx_outputs;
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

            virtual InputModel::Ptr load (const std::string& path) const
            {
                return std::make_shared<InputModelONNX>(path);
            }

            virtual std::shared_ptr<ngraph::Function> convert (InputModel::Ptr model) const
            {
                return import_onnx_model(std::dynamic_pointer_cast<InputModelONNX>(model)->editor);
            }
        };

        InputModel::Ptr FrontEnd::loadFromPaths (const std::vector<std::string>& paths) const
        {
            FRONT_END_NOT_IMPLEMENTED(loadFromPaths);
        }

        InputModel::Ptr FrontEnd::loadFromMemory (const void* model)
        {
            FRONT_END_NOT_IMPLEMENTED(loadFromMemory);
        }

        InputModel::Ptr FrontEnd::loadFromMemoryFragments (const std::vector<const void*> modelParts)
        {
            FRONT_END_NOT_IMPLEMENTED(loadFromMemoryFragments);
        }

        std::shared_ptr<ngraph::Function> FrontEnd::convertPartially (InputModel::Ptr model) const
        {
            FRONT_END_NOT_IMPLEMENTED(convertPartially);
        }

        void FrontEnd::normalize (std::shared_ptr<ngraph::Function> function) const
        {
            FRONT_END_NOT_IMPLEMENTED(normalize);
        }

        FrontEnd::Ptr FrontEndManager::loadByFramework (const std::string& framework, FrontEndCapabilities fec)
        {
            NGRAPH_CHECK(framework == "onnx");
            return std::make_shared<FrontEndONNX>();
        }

        FrontEnd::Ptr FrontEndManager::loadByModel (const std::string& path, FrontEndCapabilities fec)
        {
            return loadByFramework("onnx");
        }

        std::vector<std::string> FrontEndManager::availableFrontEnds () const
        {
            return std::vector<std::string>(1, "onnx");
        }
    } // namespace frontend

} // namespace ngraph
