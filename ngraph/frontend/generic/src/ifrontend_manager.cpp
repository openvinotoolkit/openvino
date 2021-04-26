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

#if defined(NGRAPH_ONNX_IMPORT_ENABLE) && defined(NGRAPH_ONNX_EDITOR_ENABLE)
        class PlaceInputEdgeONNX : public IPlace
        {
        public:

            onnx_editor::InputEdge edge;

            PlaceInputEdgeONNX (const std::string& _sourceTensorName, int _operationNodeIndex) :
                    edge(_operationNodeIndex, _sourceTensorName)
            {}
        };

        class PlaceOutputEdgeONNX : public IPlace
        {
        public:

            onnx_editor::OutputEdge edge;

            PlaceOutputEdgeONNX (int _operationNodeIndex, const std::string& _targetTensorName) :
                    edge(_operationNodeIndex, _targetTensorName)
            {}


        };

        class InputModelONNX;

        class PlaceTensorONNX : public IPlace
        {
            std::string tensorName;
            const InputModelONNX* model;

        public:

            PlaceTensorONNX (const std::string& _tensorName, const InputModelONNX* _model) : tensorName(_tensorName), model(_model){}

            virtual std::vector<std::string> getNames () const override
            {
                return std::vector<std::string>(1, tensorName);
            }

            virtual std::vector<IPlace::Ptr> getConsumingPorts () const override;
            virtual IPlace::Ptr getProducingPort () const override;
        };

        class InputModelONNX : public IInputModel
        {

        public:
            // TODO: Move to private
            onnx_editor::ONNXModelEditor editor;

            InputModelONNX (const std::string& model_path) : editor(model_path) {}

            IPlace::Ptr getPlaceByTensorName (const std::string& tensorName) const override
            {
                if(!editor.validate_tensor_name(tensorName)) {
                    std::cerr << " [ ERROR ] Node with name " << tensorName << " is not valid for a given model\n";
                    return nullptr;
                }
                return std::make_shared<PlaceTensorONNX>(tensorName, this);
            }

            std::vector<IPlace::Ptr> getInputs () const override {
                auto inputs = editor.model_inputs();
                std::vector<IPlace::Ptr> outputs;
                outputs.reserve(inputs.size());
                for(auto const& input: inputs)
                {
                    outputs.push_back(std::make_shared<PlaceTensorONNX>(input, this));
                }
                return outputs;
            }

            void setPartialShape (IPlace::Ptr IPlace, const ngraph::PartialShape& shape) override
            {
                std::map<std::string, ngraph::PartialShape> m;
                m[IPlace->getNames()[0]] = shape;
                editor.set_input_shapes(m);
            }

            void extractSubgraph (const std::vector<IPlace::Ptr>& inputs, const std::vector<IPlace::Ptr>& outputs)
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

        std::vector<IPlace::Ptr> PlaceTensorONNX::getConsumingPorts () const
        {
            // ONNX specific code to find a node indices for all operations that consume a given tensor name
            std::vector<IPlace::Ptr> result;
            for(int i: model->editor.find_consumeing_node_idxs(tensorName))
            {
                result.push_back(std::make_shared<PlaceInputEdgeONNX>(tensorName, i));
            }
            return result;
        }

        IPlace::Ptr PlaceTensorONNX::getProducingPort () const
        {
            return std::make_shared<PlaceOutputEdgeONNX>(model->editor.find_producing_node_idx(tensorName), tensorName);
        }

        class FrontEndONNX : public FrontEnd
        {
        public:

            FrontEndONNX ()
            {
            }

            virtual IInputModel::Ptr loadFromFile (const std::string& path) const override
            {
                return std::make_shared<InputModelONNX>(path);
            }

            virtual std::shared_ptr<ngraph::Function> convert (IInputModel::Ptr model) const override
            {
                return onnx_import::import_onnx_model(std::dynamic_pointer_cast<InputModelONNX>(model)->editor.model_path());
            }
        };
#endif

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
