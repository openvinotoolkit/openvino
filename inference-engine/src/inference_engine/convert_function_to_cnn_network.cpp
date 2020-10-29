// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_function_to_cnn_network.hpp"

#include <string>
#include <memory>
#include <vector>
#include <unordered_set>

#include "ngraph_ops/convolution_ie.hpp"
#include "ngraph_ops/deconvolution_ie.hpp"
#include "ngraph_ops/eltwise.hpp"
#include "ngraph_ops/fully_connected.hpp"
#include "ngraph_ops/gather_ie.hpp"
#include "ngraph_ops/gather_tree_ie.hpp"
#include "ngraph_ops/interp.hpp"
#include "ngraph_ops/lrn_ie.hpp"
#include "ngraph_ops/lstm_cell_ie.hpp"
#include "ngraph_ops/normalize_ie.hpp"
#include "ngraph_ops/pad_ie.hpp"
#include "ngraph_ops/onehot_ie.hpp"
#include "ngraph_ops/power.hpp"
#include "ngraph_ops/prior_box_clustered_ie.hpp"
#include "ngraph_ops/prior_box_ie.hpp"
#include "ngraph_ops/proposal_ie.hpp"
#include "ngraph_ops/relu_ie.hpp"
#include "ngraph_ops/scaleshift.hpp"
#include "ngraph_ops/tile_ie.hpp"
#include "ngraph_ops/hard_sigmoid_ie.hpp"
#include "ngraph_ops/nms_ie.hpp"
#include "ngraph_ops/crop_ie.hpp"
#include "ngraph_ops/selu_ie.hpp"
#include "ngraph_ops/strided_slice_ie.hpp"
#include "ngraph_ops/topk_ie.hpp"
#include "generic_ie.hpp"

#include "ie_profiling.hpp"
#include "ie_cnn_layer_builder_ngraph.h"

#include "transformations/convert_opset1_to_legacy/convert_opset1_to_legacy.hpp"
#include "transformations/utils/utils.hpp"

namespace InferenceEngine {
namespace details {
std::shared_ptr<CNNNetworkImpl> convertFunctionToICNNNetwork(const std::shared_ptr<const ::ngraph::Function>& graph, const CNNNetworkNGraphImpl &nGraphImpl) {
    IE_PROFILING_AUTO_SCOPE(convertFunctionToICNNNetwork)
    const auto createCNNLayer = [](const std::shared_ptr<::ngraph::Node> &node) -> CNNLayerPtr {
        class NGraphCNNLayer: public CNNLayer {
        public:
            void setNode(const std::shared_ptr<::ngraph::Node>& node) {
                this->node = node;
            }
        };
        static std::vector<std::shared_ptr<Builder::INodeConverter>> convertors = {
                std::make_shared<Builder::NodeConverter<::ngraph::op::Abs>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Acos>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::v1::Add>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Asin>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Atan>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::v1::AvgPool>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::BatchNormInference>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::v1::Broadcast>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Clamp>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Concat>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Constant>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::ConvolutionIE>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::DeconvolutionIE>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Cos>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Cosh>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::CropIE>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Convert>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::CTCGreedyDecoder>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::DetectionOutput>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::v1::DeformableConvolution>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::v1::DeformablePSROIPooling>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::v1::Divide>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::v1::Reshape>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Eltwise>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Elu>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Erf>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Exp>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::FakeQuantize>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Floor>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Ceiling>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::GatherIE>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::v1::GatherTree>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::GatherTreeIE>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Interp>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Interpolate>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Log>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::LRN>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::LRN_IE>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::MVN>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::FullyConnected>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::MatMul>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::GenericIE>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::GRN>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::v1::MaxPool>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::v1::Maximum>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::v1::Minimum>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::v1::Multiply>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::v1::NonMaxSuppression>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::NonMaxSuppressionIE>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::NormalizeL2>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::NormalizeIE>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::OneHotIE>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::PRelu>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::PadIE>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::v1::Power>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::PowerIE>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::PriorBox>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::PriorBoxClustered>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::PriorBoxClusteredIE>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::PriorBoxIE>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Proposal>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::ProposalIE>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Relu>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::SeluIE>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::ReLUIE>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Range>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::ReverseSequence>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::v1::ReduceMin>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::v1::ReduceMax>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::v1::ReduceMean>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::v1::ReduceProd>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::v1::ReduceSum>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::ResampleV2>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::RegionYolo>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::ReorgYolo>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::ROIPooling>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::PSROIPooling>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::ScaleShiftIE>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::ShapeOf>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Sigmoid>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Sin>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Sign>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Sinh>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::SquaredDifference>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::v1::Softmax>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::v1::Split>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::VariadicSplit>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::v1::StridedSlice>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::StridedSliceIE>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Squeeze>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Sqrt>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Subtract>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Tan>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Tanh>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::TileIE>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::v1::TopK>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::TopKIE>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Transpose>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Unsqueeze>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::TensorIterator>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::LSTMCellIE>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::HardSigmoid>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::HardSigmoid_IE>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::v1::LogicalNot>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::v1::ReduceLogicalAnd>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::v1::ReduceLogicalOr>>(),
        };
        CNNLayerPtr result;

        for (auto &convertor : convertors) {
            if (!convertor->canCreate(node)) continue;
            result = convertor->createLayer(node);
            break;
        }

        if (!result) {
            CNNLayerCreator visitor(node);
            if (node->visit_attributes(visitor)) result = visitor.create();
        }

        if (!result)
            THROW_IE_EXCEPTION << "Cannot cast ngraph node " << node->get_friendly_name() << " to CNNLayer!";
        NGraphCNNLayer * layer = reinterpret_cast<NGraphCNNLayer*>(result.get());
        layer->setNode(node);
        return result;
    };

    const auto isInternalConstLayer = [](const std::shared_ptr<::ngraph::op::Constant> &constLayer,
                                         const std::shared_ptr<::ngraph::Node> &consumerLayer,
                                         bool keep_constants) -> bool {
        if (((::ngraph::as_type_ptr<::ngraph::op::ConvolutionIE>(consumerLayer) ||
              ::ngraph::as_type_ptr<::ngraph::op::FullyConnected>(consumerLayer)) && !keep_constants) ||
            ::ngraph::as_type_ptr<::ngraph::op::v1::BinaryConvolution>(consumerLayer) ||
            ::ngraph::as_type_ptr<::ngraph::op::DeconvolutionIE>(consumerLayer) ||
            ::ngraph::as_type_ptr<::ngraph::op::v1::DeformableConvolution>(consumerLayer) ||
            ::ngraph::as_type_ptr<::ngraph::op::Elu>(consumerLayer) ||
            ::ngraph::as_type_ptr<::ngraph::op::NormalizeIE>(consumerLayer) ||
            ::ngraph::as_type_ptr<::ngraph::op::PRelu>(consumerLayer) ||
            ::ngraph::as_type_ptr<::ngraph::op::v1::Split>(consumerLayer) ||
            ::ngraph::as_type_ptr<::ngraph::op::VariadicSplit>(consumerLayer) ||
            ::ngraph::as_type_ptr<::ngraph::op::ScaleShiftIE>(consumerLayer) ||
            ::ngraph::as_type_ptr<::ngraph::op::Transpose>(consumerLayer)) {
            // Check that all input nodes except zero input are Constants for all ops except DeformableConvolutions
            // for which the input with index 1 is also dynamic
            size_t inputID = ::ngraph::as_type_ptr<::ngraph::op::v1::DeformableConvolution>(consumerLayer) ? 2 : 1;
            for (; inputID < consumerLayer->inputs().size(); ++inputID) {
                auto inputLayer = consumerLayer->input(inputID).get_source_output().get_node_shared_ptr();
                if (inputLayer == constLayer) {
                    return true;
                }
            }
        } else if (::ngraph::as_type_ptr<::ngraph::op::LSTMCellIE>(consumerLayer)) {
            for (size_t inputID = 3; inputID < consumerLayer->inputs().size(); ++inputID) {
                auto inputLayer = consumerLayer->input(inputID).get_source_output().get_node_shared_ptr();
                if (inputLayer == constLayer) {
                    return true;
                }
            }
        }
        return false;
    };

    // Checks that node is internal layer for all layers from specific function
    const auto isInternalLayer = [=](const std::shared_ptr<::ngraph::Node> &node,
                                     const std::unordered_set<std::string> &names,
                                     bool keep_constant) -> bool {
        if (auto constantNode = ::ngraph::as_type_ptr<::ngraph::op::Constant>(node)) {
            for (const auto &consumerInputPort : constantNode->get_outputs()[0].get_inputs()) {
                const auto &consumerLayer = consumerInputPort->get_node();
                if (names.find(consumerLayer->get_name()) == names.end())
                    continue;
                if (!isInternalConstLayer(constantNode, consumerLayer, keep_constant))
                    return false;
            }
            return true;
        }

        return ::ngraph::as_type_ptr<::ngraph::op::Result>(node) != nullptr;
    };

    const auto keep_input_info = [](std::shared_ptr<details::CNNNetworkImpl> &network, const DataPtr &inData) {
        InputInfo::Ptr info(new InputInfo());
        info->setInputData(inData);
        network->setInputInfo(info);
    };

    InputsDataMap thisInputDataMap;
    nGraphImpl.getInputsInfo(thisInputDataMap);

    // Create network
    auto cnnNetworkImpl = std::make_shared<details::CNNNetworkImpl>();
    cnnNetworkImpl->setName(graph->get_friendly_name());
    // In generic case all nGraph functions have MIXED precision
    // Network precision should be deprecated
    cnnNetworkImpl->setPrecision(Precision::MIXED);

    // Collect all names from current graph
    // It is necessary in order to differentiate outputs from constant layers when we share constants
    // (Constant operations contains outputs for converted and original functions)
    std::unordered_set<std::string> op_names;
    for (const auto &layer : graph->get_ops())
        op_names.insert(layer->get_name());

    bool keep_constants = ::ngraph::op::util::has_op_with_type<::ngraph::op::FakeQuantize>(graph);

    // Create layers and output data
    for (const auto &layer : graph->get_ops()) {
        if (isInternalLayer(layer, op_names, keep_constants)) continue;

        // TODO: remove this rt info when all blobs will be inputs
        InferenceEngine::Parameter attr(keep_constants);
        auto &rt_info = layer->get_rt_info();
        rt_info["keep_constants"] = attr.asVariant();

        CNNLayerPtr cnnLayer = createCNNLayer(layer);
        for (const auto &rt : layer->get_rt_info()) {
            Parameter param(rt.second);
            if (param.empty()) continue;
            if (details::CaselessEq<std::string>()(rt.first, "affinity")) {
                cnnLayer->affinity = param.as<std::string>();
            } else if (param.is<std::string>()) {
                cnnLayer->params[rt.first] = param.as<std::string>();
            }
        }
        size_t inputCount(0);
        for (size_t i = 0; i < layer->get_input_size(); i++) {
            const auto &input = layer->get_inputs()[i];
            if (isInternalLayer(input.get_output().get_node(), op_names, keep_constants)) continue;
            inputCount++;
        }
        cnnLayer->insData.resize(inputCount);
        for (size_t i = 0; i < layer->get_output_size(); i++) {
            std::string outName = layer->get_friendly_name();
            if (layer->get_output_size() != 1) outName += "." + std::to_string(i);
            DataPtr &ptr = cnnNetworkImpl->getData(outName.c_str());

            SizeVector dims;
            dims = layer->get_output_shape(i);
            for (const auto &dim : dims) {
                if (!dim)
                    THROW_IE_EXCEPTION << cnnLayer->type << " layer " << cnnLayer->name
                                       << " has incorrect dimensions in the output data " << i;
            }

            if (!ptr && nGraphImpl._data.find(outName) != nGraphImpl._data.end()) {
                ptr = nGraphImpl._data.at(outName);
                if (auto nData = std::dynamic_pointer_cast<InferenceEngine::details::NGraphData>(ptr)) {
                    const auto layout =
                            dims.size() == nData->getTensorDesc().getDims().size() ?
                            nData->getTensorDesc().getLayout() :
                            TensorDesc::getLayoutByDims(dims);

                    nData->reset();
                    nData->reshape(dims, layout);
                }
                cnnNetworkImpl->addData(outName.c_str(), ptr);
            }
            if (!ptr) {
                ptr.reset(new Data(outName,
                                   {details::ngraph::convertPrecision(layer->get_output_element_type(i)), dims,
                                    TensorDesc::getLayoutByDims(dims)}));
            }

            ptr->getCreatorLayer() = cnnLayer;
            cnnLayer->outData.push_back(ptr);
            if (std::dynamic_pointer_cast<::ngraph::op::Parameter>(layer)) {
                keep_input_info(cnnNetworkImpl, ptr);
            }
        }
        cnnNetworkImpl->addLayer(cnnLayer);
    }

    // Set input data
    for (const auto &layer : graph->get_ordered_ops()) {
        if (std::dynamic_pointer_cast<::ngraph::op::Result>(layer)) {
            IE_ASSERT(layer->get_inputs().size() == 1);
            const auto &input = layer->input_value(0);
            std::string outName = input.get_node_shared_ptr()->get_friendly_name();
            if (input.get_node_shared_ptr()->get_output_size() != 1)
                outName += "." + std::to_string(input.get_index());
            cnnNetworkImpl->addOutput(outName);
            continue;
        }

        uint64_t count_of_skipped = 0;
        for (size_t i = 0; i < layer->get_input_size(); i++) {
            const auto &output_port = layer->input_value(i);
            const auto &input = output_port.get_node_shared_ptr();

            if (auto const_node = std::dynamic_pointer_cast<::ngraph::op::Constant>(input)) {
                if (isInternalConstLayer(const_node, layer, keep_constants)) {
                    count_of_skipped++;
                    continue;
                }
            }

            CNNLayerPtr prevCnnLayer;
            StatusCode ret = cnnNetworkImpl->getLayerByName(input->get_friendly_name().c_str(), prevCnnLayer, nullptr);
            if (ret != OK)
                THROW_IE_EXCEPTION << "Cannot find layer with name: " << input->get_friendly_name();

            CNNLayerPtr cnnLayer;
            ret = cnnNetworkImpl->getLayerByName(layer->get_friendly_name().c_str(), cnnLayer, nullptr);
            if (ret != OK) THROW_IE_EXCEPTION << "Cannot find layer with name: " << layer->get_friendly_name();

            auto inIndex = layer->input(i).get_index();
            if (cnnLayer->insData.size() <= (inIndex - count_of_skipped) ||
                prevCnnLayer->outData.size() <= output_port.get_index() || count_of_skipped > inIndex)
                THROW_IE_EXCEPTION << "Cannot create ICNNNetwork. Network structure is incorrect! "
                                   << "Input port " << inIndex << " (max " << cnnLayer->insData.size() << ") of "
                                   << cnnLayer->type << " layer " << cnnLayer->name
                                   << " cannot be connected with output port " << output_port.get_index()
                                   << " (max " << prevCnnLayer->outData.size() << ") of " << prevCnnLayer->type
                                   << " layer " << prevCnnLayer->name;
            cnnLayer->insData[inIndex - count_of_skipped] = prevCnnLayer->outData[output_port.get_index()];
            prevCnnLayer->outData[output_port.get_index()]->getInputTo()[cnnLayer->name] = cnnLayer;
        }
    }

    // check all input ports are occupied
    for (const auto &kvp : cnnNetworkImpl->allLayers()) {
        const CNNLayer::Ptr &layer = kvp.second;
        size_t inSize = layer->insData.size();

        for (unsigned i = 0; i < inSize; i++) {
            if (!layer->insData[i].lock()) {
                THROW_IE_EXCEPTION << "Layer " << layer->name.c_str() << " input port " << i
                                   << " is not connected to any data";
            }
        }
        layer->validateLayer();
    }

    if (!cnnNetworkImpl) THROW_IE_EXCEPTION << "Cannot convert nGraph function to CNNNetworkImpl!";

    // update input preprocessing info
    InputsDataMap resultInputDataMap;
    cnnNetworkImpl->getInputsInfo(resultInputDataMap);
    IE_ASSERT(resultInputDataMap.size() == thisInputDataMap.size());
    for (auto i : resultInputDataMap) {
        auto &thisInputData = *thisInputDataMap[i.first];
        i.second->setPrecision(thisInputData.getPrecision());
        i.second->setLayout(thisInputData.getLayout());
        i.second->getPreProcess() = thisInputData.getPreProcess();
    }

    for (const auto &ext : ::ngraph::op::GenericIE::getExtensions(graph)) {
        cnnNetworkImpl->AddExtension(ext, nullptr);
    }

    return cnnNetworkImpl;
}
}  // namespace details
}  // namespace InferenceEngine
