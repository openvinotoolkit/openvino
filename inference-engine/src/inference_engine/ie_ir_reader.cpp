// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <fstream>
#include <sstream>
#include <memory>
#include <vector>
#include <map>

#include <ngraph/function.hpp>
#include <ngraph/op/add.hpp>
#include <ngraph/op/avg_pool.hpp>
#include <ngraph/op/broadcast.hpp>
#include <ngraph/op/concat.hpp>
#include <ngraph/op/constant.hpp>
#include <ngraph/op/convert.hpp>
#include <ngraph/op/convolution.hpp>
#include <ngraph/op/divide.hpp>
#include <ngraph/op/dot.hpp>
#include <ngraph/op/exp.hpp>
#include <ngraph/op/experimental/dyn_reshape.hpp>
#include <ngraph/op/experimental/layers/detection_output.hpp>
#include <ngraph/op/experimental/layers/prior_box.hpp>
#include <ngraph/op/experimental/layers/prior_box_clustered.hpp>
#include <ngraph/op/experimental/layers/proposal.hpp>
#include <ngraph/op/experimental/shape_of.hpp>
#include <ngraph/op/experimental/transpose.hpp>
#include <ngraph/op/fused/clamp.hpp>
#include <ngraph/op/fused/conv_fused.hpp>
#include <ngraph/op/fused/elu.hpp>
#include <ngraph/op/fused/leaky_relu.hpp>
#include <ngraph/op/fused/mvn.hpp>
#include <ngraph/op/fused/prelu.hpp>
#include <ngraph/op/fused/split.hpp>
#include <ngraph/op/fused/squeeze.hpp>
#include <ngraph/op/fused/unsqueeze.hpp>
#include <ngraph/op/get_output_element.hpp>
#include <ngraph/op/lrn.hpp>
#include <ngraph/op/max_pool.hpp>
#include <ngraph/op/maximum.hpp>
#include <ngraph/op/multiply.hpp>
#include <ngraph/op/pad.hpp>
#include <ngraph/op/parameter.hpp>
#include <ngraph/op/power.hpp>
#include <ngraph/op/relu.hpp>
#include <ngraph/op/reshape.hpp>
#include <ngraph/op/result.hpp>
#include <ngraph/op/sigmoid.hpp>
#include <ngraph/op/softmax.hpp>
#include <ngraph/op/subtract.hpp>
#include <ngraph/op/tanh.hpp>
#include <ngraph/op/topk.hpp>

#include <transform/transformations/conv_bias_fusion.hpp>
#include <transform/transformations/convert_mul_add_to_scaleshift_or_power.hpp>
#include <transform/transformations/convert_mul_or_add_finally.hpp>
#include <transform/transformations/matmul_bias_fusion.hpp>
#include <transform/transformations/quantizeconv_dequantize_fusion.hpp>
#include <transform/transformations/constant_eltwise_reduction.hpp>
#include <transform/transformations/convert_quantize_conv_elimination.hpp>
#include <transform/transformations/convert_broadcast_to_tiles.hpp>
#include <transform/transformations/convert_tile_to_ie_tile.hpp>
#include <transform/transformations/reshape_constant_folding.hpp>
#include <transform/transformations/convert_prior_clustered_to_ie_clustered.hpp>
#include <transform/transformations/convert_pror_to_ie_prior.hpp>
#include <transform/transformations/convert_interpolate_to_interp_or_resample.hpp>
#include <transform/transformations/convert_strided_slice_to_crop.hpp>

#include <ngraph/pass/constant_folding.hpp>
#include <ngraph/pass/visualize_tree.hpp>

#include "ngraph_ops/dummy.hpp"
#include "ngraph_ops/eltwise.hpp"
#include "ngraph_ops/group_conv_bias.hpp"
#include "ngraph_ops/matmul_bias.hpp"
#include "ngraph_ops/power.hpp"
#include "ngraph_ops/prior_box_clustered_ie.hpp"
#include "ngraph_ops/prior_box_ie.hpp"
#include "ngraph_ops/quantize_conv_bias_fused.hpp"
#include "ngraph_ops/scaleshift.hpp"
#include "ngraph_ops/tile_ie.hpp"
#include "ngraph_ops/interp.hpp"

#include <xml_parse_utils.h>
#include <ie_ir_reader.hpp>
#include "ie_ngraph_utils.hpp"
#include "ie_cnn_layer_builder.h"
#include "cnn_network_impl.hpp"
#include "description_buffer.hpp"
#include "ie_ir_parser.hpp"
#include <file_utils.h>
#include <ngraph.hpp>

using namespace InferenceEngine;

static size_t GetIRVersion(pugi::xml_node& root) {
    return XMLParseUtils::GetUIntAttr(root, "version", 0);
}

std::shared_ptr<ngraph::Function> IRReader::read(const std::string& modelPath) {
    std::string binPath = modelPath;
    auto pos = modelPath.rfind('.');
    if (pos != std::string::npos)
        binPath = binPath.substr(0, pos);
    binPath += ".bin";
    if (!FileUtils::fileExist(binPath))
        binPath.clear();
    return read(modelPath, binPath);
}

std::shared_ptr<ngraph::Function> IRReader::read(const std::string& modelPath, const std::string& binPath) {
    std::ifstream modelFile(modelPath);
    if (!modelFile.is_open())
        THROW_IE_EXCEPTION << "File " << modelPath << " cannot be openned!";

    std::stringstream modelBuf;
    modelBuf << modelFile.rdbuf();

    Blob::Ptr weights;
    if (!binPath.empty()) {
        int64_t fileSize = FileUtils::fileSize(binPath);

        if (fileSize < 0)
            THROW_IE_EXCEPTION << "Filesize for: " << binPath << " - " << fileSize
                << " < 0. Please, check weights file existence.";

        size_t ulFileSize = static_cast<size_t>(fileSize);

        weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {ulFileSize}, Layout::C));
        weights->allocate();
        FileUtils::readAllFile(binPath, weights->buffer(), ulFileSize);
    }

    return read(modelBuf.str(), weights);
}

std::shared_ptr<ngraph::Function> IRReader::read(const std::string& model, const Blob::CPtr& weights) {
    pugi::xml_document xmlDoc;
    pugi::xml_parse_result res = xmlDoc.load_buffer(model.data(), model.length());
    if (res.status != pugi::status_ok) {
        THROW_IE_EXCEPTION << res.description() << "at offset " << res.offset;
    }
    return readXml(xmlDoc, weights);
}

std::shared_ptr<ngraph::Function> IRReader::readXml(const pugi::xml_document& xmlDoc, const Blob::CPtr& weights) {
    try {
        // check which version it is...
        pugi::xml_node root = xmlDoc.document_element();

        auto version = GetIRVersion(root);
        IRParser parser(version);
        return parser.parse(root, weights);
    } catch (const std::string& err) {
        THROW_IE_EXCEPTION << err;
    } catch (const details::InferenceEngineException& e) {
        throw;
    } catch (const std::exception& e) {
        THROW_IE_EXCEPTION << e.what();
    } catch (...) {
        THROW_IE_EXCEPTION << "Unknown exception thrown";
    }
}

ICNNNetwork::Ptr InferenceEngine::convertFunctionToICNNNetwork(const std::shared_ptr<ngraph::Function>& nGraph) {
    const auto createCNNLayer = [](const std::shared_ptr<ngraph::Node>& node, const Precision& netPrc) -> CNNLayerPtr {
        static std::vector<std::shared_ptr<Builder::INodeConverter>> convertors = {
            std::make_shared<Builder::NodeConverter<ngraph::op::Add>>(),
            std::make_shared<Builder::NodeConverter<ngraph::op::AvgPool>>(),
            std::make_shared<Builder::NodeConverter<ngraph::op::Broadcast>>(),
            std::make_shared<Builder::NodeConverter<ngraph::op::Clamp>>(),
            std::make_shared<Builder::NodeConverter<ngraph::op::Concat>>(),
            std::make_shared<Builder::NodeConverter<ngraph::op::Constant>>(),
            std::make_shared<Builder::NodeConverter<ngraph::op::ConvolutionBackpropData>>(),
            std::make_shared<Builder::NodeConverter<ngraph::op::CropIE>>(),
            std::make_shared<Builder::NodeConverter<ngraph::op::DetectionOutput>>(),
            std::make_shared<Builder::NodeConverter<ngraph::op::Divide>>(),
            std::make_shared<Builder::NodeConverter<ngraph::op::Dot>>(),
            std::make_shared<Builder::NodeConverter<ngraph::op::DynReshape>>(),
            std::make_shared<Builder::NodeConverter<ngraph::op::Eltwise>>(),
            std::make_shared<Builder::NodeConverter<ngraph::op::Elu>>(),
            std::make_shared<Builder::NodeConverter<ngraph::op::Exp>>(),
            std::make_shared<Builder::NodeConverter<ngraph::op::Convolution>>(),
            std::make_shared<Builder::NodeConverter<ngraph::op::ConvolutionBias>>(),
            std::make_shared<Builder::NodeConverter<ngraph::op::GroupConvolution>>(),
            std::make_shared<Builder::NodeConverter<ngraph::op::GroupConvolutionBias>>(),
            std::make_shared<Builder::NodeConverter<ngraph::op::Interp>>(),
            std::make_shared<Builder::NodeConverter<ngraph::op::Interpolate>>(),
            std::make_shared<Builder::NodeConverter<ngraph::op::LRN>>(),
            std::make_shared<Builder::NodeConverter<ngraph::op::LeakyRelu>>(),
            std::make_shared<Builder::NodeConverter<ngraph::op::MVN>>(),
            std::make_shared<Builder::NodeConverter<ngraph::op::MatmulBias>>(),
            std::make_shared<Builder::NodeConverter<ngraph::op::MaxPool>>(),
            std::make_shared<Builder::NodeConverter<ngraph::op::Maximum>>(),
            std::make_shared<Builder::NodeConverter<ngraph::op::Multiply>>(),
            std::make_shared<Builder::NodeConverter<ngraph::op::PRelu>>(),
            std::make_shared<Builder::NodeConverter<ngraph::op::Pad>>(),
            std::make_shared<Builder::NodeConverter<ngraph::op::Parameter>>(),
            std::make_shared<Builder::NodeConverter<ngraph::op::Power>>(),
            std::make_shared<Builder::NodeConverter<ngraph::op::PowerIE>>(),
            std::make_shared<Builder::NodeConverter<ngraph::op::PriorBox>>(),
            std::make_shared<Builder::NodeConverter<ngraph::op::PriorBoxClustered>>(),
            std::make_shared<Builder::NodeConverter<ngraph::op::PriorBoxClusteredIE>>(),
            std::make_shared<Builder::NodeConverter<ngraph::op::PriorBoxIE>>(),
            std::make_shared<Builder::NodeConverter<ngraph::op::Proposal>>(),
            std::make_shared<Builder::NodeConverter<ngraph::op::Relu>>(),
            std::make_shared<Builder::NodeConverter<ngraph::op::Reshape>>(),
            std::make_shared<Builder::NodeConverter<ngraph::op::ScaleShiftIE>>(),
            std::make_shared<Builder::NodeConverter<ngraph::op::ShapeOf>>(),
            std::make_shared<Builder::NodeConverter<ngraph::op::Sigmoid>>(),
            std::make_shared<Builder::NodeConverter<ngraph::op::Softmax>>(),
            std::make_shared<Builder::NodeConverter<ngraph::op::Split>>(),
            std::make_shared<Builder::NodeConverter<ngraph::op::Squeeze>>(),
            std::make_shared<Builder::NodeConverter<ngraph::op::Subtract>>(),
            std::make_shared<Builder::NodeConverter<ngraph::op::Tanh>>(),
            std::make_shared<Builder::NodeConverter<ngraph::op::TileIE>>(),
            // std::make_shared<Builder::NodeConverter<ngraph::op::TopK>>(),
            std::make_shared<Builder::NodeConverter<ngraph::op::Transpose>>(),
            std::make_shared<Builder::NodeConverter<ngraph::op::Unsqueeze>>(),
        };
        Precision precision = Precision::UNSPECIFIED;
        for (const auto& port : node->get_outputs()) {
            auto prc = details::ngraph::convertPrecision(port.get_element_type());
            if (prc == Precision::UNSPECIFIED) {
                precision = prc;
                break;
            }
        }
        for (const auto& port : node->get_inputs()) {
            auto prc = details::ngraph::convertPrecision(port.get_element_type());
            if (prc == Precision::UNSPECIFIED) {
                precision = prc;
                break;
            }
        }
        if (precision == Precision::UNSPECIFIED) {
            precision = netPrc;
        }
        for (auto &convertor : convertors) {
            if (!convertor->canCreate(node))
                continue;
            return convertor->createLayer(node, precision);
        }
        THROW_IE_EXCEPTION << "Cannot cast ngraph node " << node->get_friendly_name() << " to CNNLayer!";
    };

    const auto isInternalLayer = [](const std::shared_ptr<ngraph::Node>& layer) -> bool {
        bool internalLayer = std::dynamic_pointer_cast<ngraph::op::Constant>(layer) != nullptr;
        for (const auto& output : layer->get_outputs()) {
            if (!internalLayer)
                break;
            for (const auto& input : output.get_inputs()) {
                const auto& inputLayer = input->get_node();
                internalLayer = internalLayer && (input->get_index() ||
                        std::dynamic_pointer_cast<ngraph::op::ConvolutionBackpropData>(inputLayer) != nullptr) &&
                    (std::dynamic_pointer_cast<ngraph::op::GroupConvolution>(inputLayer) != nullptr ||
                     std::dynamic_pointer_cast<ngraph::op::GroupConvolutionBias>(inputLayer) != nullptr ||
                     std::dynamic_pointer_cast<ngraph::op::Convolution>(inputLayer) != nullptr ||
                     std::dynamic_pointer_cast<ngraph::op::ConvolutionBias>(inputLayer) != nullptr ||
                     std::dynamic_pointer_cast<ngraph::op::Elu>(inputLayer) != nullptr ||
                     std::dynamic_pointer_cast<ngraph::op::LeakyRelu>(inputLayer) != nullptr ||
                     std::dynamic_pointer_cast<ngraph::op::ConvolutionBackpropData>(inputLayer) != nullptr ||
                     std::dynamic_pointer_cast<ngraph::op::MatmulBias>(inputLayer) != nullptr ||
                     std::dynamic_pointer_cast<ngraph::op::Pad>(inputLayer) != nullptr ||
                     std::dynamic_pointer_cast<ngraph::op::PRelu>(inputLayer) != nullptr ||
                     std::dynamic_pointer_cast<ngraph::op::ScaleShiftIE>(inputLayer) != nullptr ||
                     std::dynamic_pointer_cast<ngraph::op::Transpose>(inputLayer) != nullptr ||
                     std::dynamic_pointer_cast<ngraph::op::TopK>(inputLayer) != nullptr ||
                     std::dynamic_pointer_cast<ngraph::op::Dot>(inputLayer) != nullptr);
            }
        }
        internalLayer = internalLayer || std::dynamic_pointer_cast<ngraph::op::Result>(layer) != nullptr ||
            std::dynamic_pointer_cast<ngraph::op::Dummy>(layer) != nullptr;
        return internalLayer;
    };

    const auto keep_input_info = [](std::unique_ptr<details::CNNNetworkImpl>& network, const DataPtr& inData) {
        InputInfo::Ptr info(new InputInfo());
        info->setInputData(inData);
        Precision prc = info->getPrecision();

        // Convert precision into native format (keep element size)
        prc = prc == Precision::Q78 ? Precision::I16 :
            prc == Precision::FP16 ? Precision::FP32 :
            static_cast<Precision::ePrecision>(prc);

        info->setPrecision(prc);
        network->setInputInfo(info);
    };

    std::shared_ptr<ngraph::Function> graph = nGraph;

    ngraph::pass::ConvertPriorBox().run_on_function(graph);
    ngraph::pass::ConvertPriorBoxClustered().run_on_function(graph);

    ngraph::pass::ConvBiasFusion().run_on_function(graph);
    ngraph::pass::ReshapeConstanFolding().run_on_function(graph);

    ngraph::pass::MatMulBiasFusion().run_on_function(graph);
    ngraph::pass::ReshapeConstanFolding().run_on_function(graph);

    ngraph::pass::ConvertElimination().run_on_function(graph);
    ngraph::pass::ConstantEltwiseReduction().run_on_function(graph);
    ngraph::pass::ConvertMulAddToScaleShiftOrPower().run_on_function(graph);
    ngraph::pass::ConvertMulOrAddFinally().run_on_function(graph);
    ngraph::pass::ReshapeConstanFolding().run_on_function(graph);

    ngraph::pass::ConvertBroadcastToTiles().run_on_function(graph);
    ngraph::pass::ConvertTileToIETile().run_on_function(graph);
    ngraph::pass::ConvertInterpolateToInterpOrResample().run_on_function(graph);
    ngraph::pass::ConvertStridedSliceToCrop().run_on_function(graph);

    graph->validate_nodes_and_infer_types();

    // Detect correct precision
    Precision detectedPrecision = Precision::UNSPECIFIED;
    for (const auto& layer : graph->get_ops()) {
        for (const auto& port : layer->get_inputs()) {
            auto prc = details::ngraph::convertPrecision(port.get_element_type());
            if (prc != Precision::UNSPECIFIED) {
                detectedPrecision = prc;
                break;
            }
        }
        for (const auto& port : layer->get_outputs()) {
            auto prc = details::ngraph::convertPrecision(port.get_element_type());
            if (prc != Precision::UNSPECIFIED) {
                detectedPrecision = prc;
                break;
            }
        }
        if (detectedPrecision != Precision::UNSPECIFIED)
            break;
    }
    if (detectedPrecision == Precision::UNSPECIFIED)
        detectedPrecision = Precision::FP32;

    // Create network
    std::unique_ptr<details::CNNNetworkImpl> cnnNetworkImpl(new details::CNNNetworkImpl());
    cnnNetworkImpl->setName(graph->get_friendly_name());
    cnnNetworkImpl->setPrecision(detectedPrecision);

    // Create layers and output data
    for (const auto& layer : graph->get_ops()) {
        if (isInternalLayer(layer))
            continue;
        CNNLayerPtr cnnLayer = createCNNLayer(layer, detectedPrecision);
        size_t inputCount(0);
        for (size_t i = 0; i < layer->get_input_size(); i++) {
            const auto& input = layer->get_inputs()[i];
            if (isInternalLayer(input.get_output().get_node()))
                continue;
            inputCount++;
        }
        cnnLayer->insData.resize(inputCount);
        for (size_t i = 0; i < layer->get_output_size(); i++) {
            std::string outName = layer->get_friendly_name();
            if (layer->get_output_size() != 1)
                outName += "." + std::to_string(i);
            DataPtr& ptr = cnnNetworkImpl->getData(outName.c_str());
            if (!ptr) {
                SizeVector dims;
                dims = layer->get_output_shape(i);
                for (const auto& dim : dims) {
                    if (!dim)
                        THROW_IE_EXCEPTION << cnnLayer->type << " layer has incorrect dimensions in the output data " << i;
                }
                ptr.reset(new Data(outName, dims, details::ngraph::convertPrecision(layer->get_output_element_type(i)),
                            TensorDesc::getLayoutByDims(dims)));
                ptr->setDims(dims);
            }
            if (ptr->getCreatorLayer().lock())
                THROW_IE_EXCEPTION << "two layers set to the same output [" << outName << "]";

            ptr->getCreatorLayer() = cnnLayer;
            cnnLayer->outData.push_back(ptr);
            if (std::dynamic_pointer_cast<ngraph::op::Parameter>(layer)) {
                keep_input_info(cnnNetworkImpl, ptr);
            }
        }
        cnnNetworkImpl->addLayer(cnnLayer);
    }

    // Set input data
    for (const auto& layer : graph->get_ops()) {
        for (size_t i = 0; i < layer->get_input_size(); i++) {
            const auto& input = layer->get_inputs()[i];
            if (isInternalLayer(input.get_output().get_node()))
                continue;
            CNNLayerPtr prevCnnLayer;
            StatusCode ret = cnnNetworkImpl->getLayerByName(input.get_output().get_node()->get_friendly_name().c_str(), prevCnnLayer, nullptr);
            if (ret != OK)
                THROW_IE_EXCEPTION << "Cannot find layer with name: " << input.get_output().get_node()->get_friendly_name();
            auto output = std::dynamic_pointer_cast<ngraph::op::Result>(layer);
            if (output) {
                std::string outName = input.get_output().get_node()->get_friendly_name();
                if (input.get_output().get_node()->get_output_size() != 1)
                    outName += "." + std::to_string(input.get_output().get_index());
                cnnNetworkImpl->addOutput(outName);
            } else {
                CNNLayerPtr cnnLayer;
                ret = cnnNetworkImpl->getLayerByName(layer->get_friendly_name().c_str(), cnnLayer, nullptr);
                if (ret != OK)
                    THROW_IE_EXCEPTION << "Cannot find layer with name: " << layer->get_friendly_name();
                auto inIndex = input.get_index();
                // FIXME: WA for deconvolution
                if (cnnLayer->type == "Deconvolution")
                    inIndex--;
                if (cnnLayer->insData.size() <= inIndex || prevCnnLayer->outData.size() <= input.get_output().get_index())
                    THROW_IE_EXCEPTION << "Cannot create ICNNNetwork. Network structure is incorrect! "
                        << "Input port " << inIndex << " (max " << cnnLayer->insData.size() << ") of "
                        << cnnLayer->type << " layer " << cnnLayer->name << " cannot be connected with output port "
                        << input.get_output().get_index() << " (max " << prevCnnLayer->outData.size() << ") of "
                        << prevCnnLayer->type << " layer " << prevCnnLayer->name;
                cnnLayer->insData[inIndex] = prevCnnLayer->outData[input.get_output().get_index()];
                prevCnnLayer->outData[input.get_output().get_index()]->getInputTo()[cnnLayer->name] = cnnLayer;
            }
        }
    }

    // check all input ports are occupied
    for (const auto& kvp : cnnNetworkImpl->allLayers()) {
        const CNNLayer::Ptr& layer = kvp.second;
        size_t inSize = layer->insData.size();

        for (unsigned i = 0; i < inSize; i++) {
            if (!layer->insData[i].lock()) {
                THROW_IE_EXCEPTION << "Layer " << layer->name.c_str() << " input port "
                                   << i << " is not connected to any data";
            }
        }
        layer->validateLayer();
    }

    // Set default output precision to FP32 (for back-compatibility)
    OutputsDataMap outputsInfo;
    cnnNetworkImpl->getOutputsInfo(outputsInfo);
    for (auto outputInfo : outputsInfo) {
        if (outputInfo.second->getPrecision() != Precision::FP32 &&
            outputInfo.second->getPrecision() != Precision::I32) {
            outputInfo.second->setPrecision(Precision::FP32);
        }
    }

    return std::shared_ptr<ICNNNetwork>(cnnNetworkImpl.release());
}
