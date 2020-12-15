// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fstream>
#include <memory>
#include <string>
#include <algorithm>
#include <vector>

#include "functional_test_utils/test_model/test_model.hpp"
#include "common_test_utils/xml_net_builder/ir_net.hpp"
#include "common_test_utils/xml_net_builder/xml_filler.hpp"
#include "common_test_utils/common_layers_params.hpp"
#include "functional_test_utils/precision_utils.hpp"

IE_SUPPRESS_DEPRECATED_START

namespace FuncTestUtils {
namespace TestModel {

/**
 * @brief generates IR files (XML and BIN files) with the test model.
 *        Passed reference vector is filled with CNN layers to validate after the network reading.
 * @param modelPath used to serialize the generated network
 * @param weightsPath used to serialize the generated weights
 * @param netPrc precision of the generated network
 * @param inputDims dims on the input layer of the generated network
 * @param refLayersVec pointer to a vector of reference CNN layers
 * @return none
 */
void generateTestModel(const std::string &modelPath,
                       const std::string &weightsPath,
                       const InferenceEngine::Precision &netPrc,
                       const InferenceEngine::SizeVector &inputDims,
                       std::vector<InferenceEngine::CNNLayerPtr> *refLayersVec) {
    std::string precision = netPrc.name();
    std::string modelName = modelPath;
    /* remove ".xml" extension from file path to get the model name */
    modelName.erase(modelName.length() - std::string(".xml").size(), std::string(".xml").size());

    CommonTestUtils::IRBuilder_v10 ir_builder_v10(modelName);

    /* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> */
    /* Create Input node and add it to the reference */
    auto inputLayerPtr = std::make_shared<InferenceEngine::CNNLayer>(
            InferenceEngine::LayerParams{"Input0", "Input", netPrc});
    inputLayerPtr->outData.resize(1);
    auto inputOutData = std::make_shared<InferenceEngine::Data>("InputOutData",
                                                                InferenceEngine::TensorDesc{netPrc, inputDims,
                                                                                            InferenceEngine::Layout::NCHW});
    getCreatorLayer(inputOutData) = inputLayerPtr;
    inputLayerPtr->outData[0] = inputOutData;
    if (refLayersVec) refLayersVec->emplace_back(inputLayerPtr);

    std::ostringstream shapeStr("");
    std::copy(inputDims.begin(), inputDims.end() - 1, std::ostream_iterator<size_t>(shapeStr, ","));
    shapeStr << inputDims.back();
    auto inputLayerXML = ir_builder_v10
            .AddLayer(inputLayerPtr->name, "Parameter", {{"shape",        shapeStr.str()},
                                                         {"element_type", FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(
                                                                 netPrc).get_type_name()}})
            .AddOutPort(netPrc, inputDims)
            .getLayer();
    /* <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< */

    /* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> */
    /* Create Conv1 node and add it to the reference */
    InferenceEngine::LayerParams conv1CommonParams = {"Convolution1", "Convolution", netPrc};
    auto conv1LayerPtr = std::make_shared<InferenceEngine::ConvolutionLayer>(conv1CommonParams);
    CommonTestUtils::conv_common_params conv1Params{
            {{4, 4}},
            {{11, 11}},
            {{0, 0}},
            {{0, 0}},
            {{1, 1}},
            "",
            1,
            96,
            false,
            true
    };
    conv1LayerPtr->params = CommonTestUtils::convertConvParamToMap(conv1Params);
    if (refLayersVec) refLayersVec->emplace_back(conv1LayerPtr);

    InferenceEngine::SizeVector conv1OutShape(inputDims.size());
    CommonTestUtils::getConvOutShape(inputDims, conv1Params, conv1OutShape);

    conv1LayerPtr->insData.resize(1);
    conv1LayerPtr->outData.resize(1);
    getInputTo(inputOutData)[conv1LayerPtr->name] = conv1LayerPtr;
    conv1LayerPtr->insData[0] = inputOutData;
    auto conv1OutData = std::make_shared<InferenceEngine::Data>("Conv1OutData",
                                                                InferenceEngine::TensorDesc{netPrc, conv1OutShape,
                                                                                            InferenceEngine::Layout::NCHW});
    getCreatorLayer(conv1OutData) = conv1LayerPtr;
    conv1LayerPtr->outData[0] = conv1OutData;

    shapeStr.str("");
    InferenceEngine::SizeVector constShape {conv1Params.out_c, inputDims[1], conv1Params.kernel[0], conv1Params.kernel[1]};
    std::copy(constShape.begin(), constShape.end() - 1, std::ostream_iterator<size_t>(shapeStr, ","));
    shapeStr << constShape.back();

    auto conv1ParamConstLayerXML = ir_builder_v10
            .AddLayer("Conv1_Param_Const", "Const",
                      {{"size", std::to_string(CommonTestUtils::getConvWeightsSize(
                              inputDims,
                              conv1Params,
                              netPrc.name()))},
                        {"offset", "0"},
                        {"element_type", FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrc).get_type_name()},
                        {"shape", shapeStr.str()}})
            .AddOutPort(netPrc, {conv1Params.out_c, inputDims[1], conv1Params.kernel[0], conv1Params.kernel[1]})
            .getLayer();

    auto conv1LayerXML = ir_builder_v10
            .AddLayer(conv1LayerPtr->name, conv1LayerPtr->type,
                      {{"dilations",  "1,1"},
                       {"pads_begin", "0,0"},
                       {"pads_end",   "0,0"},
                       {"strides",    "4,4"}}
            )
            .AddInPort(netPrc, inputDims)
            .AddInPort(netPrc, {conv1Params.out_c, inputDims[1], conv1Params.kernel[0], conv1Params.kernel[1]})
            .AddOutPort(netPrc, conv1OutShape)
            .getLayer();
    inputLayerXML.out(0).connect(conv1LayerXML.in(0));
    conv1ParamConstLayerXML.out(0).connect(conv1LayerXML.in(1));
    /* <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< */

    /* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> */
    /* Create Relu1 node and add it to the reference */
    InferenceEngine::LayerParams relu1CommonParams = {"Relu1", "ReLU", netPrc};
    auto relu1LayerPtr = std::make_shared<InferenceEngine::ReLULayer>(relu1CommonParams);
    if (refLayersVec) refLayersVec->emplace_back(relu1LayerPtr);

    relu1LayerPtr->insData.resize(1);
    relu1LayerPtr->outData.resize(1);
    getInputTo(conv1OutData)[relu1LayerPtr->name] = relu1LayerPtr;
    relu1LayerPtr->insData[0] = conv1OutData;
    auto relu1OutData = std::make_shared<InferenceEngine::Data>("Relu1OutData",
                                                                InferenceEngine::TensorDesc{netPrc, conv1OutShape,
                                                                                            InferenceEngine::Layout::NCHW});
    getCreatorLayer(relu1OutData) = relu1LayerPtr;
    relu1LayerPtr->outData[0] = relu1OutData;

    auto relu1LayerXML = ir_builder_v10
            .AddLayer(relu1LayerPtr->name, relu1LayerPtr->type, relu1LayerPtr->params)
            .AddInPort(netPrc, conv1OutShape)
            .AddOutPort(netPrc, conv1OutShape)
            .getLayer();
    conv1LayerXML.out(0).connect(relu1LayerXML.in(0));
    /* <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< */

    /* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> */
    /* Create Lrn1 node and add it to the reference */
    InferenceEngine::LayerParams lrn1CommonParams = {"Lrn1", "Norm", netPrc};
    auto lrn1LayerPtr = std::make_shared<InferenceEngine::NormLayer>(lrn1CommonParams);
    if (refLayersVec) refLayersVec->emplace_back(lrn1LayerPtr);
    lrn1LayerPtr->params = {
            {"alpha",      "9.9999997e-05"},
            {"beta",       "0.75"},
            {"k",          "1"},
            {"local-size", "5"},
            {"region",     "across"},
    };

    lrn1LayerPtr->insData.resize(1);
    lrn1LayerPtr->outData.resize(1);
    getInputTo(relu1OutData)[lrn1LayerPtr->name] = lrn1LayerPtr;
    lrn1LayerPtr->insData[0] = relu1OutData;
    auto lrn1OutData = std::make_shared<InferenceEngine::Data>("Lrn1OutData",
                                                               InferenceEngine::TensorDesc{netPrc, conv1OutShape,
                                                                                           InferenceEngine::Layout::NCHW});
    getCreatorLayer(lrn1OutData) = lrn1LayerPtr;
    lrn1LayerPtr->outData[0] = lrn1OutData;
    size_t offset = CommonTestUtils::getConvWeightsSize(inputDims, conv1Params, netPrc.name());

    auto lrn1ParamConstLayerXML = ir_builder_v10
            .AddLayer("Lrn1_Param_Const", "Const", {{"size", "8"},
                                                    {"element_type", "i64"},
                                                    {"shape", "1"},
                                                    {"offset", std::to_string(offset)}})
            .AddOutPort(InferenceEngine::Precision::I64, {1})
            .getLayer();

    auto lrn1LayerXML = ir_builder_v10
            .AddLayer(lrn1LayerPtr->name, "LRN",
                      {{"alpha", "9.9999997e-05"},
                       {"beta",  "0.75"},
                       {"size",  "5"},
                       {"bias",  "1"}}
            )
            .AddInPort(netPrc, conv1OutShape)
            .AddInPort(InferenceEngine::Precision::I64, {1})
            .AddOutPort(netPrc, conv1OutShape)
            .getLayer();
    relu1LayerXML.out(0).connect(lrn1LayerXML.in(0));
    lrn1ParamConstLayerXML.out(0).connect(lrn1LayerXML.in(1));
    /* <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< */

    /* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> */
    /* Create Pool1 node and add it to the reference */
    InferenceEngine::LayerParams pool1CommonParams = {"Pooling1", "Pooling", netPrc};
    auto pool1LayerPtr = std::make_shared<InferenceEngine::PoolingLayer>(pool1CommonParams);
    CommonTestUtils::pool_common_params pool1Params = {
            {{2, 2}},
            {{3, 3}},
            {{0, 0}},
            {{0, 0}},
            "valid",
            false,
            false,
            "floor"
    };
    pool1LayerPtr->params = CommonTestUtils::convertPoolParamToMap(pool1Params);
    if (refLayersVec) refLayersVec->emplace_back(pool1LayerPtr);
    InferenceEngine::SizeVector pool1OutShape(conv1OutShape.size());
    getPoolOutShape(conv1OutShape, pool1Params, pool1OutShape);

    pool1LayerPtr->insData.resize(1);
    pool1LayerPtr->outData.resize(1);
    getInputTo(lrn1OutData)[pool1LayerPtr->name] = pool1LayerPtr;
    pool1LayerPtr->insData[0] = lrn1OutData;
    auto pool1OutData = std::make_shared<InferenceEngine::Data>("Pool1OutData",
                                                                InferenceEngine::TensorDesc{netPrc, pool1OutShape,
                                                                                            InferenceEngine::Layout::NCHW});
    getCreatorLayer(pool1OutData) = pool1LayerPtr;
    pool1LayerPtr->outData[0] = pool1OutData;

    auto pool1LayerXML = ir_builder_v10
            .AddLayer(pool1LayerPtr->name, "MaxPool",
                      {{"auto_pad",   "valid"},
                       {"kernel",     "3,3"},
                       {"pads_begin", "0,0"},
                       {"pads_end",   "0,0"},
                       {"strides",    "2,2"}}
            )
            .AddInPort(netPrc, conv1OutShape)
            .AddOutPort(netPrc, pool1OutShape)
            .getLayer();
    lrn1LayerXML.out(0).connect(pool1LayerXML.in(0));
    /* <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< */

    /* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> */
    /* Create Split1 node and add it to the reference */
    InferenceEngine::LayerParams split1CommonParams = {"Conv_split1", "Split", netPrc};
    auto split1LayerPtr = std::make_shared<InferenceEngine::SplitLayer>(split1CommonParams);
    if (refLayersVec) refLayersVec->emplace_back(split1LayerPtr);

    size_t axis = 1;
    size_t numSplit = 2;
    InferenceEngine::SizeVector split1OutShape(pool1OutShape);
    if (split1OutShape.size() != 4) {
        throw std::logic_error("unexpected split1 input shape");
    }
    split1OutShape[axis] /= numSplit;

    split1LayerPtr->params = {
            {"axis",      std::to_string(axis)},
            {"out_sizes", std::to_string(split1OutShape[axis]) + "," + std::to_string(split1OutShape[axis])}
    };

    split1LayerPtr->insData.resize(1);
    split1LayerPtr->outData.resize(2);
    getInputTo(pool1OutData)[split1LayerPtr->name] = split1LayerPtr;
    split1LayerPtr->insData[0] = pool1OutData;
    auto split1OutData0 = std::make_shared<InferenceEngine::Data>("Split1OutData0",
                                                                  InferenceEngine::TensorDesc{netPrc,
                                                                                              split1OutShape,
                                                                                              InferenceEngine::Layout::NCHW});
    auto split1OutData1 = std::make_shared<InferenceEngine::Data>("Split1OutData1",
                                                                  InferenceEngine::TensorDesc{netPrc,
                                                                                              split1OutShape,
                                                                                              InferenceEngine::Layout::NCHW});
    getCreatorLayer(split1OutData0) = split1LayerPtr;
    getCreatorLayer(split1OutData1) = split1LayerPtr;
    split1LayerPtr->outData[0] = split1OutData0;
    split1LayerPtr->outData[1] = split1OutData1;

    offset = offset + 8;

    auto split1ParamConstLayerXML = ir_builder_v10
            .AddLayer("Split1_Param_Const", "Const", {{"size", "8"},
                                                      {"element_type", "i64"},
                                                      {"shape", ""},
                                                      {"offset", std::to_string(offset)}})
            .AddOutPort(InferenceEngine::Precision::I64, {})
            .getLayer();

    auto split1LayerXMLBuilder = ir_builder_v10
            .AddLayer(split1LayerPtr->name, split1LayerPtr->type, {{"num_splits", std::to_string(numSplit)}})
            .AddInPort(netPrc, pool1OutShape)
            .AddInPort(InferenceEngine::Precision::I64, {});
    for (size_t i = 0; i < numSplit; i++) {
        split1LayerXMLBuilder.AddOutPort(netPrc, split1OutShape);
    }
    auto split1LayerXML = split1LayerXMLBuilder.getLayer();
    pool1LayerXML.out(0).connect(split1LayerXML.in(0));
    split1ParamConstLayerXML.out(0).connect(split1LayerXML.in(1));
    /* <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< */

    /* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> */
    /* Create Conv2 node and add it to the reference */
    InferenceEngine::LayerParams conv2CommonParams = {"Convolution2", "Convolution", netPrc};
    auto conv2LayerPtr = std::make_shared<InferenceEngine::ConvolutionLayer>(conv2CommonParams);
    CommonTestUtils::conv_common_params conv2Params = {
            {{1, 1}},
            {{5, 5}},
            {{2, 2}},
            {{2, 2}},
            {{1, 1}},
            "",
            1,
            128,
            false,
            true
    };
    conv2LayerPtr->params = CommonTestUtils::convertConvParamToMap(conv2Params);
    if (refLayersVec) refLayersVec->emplace_back(conv2LayerPtr);

    InferenceEngine::SizeVector conv2OutShape(split1OutShape.size());
    getConvOutShape(split1OutShape, conv2Params, conv2OutShape);

    conv2LayerPtr->insData.resize(1);
    conv2LayerPtr->outData.resize(1);
    getInputTo(split1OutData0)[conv2LayerPtr->name] = conv2LayerPtr;
    conv2LayerPtr->insData[0] = split1OutData0;
    auto conv2OutData = std::make_shared<InferenceEngine::Data>("Conv2OutData",
                                                                InferenceEngine::TensorDesc{netPrc, conv2OutShape,
                                                                                            InferenceEngine::Layout::NCHW});
    getCreatorLayer(conv2OutData) = conv2LayerPtr;
    conv2LayerPtr->outData[0] = conv2OutData;

    shapeStr.str("");
    InferenceEngine::SizeVector conv2ConstShape {conv2Params.out_c, split1OutShape[1], conv2Params.kernel[0], conv2Params.kernel[1]};
    std::copy(conv2ConstShape.begin(), conv2ConstShape.end() - 1, std::ostream_iterator<size_t>(shapeStr, ","));
    shapeStr << conv2ConstShape.back();
    offset = offset + 8;

    auto conv2ParamConstLayerXML = ir_builder_v10
            .AddLayer("Conv2_Param_Const", "Const",
                      {{"size", std::to_string(CommonTestUtils::getConvWeightsSize(
                              split1OutShape,
                              conv2Params,
                              netPrc.name()))},
                        {"offset", std::to_string(offset)},
                        {"element_type", FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrc).get_type_name()},
                        {"shape", shapeStr.str()}})
            .AddOutPort(netPrc,
                        {conv2Params.out_c, split1OutShape[1], conv2Params.kernel[0], conv2Params.kernel[1]})
            .getLayer();

    auto conv2LayerXML = ir_builder_v10
            .AddLayer(conv2LayerPtr->name, conv2LayerPtr->type,
                      {{"dilations",  "1,1"},
                       {"pads_begin", "2,2"},
                       {"pads_end",   "2,2"},
                       {"strides",    "1,1"}}
            )
            .AddInPort(netPrc, split1OutShape)
            .AddInPort(netPrc, {conv2Params.out_c, split1OutShape[1], conv2Params.kernel[0], conv2Params.kernel[1]})
            .AddOutPort(netPrc, conv2OutShape)
            .getLayer();
    split1LayerXML.out(0).connect(conv2LayerXML.in(0));
    conv2ParamConstLayerXML.out(0).connect(conv2LayerXML.in(1));
    /* <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< */

    /* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> */
    /* Create Conv3 node and add it to the reference */
    InferenceEngine::LayerParams conv3CommonParams = {"Convolution3", "Convolution", netPrc};
    auto conv3LayerPtr = std::make_shared<InferenceEngine::ConvolutionLayer>(conv3CommonParams);
    CommonTestUtils::conv_common_params conv3Params = {
            {{1, 1}},
            {{5, 5}},
            {{2, 2}},
            {{2, 2}},
            {{1, 1}},
            "",
            1,
            128,
            false,
            true
    };
    conv3LayerPtr->params = CommonTestUtils::convertConvParamToMap(conv3Params);
    if (refLayersVec) refLayersVec->emplace_back(conv3LayerPtr);

    InferenceEngine::SizeVector conv3OutShape(split1OutShape.size());
    getConvOutShape(split1OutShape, conv3Params, conv3OutShape);

    conv3LayerPtr->insData.resize(1);
    conv3LayerPtr->outData.resize(1);
    getInputTo(split1OutData1)[conv3LayerPtr->name] = conv3LayerPtr;
    conv3LayerPtr->insData[0] = split1OutData1;
    auto conv3OutData = std::make_shared<InferenceEngine::Data>("Conv3OutData",
                                                                InferenceEngine::TensorDesc{netPrc, conv3OutShape,
                                                                                            InferenceEngine::Layout::NCHW});
    getCreatorLayer(conv3OutData) = conv3LayerPtr;
    conv3LayerPtr->outData[0] = conv3OutData;

    shapeStr.str("");
    InferenceEngine::SizeVector conv3ConstShape {conv3Params.out_c, split1OutShape[1], conv3Params.kernel[0], conv3Params.kernel[1]};
    std::copy(conv3ConstShape.begin(), conv3ConstShape.end() - 1, std::ostream_iterator<size_t>(shapeStr, ","));
    shapeStr << conv3ConstShape.back();
    offset = offset + CommonTestUtils::getConvWeightsSize(split1OutShape, conv2Params, netPrc.name());

    auto conv3ParamConstLayerXML = ir_builder_v10
            .AddLayer("Conv3_Param_Const", "Const",
                      {{"size", std::to_string(CommonTestUtils::getConvWeightsSize(
                              split1OutShape,
                              conv3Params,
                              netPrc.name()))},
                        {"offset", std::to_string(offset)},
                        {"element_type", FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrc).get_type_name()},
                        {"shape", shapeStr.str()}})
            .AddOutPort(netPrc,
                        {conv3Params.out_c, split1OutShape[1], conv3Params.kernel[0], conv3Params.kernel[1]})
            .getLayer();

    auto conv3LayerXML = ir_builder_v10
            .AddLayer(conv3LayerPtr->name, conv3LayerPtr->type,
                      {{"dilations",  "1,1"},
                       {"pads_begin", "2,2"},
                       {"pads_end",   "2,2"},
                       {"strides",    "1,1"}}
            )
            .AddInPort(netPrc, split1OutShape)
            .AddInPort(netPrc, {conv3Params.out_c, split1OutShape[1], conv3Params.kernel[0], conv3Params.kernel[1]})
            .AddOutPort(netPrc, conv3OutShape)
            .getLayer();
    split1LayerXML.out(1).connect(conv3LayerXML.in(0));
    conv3ParamConstLayerXML.out(0).connect(conv3LayerXML.in(1));
    /* <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< */

    /* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> */
    /* Create Concat1 node and add it to the reference */
    InferenceEngine::LayerParams concat1CommonParams = {"Conv_merge1", "Concat", netPrc};
    auto concat1LayerPtr = std::make_shared<InferenceEngine::ConcatLayer>(concat1CommonParams);
    concat1LayerPtr->params = {
            {"axis", std::to_string(axis)}
    };
    if (refLayersVec) refLayersVec->emplace_back(concat1LayerPtr);

    InferenceEngine::SizeVector concat1OutShape = conv3OutShape;
    if (concat1OutShape.size() != 4) {
        throw std::logic_error("unexpected concat1 input shape");
    }
    concat1OutShape[axis] *= numSplit;

    concat1LayerPtr->insData.resize(2);
    concat1LayerPtr->outData.resize(1);
    getInputTo(conv2OutData)[concat1LayerPtr->name] = concat1LayerPtr;
    getInputTo(conv3OutData)[concat1LayerPtr->name] = concat1LayerPtr;
    concat1LayerPtr->insData[0] = conv2OutData;
    concat1LayerPtr->insData[1] = conv3OutData;
    auto concat1OutData = std::make_shared<InferenceEngine::Data>("Concat1OutData",
                                                                  InferenceEngine::TensorDesc{netPrc,
                                                                                              concat1OutShape,
                                                                                              InferenceEngine::Layout::NCHW});
    getCreatorLayer(concat1OutData) = concat1LayerPtr;
    concat1LayerPtr->outData[0] = concat1OutData;

    auto concat1LayerXML = ir_builder_v10
            .AddLayer(concat1LayerPtr->name, concat1LayerPtr->type, concat1LayerPtr->params)
            .AddInPort(netPrc, conv3OutShape)
            .AddInPort(netPrc, conv3OutShape)
            .AddOutPort(netPrc, concat1OutShape)
            .getLayer();
    conv2LayerXML.out(0).connect(concat1LayerXML.in(0));
    conv3LayerXML.out(0).connect(concat1LayerXML.in(1));
    /* <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< */

    /* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> */
    /* Create Output node                            */
    auto resultLayerXML = ir_builder_v10
            .AddLayer("Output0", "Result")
            .AddInPort(netPrc, concat1OutShape)
            .getLayer();
    concat1LayerXML.out(0).connect(resultLayerXML.in(0));
    /* <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< */

    std::ofstream ofsXml;
    ofsXml.open(modelPath);
    if (!ofsXml) {
        THROW_IE_EXCEPTION << "File '" << modelPath << "' can not be opened as out file stream";
    }
    ofsXml << ir_builder_v10.serialize();
    ofsXml.close();
    if (!ofsXml.good()) {
        THROW_IE_EXCEPTION << "Error during '" << modelPath << "' closing";
    }

    /* Allocate conv blobs and store them to the reference with the required tensor desc */
    conv1LayerPtr->blobs["weights"] = CommonTestUtils::getConvWeightsBlob(
            {conv1Params.out_c, inputDims[1], conv1Params.kernel[0], conv1Params.kernel[1]},
            conv1Params, precision, true);

    conv2LayerPtr->blobs["weights"] = CommonTestUtils::getConvWeightsBlob(
            {conv2Params.out_c, split1OutShape[1], conv2Params.kernel[0], conv2Params.kernel[1]},
            conv2Params, precision, true);

    conv3LayerPtr->blobs["weights"] = CommonTestUtils::getConvWeightsBlob(
            {conv3Params.out_c, split1OutShape[1], conv3Params.kernel[0], conv3Params.kernel[1]},
            conv3Params, precision, true);


    std::ofstream ofsBin;
    ofsBin.open(weightsPath, std::ofstream::out | std::ofstream::binary);
    if (!ofsBin) {
        THROW_IE_EXCEPTION << "File '" << weightsPath << "' can not be opened as out file stream";
    }

    /* Write weights for Conv1 layer to the bin file */
    ofsBin.write(conv1LayerPtr->blobs["weights"]->buffer().as<char *>(),
                 conv1LayerPtr->blobs["weights"]->byteSize());

    /* Write weights for Lrn1 layer to the bin file */
    InferenceEngine::Blob::Ptr lrn1Weights = InferenceEngine::make_shared_blob<uint8_t>(
            {InferenceEngine::Precision::U8,
             {axis * sizeof(InferenceEngine::PrecisionTrait<InferenceEngine::Precision::I64>::value_type)},
             InferenceEngine::Layout::C});
    lrn1Weights->allocate();
    lrn1Weights->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::I64>::value_type *>()[0] = 1;
    ofsBin.write(lrn1Weights->buffer().as<char *>(), lrn1Weights->byteSize());

    /* Write weights for Split1 layer to the bin file */
    InferenceEngine::Blob::Ptr split1Weights = InferenceEngine::make_shared_blob<uint8_t>(
            {InferenceEngine::Precision::U8,
             {axis * sizeof(InferenceEngine::PrecisionTrait<InferenceEngine::Precision::I64>::value_type)},
             InferenceEngine::Layout::C});
    split1Weights->allocate();
    split1Weights->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::I64>::value_type *>()[0] = axis;
    ofsBin.write(split1Weights->buffer().as<char *>(), split1Weights->byteSize());

    /* Write weights for Conv2 layer to the bin file */
    ofsBin.write(conv2LayerPtr->blobs["weights"]->buffer().as<char *>(),
                 conv2LayerPtr->blobs["weights"]->byteSize());
    /* Write weights for Conv3 layer to the bin file */
    ofsBin.write(conv3LayerPtr->blobs["weights"]->buffer().as<char *>(),
                 conv3LayerPtr->blobs["weights"]->byteSize());

    if (!ofsBin.good()) {
        THROW_IE_EXCEPTION << "Error during writing blob weights";
    }
    ofsBin.close();
    if (!ofsBin.good()) {
        THROW_IE_EXCEPTION << "Error during '" << weightsPath << "' closing";
    }
}

std::string getRawConvReluNormPoolFcModel() {
    return (R"V0G0N(
<net name="_NAME_" version="_VER_" batch="1">
    <layers>
        <layer name="data" type="Input" precision="_PRC_" id="0">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>227</dim>
                    <dim>227</dim>
                </port>
            </output>
        </layer>
        <layer name="conv1" type="Convolution" precision="_PRC_" id="1">
            <convolution_data stride-x="4" stride-y="4" pad-x="0" pad-y="0" kernel-x="11" kernel-y="11" output="16" group="1"/>
            <input>
                <port id="1">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>227</dim>
                    <dim>227</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>16</dim>
                    <dim>55</dim>
                    <dim>55</dim>
                </port>
            </output>
            <weights offset="0" size="_CONV_WS_"/>
            <biases offset="_CONV_WS_" size="_CONV_BS_"/>
        </layer>
        <layer name="relu1" type="ReLU" precision="_PRC_" id="2">
            <input>
                <port id="3">
                    <dim>1</dim>
                    <dim>16</dim>
                    <dim>55</dim>
                    <dim>55</dim>
                </port>
            </input>
            <output>
                <port id="4">
                    <dim>1</dim>
                    <dim>16</dim>
                    <dim>55</dim>
                    <dim>55</dim>
                </port>
            </output>
        </layer>
        <layer name="norm1" type="Norm" precision="_PRC_" id="3">
            <norm_data alpha="9.9999997e-05" beta="0.75" local-size="5" region="across"/>
            <input>
                <port id="5">
                    <dim>1</dim>
                    <dim>16</dim>
                    <dim>55</dim>
                    <dim>55</dim>
                </port>
            </input>
            <output>
                <port id="6">
                    <dim>1</dim>
                    <dim>16</dim>
                    <dim>55</dim>
                    <dim>55</dim>
                </port>
            </output>
        </layer>
        <layer name="pool1" type="Pooling" precision="_PRC_" id="4">
            <pooling_data kernel-x="3" kernel-y="3" pad-x="0" pad-y="0" stride-x="2" stride-y="2" rounding-type="ceil" pool-method="max"/>
            <input>
                <port id="7">
                    <dim>1</dim>
                    <dim>16</dim>
                    <dim>55</dim>
                    <dim>55</dim>
                </port>
            </input>
            <output>
                <port id="8">
                    <dim>1</dim>
                    <dim>16</dim>
                    <dim>27</dim>
                    <dim>27</dim>
                </port>
            </output>
        </layer>
        <layer name="fc6" type="FullyConnected" precision="_PRC_" id="5">
            <fc_data out-size="10"/>
            <input>
                <port id="9">
                    <dim>1</dim>
                    <dim>16</dim>
                    <dim>27</dim>
                    <dim>27</dim>
                </port>
            </input>
            <output>
                <port id="10">
                    <dim>1</dim>
                    <dim>10</dim>
                </port>
            </output>
            <weights offset="_FC_W_OFFS_" size="_FC_WS_"/>
            <biases offset="_FC_B_OFFS_" size="_FC_BS_"/>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="3"/>
        <edge from-layer="2" from-port="4" to-layer="3" to-port="5"/>
        <edge from-layer="3" from-port="6" to-layer="4" to-port="7"/>
        <edge from-layer="4" from-port="8" to-layer="5" to-port="9"/>
    </edges>
</net>
)V0G0N");
}

TestModel getConvReluNormPoolFcModel(InferenceEngine::Precision netPrc) {
    std::string model_str = getRawConvReluNormPoolFcModel();
    /* Default values for FP16 are used */
    size_t convWeigthsLen = 5808;  // kernel_x * kernel_y * in_channels * out_channels
    size_t convWeigthsSize = convWeigthsLen * 2;  // 2 (bytes in FP16)
    size_t convBiasesLen = 16;  // out_channels
    size_t convBiasesSize = convBiasesLen * 2;
    size_t fcWeigthsLen = 116640;  // fc_in_channels * fc_h * fc_w * fc_out_channels
    size_t fcWeigthsSize = fcWeigthsLen * 2;
    size_t fcBiasesLen = 10;  // fc_out_channels
    size_t fcBiasesSize = fcBiasesLen * 2;
    switch (netPrc) {
        case InferenceEngine::Precision::FP16:
        case InferenceEngine::Precision::Q78:
            break;
        case InferenceEngine::Precision::FP32:
            convWeigthsSize *= 2;  // 4 bytes in FP32
            convBiasesSize *= 2;
            fcWeigthsSize *= 2;
            fcBiasesSize *= 2;
            break;
        default:
            std::string err = "ConvReluNormPoolFcModel can not be constructed with precision ";
            err += netPrc.name();
            throw std::runtime_error(err);
    }
    std::string irName = std::string("ConvReluNormPoolFcModel") + netPrc.name();
    REPLACE_WITH_STR(model_str, "_NAME_", irName);
    REPLACE_WITH_NUM(model_str, "_VER_", 2);
    REPLACE_WITH_STR(model_str, "_PRC_", netPrc.name());
    REPLACE_WITH_NUM(model_str, "_CONV_WS_", convWeigthsSize);
    REPLACE_WITH_NUM(model_str, "_CONV_BS_", convBiasesSize);
    REPLACE_WITH_NUM(model_str, "_FC_W_OFFS_", convWeigthsSize + convBiasesSize);
    REPLACE_WITH_NUM(model_str, "_FC_WS_", fcWeigthsSize);
    REPLACE_WITH_NUM(model_str, "_FC_B_OFFS_", convWeigthsSize + convBiasesSize + fcWeigthsSize);
    REPLACE_WITH_NUM(model_str, "_FC_BS_", fcBiasesSize);
    return TestModel(model_str, CommonTestUtils::getWeightsBlob(
            convWeigthsSize + convBiasesSize + fcWeigthsSize + fcBiasesSize));
}


TestModel getModelWithMemory(InferenceEngine::Precision netPrc) {
    CommonTestUtils::IRBuilder_v6 test_model_builder("model");

    auto Memory_1_layer =
            test_model_builder.AddLayer("Memory_1", "Memory", netPrc, {{"id",    "r_1-3"},
                                                                       {"index", "1"},
                                                                       {"size",  "2"}})
                    .AddOutPort({1, 200})
                    .getLayer();
    auto Input_2_layer = test_model_builder.AddLayer("Input_2", "input", netPrc).AddOutPort({1, 200}).getLayer();
    auto Eltwise_3_layer = test_model_builder.AddLayer("Eltwise_3", "Eltwise", netPrc, {{"operation", "mul"}})
            .AddInPort({1, 200})
            .AddInPort({1, 200})
            .AddOutPort({1, 200})
            .getLayer();

    auto Activation_4_layer =
            test_model_builder.AddLayer("Activation_4", "Activation", netPrc, {{"type", "sigmoid"}})
                    .AddInPort({1, 200})
                    .AddOutPort({1, 200})
                    .getLayer();
    auto Memory_5_layer =
            test_model_builder.AddLayer("Memory_5", "Memory", netPrc, {{"id",    "r_1-3"},
                                                                       {"index", "0"},
                                                                       {"size",  "2"}})
                    .AddInPort({1, 200})
                    .getLayer();

    test_model_builder.AddEdge(Memory_1_layer.out(0), Eltwise_3_layer.in(0));
    test_model_builder.AddEdge(Input_2_layer.out(0), Eltwise_3_layer.in(1));
    test_model_builder.AddEdge(Eltwise_3_layer.out(0), Activation_4_layer.in(0));
    test_model_builder.AddEdge(Activation_4_layer.out(0), Memory_5_layer.in(0));

    auto serial = test_model_builder.serialize();

    return TestModel(serial, CommonTestUtils::getWeightsBlob(0));
}
TestModel getModelWithMultipleMemoryConnections(InferenceEngine::Precision netPrc) {
    CommonTestUtils::IRBuilder_v6 test_model_builder("model");

    auto Memory_1_layer =
        test_model_builder.AddLayer("Memory_1", "Memory", netPrc, { {"id",    "r_1-3"},
                                                                    {"index", "1"},
                                                                    {"size",  "2"} })
        .AddOutPort({ 1, 200 })
        .getLayer();
    auto Input_1_layer = test_model_builder.AddLayer("Input_1", "input", netPrc).AddOutPort({ 1, 200 }).getLayer();
    auto Eltwise_1_layer = test_model_builder.AddLayer("Eltwise_1", "Eltwise", netPrc, { {"operation", "mul"} })
        .AddInPort({ 1, 200 })
        .AddInPort({ 1, 200 })
        .AddOutPort({ 1, 200 })
        .getLayer();

    auto Memory_2_layer =
        test_model_builder.AddLayer("Memory_2", "Memory", netPrc, { {"id",    "c_1-3"},
                                                                    {"index", "1"},
                                                                    {"size",  "2"} })
        .AddOutPort({ 1, 200 })
        .getLayer();
    auto Eltwise_2_layer = test_model_builder.AddLayer("Eltwise_2", "Eltwise", netPrc, { {"operation", "mul"} })
        .AddInPort({ 1, 200 })
        .AddInPort({ 1, 200 })
        .AddOutPort({ 1, 200 })
        .getLayer();
    auto Memory_3_layer =
        test_model_builder.AddLayer("Memory_3", "Memory", netPrc, { {"id",    "c_1-3"},
                                                                   {"index", "0"},
                                                                   {"size",  "2"} })
        .AddInPort({ 1, 200 })
        .getLayer();

    auto Activation_1_layer =
        test_model_builder.AddLayer("Activation_1", "Activation", netPrc, { {"type", "sigmoid"} })
        .AddInPort({ 1, 200 })
        .AddOutPort({ 1, 200 })
        .getLayer();
    auto Memory_4_layer =
        test_model_builder.AddLayer("Memory_4", "Memory", netPrc, { {"id",    "r_1-3"},
                                                                   {"index", "0"},
                                                                   {"size",  "2"} })
        .AddInPort({ 1, 200 })
        .getLayer();

    test_model_builder.AddEdge(Memory_1_layer.out(0), Eltwise_1_layer.in(0));
    test_model_builder.AddEdge(Input_1_layer.out(0), Eltwise_1_layer.in(1));
    test_model_builder.AddEdge(Eltwise_1_layer.out(0), Eltwise_2_layer.in(1));
    test_model_builder.AddEdge(Memory_2_layer.out(0), Eltwise_2_layer.in(0));
    test_model_builder.AddEdge(Eltwise_2_layer.out(0), Memory_3_layer.in(0));
    test_model_builder.AddEdge(Eltwise_2_layer.out(0), Activation_1_layer.in(0));
    test_model_builder.AddEdge(Activation_1_layer.out(0), Memory_4_layer.in(0));

    auto serial = test_model_builder.serialize();

    return TestModel(serial, CommonTestUtils::getWeightsBlob(0));
}
}  // namespace TestModel
}  // namespace FuncTestUtils
