// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_ir_parser.hpp"
#include "ie_ir_itt.hpp"

#include <memory>
#include <ngraph/ngraph.hpp>
#include <ngraph/op/util/sub_graph_base.hpp>
#include <ir_frontend/frontend.hpp>
#include <ngraph/ops.hpp>
#include <sstream>
#include <string>
#include <typeinfo>
#include <unordered_set>
#include <vector>

#include <cpp/ie_cnn_network.h>
#include "blob_factory.hpp"

using namespace XMLParseUtils;
namespace InferenceEngine {

IRParser::IRParser(size_t version) : IRParser(version, {}) {}

IRParser::IRParser(size_t version, const std::vector<InferenceEngine::IExtensionPtr>& exts) {
    switch (version) {
    case 10:
        parser = std::make_shared<V10Parser>(exts);
        break;
    default:
        IE_THROW() << "Unsupported IR version: " << version;
    }
}

CNNNetwork IRParser::parse(
    const pugi::xml_node& root, const Blob::CPtr& weights) {
    return parser->parse(root, weights);
}

V10Parser::V10Parser(const std::vector<IExtensionPtr>& exts) : _exts(exts) {}

CNNNetwork V10Parser::parse(
    const pugi::xml_node& root, const Blob::CPtr& weights) {

    auto ir_fe = std::make_shared<ngraph::frontend::FrontEndIR>();
    auto model = ir_fe->load(root, weights, _exts);
    auto f = ir_fe->convert(model);

    OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::V10Reader_RT, "ConstructCNNNetwork");

    CNNNetwork net(f, _exts);
    parsePreProcess(net, root, weights);

    return net;
}

void V10Parser::parsePreProcess(
    CNNNetwork& network, const pugi::xml_node& root, const Blob::CPtr& weights) {
    /*
        <pre-process mean-precision="FP32">
        <channel id = ”0”>
        <mean offset = "121930449" size = "51529" / >  // in case of array – ref to the .bin file
        </channel>
        </pre-process>
    */

    auto ppNode = root.child("pre-process");
    if (ppNode.empty()) {
        return;
    }
    // find out to what input this belongs to
    std::string inputName;
    InputInfo::Ptr preProcessInput;

    inputName = GetStrAttr(ppNode, "reference-layer-name", "");
    inputName = ngraph::trim(inputName);
    if (inputName.empty()) {
        // fallback (old format), look for the picture in the inputs
        InputsDataMap inputs = network.getInputsInfo();

        if (inputs.empty()) IE_THROW() << "network has no input";

        for (auto i : inputs) {
            if (i.second->getTensorDesc().getDims().size() == 4) {
                preProcessInput = i.second;
                break;
            }
        }
        if (!preProcessInput) {
            preProcessInput = inputs.begin()->second;
        }

        inputName = preProcessInput->name();
    } else {
        preProcessInput = network.getInputsInfo()[inputName];
        if (!preProcessInput)
            IE_THROW() << "pre-process name ref '" << inputName
                               << "' refers to un-existing input";
    }

    // dims vector without batch size
    SizeVector inputDims = preProcessInput->getTensorDesc().getDims();
    size_t noOfChannels = 0, width = 0, height = 0;

    if (inputDims.size() < 2) {
        IE_THROW() << "network did not define input dimensions properly";
    } else if (inputDims.size() == 2) {  // NC
        noOfChannels = inputDims[1];
        width = inputDims[1];
        height = inputDims[0];
    } else if (inputDims.size() == 3) {
        width = inputDims[2];
        height = inputDims[1];
        noOfChannels = inputDims[0];
    } else if (inputDims.size() == 4) {
        width = inputDims[3];
        height = inputDims[2];
        noOfChannels = inputDims[1];
    } else if (inputDims.size() == 5) {
        width = inputDims[4];
        height = inputDims[3];
        noOfChannels = inputDims[2];
    }

    PreProcessInfo& pp = preProcessInput->getPreProcess();
    pp.init(noOfChannels);

    auto meanSegmentPrecision = GetPrecisionAttr(ppNode, "mean-precision", Precision::UNSPECIFIED);
    if (!meanSegmentPrecision || meanSegmentPrecision == Precision::MIXED)
        IE_THROW() << "mean blob defined without specifying precision.";

    int lastChanNo = -1;
    std::unordered_set<int> idsForMeanImage;

    FOREACH_CHILD(chan, ppNode, "channel") {
        int chanNo = GetIntAttr(chan, "id", lastChanNo + 1);
        if (chanNo >= static_cast<int>(noOfChannels) || chanNo < 0) {
            IE_THROW() << "Pre-process channel id invalid: " << chanNo;
        }
        lastChanNo = chanNo;

        auto meanNode = chan.child("mean");
        if (!meanNode.empty()) {
            if (!meanNode.attribute("size")) {
                IE_THROW() << "mean should have the attribute: size";
            }
            if (meanNode.attribute("size")) {
                idsForMeanImage.insert(chanNo);
                size_t size = static_cast<size_t>(GetIntAttr(meanNode, "size"));
                size_t offset = static_cast<size_t>(GetIntAttr(meanNode, "offset"));
                if (width * height * meanSegmentPrecision.size() != size) {
                    IE_THROW() << "mean blob size mismatch expected input, got: " << size
                                       << " extpecting " << width << " x " << height << " x "
                                       << meanSegmentPrecision.size();
                }
                auto meanData = make_blob_with_precision(
                    TensorDesc(meanSegmentPrecision, {height, width}, Layout::HW));
                meanData->allocate();
                auto lockedMem = meanData->buffer();
                char* data = lockedMem.as<char*>();
                uint8_t* src_data = weights->cbuffer().as<uint8_t*>() + offset;
                memcpy(data, src_data, size);

                pp.setMeanImageForChannel(meanData, chanNo);
            }
        }
    }

    if (idsForMeanImage.size() == noOfChannels) {
        pp.setVariant(MEAN_IMAGE);
    } else if (idsForMeanImage.size() == 0) {
        pp.setVariant(NONE);
    } else {
        std::string validMeanImageIds = "";
        for (auto id : idsForMeanImage) {
            validMeanImageIds += std::to_string(id) + " ";
        }
        IE_THROW() << "mean is not provided for all channels\n"
                              "Provided mean image for: "
                           << validMeanImageIds;
    }
}
}  // namespace InferenceEngine
