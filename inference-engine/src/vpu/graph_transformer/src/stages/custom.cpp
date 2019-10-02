// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <vector>
#include <memory>
#include <string>
#include <map>
#include <unordered_set>
#include <utility>
#include <algorithm>
#include <tuple>

#include <vpu/custom_layer.hpp>
#include <vpu/utils/simple_math.hpp>

namespace vpu {

namespace {

class KernelBinaryContent final : public DataContent {
public:
    explicit KernelBinaryContent(const std::string& blob) : _blob(blob) {
        IE_ASSERT(!_blob.empty());
    }

    const void* getRaw() const override {
        IE_ASSERT(desc().totalDimSize() * desc().elemSize() == _blob.length());
        return _blob.data();
    }

private:
    std::string _blob;
};

class CustomStage final : public StageNode {
public:
    using StageNode::StageNode;

private:
    StagePtr cloneImpl() const override {
        return std::make_shared<CustomStage>(*this);
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
        const auto& inputOrders = attrs().get<std::map<int, DimsOrder>>("inputOrders");
        const auto& outputOrders = attrs().get<std::map<int, DimsOrder>>("outputOrders");

        for (const auto& inEdge : inputEdges()) {
            // last input is always OpenCL binary, so use it as is.
            if (inEdge->portInd() == numInputs() - 1) {
                break;
            }

            auto it = inputOrders.find(inEdge->portInd());
            if (it != inputOrders.end()) {
                auto requiredOrder = it->second;
                orderInfo.setInput(inEdge, requiredOrder);
            }
        }

        for (const auto& outEdge : outputEdges()) {
            auto it = outputOrders.find(outEdge->portInd());
            if (it != outputOrders.end()) {
                auto requiredOrder = it->second;
                orderInfo.setOutput(outEdge, requiredOrder);
            }
        }
    }

    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) override {
        for (const auto& inEdge : inputEdges()) {
            // last input is always OpenCL binary, so use it as is.
            if (inEdge->portInd() == numInputs() - 1) {
                break;
            }

            stridesInfo.setInput(inEdge, StridesRequirement::compact());
        }
        for (const auto& outEdge : outputEdges()) {
            stridesInfo.setOutput(outEdge, StridesRequirement::compact());
        }
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& batchInfo) override {
        std::vector<CustomDataFormat> formats = attrs().get<std::vector<CustomDataFormat>>("formats");

        for (const auto& inEdge : inputEdges()) {
            IE_ASSERT(inEdge->portInd() < formats.size());

            // last input is always OpenCL binary, so use it as is.
            if ((inEdge->portInd() == numInputs() - 1) || (formats[inEdge->portInd()] == CustomDataFormat::Any)) {
                break;
            }

            batchInfo.setInput(inEdge, BatchSupport::Split);
        }
        for (const auto& outEdge : outputEdges()) {
            batchInfo.setOutput(outEdge, BatchSupport::Split);
        }
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        const auto& customLayer = attrs().get<CustomLayer::Ptr>("customLayer");
        const auto& gws = attrs().get<SmallVector<int, 3>>("gws");
        const auto& lws = attrs().get<SmallVector<int, 3>>("lws");
        const auto& ports = attrs().get<std::map<std::string, int>>("ports");

        //
        // GWG, LWG, Offs
        //

        for (auto x : gws) {
            serializer.append(static_cast<uint32_t>(x));
        }

        for (auto x : lws) {
            serializer.append(static_cast<uint32_t>(x));
        }

        for (int i = 0; i < lws.size(); ++i) {
            serializer.append(static_cast<uint32_t>(0));
        }

        serializer.append(static_cast<uint32_t>(customLayer->maxShaves()));

        //
        // Entry point
        //
        IE_ASSERT(customLayer->stageNumInputs() >= 0);
        serializer.append(static_cast<uint32_t>(customLayer->stageNumInputs()));
        serializer.append(static_cast<uint32_t>(customLayer->kernelAddress(lws[0])));

        //
        // Total number of blobs
        //

        serializer.append(static_cast<int32_t>(numInputs() + numOutputs()));

        //
        // Number of kernel parameters
        //

        serializer.append(static_cast<uint32_t>(customLayer->parameters().size()));

        //
        // Parameters & relocation info
        //

        std::map<std::string, CustomLayer::KernelParam> b2b;
        for (const auto& kp : customLayer->bindings()) {
            b2b[kp.argName] = kp;
        }

        IE_ASSERT(origLayer() != nullptr);

        for (const auto& kp : customLayer->parameters()) {
            const auto& parameter = b2b[kp];

            switch (parameter.type) {
                case CustomParamType::Input:
                case CustomParamType::Output:
                case CustomParamType::InputBuffer:
                case CustomParamType::OutputBuffer:
                case CustomParamType::Data:
                {
                    if (ports.find(kp) == ports.end()) {
                        VPU_THROW_EXCEPTION
                            << "Unable to bind parameter " << parameter.argName << " for "
                            << origLayer()->type <<" layer. Name is: " << origLayer()->name;
                    }
                    int id = ports.find(kp)->second;
                    serializer.append(static_cast<uint32_t>(0));
                    serializer.append(static_cast<uint32_t>(id));

                    break;
                }
                case CustomParamType::Int:
                case CustomParamType::Float:
                {
                    if (origLayer()->params.find(parameter.irSource) != origLayer()->params.end()) {
                        std::stringstream parameterStream(origLayer()->params[parameter.irSource]);
                        std::string param;
                        for (int i = 0; i <= parameter.portIndex; i++) {
                            getline(parameterStream, param, ',');
                        }

                        if (parameter.type == CustomParamType::Int) {
                            serializer.append(static_cast<int32_t>(std::stoi(param)));
                            serializer.append(static_cast<int32_t>(-1));
                        } else {
                            serializer.append(static_cast<float>(std::stof(param) ));
                            serializer.append(static_cast<int32_t>(-2));
                        }
                        break;
                    } else {
                        auto pos = parameter.irSource.find_first_of('.');
                        if (pos != std::string::npos) {
                            auto blob = parameter.irSource.substr(0, pos);
                            auto dim = parameter.irSource.substr(pos + 1, std::string::npos);

                            IE_ASSERT(dim.length() == 1)
                                    << "Unable to deduce parameter " << parameter.argName << " for "
                                    << origLayer()->type <<" layer. Name is: " << origLayer()->name;
                            char dimLetter = dim[0];

                            ie::DataPtr origData;
                            if (blob == "I") {
                                origData = origLayer()->insData[parameter.portIndex].lock();
                            } else {
                                origData = origLayer()->outData[parameter.portIndex];
                            }
                            IE_ASSERT(origData != nullptr);

                            auto dims = origData->getDims();
                            int ndims = dims.size();

                            if (ndims > 4)
                                VPU_THROW_EXCEPTION
                                    << "Unable to deduce parameter " << parameter.argName << " for "
                                    << origLayer()->type <<" layer. Name is: " << origLayer()->name;

                            const std::map<char, int> vars = {
                                { 'b', 0 }, { 'B', 0 },
                                { 'f', 1 }, { 'F', 1 },
                                { 'y', 2 }, { 'Y', 2 },
                                { 'x', 3 }, { 'X', 3 },
                            };

                            auto var = vars.find(dimLetter);
                            if (var != vars.end()) {
                                auto res = dims.at(var->second-4+ndims);

                                serializer.append(static_cast<uint32_t>(res));
                                serializer.append(static_cast<int32_t>(-1));
                            } else {
                                VPU_THROW_EXCEPTION
                                    << "Unable to deduce parameter " << parameter.argName << " for "
                                    << origLayer()->type <<" layer. Name is: " << origLayer()->name;
                            }

                            break;
                        } else {
                            try {
                                if (parameter.type == CustomParamType::Int) {
                                    serializer.append(static_cast<int32_t>(std::stoi(parameter.irSource)));
                                    serializer.append(static_cast<int32_t>(-1));
                                } else {
                                    serializer.append(static_cast<float>(std::stof(parameter.irSource) ));
                                    serializer.append(static_cast<int32_t>(-2));
                                }
                                break;
                            }
                            catch (const std::invalid_argument&) {
                                VPU_THROW_EXCEPTION
                                    << "Unable to deduce parameter " << parameter.argName << " for "
                                    << origLayer()->type <<" layer. Name is: " << origLayer()->name
                                    <<", parameter is: " << parameter.irSource;
                            }
                        }
                    }
                }
                default:
                    VPU_THROW_EXCEPTION
                        << "Unable to deduce parameter " << parameter.argName << " for "
                        << origLayer()->type <<" layer. Name is: " << origLayer()->name;
            }
        }
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        IE_ASSERT(numTempBuffers() == 1);

        for (const auto& inEdge : inputEdges()) {
            inEdge->input()->serializeOldBuffer(this, serializer);
        }

        for (const auto& outEdge : outputEdges()) {
            outEdge->output()->serializeOldBuffer(this, serializer);
        }

        for (const auto& tempEdge : tempBufferEdges()) {
            tempEdge->tempBuffer()->serializeOldBuffer(this, serializer);
        }
    }
};

}  // namespace

static void calcSizesFromParams(const DataDesc &desc, const SmallVector<std::string> &bufferSizeRules, SmallVector<int, 3> &sizes) {
    // assume output tensor is dimension source by default
    auto batchDim = desc.dim(Dim::N, 1);
    auto featureDim = desc.dim(Dim::C, 1);
    auto yDim = desc.dim(Dim::H, 1);
    auto xDim = desc.dim(Dim::W, 1);

    const std::map<char, int> vars = {
        { 'b', batchDim },   { 'B', batchDim },
        { 'f', featureDim }, { 'F', featureDim },
        { 'y', yDim },       { 'Y', yDim },
        { 'x', xDim },       { 'X', xDim },
    };

    sizes.reserve(std::max<size_t>(bufferSizeRules.size(), 3));
    for (const auto& rule : bufferSizeRules) {
        SimpleMathExpression expr;
        expr.setVariables(vars);
        expr.parse(rule);
        sizes.emplace_back(expr.evaluate());
    }
    while (sizes.size() < 3) {
        sizes.emplace_back(1);
    }
}

static CustomLayer::Ptr chooseSuitable(const std::vector<CustomLayer::Ptr>& customLayers,
                                       const std::map<std::string, std::string>& layerParams) {
    ie::details::CaselessEq<std::string> cmp;

    for (const auto& customLayer : customLayers) {
        bool suitable = true;
        for (const auto& whereParam : customLayer->whereParams()) {
            if (layerParams.find(whereParam.first) == layerParams.end() ||
                !cmp(layerParams.find(whereParam.first)->second, whereParam.second)) {
                suitable = false;
            }
        }
        if (suitable) {
            return customLayer;
        }
    }

    IE_ASSERT(false);
    return CustomLayer::Ptr(nullptr);
}

void FrontEnd::parseCustom(
        const Model::Ptr& model,
        const ie::CNNLayerPtr& layer,
        const DataVector& inputs,
        const DataVector& outputs) {
    IE_ASSERT(layer != nullptr);
    IE_ASSERT(outputs.size() == 1);

    std::vector<CustomLayer::Ptr> customLayersForType;
    if (_customLayers.count(layer->type) > 0) {
        customLayersForType.push_back(chooseSuitable(_customLayers.find(layer->type)->second, layer->params));
    } else if (_customLayers.count(layer->type + "@stage_0") > 0) {
        int stageNum = 0;
        while (_customLayers.count(layer->type + "@stage_" + std::to_string(stageNum)) > 0) {
            customLayersForType.push_back(chooseSuitable(_customLayers.find(layer->type + "@stage_" + std::to_string(stageNum))->second,
                                                         layer->params));
            stageNum++;
        }
    } else {
        IE_ASSERT(false);
    }

    // Get all buffers, buffers must be unique associated by port index
    std::map<int, Data> tempBuffsMap;
    for (size_t stageNum = 0; stageNum < customLayersForType.size(); stageNum++) {
        for (auto& param : customLayersForType[stageNum]->bindings()) {
            if (param.type == CustomParamType::InputBuffer || param.type == CustomParamType::OutputBuffer) {
                SmallVector<int, 3> sizes;
                auto desc = (param.dimSource == CustomDimSource::Input) ? inputs[param.dimIdx]->desc() : outputs[param.dimIdx]->desc();
                calcSizesFromParams(desc, param.bufferSizeRules, sizes);
                auto buf = model->addNewData("custom_" + layer->type + "_buf", DataDesc({sizes[0], sizes[1], sizes[2], 1}));
                if (tempBuffsMap.find(param.portIndex) == tempBuffsMap.end()) {
                    tempBuffsMap[param.portIndex] = buf;
                }
            }
        }
    }

    // Gather inputs and outputs for each stage for the layer
    for (int stage_num = 0; stage_num < customLayersForType.size(); stage_num++) {
        auto customLayer = customLayersForType[stage_num];

        std::map<std::string, int> ports;
        std::vector<CustomDataFormat> formats;

        // Gather inputs
        DataVector stageInputs;
        for (auto& param : customLayer->bindings()) {
            if (param.type == CustomParamType::Input) {
                ports[param.argName] = stageInputs.size();
                formats.emplace_back(param.format);
                stageInputs.emplace_back(inputs[param.portIndex]);
            } else if (param.type == CustomParamType::InputBuffer) {
                ports[param.argName] = stageInputs.size();
                formats.emplace_back(CustomDataFormat::BFYX);
                stageInputs.emplace_back(tempBuffsMap[param.portIndex]);
            }
        }

        // Gather data blobs
        for (auto& param : customLayer->bindings()) {
            if (param.type == CustomParamType::Data) {
                auto blobIterator = layer->blobs.find(param.irSource);
                if (blobIterator != layer->blobs.end()) {
                    auto origBlob = blobIterator->second;
                    auto customBlob = model->addConstData(
                        layer->name + "@" + param.irSource,
                        DataDesc({origBlob->size()}),
                        ieBlobContent(origBlob));
                    ports[param.argName] = stageInputs.size();
                    formats.emplace_back(param.format);
                    stageInputs.emplace_back(std::move(customBlob));
                }
            }
        }

        customLayer->setStageNumInputs(stageInputs.size());
        formats.emplace_back(CustomDataFormat::Any);

        // Get kernel binary
        auto kernelNode = kernelNodes.find(customLayer->kernelBinary());
        if (kernelNode != kernelNodes.end()) {
            stageInputs.emplace_back((kernelNode->second));
        } else {
            auto kernelBinaryDesc = DataDesc({customLayer->kernelBinary().length()});
            kernelBinaryDesc.setType(DataType::U8);

            auto kernelBinary = model->addConstData(
                layer->type + "@kernelBinary",
                kernelBinaryDesc,
                std::make_shared<KernelBinaryContent>(customLayer->kernelBinary()));
            stageInputs.emplace_back((kernelBinary));
            kernelNodes[customLayer->kernelBinary()] = kernelBinary;
        }

        DataVector stageOutputs;
        for (auto& param : customLayer->bindings()) {
            if (param.type == CustomParamType::Output) {
                ports[param.argName] = stageInputs.size() + stageOutputs.size();
                stageOutputs.emplace_back(outputs[param.portIndex]);
            } else if (param.type == CustomParamType::OutputBuffer) {
                ports[param.argName] = stageInputs.size() + stageOutputs.size();
                stageOutputs.emplace_back(tempBuffsMap[param.portIndex]);
            }
        }

        auto stage = model->addNewStage<CustomStage>(
            layer->name + ((customLayersForType.size() == 1) ? "" : "@stage_" + std::to_string(stage_num)),
            StageType::Custom,
            layer,
            stageInputs,
            stageOutputs);

        stage->attrs().set("customLayer", customLayer);
        stage->attrs().set("ports", ports);
        stage->attrs().set("formats", formats);

        SmallVector<int, 3> gws;
        SmallVector<int, 3> lws;
        auto dimSource = (customLayer->dimSource() == CustomDimSource::Input) ? inputs : outputs;
        calcSizesFromParams(dimSource[customLayer->dimSourceIndex()]->desc(), customLayer->globalSizeRules(), gws);
        calcSizesFromParams(dimSource[customLayer->dimSourceIndex()]->desc(), customLayer->localSizeRules(), lws);

        stage->attrs().set("gws", gws);
        stage->attrs().set("lws", lws);

        std::map<int, DimsOrder> inputOrders;
        std::map<int, DimsOrder> outputOrders;

        std::map<std::string, CustomLayer::KernelParam> b2b;
        for (const auto& kp : customLayer->bindings()) {
            b2b[kp.argName] = kp;
        }

        const std::map<CustomDataFormat, DimsOrder> formatsMap = {
            { CustomDataFormat::BYXF, DimsOrder::NHWC },
            { CustomDataFormat::BFYX, DimsOrder::NCHW },
            { CustomDataFormat::YXF, DimsOrder::HWC },
            { CustomDataFormat::FYX, DimsOrder::CHW }
        };

        for (const auto& kp : customLayer->parameters()) {
            const auto& parameter = b2b[kp];

            if (parameter.type == CustomParamType::Input) {
                auto it = formatsMap.find(parameter.format);
                if (it != formatsMap.end()) {
                    auto requiredOrder = it->second;
                    inputOrders[parameter.portIndex] = requiredOrder;
                }
            }

            if (parameter.type == CustomParamType::Output) {
                auto it = formatsMap.find(parameter.format);
                if (it != formatsMap.end()) {
                    auto requiredOrder = it->second;
                    outputOrders[parameter.portIndex] = requiredOrder;
                }
            }
        }

        stage->attrs().set("inputOrders", std::move(inputOrders));
        stage->attrs().set("outputOrders", std::move(outputOrders));

        int buffer_size = customLayer->kernelBinary().length() + 1024;
        model->addTempBuffer(
            stage,
            DataDesc({buffer_size}));
    }
}

}  // namespace vpu
