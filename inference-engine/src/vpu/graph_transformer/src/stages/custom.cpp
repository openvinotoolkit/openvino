// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <vpu/frontend/custom_layer.hpp>
#include <vpu/utils/simple_math.hpp>
#include <vpu/model/data_contents/kernel_binary_content.hpp>
#include <vpu/model/data_contents/ie_blob_content.hpp>

#include <vector>
#include <memory>
#include <string>
#include <map>
#include <utility>
#include <algorithm>
#include <tuple>

namespace vpu {

static SmallVector<int> calcSizesFromParams(const DataDesc& desc, const SmallVector<std::string>& bufferSizeRules,
                                            std::map<std::string, std::string> layerParams);

namespace {

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
        const auto& kernel = attrs().get<CustomKernel>("customKernel");
        const auto& gws = attrs().get<SmallVector<int>>("gws");
        const auto& lws = attrs().get<SmallVector<int>>("lws");
        const auto& ports = attrs().get<std::map<std::string, int>>("ports");
        const auto& localDataSizes = attrs().get<std::map<std::string, int>>("localDataSizes");

        for (int i = 0; i < gws.size(); ++i) {
            serializer.append(static_cast<uint32_t>(gws[i] / lws[i]));
        }

        for (auto x : lws) {
            serializer.append(static_cast<uint32_t>(x));
        }

        for (int i = 0; i < lws.size(); ++i) {
            serializer.append(static_cast<uint32_t>(0));
        }

        serializer.append(static_cast<uint32_t>(kernel.maxShaves()));
        serializer.append(static_cast<uint32_t>(kernel.kernelId()));
        serializer.append(static_cast<uint32_t>(kernel.inputDataCount()));
        serializer.append(static_cast<int32_t>(numInputs() + numOutputs()));
        serializer.append(static_cast<uint32_t>(kernel.parameters().size()));

        std::map<std::string, CustomKernel::KernelParam> b2b;
        for (const auto& kp : kernel.bindings()) {
            b2b[kp.argName] = kp;
        }

        IE_ASSERT(origLayer() != nullptr);

        for (const auto& kp : kernel.parameters()) {
            const auto& parameter = b2b[kp];

            switch (parameter.type) {
            case CustomParamType::Input:
            case CustomParamType::Output:
            case CustomParamType::InputBuffer:
            case CustomParamType::OutputBuffer:
            case CustomParamType::Data: {
                VPU_THROW_UNLESS(ports.find(kp) != ports.end(),
                    "XML specification for %s layer has no definition for '%s' parameter. Layer name: %s",
                    origLayer()->type, kp, origLayer()->name);

                int id = ports.find(kp)->second;
                serializer.append(static_cast<uint32_t>(0));
                serializer.append(static_cast<uint32_t>(id));
                break;
            }
            case CustomParamType::Int:
            case CustomParamType::Float: {
                const auto cnnParam = origLayer()->params.find(parameter.irSource);
                if (cnnParam != origLayer()->params.end()) {
                    const auto param = [&]() -> std::string {
                        if (parameter.portIndex < 0) {
                            return cnnParam->second;
                        }

                        VPU_THROW_UNLESS(cnnParam->second.find(',') != std::string::npos,
                            "Error while parsing CNNetwork parameter '%s' for '%s' layer: port-index=%d is set, "
                            "but parameter is neither a tensor, nor an array type.",
                            cnnParam->first, origLayer()->type, parameter.portIndex);

                        std::string value;
                        std::stringstream parameterStream{cnnParam->second};
                        for (int i = 0; i <= parameter.portIndex; i++) {
                            getline(parameterStream, value, ',');
                        }
                        return value;
                    }();

                    if (parameter.type == CustomParamType::Int) {
                        serializer.append(static_cast<int32_t>(std::stoi(param)));
                        serializer.append(static_cast<int32_t>(-1));
                    } else {
                        serializer.append(static_cast<float>(std::stof(param)));
                        serializer.append(static_cast<int32_t>(-2));
                    }
                    break;
                } else {
                    auto pos = parameter.irSource.find_first_of('.');
                    if (pos != std::string::npos) {
                        auto blob = parameter.irSource.substr(0, pos);
                        auto dim = parameter.irSource.substr(pos + 1, std::string::npos);

                        VPU_THROW_UNLESS(dim.length() == 1,
                            "Unable to deduce parameter '%s' for '%s' layer. Name is: '%s'",
                            parameter.argName, origLayer()->type, origLayer()->name);

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

                        if (ndims > 4) {
                            VPU_THROW_UNLESS(dim.length() == 1,
                                 "Unable to deduce parameter '%s' for '%s' layer. Name is: '%s'",
                                 parameter.argName, origLayer()->type, origLayer()->name);
                        }
                        const std::map<char, int> vars = {
                            {'b', 0}, {'B', 0},
                            {'f', 1}, {'F', 1},
                            {'y', 2}, {'Y', 2},
                            {'x', 3}, {'X', 3},
                        };

                        auto var = vars.find(dimLetter);
                        if (var != vars.end()) {
                            auto res = dims.at(var->second - 4 + ndims);

                            serializer.append(static_cast<uint32_t>(res));
                            serializer.append(static_cast<int32_t>(-1));
                        } else {
                            VPU_THROW_FORMAT("Unable to deduce parameter '%s' for '%s' layer. Name is: '%s'",
                                parameter.argName, origLayer()->type, origLayer()->name);
                        }

                        break;
                    } else {
                        VPU_THROW_UNLESS(parameter.portIndex < 0,
                            "Unable to deduce parameter '%s' for '%s' layer: port-index=%d is set, "
                            "but parameter is neither a tensor, nor an array type.",
                            parameter.argName, origLayer()->type, parameter.portIndex);
                        try {
                            if (parameter.type == CustomParamType::Int) {
                                serializer.append(static_cast<int32_t>(std::stoi(parameter.irSource)));
                                serializer.append(static_cast<int32_t>(-1));
                            } else {
                                serializer.append(static_cast<float>(std::stof(parameter.irSource)));
                                serializer.append(static_cast<int32_t>(-2));
                            }
                            break;
                        } catch (const std::invalid_argument&) {
                            VPU_THROW_FORMAT("Unable to deduce parameter '%s' for '%s' layer. "
                                "Name is: '%s', parameter is: '%s'",
                                parameter.argName, origLayer()->type, origLayer()->name, parameter.irSource);
                        }
                    }
                }
            }
            case CustomParamType::LocalData: {
                const auto size = localDataSizes.at(parameter.argName);
                serializer.append(static_cast<int32_t>(size));
                serializer.append(static_cast<int32_t>(-3));

                break;
            }
            default:
                VPU_THROW_FORMAT("Unable to deduce parameter '%s' for '%s' layer. Name is: '%s'",
                    parameter.argName, origLayer()->type, origLayer()->name);
            }
        }
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        IE_ASSERT(numTempBuffers() == 1);

        for (const auto& inEdge : inputEdges()) {
            inEdge->input()->serializeBuffer(serializer);
        }

        for (const auto& outEdge : outputEdges()) {
            outEdge->output()->serializeBuffer(serializer);
        }

        for (const auto& tempEdge : tempBufferEdges()) {
            tempEdge->tempBuffer()->serializeBuffer(serializer);
        }
    }
};

}  // namespace

static SmallVector<int> calcSizesFromParams(const DataDesc& desc, const SmallVector<std::string>& bufferSizeRules,
                                            std::map<std::string, std::string> layerParams) {
    {
        const auto B = std::to_string(desc.dim(Dim::N, 1));
        const auto F = std::to_string(desc.dim(Dim::C, 1));
        const auto Y = std::to_string(desc.dim(Dim::H, 1));
        const auto X = std::to_string(desc.dim(Dim::W, 1));

        auto sizes = std::vector<std::pair<std::string, std::string>> {
            {"b", B}, {"B", B},
            {"f", F}, {"F", F},
            {"y", Y}, {"Y", Y},
            {"x", X}, {"X", X},
        };

        std::move(begin(sizes), end(sizes), inserter(layerParams, end(layerParams)));
    }

    MathExpression expr;
    expr.setVariables(layerParams);
    const auto parseSizeRule = [&expr](const std::string& rule) {
        expr.parse(rule);
        return expr.evaluate();
    };

    auto sizes = SmallVector<int>{};
    sizes.reserve(bufferSizeRules.size());
    std::transform(begin(bufferSizeRules), end(bufferSizeRules), std::back_inserter(sizes), parseSizeRule);

    return sizes;
}

void FrontEnd::parseCustom(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) {
    IE_ASSERT(layer != nullptr);
    IE_ASSERT(outputs.size() == 1);

    const auto suitableLayer = [&] {
        const auto customLayersForType = _customLayers.find(layer->type);
        IE_ASSERT(customLayersForType != _customLayers.end());
        return getSuitableCustomLayer(customLayersForType->second, layer);
    }();
    IE_ASSERT(suitableLayer);

    const auto kernels = suitableLayer->kernels();
    // Get all buffers, buffers must be unique associated by port index
    std::map<int, Data> tempBuffsMap;
    for (const auto& kernel : kernels) {
        for (const auto& param : kernel.bindings()) {
            if (param.type == CustomParamType::InputBuffer || param.type == CustomParamType::OutputBuffer) {
                const auto desc = (param.dimSource == CustomDimSource::Input) ? inputs[param.dimIdx]->desc()
                                                                              : outputs[param.dimIdx]->desc();
                const auto sizes = calcSizesFromParams(desc, { param.bufferSizeRule }, layer->params);
                const auto buf = model->addNewData("custom_" + layer->type + "_buf", DataDesc({sizes[0], 1, 1, 1}));
                if (tempBuffsMap.find(param.portIndex) == tempBuffsMap.end()) {
                    tempBuffsMap[param.portIndex] = buf;
                }
            }
        }
    }

    // Gather inputs and outputs for each stage for the layer
    for (int stage_num = 0; stage_num < kernels.size(); stage_num++) {
        const auto& kernel = kernels[stage_num];

        std::map<std::string, int> ports;
        std::vector<CustomDataFormat> formats;

        // Gather inputs
        DataVector stageInputs;
        for (auto& param : kernel.bindings()) {
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
        for (auto& param : kernel.bindings()) {
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

        formats.emplace_back(CustomDataFormat::Any);

        // Get kernel binary
        auto kernelNode = _kernelNodes.find(kernel.kernelBinary());
        if (kernelNode != _kernelNodes.end()) {
            stageInputs.emplace_back((kernelNode->second));
        } else {
            auto kernelBinaryDesc = DataDesc({kernel.kernelBinary().length()});
            kernelBinaryDesc.setType(DataType::U8);

            auto kernelBinary = model->addConstData(
                layer->type + "@kernelBinary",
                kernelBinaryDesc,
                std::make_shared<KernelBinaryContent>(kernel.kernelBinary()));
            stageInputs.emplace_back((kernelBinary));
            _kernelNodes[kernel.kernelBinary()] = kernelBinary;
        }

        DataVector stageOutputs;
        for (auto& param : kernel.bindings()) {
            if (param.type == CustomParamType::Output) {
                ports[param.argName] = stageInputs.size() + stageOutputs.size();
                stageOutputs.emplace_back(outputs[param.portIndex]);
            } else if (param.type == CustomParamType::OutputBuffer) {
                ports[param.argName] = stageInputs.size() + stageOutputs.size();
                stageOutputs.emplace_back(tempBuffsMap[param.portIndex]);
            }
        }

        auto stage = model->addNewStage<CustomStage>(
            layer->name + ((kernels.size() == 1) ? "" : "@stage_" + std::to_string(stage_num)),
            StageType::Custom,
            layer,
            stageInputs,
            stageOutputs);

        stage->attrs().set("customKernel", suitableLayer->kernels()[stage_num]);
        stage->attrs().set("ports", ports);
        stage->attrs().set("formats", formats);

        const auto& dimSource = (kernel.dimSource() == CustomDimSource::Input) ? inputs : outputs;
        const auto& dataDesc = dimSource[kernel.dimSourceIndex()]->desc();

        const auto gws = calcSizesFromParams(dataDesc, kernel.globalGridSizeRules(), layer->params);
        const auto lws = calcSizesFromParams(dataDesc, kernel.localGridSizeRules(), layer->params);

        stage->attrs().set("gws", gws);
        stage->attrs().set("lws", lws);

        const auto localDataSizes = [&] {
            auto sizes = std::map<std::string, int>{};
            for (const auto& bind : kernel.bindings()) {
                if (bind.type == CustomParamType::LocalData) {
                    const auto& source = bind.dimSource == CustomDimSource::Input ? inputs : outputs;
                    const auto& desc = source[bind.dimIdx]->desc();
                    const auto size = calcSizesFromParams(desc, { bind.bufferSizeRule }, layer->params);
                    sizes.emplace(bind.argName, size[0]);
                }
            }
            return sizes;
        }();

        stage->attrs().set("localDataSizes", localDataSizes);

        std::map<int, DimsOrder> inputOrders;
        std::map<int, DimsOrder> outputOrders;

        std::map<std::string, CustomKernel::KernelParam> b2b;
        for (const auto& kp : kernel.bindings()) {
            b2b[kp.argName] = kp;
        }

        const std::map<CustomDataFormat, DimsOrder> formatsMap = {
            { CustomDataFormat::BYXF, DimsOrder::NHWC },
            { CustomDataFormat::BFYX, DimsOrder::NCHW },
            { CustomDataFormat::YXF, DimsOrder::HWC },
            { CustomDataFormat::FYX, DimsOrder::CHW }
        };

        for (const auto& kp : kernel.parameters()) {
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

        int buffer_size = kernel.kernelBinary().length() + 1024;
        model->addTempBuffer(
            stage,
            DataDesc({buffer_size}));
    }
}

}  // namespace vpu
