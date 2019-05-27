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
        IE_ASSERT(_desc.totalDimSize() * _desc.elemSize() == _blob.length());
        return _blob.data();
    }

private:
    std::string _blob;
};

void printTo(std::ostream& os, const CustomLayer::Ptr& obj) {
    os << obj->kernelAddress();
}

void printTo(DotLabel& lbl, const CustomLayer::Ptr& obj) {
    DotLabel subLbl(lbl);
    subLbl.appendPair("kernelAddress", obj->kernelAddress());
}

class CustomStage final : public StageNode {
private:
    StagePtr cloneImpl() const override {
        return std::make_shared<CustomStage>(*this);
    }

    DataMap<float> propagateScaleFactorsImpl(
            const DataMap<float>&,
            ScalePropagationStep) override {
        DataMap<float> out;

        for (const auto& inEdge : _inputEdges) {
            out[inEdge->input()] = 1.0f;
        }
        for (const auto& outEdge : _outputEdges) {
            out[outEdge->output()] = 1.0f;
        }

        return out;
    }

    DataMap<DimsOrder> propagateDataOrderImpl() const override {
        const auto& inputOrders = attrs().get<std::map<int, DimsOrder>>("inputOrders");
        const auto& outputOrders = attrs().get<std::map<int, DimsOrder>>("outputOrders");

        DataMap<DimsOrder> out;

        // last input is always OpenCL binary, so use it as is.
        for (int i = 0; i < _inputEdges.size() - 1; i++) {
            auto input = _inputEdges[i]->input();
            IE_ASSERT(input != nullptr);

            auto it = inputOrders.find(i);
            if (it != inputOrders.end()) {
                auto requiredOrder = it->second;
                out[input] = requiredOrder;
            }
        }

        for (const auto& outEdge : _outputEdges) {
            auto it = outputOrders.find(outEdge->portInd());
            if (it != outputOrders.end()) {
                auto requiredOrder = it->second;
                out[outEdge->output()] = requiredOrder;
            }
        }

        return out;
    }

    DataMap<StridesRequirement> getDataStridesRequirementsImpl() const override {
        DataMap<StridesRequirement> out;

        // last input is always OpenCL binary, so use it as is.
        for (int i = 0; i < _inputEdges.size() - 1; i++) {
            auto input = _inputEdges[i]->input();
            IE_ASSERT(input != nullptr);

            out[input] = StridesRequirement::compact();
        }

        for (const auto& outEdge : _outputEdges) {
            out[outEdge->output()] = StridesRequirement::compact();
        }

        return out;
    }

    void finalizeDataLayoutImpl() override {
    }

    DataMap<BatchSupport> getBatchSupportInfoImpl() const override {
        DataMap<BatchSupport> out;

        // Last input is always OpenCL binary, so use it as is.
        for (int i = 0; i < _inputEdges.size() - 1; i++) {
            auto input = _inputEdges[i]->input();
            IE_ASSERT(input != nullptr);

            out[input] = BatchSupport::Split;
        }
        for (const auto& outEdge : _outputEdges) {
            out[outEdge->output()] = BatchSupport::Split;
        }

        return out;
    }

    void finalCheckImpl() const override {
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        const auto& customLayer = attrs().get<CustomLayer::Ptr>("customLayer");
        const auto& gws = attrs().get<std::vector<int>>("gws");
        const auto& lws = attrs().get<std::vector<int>>("lws");

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

        //
        // Entry point
        //

        serializer.append(static_cast<uint32_t>(customLayer->kernelAddress(lws[0])));

        //
        // Total number of blobs
        //

        serializer.append(static_cast<int32_t>(_inputEdges.size() + _outputEdges.size()));

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

        IE_ASSERT(_origLayer != nullptr);

        for (const auto& kp : customLayer->parameters()) {
            const auto& parameter = b2b[kp];

            switch (parameter.type) {
                case CustomParamType::Input:
                {
                    serializer.append(static_cast<uint32_t>(0));
                    serializer.append(static_cast<uint32_t>(parameter.portIndex));
                    break;
                }
                case CustomParamType::Output:
                {
                    serializer.append(static_cast<uint32_t>((uint32_t)0));
                    serializer.append(static_cast<uint32_t>(_inputEdges.size() + parameter.portIndex));
                    break;
                }
                case CustomParamType::Data:
                {
                    // TODO: handle data
                    break;
                }
                case CustomParamType::Int:
                case CustomParamType::Float:
                {
                    if (_origLayer->params.find(parameter.irSource) != _origLayer->params.end()) {
                        if (parameter.type == CustomParamType::Int) {
                            serializer.append(static_cast<int32_t>(std::stoi(_origLayer->params[parameter.irSource]) ));
                            serializer.append(static_cast<int32_t>(-1));
                        } else {
                            serializer.append(static_cast<float>(std::stof(_origLayer->params[parameter.irSource]) ));
                            serializer.append(static_cast<int32_t>(-2));
                        }
                        break;
                    } else {
                        auto pos = parameter.irSource.find_first_of('.');
                        if (pos != std::string::npos) {
                            auto blob = parameter.irSource.substr(0, pos);
                            auto dim = parameter.irSource.substr(pos + 1, std::string::npos);

                            ie::DataPtr origData;
                            if (blob == "I") {
                                origData = _origLayer->insData[0].lock();
                            } else {
                                origData = _origLayer->outData[0];
                            }
                            IE_ASSERT(origData != nullptr);

                            auto dims = origData->dims;

                            const std::map<char, int> vars = {
                                { 'b', 3 }, { 'B', 3 },
                                { 'f', 2 }, { 'F', 2 },
                                { 'y', 1 }, { 'Y', 1 },
                                { 'x', 0 }, { 'X', 0 },
                            };

                            if (vars.find(dim[0]) != vars.end()) {
                                auto res = dims[vars.at(dim[0])];

                                serializer.append(static_cast<uint32_t>(res));
                                serializer.append(static_cast<int32_t>(-1));
                            } else {
                                VPU_THROW_EXCEPTION
                                    << "Unable to deduce parameter " << parameter.argName << " for "
                                    << _origLayer->type <<" layer. Name is: " << _origLayer->name;
                            }

                            break;
                        }

                        VPU_THROW_EXCEPTION
                            << "Unable to deduce parameter " << parameter.argName << " for "
                            << _origLayer->type <<" layer. Name is: " << _origLayer->name;
                    }
                }
                default:
                    VPU_THROW_EXCEPTION
                        << "Unable to deduce parameter " << parameter.argName << " for "
                        << _origLayer->type <<" layer. Name is: " << _origLayer->name;
            }
        }
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        IE_ASSERT(_tempBufferEdges.empty());

        for (const auto& inEdge : _inputEdges) {
            inEdge->input()->serializeOldBuffer(handle_from_this(), serializer);
        }

        for (const auto& outEdge : _outputEdges) {
            outEdge->output()->serializeOldBuffer(handle_from_this(), serializer);
        }
    }
};

}  // namespace

void FrontEnd::parseCustom(
        const Model::Ptr& model,
        const ie::CNNLayerPtr& layer,
        const DataVector& inputs,
        const DataVector& outputs) {
    IE_ASSERT(layer != nullptr);
    IE_ASSERT(outputs.size() == 1);

    auto customLayerIt = _customLayers.find(layer->type);
    IE_ASSERT(customLayerIt != _customLayers.end());

    auto customLayer = customLayerIt->second;

    auto kernelBinaryDesc = DataDesc({customLayer->kernelBinary().length()});
    kernelBinaryDesc.setType(DataType::U8);

    auto kernelBinary = model->addConstData(
        layer->name + "@kernelBinary",
        kernelBinaryDesc,
        std::make_shared<KernelBinaryContent>(customLayer->kernelBinary()));

    auto allInputs = inputs;
    allInputs.emplace_back(std::move(kernelBinary));

    auto stage = model->addNewStage<CustomStage>(
        layer->name,
        StageType::Custom,
        layer,
        allInputs,
        outputs);

    stage->attrs().set("customLayer", customLayer);

    auto dims = layer->outData[0]->getTensorDesc().getDims();
    std::reverse(dims.begin(), dims.end());

    // assume output tensor is dimension source by default
    auto batchDim = (dims.size() > 0) ? dims[0] : 1;
    auto featureDim = (dims.size() > 1) ? dims[1] : 1;
    auto yDim = (dims.size() > 2) ? dims[2] : 1;
    auto xDim = (dims.size() > 3) ? dims[3] : 1;

    int iidx = customLayer->inputDimSourceIndex();
    if (iidx >= 0) {
        IE_ASSERT(iidx < layer->insData.size());

        auto origData = layer->insData[iidx].lock();
        IE_ASSERT(origData != nullptr);

        auto inputDims = origData->dims;

        batchDim = featureDim = yDim = 0;
        xDim = inputDims[0];

        if (dims.size() > 1)
            yDim = inputDims[1];
        if (dims.size() > 2)
            featureDim = inputDims[2];
        if (dims.size() > 3)
            batchDim = inputDims[3];
    }

    // evaluate work sizes rules
    std::vector<int> gws;
    std::vector<int> lws;

    const std::map<char, int> vars = {
        { 'b', batchDim },   { 'B', batchDim },
        { 'f', featureDim }, { 'F', featureDim },
        { 'y', yDim },       { 'Y', yDim },
        { 'x', xDim },       { 'X', xDim },
    };

    for (const auto& rule : customLayer->globalSizeRules()) {
        SimpleMathExpression expr;
        expr.setVariables(vars);
        expr.parse(rule);
        gws.emplace_back(expr.evaluate());
    }
    while (gws.size() < 3) {
        gws.emplace_back(1);
    }

    for (const auto& rule : customLayer->localSizeRules()) {
        SimpleMathExpression expr;
        expr.setVariables(vars);
        expr.parse(rule);
        lws.emplace_back(expr.evaluate());
    }
    while (lws.size() < 3) {
        lws.emplace_back(1);
    }

    stage->attrs().set("gws", gws);
    stage->attrs().set("lws", lws);

    std::map<int, DimsOrder> inputOrders;
    std::map<int, DimsOrder> outputOrders;

    std::map<std::string, CustomLayer::KernelParam> b2b;
    for (const auto& kp : customLayer->bindings()) {
        b2b[kp.argName] = kp;
    }

    const std::map<CustomDataFormat, DimsOrder> formats = {
        { CustomDataFormat::BYXF, DimsOrder::NHWC },
        { CustomDataFormat::BFYX, DimsOrder::NCHW }
    };

    for (const auto& kp : customLayer->parameters()) {
        const auto& parameter = b2b[kp];

        if (parameter.type == CustomParamType::Input) {
            auto it = formats.find(parameter.format);
            if (it != formats.end()) {
                auto requiredOrder = it->second;
                inputOrders[parameter.portIndex] = requiredOrder;
            }
        }

        if (parameter.type == CustomParamType::Output) {
            auto it = formats.find(parameter.format);
            if (it != formats.end()) {
                auto requiredOrder = it->second;
                outputOrders[parameter.portIndex] = requiredOrder;
            }
        }
    }

    stage->attrs().set("inputOrders", inputOrders);
    stage->attrs().set("outputOrders", outputOrders);
}

}  // namespace vpu
