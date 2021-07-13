// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <vpu/graph_transformer.hpp>
#include <vpu/compile_env.hpp>
#include <vpu/utils/file_system.hpp>
#include <vpu/model/data_contents/mtcnn_blob_content.hpp>

#include <vpu/configuration/options/hw_acceleration.hpp>

#include <vector>
#include <fstream>
#include <string>
#include <utility>
#include <memory>
#include <set>

namespace vpu {

// Must be synchronized with MvTensor
VPU_DECLARE_ENUM(MTCNN_Mode,
    AVA_FaceDetector = 0,
    Public = 1)

namespace {

class MTCNNStage final : public StageNode {
public:
    using StageNode::StageNode;

private:
    StagePtr cloneImpl() const override {
        return std::make_shared<MTCNNStage>(*this);
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
        auto input = inputEdge(0)->input();
        auto output = outputEdge(0)->output();

        orderInfo.setInput(inputEdge(0), input->desc().dimsOrder().createMovedDim(Dim::C, 2));
        orderInfo.setOutput(outputEdge(0), output->desc().dimsOrder().createMovedDim(Dim::C, 0));
    }

    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) override {
        stridesInfo.setInput(inputEdge(0), StridesRequirement::compact());
        stridesInfo.setOutput(outputEdge(0), StridesRequirement::compact());
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& batchInfo) override {
        batchInfo.setInput(inputEdge(0), BatchSupport::Split);
        batchInfo.setOutput(outputEdge(0), BatchSupport::Split);
    }

    void initialCheckImpl() const override {
        assertInputsOutputsTypes(this,
            {{DataType::U8, DataType::FP16}, {DataType::U8, DataType::FP16}},
            {{DataType::FP16}});
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        auto debug_pnet_post_nms = attrs().get<int>("debug_pnet_post_nms");
        auto debug_rnet_post_nms = attrs().get<int>("debug_rnet_post_nms");
        auto mode = attrs().get<MTCNN_Mode>("mode");
        const auto& pyramid = attrs().get<SmallVector<std::pair<int, int>>>("pyramid");
        auto stage2_zdir_batch_size = attrs().get<int>("stage2_zdir_batch_size");

        serializer.append(static_cast<int32_t>(pyramid.size()));
        for (const auto& elem : pyramid) {
            serializer.append(static_cast<int32_t>(elem.first));
            serializer.append(static_cast<int32_t>(elem.second));
        }

        serializer.append(static_cast<int32_t>(debug_pnet_post_nms));
        serializer.append(static_cast<int32_t>(debug_rnet_post_nms));
        serializer.append(static_cast<int32_t>(mode));
        serializer.append(static_cast<int32_t>(stage2_zdir_batch_size));
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        auto input0 = inputEdge(0)->input();
        auto input1 = inputEdge(1)->input();
        auto output = outputEdge(0)->output();

        input0->serializeBuffer(serializer);
        output->serializeBuffer(serializer);

        IE_ASSERT(inputEdge(1)->input()->desc().dimsOrder() == DimsOrder::C);
        input1->serializeBuffer(serializer);
    }
};

std::pair<int, int> getResolution(const std::string& str) {
    std::istringstream stream(str);
    std::string output;
    std::getline(stream, output, 'x');
    auto width = std::stoi(output);
    std::getline(stream, output, 'x');
    auto height = std::stoi(output);
    return std::make_pair(width, height);
}

ie::CNNNetwork loadSubNetwork(
        const std::string& fileName,
        const std::pair<int, int>& imgSize,
        const std::shared_ptr<ie::ICore> core,
        int* zdir_batchsize = nullptr) {
    //
    // Load network
    //

    auto network = core->ReadNetwork(fileName, std::string());

    //
    // Set precision of input/output
    //

    auto networkInputs = network.getInputsInfo();
    IE_ASSERT(networkInputs.size() == 1);

    auto networkOutputs = network.getOutputsInfo();
    IE_ASSERT(networkOutputs.size() == 1);

    networkInputs.begin()->second->setPrecision(ie::Precision::FP16);
    networkInputs.begin()->second->setLayout(ie::Layout::NCHW);

    networkOutputs.begin()->second->setPrecision(ie::Precision::FP16);
    networkOutputs.begin()->second->setLayout(ie::Layout::NCHW);

    //
    // Change input shape
    //

    auto inputShapes = network.getInputShapes();
    IE_ASSERT(inputShapes.size() == 1);

    std::string inputName;
    ie::SizeVector inputShape;
    std::tie(inputName, inputShape) = *inputShapes.begin();
    if (zdir_batchsize != nullptr)
        *zdir_batchsize = static_cast<int>(inputShape[1]/3);
    inputShape[0] = 1;                // set batch size to the first input dimension
    inputShape[2] = imgSize.second;   // changes input height to the image one
    inputShape[3] = imgSize.first;    // changes input width to the image one
    inputShapes[inputName] = inputShape;

    network.reshape(inputShapes);

    return network;
}

}  // namespace

void FrontEnd::parseMTCNN(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const {
    const auto& env = CompileEnv::get();

    ie::details::CaselessEq<std::string> cmp;

    IE_ASSERT(inputs.size() == 1);
    IE_ASSERT(outputs.size() == 1);

    if (!env.config.get<HwAccelerationOption>()) {
        VPU_THROW_EXCEPTION << "MTCNN layer supports Myriad X with NCE only";
    }

    auto input = inputs[0];
    auto output = outputs[0];

    auto modeStr = layer->GetParamAsString("mode", "PUBLIC_MTCNN");

    auto pnet_ir_name = layer->GetParamAsString("pnet_ir");
    auto rnet_ir_name = layer->GetParamAsString("rnet_ir");
    auto onet_ir_name = layer->GetParamAsString("onet_ir");
    auto pnet_resolutions_str = layer->GetParamAsString("pnet_resolutions");

    std::pair<int, int> r_net_input = {24, 24};
    std::pair<int, int> o_net_input = {48, 48};

    SmallVector<std::pair<int, int>> pyramid;

    std::istringstream stream(pnet_resolutions_str);
    std::string str;
    while (getline(stream, str, ',')) {
        pyramid.emplace_back(getResolution(str));
    }

    // Assert that the first stage in the pyramid is the largest one
    for (const auto& p_net_input : pyramid) {
        if (p_net_input.first > pyramid[0].first || p_net_input.second > pyramid[0].second) {
            VPU_THROW_EXCEPTION << "MTCNN layer: first stage in pyramid should be the largest one";
        }
    }

    SmallVector<CompiledGraph::Ptr> compiledSubNetworks;
    compiledSubNetworks.reserve(pyramid.size() + 2);

    //
    // Compile sub-networks with std::async to avoid current CompileEnv modification.
    //
    size_t mergedBlobSize = 0;

    // Convert p-nets
    for (const auto& p_net_input : pyramid) {
        auto pNet = loadSubNetwork(pnet_ir_name, p_net_input, _core);
        auto res = compileSubNetwork(pNet, env.config, _core);
        mergedBlobSize += res->blob.size();
        compiledSubNetworks.emplace_back(std::move(res));
    }

    int stage2_zdir_batchsize = 1;
    // Convert r-net
    {
        auto rNet = loadSubNetwork(rnet_ir_name, r_net_input, _core, &stage2_zdir_batchsize);
        auto res = compileSubNetwork(rNet, env.config, _core);
        mergedBlobSize += res->blob.size();
        compiledSubNetworks.emplace_back(std::move(res));
    }

    // Convert o-net
    {
        auto oNet = loadSubNetwork(onet_ir_name, o_net_input, _core);
        auto res = compileSubNetwork(oNet, env.config, _core);
        mergedBlobSize += res->blob.size();
        compiledSubNetworks.emplace_back(std::move(res));
    }

    //
    // Merge sub networks blobs
    //

    std::vector<char> mergedBlob(mergedBlobSize);

    size_t curOffset = 0;
    for (const auto& subRes : compiledSubNetworks) {
        std::copy_n(subRes->blob.data(), subRes->blob.size(), mergedBlob.data() + curOffset);
        curOffset += subRes->blob.size();
    }

    auto innerGraphsDesc = DataDesc({mergedBlob.size()});
    innerGraphsDesc.setType(DataType::U8);

    auto innerGraphs = model->addConstData(layer->name + "@innerGraphs", innerGraphsDesc, std::make_shared<MTCNNBlobContent>(mergedBlob));

    auto stage = model->addNewStage<MTCNNStage>(layer->name, StageType::MTCNN, layer, {input, innerGraphs}, {output});
    stage->attrs().set("pyramid", pyramid);
    stage->attrs().set<int>("debug_pnet_post_nms", layer->GetParamAsInt("debug_pnet_post_nms", 0));
    stage->attrs().set<int>("debug_rnet_post_nms", layer->GetParamAsInt("debug_rnet_post_nms", 0));
    stage->attrs().set<MTCNN_Mode>("mode", cmp(modeStr, "AVA_FaceDetector") ? MTCNN_Mode::AVA_FaceDetector : MTCNN_Mode::Public);
    stage->attrs().set<int>("stage2_zdir_batch_size", stage2_zdir_batchsize);
}

}  // namespace vpu
