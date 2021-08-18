// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <ie_common.h>

#include "mkldnn_reorder_node.h"
#include "mkldnn_input_node.h"
#include "mkldnn_edge.h"
#include "mkldnn_node.h"

namespace {
size_t getNumElems(const std::vector<size_t>& dims) {
    size_t result = 1;
    for (auto d : dims)
        result *= d;
    return result;
}
template<typename T>
bool resultIsCorrect(size_t blockSize, size_t numElems, const std::vector<T>& dstData) {
    bool isCorrect = true;
    auto b = dstData.begin();
    for ( int i = 1; i <= numElems/blockSize; i++ ) {
        isCorrect = isCorrect && std::all_of(b, b + blockSize, [i](float x) {return x == i % 2;});
        b += blockSize;
    }
    return isCorrect;
}
void executeReorderNode(const std::vector<size_t>& srcDims,
                         const std::vector<size_t>& srcOrder,
                         const std::vector<size_t>& dstDims,
                         const std::vector<size_t>& dstOrder,
                         const InferenceEngine::Precision prec,
                         const void* srcData,
                         void* dstData) {
    auto getBlockedDims = [](const std::vector<size_t>& dims, const std::vector<size_t>& order){
        std::vector<size_t> result;
        result.reserve(order.size());
        for (auto i : order)
            result.push_back(dims[i]);
        return result;
    };
    auto getStrides = [](const std::vector<size_t>& dims){
        std::vector<size_t> result(dims.size());
        result[dims.size() - 1] = 1;
        for (int i = dims.size() - 2; i >= 0; --i) {
            result[i] = result[i+1] * dims[i+1];
        }
        return result;
    };
    const mkldnn::engine cpuEngine(dnnl::engine::kind::cpu, 0);
    MKLDNNPlugin::MKLDNNWeightsSharing::Ptr weightsCache;

    auto inputNode = std::make_shared<MKLDNNPlugin::MKLDNNInputNode>(MKLDNNPlugin::Shape(srcDims),
                                                                      prec,
                                                                      "Reorder_Input", "Input",
                                                                      cpuEngine, weightsCache);
    auto reorderNode = std::make_shared<MKLDNNPlugin::MKLDNNReorderNode>("Reorder", cpuEngine, weightsCache);
    auto outputNode = std::make_shared<MKLDNNPlugin::MKLDNNInputNode>(MKLDNNPlugin::Shape(dstDims),
                                                                       prec,
                                                                       "Reorder_Output", "Output",
                                                                       cpuEngine, weightsCache);

    auto parentEdge = std::make_shared<MKLDNNPlugin::MKLDNNEdge>(inputNode, reorderNode, 0, 0);
    auto childEdge = std::make_shared<MKLDNNPlugin::MKLDNNEdge>(reorderNode, outputNode, 0, 0);
    parentEdge->changeStatus(MKLDNNPlugin::MKLDNNEdge::Status::NeedAllocation);
    childEdge->changeStatus(MKLDNNPlugin::MKLDNNEdge::Status::NeedAllocation);
    reorderNode->addEdge(parentEdge);
    reorderNode->addEdge(childEdge);

    const std::vector<size_t> srcBlockedDims = getBlockedDims(srcDims, srcOrder);
    const std::vector<size_t> srcStrides = getStrides(srcBlockedDims);
    const std::vector<size_t> offsetPaddingToData(srcDims.size(), 0);

    const std::vector<size_t> dstBlockedDims = getBlockedDims(dstDims, dstOrder);
    const std::vector<size_t> dstStrides = getStrides(dstBlockedDims);

    const MKLDNNPlugin::BlockedMemoryDesc inputDesc(prec, srcDims, srcBlockedDims,
                                                     srcOrder, 0, offsetPaddingToData, srcStrides);

    const MKLDNNPlugin::BlockedMemoryDesc outputDesc(prec, srcDims, getBlockedDims(srcDims, dstOrder),
                                                      dstOrder, 0, offsetPaddingToData, dstStrides);

    auto parent_memory = std::make_shared<MKLDNNPlugin::MKLDNNMemory>(cpuEngine);
    auto childMemory = std::make_shared<MKLDNNPlugin::MKLDNNMemory>(cpuEngine);
    parent_memory->Create(inputDesc, srcData);
    childMemory->Create(outputDesc, dstData);
    parentEdge->reuse(parent_memory);
    childEdge->reuse(childMemory);

    reorderNode->setDescs(inputDesc, outputDesc);
    std::vector<std::shared_ptr<MKLDNNPlugin::MKLDNNNode>> nodes {inputNode, reorderNode, outputNode};
    for (auto &n : nodes) {
        n->init();
        n->getSupportedDescriptors();
        n->initSupportedPrimitiveDescriptors();
        n->selectPrimitiveDescriptorByIndex(0);
    }
    auto config = outputNode->getSelectedPrimitiveDescriptor()->getConfig();
    config.inConfs.resize(1);
    config.inConfs[0].inPlace = 1;
    outputNode->getSelectedPrimitiveDescriptor()->setConfig(config);
    reorderNode->createPrimitive();

    mkldnn::stream strm(cpuEngine);
    reorderNode->execute(strm);
    return;
}
}// namespace
/*
 * The tests check that dst batch strides are correctly taken into account by
 * MKLDNNReorderNode::optimizedNcsp2Nspc() and MKLDNNReorderNode::optimizedNspc2Ncsp().
 * This represents the case when the reorder is followed by an inplace concat.
 */
TEST(InplaceChildReorderTest, NSPC2NCSP) {
    const std::vector<size_t> srcDims{2, 16, 8, 8};
    const std::vector<size_t> srcOrder{0, 2, 3, 1};
    const std::vector<size_t> dstDims{srcDims[0], srcDims[1] * 2, srcDims[2], srcDims[3]};
    const std::vector<size_t> dstOrder{0, 1, 2, 3};
    const InferenceEngine::Precision prec{InferenceEngine::Precision::FP32};
    const std::vector<float> srcData(getNumElems(srcDims), 1);
    std::vector<float> dstData(getNumElems(dstDims), 0);

    executeReorderNode(srcDims, srcOrder, dstDims, dstOrder,
                         prec, srcData.data(), dstData.data());

    EXPECT_EQ(resultIsCorrect(srcData.size() / srcDims[0], srcData.size(), dstData), true);
}

TEST(InplaceChildReorderTest, NCSP2NSPC) {
    const std::vector<size_t> srcDims{2, 16, 8, 8};
    const std::vector<size_t> srcOrder{0, 1, 2, 3};
    const std::vector<size_t> dstDims{srcDims[0], srcDims[1] * 2, srcDims[2], srcDims[3]};
    const std::vector<size_t> dstOrder{0, 2, 3, 1};
    const InferenceEngine::Precision prec{InferenceEngine::Precision::U8};
    const std::vector<uint8_t> srcData(getNumElems(srcDims), 1);
    std::vector<uint8_t> dstData(getNumElems(dstDims), 0);

    executeReorderNode(srcDims, srcOrder, dstDims, dstOrder,
                             prec, srcData.data(), dstData.data());

    EXPECT_EQ(resultIsCorrect(srcDims[1], srcData.size(), dstData), true);
}