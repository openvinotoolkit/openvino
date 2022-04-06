// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <ie_common.h>

#include <nodes/reorder.h>
#include "nodes/input.h"
#include <edge.h>
#include <node.h>
#include "cache/multi_cache.h"

/*
 * Test Reorder::optimizedNcsp2Nspc() and Reorder::optimizedNspc2Ncsp() for
 * inPlace and non-inPlace cases. Specifically, the test checks that dst batch strides are
 * correctly taken into account by the custom impls (the case when the reorder is followed by an inplace concat).
 */
typedef std::tuple<
        std::vector<size_t>, // srcDims
        bool>                // forceInplace;
        ReorderCustomImplTestParamSet;

class ReorderCustomImplTestBase: public ::testing::Test {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ReorderCustomImplTestParamSet> &obj) {
        std::vector<size_t> srcDims;
        bool inPlace;
        std::tie(srcDims, inPlace) = obj.param;
        std::ostringstream result;
        result << "IS=(";
        for (const auto s : srcDims)
            result << s << ".";
        result.seekp(-1, result.cur);
        result << ")";
        result << "_InPlace=" << inPlace;
        return result.str();
    }

protected:
    void executeReorderNode(const void* srcData, void* dstData) {
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
        const dnnl::engine cpuEngine(dnnl::engine::kind::cpu, 0);
        ov::intel_cpu::WeightsSharing::Ptr weightsCache;

        auto inputNode = std::make_shared<ov::intel_cpu::node::Input>(ov::intel_cpu::Shape(srcDims),
                                                                                prec,
                                                                                "Reorder_Input", "Input",
                                                                                cpuEngine, weightsCache);
        auto reorderNode = std::make_shared<ov::intel_cpu::node::Reorder>("Reorder", cpuEngine, weightsCache);
        auto outputNode = std::make_shared<ov::intel_cpu::node::Input>(ov::intel_cpu::Shape(dstDims),
                                                                                 prec,
                                                                                 "Reorder_Output", "Output",
                                                                                 cpuEngine, weightsCache);

        auto parentEdge = std::make_shared<ov::intel_cpu::Edge>(inputNode, reorderNode, 0, 0);
        auto childEdge = std::make_shared<ov::intel_cpu::Edge>(reorderNode, outputNode, 0, 0);
        parentEdge->changeStatus(ov::intel_cpu::Edge::Status::NeedAllocation);
        childEdge->changeStatus(ov::intel_cpu::Edge::Status::NeedAllocation);
        reorderNode->addEdge(parentEdge);
        reorderNode->addEdge(childEdge);
        auto rtParamsCache = std::make_shared<ov::intel_cpu::MultiCache>(100);

        const std::vector<size_t> srcBlockedDims = getBlockedDims(srcDims, srcOrder);
        const std::vector<size_t> srcStrides = getStrides(srcBlockedDims);
        const std::vector<size_t> offsetPaddingToData(srcDims.size(), 0);

        const std::vector<size_t> dstBlockedDims = getBlockedDims(dstDims, dstOrder);
        const std::vector<size_t> dstStrides = getStrides(dstBlockedDims);

        const ov::intel_cpu::CpuBlockedMemoryDesc inputDesc(prec, ov::intel_cpu::Shape(srcDims),
                                                           srcBlockedDims, srcOrder,
                                                           0, offsetPaddingToData, srcStrides);

        const ov::intel_cpu::CpuBlockedMemoryDesc outputDesc(prec, ov::intel_cpu::Shape(srcDims),
                                                            getBlockedDims(srcDims, dstOrder), dstOrder,
                                                            0, offsetPaddingToData, dstStrides);

        auto parentMemory = std::make_shared<ov::intel_cpu::Memory>(cpuEngine);
        auto childMemory = std::make_shared<ov::intel_cpu::Memory>(cpuEngine);
        parentMemory->Create(inputDesc, srcData);
        childMemory->Create(outputDesc, dstData);
        parentEdge->reuse(parentMemory);
        childEdge->reuse(childMemory);

        reorderNode->setDescs(inputDesc, outputDesc);
        reorderNode->setRuntimeCache(rtParamsCache);
        std::vector<std::shared_ptr<ov::intel_cpu::Node>> nodes {inputNode, reorderNode, outputNode};
        for (auto &n : nodes) {
            n->init();
            n->getSupportedDescriptors();
            n->initSupportedPrimitiveDescriptors();
            n->selectPrimitiveDescriptorByIndex(0);
        }
        auto config = outputNode->getSelectedPrimitiveDescriptor()->getConfig();
        config.inConfs.resize(1);
        config.inConfs[0].inPlace(forceInplace ? 0 : -1);
        outputNode->getSelectedPrimitiveDescriptor()->setConfig(config);
        reorderNode->createPrimitive();

        dnnl::stream strm(cpuEngine);
        reorderNode->execute(strm);
        return;
    }

    template<typename T>
    void Run(const std::vector<T>& srcData, std::vector<T>& dstData) {
        fillData();
        executeReorderNode(srcData.data(), dstData.data());
        EXPECT_TRUE(resultIsCorrect(dstData));
    }
    // Fill srcData so that the results of NSPC2NCSP and NCSP2NSPC reorders are incremental numbers 0,1,2,...
    // Fill dstData with zeros
    virtual void fillData() = 0;
    template<typename T>
    bool resultIsCorrect(const std::vector<T>& dstData) {
        const size_t numElems = getNumElems(dstDims);
        auto b = dstData.begin();
        std::vector<T> expectedData(blockSize);
        for (int i = 0; i < numElems / blockSize; i++, b += blockSize) {
            if (i % 2 == 0) {
                std::iota(expectedData.begin(), expectedData.end(), i / 2 * blockSize);
                if (!std::equal(b, b + blockSize, expectedData.begin()))
                    return false;
            } else if (!std::all_of(b, b + blockSize, [](T x){return x == 0;})) {
                return false;
            }
        }
        return true;
    }
    size_t getNumElems(const std::vector<size_t>& dims) {
        size_t result = 1;
        for (auto d : dims)
            result *= d;
        return result;
    }
    std::vector<size_t> srcDims;
    std::vector<size_t> srcOrder;
    std::vector<size_t> dstDims;
    std::vector<size_t> dstOrder;
    InferenceEngine::Precision prec;
    bool forceInplace;
    size_t blockSize;
};

class ReorderNSPC2NCSPTest: public testing::WithParamInterface<ReorderCustomImplTestParamSet>,
                            public ReorderCustomImplTestBase{
protected:
    void SetUp() override {
        std::tie(srcDims, forceInplace) = this->GetParam();
        // The custom NSPC2NCSP  impl is used only if an input shape complies with:
        assert(srcDims[1] <= 64 &&  srcDims[1] >= 16 &&   (getNumElems(srcDims) / srcDims[1]) >= 128);
        // The custom NSPC2NCSP  impl is used only for FP32
        prec = InferenceEngine::Precision::FP32;
        srcOrder = std::vector<size_t> {0, 2, 3, 1};
        dstOrder = std::vector<size_t> {0, 1, 2, 3};
        dstDims = srcDims;
        blockSize = getNumElems(srcDims);
        // Create channel-strided dst layout for the inPlace case
        // Other dstDims could also be supported, but fillData() and resultIsCorrect() should be updated accordingly.
        if (forceInplace) {
            dstDims[1] *= 2;
            blockSize /=  srcDims[0];
        }
    }
    void Run() {
        ReorderCustomImplTestBase::Run(srcData, dstData);
    }
    void fillData() override {
        dstData.resize(getNumElems(dstDims));
        std::fill(dstData.begin(), dstData.end(), 0);
        srcData.resize(getNumElems(srcDims));
        const int numChannels = srcDims[1];
        const int spBlockSize = srcDims[2] * srcDims[3];
        const int batchSize = spBlockSize * numChannels;
        int i = 0;
        for (int n = 0; n < getNumElems(srcDims); n += batchSize) {
            for (int sp = n; sp < n + spBlockSize; sp++) {
                for (int c = sp; c < sp + batchSize; c += spBlockSize) {
                    srcData[i++] = static_cast<float>(c);
                }
            }
        }
    }
    std::vector<float> dstData;
    std::vector<float> srcData;
};

class ReorderNCSP2NSPCTest: public testing::WithParamInterface<ReorderCustomImplTestParamSet>,
                            public ReorderCustomImplTestBase{
protected:
    void SetUp() override {
        std::tie(srcDims, forceInplace) = this->GetParam();
        // Avoid uint8_t overflow or modify fillNCSP2NSPC() and resultIsCorrect()
        assert(getNumElems(srcDims) <= 256);
        srcOrder = std::vector<size_t> {0, 1, 2, 3};
        dstOrder = std::vector<size_t> {0, 2, 3, 1};
        // The custom NSPC2NCSP  impl is used only for U8
        prec = InferenceEngine::Precision::U8;
        dstDims = srcDims;
        blockSize = getNumElems(srcDims);
        // Create channel-strided dst layout for the inPlace case
        // Other dstDims could also be supported, but fillData() and resultIsCorrect() should be updated accordingly.
        if (forceInplace) {
            dstDims[1] *= 2;
            blockSize = srcDims[1];
        }
    }
    void Run() {
        ReorderCustomImplTestBase::Run(srcData, dstData);
    }
    void fillData() override {
        dstData.resize(getNumElems(dstDims));
        std::fill(dstData.begin(), dstData.end(), 0);
        srcData.resize(getNumElems(srcDims));
        const int numChannels = srcDims[1];
        const int batchSize = srcDims[2] * srcDims[3] * numChannels;
        int i = 0;
        for (int n = 0; n < getNumElems(srcDims); n += batchSize) {
            for (int c = n; c < n + numChannels; c ++) {
                for (int sp = c; sp < c + batchSize; sp += numChannels) {
                    srcData[i++] = static_cast<uint8_t>(sp);
                }
            }
        }
    }
    std::vector<uint8_t> dstData;
    std::vector<uint8_t> srcData;
};

TEST_P(ReorderNSPC2NCSPTest, NSPC2NCSP) {
    Run();
}

TEST_P(ReorderNCSP2NSPCTest, NCSP2NSPC) {
    Run();
}

const std::vector<bool> forceInplace {false, true};
const auto NSPC2NCSPparams =::testing::Combine(
                ::testing::Values(std::vector<size_t> {2, 16, 8, 8}),
                ::testing::ValuesIn(forceInplace));

INSTANTIATE_TEST_SUITE_P(smoke_ReorderTestCustomNSPC, ReorderNSPC2NCSPTest, NSPC2NCSPparams,
                         ReorderCustomImplTestBase::getTestCaseName);

const auto NCSP2NSPCparams =::testing::Combine(
        ::testing::Values(std::vector<size_t> {2, 8, 4, 4}),
        ::testing::ValuesIn(forceInplace));

INSTANTIATE_TEST_SUITE_P(smoke_ReorderTestCustomNCSP, ReorderNCSP2NSPCTest, NCSP2NSPCparams,
                         ReorderCustomImplTestBase::getTestCaseName);