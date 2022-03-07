// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common/blocked_desc_creator.h>
#include <cpu_types.h>
#include <gtest/gtest.h>
#include <ie_common.h>
#include <memory_desc/cpu_memory_desc_utils.h>
#include <memory_desc/dnnl_memory_desc.h>

#include <common/memory_desc_wrapper.hpp>
#include <dnnl.hpp>
#include <typeinfo>

#include <nodes/reorder.h>
#include "nodes/input.h"
#include <edge.h>
#include <node.h>
#include "cache/multi_cache.h"

using namespace InferenceEngine;
using namespace ov::intel_cpu;
/*
 * ReorderCustomizedStrideTest validateds MKLDNNReorderNode::optimizedNcsp2Nspc() and
 * MKLDNNReorderNode::optimizedNspc2Ncsp() for inPlace case. The customized inPlace case means reorder output C channel
 * is not dense. The non-inPlace cases would be covered by more general ReorderCPUTestParamSet test. Specifically, the
 * test checks that dst batch strides are correctly taken into account by the custom impls (the case when the reorder is
 * followed by an inplace concat).
 */
template <typename T>
struct ReorderCustomImplTestParamSet {
    // logical dimension of input
    std::vector<size_t> srcDims;
    bool isNspc2Ncsp;
    uint32_t strideFactor;
};

using f32_f32 = std::pair<float, float>;
using s8_s8 = std::pair<int8_t, int8_t>;

using ReorderCustomizedStrideParamF32 = ReorderCustomImplTestParamSet<f32_f32>;
using ReorderCustomizedStrideParamS8 = ReorderCustomImplTestParamSet<s8_s8>;

template <typename T>
struct mapped_ptr_t {
    using nonconst_type = typename std::remove_cv<T>::type;

    mapped_ptr_t(std::nullptr_t) : mem_(nullptr), ptr_(nullptr) {}
    mapped_ptr_t(const dnnl::memory* mem) : mem_(mem) {
        ptr_ = mem->map_data<nonconst_type>();
    }
    mapped_ptr_t(mapped_ptr_t&& other) : mem_(other.mem_), ptr_(other.ptr_) {
        other.mem_ = nullptr;
        other.ptr_ = nullptr;
    }

    mapped_ptr_t(const mapped_ptr_t&) = delete;
    mapped_ptr_t& operator=(const mapped_ptr_t&) = delete;

    ~mapped_ptr_t() {
        if (mem_ && ptr_)
            mem_->unmap_data(ptr_);
    }

    operator T*() {
        return ptr_;
    }
    operator const T*() const {
        return ptr_;
    }
    operator bool() const {
        return ptr_ != nullptr;
    }

private:
    const dnnl::memory* mem_;
    nonconst_type* ptr_;
};

template <typename T>
mapped_ptr_t<T> map_memory(const dnnl::memory& mem) {
    return mapped_ptr_t<T>(&mem);
}

template <typename data_i_t, typename data_o_t>
inline void check_reorder(const dnnl::memory::desc& md_i,
                          const dnnl::memory::desc& md_o,
                          dnnl::memory& src,
                          dnnl::memory& dst) {
    auto src_data = map_memory<data_i_t>(src);
    auto dst_data = map_memory<data_o_t>(dst);

    const auto ndims = md_i.data.ndims;
    const auto* dims = md_i.data.dims;
    const size_t nelems = std::accumulate(dims, dims + ndims, size_t(1), std::multiplies<size_t>());

    const dnnl::impl::memory_desc_wrapper mdw_i(md_i.data);
    const dnnl::impl::memory_desc_wrapper mdw_o(md_o.data);
    for (size_t i = 0; i < nelems; ++i) {
        auto src_offset = mdw_i.off_l(i, false);
        data_i_t s_raw = src_data[src_offset];
        data_o_t s = static_cast<data_o_t>(s_raw);
        auto dst_offset = mdw_o.off_l(i, false);
        data_o_t d = dst_data[dst_offset];
        ASSERT_EQ(s, d) << "mismatch at position " << i;
    }
}

template <typename T>
class ReorderCustomizedStrideTest : public ::testing::Test,
                                    public ::testing::WithParamInterface<ReorderCustomImplTestParamSet<T>> {
    using inputType = typename T::first_type;
    using outputType = typename T::second_type;
    using ReorderType = ReorderCustomImplTestParamSet<T>;

public:
    static std::string getTestCaseName(const testing::TestParamInfo<ReorderType>& obj) {
        ReorderType p = obj.param;
        std::ostringstream result;
        result << "IS:(";
        for (const auto s : p.srcDims)
            result << s << ".";
        result.seekp(-1, result.cur);
        result << ")";
        result << "_IsNspcToNcsp:" << p.isNspc2Ncsp;
        result << "_InputDataType:" << typeid(inputType).name();
        result << "_OutputDataType:" << typeid(outputType).name();
        result << ")";
        return result.str();
    }

    void Run() {
        buildReorderGraph();
        infer();
        validate<inputType, outputType>();
    }

protected:
    void SetUp() override {
        ReorderCustomImplTestParamSet<T> p = ::testing::TestWithParam<decltype(p)>::GetParam();
        srcDims = p.srcDims;

        if (p.isNspc2Ncsp) {
            // The custom NSPC2NCSP  impl is used only if an input shape complies with:
            assert(srcDims[1] <= 64 && srcDims[1] >= 16 && (getNumElems(srcDims) / srcDims[1]) >= 128);
            // The custom NSPC2NCSP impl is used only for FP32
            prec = InferenceEngine::Precision::FP32;
            srcOrder = std::vector<size_t>{0, 2, 3, 1};
            dstOrder = std::vector<size_t>{0, 1, 2, 3};
        } else {
            assert(getNumElems(srcDims) <= 256);
            srcOrder = std::vector<size_t>{0, 1, 2, 3};
            dstOrder = std::vector<size_t>{0, 2, 3, 1};
            // The custom NSPC2NCSP  impl is used only for U8
            prec = InferenceEngine::Precision::U8;
        }
        dstDims = srcDims;
        // Create channel-strided dst layout for the inPlace case
        // Other dstDims could also be supported, but fillData() and resultIsCorrect() should be updated accordingly.
        dstDims[1] *= p.strideFactor;
    }

    void buildReorderGraph() {
        auto getBlockedDims = [](const std::vector<size_t>& dims, const std::vector<size_t>& order) {
            std::vector<size_t> result;
            result.reserve(order.size());
            for (auto i : order)
                result.push_back(dims[i]);
            return result;
        };
        auto getStrides = [](const std::vector<size_t>& dims) {
            std::vector<size_t> result(dims.size());
            result[dims.size() - 1] = 1;
            for (int i = dims.size() - 2; i >= 0; --i) {
                result[i] = result[i + 1] * dims[i + 1];
            }
            return result;
        };
        cpuEngine = {dnnl::engine::kind::cpu, 0};
        ov::intel_cpu::MKLDNNWeightsSharing::Ptr weightsCache;

        inputNode = std::make_shared<ov::intel_cpu::MKLDNNInputNode>(ov::intel_cpu::Shape(srcDims),
                                                                    prec,
                                                                    "Reorder_Input",
                                                                    "Input",
                                                                    cpuEngine,
                                                                    weightsCache);
        reorderNode = std::make_shared<ov::intel_cpu::MKLDNNReorderNode>("Reorder", cpuEngine, weightsCache);
        auto outputNode = std::make_shared<ov::intel_cpu::MKLDNNInputNode>(ov::intel_cpu::Shape(dstDims),
                                                                          prec,
                                                                          "Reorder_Output",
                                                                          "Output",
                                                                          cpuEngine,
                                                                          weightsCache);

        parentEdge = std::make_shared<ov::intel_cpu::MKLDNNEdge>(inputNode, reorderNode, 0, 0);
        childEdge = std::make_shared<ov::intel_cpu::MKLDNNEdge>(reorderNode, outputNode, 0, 0);
        parentEdge->changeStatus(ov::intel_cpu::MKLDNNEdge::Status::NeedAllocation);
        childEdge->changeStatus(ov::intel_cpu::MKLDNNEdge::Status::NeedAllocation);
        reorderNode->addEdge(parentEdge);
        reorderNode->addEdge(childEdge);

        auto rtParamsCache = std::make_shared<ov::intel_cpu::MultiCache>(100);

        const std::vector<size_t> srcBlockedDims = getBlockedDims(srcDims, srcOrder);
        const std::vector<size_t> srcStrides = getStrides(srcBlockedDims);
        const std::vector<size_t> offsetPaddingToData(srcDims.size(), 0);

        const std::vector<size_t> dstBlockedDims = getBlockedDims(dstDims, dstOrder);
        const std::vector<size_t> dstStrides = getStrides(dstBlockedDims);

        const ov::intel_cpu::CpuBlockedMemoryDesc
            inputDesc(prec, ov::intel_cpu::Shape(srcDims), srcBlockedDims, srcOrder, 0, offsetPaddingToData, srcStrides);

        const ov::intel_cpu::CpuBlockedMemoryDesc outputDesc(prec,
                                                            ov::intel_cpu::Shape(srcDims),
                                                            getBlockedDims(srcDims, dstOrder),
                                                            dstOrder,
                                                            0,
                                                            offsetPaddingToData,
                                                            dstStrides);

        auto parentMemory = std::make_shared<ov::intel_cpu::MKLDNNMemory>(cpuEngine);
        auto childMemory = std::make_shared<ov::intel_cpu::MKLDNNMemory>(cpuEngine);
        parentMemory->Create(inputDesc, nullptr);
        childMemory->Create(outputDesc, nullptr);

        parentEdge->reuse(parentMemory);
        childEdge->reuse(childMemory);

        reorderNode->setDescs(inputDesc, outputDesc);
        reorderNode->setRuntimeCache(rtParamsCache);
        std::vector<std::shared_ptr<ov::intel_cpu::MKLDNNNode>> nodes{inputNode, reorderNode, outputNode};
        for (auto& n : nodes) {
            n->init();
            n->getSupportedDescriptors();
            n->initSupportedPrimitiveDescriptors();
        }
        //Select inputDesc as primitive descriptor for reorder node
        reorderNode->selectPrimitiveDescriptorByIndex(0);
    }

    void infer() {
        fillData();
        reorderNode->createPrimitive();
        mkldnn::stream strm(cpuEngine);
        reorderNode->execute(strm);
    }

    template <typename inputType, typename outputType>
    void validate(void) {
        auto reorderInput = parentEdge->getMemory().GetPrimitive();
        auto reorderOutput = childEdge->getMemory().GetPrimitive();
        auto dnnlMdInput = parentEdge->getMemory().GetDescWithType<DnnlMemoryDesc>();
        auto dnnlMdOutput = childEdge->getMemory().GetDescWithType<DnnlMemoryDesc>();
        auto mdInput = dnnlMdInput->getDnnlDesc();
        auto mdOutput = dnnlMdOutput->getDnnlDesc();
        check_reorder<inputType, outputType>(mdInput, mdOutput, reorderInput, reorderOutput);
    }
    // Fill srcData so that the results of NSPC2NCSP and NCSP2NSPC reorders are incremental numbers 0,1,2,...
    // Fill dstData with zeros

    void fillData() {
        auto elemNum = std::accumulate(srcDims.begin(), srcDims.end(), size_t(1), std::multiplies<size_t>());
        const auto& inputReorder = parentEdge->getMemory().GetPrimitive();
        auto inputReorderData = map_memory<inputType>(inputReorder);

        ov::intel_cpu::DnnlMemoryDescPtr dnnlMdInput = parentEdge->getMemory().GetDescWithType<DnnlMemoryDesc>();
        const dnnl::impl::memory_desc_wrapper mdInput{dnnlMdInput->getDnnlDesc().data};
        for (size_t i = 0; i < elemNum; ++i)
            inputReorderData[mdInput.off_l(i, false)] = inputType(i);
        // Set all the elements in output memory to be 0.
        elemNum = std::accumulate(dstDims.begin(), dstDims.end(), size_t(1), std::multiplies<size_t>());
        const auto& outputReorder = childEdge->getMemory().GetPrimitive();
        auto outputReorderData = map_memory<outputType>(outputReorder);
        ov::intel_cpu::DnnlMemoryDescPtr dnnlMdOutput = childEdge->getMemory().GetDescWithType<DnnlMemoryDesc>();
        const dnnl::impl::memory_desc_wrapper mdOutput{dnnlMdOutput->getDnnlDesc().data};
        for (size_t i = 0; i < elemNum; ++i)
            outputReorderData[mdOutput.off_l(i, false)] = outputType(0);
    }

    size_t getNumElems(const std::vector<size_t>& dims) {
        size_t result = 1;
        for (auto d : dims)
            result *= d;
        return result;
    }

    mkldnn::engine cpuEngine;

    std::vector<size_t> srcDims;
    std::vector<size_t> srcOrder;
    std::vector<size_t> dstDims;
    std::vector<size_t> dstOrder;
    InferenceEngine::Precision prec;

    std::shared_ptr<ov::intel_cpu::MultiCache> rtParamsCache;
    mkldnn::stream stream;
    std::shared_ptr<ov::intel_cpu::MKLDNNReorderNode> reorderNode;
    std::shared_ptr<ov::intel_cpu::MKLDNNInputNode> inputNode;
    std::shared_ptr<ov::intel_cpu::MKLDNNEdge> parentEdge;
    std::shared_ptr<ov::intel_cpu::MKLDNNEdge> childEdge;
};

using ReorderCustomizedStrideTestF32 = ReorderCustomizedStrideTest<f32_f32>;

using ReorderCustomizedStrideTestS8 = ReorderCustomizedStrideTest<s8_s8>;

TEST_P(ReorderCustomizedStrideTestF32, NSPC2NCSP) {
    Run();
}

TEST_P(ReorderCustomizedStrideTestS8, NCSP2NSPC) {
    Run();
}

// NSPC to NCSP with from
const auto NSPC2NCSPparamsFactorIs2 = ::testing::Values(ReorderCustomizedStrideParamF32{{2, 16, 8, 8}, true, 2});
const auto NSPC2NCSPparamsFactorIs3 = ::testing::Values(ReorderCustomizedStrideParamF32{{2, 16, 8, 8}, true, 3});
const auto NSPC2NCSPparamsFactorIs1 = ::testing::Values(ReorderCustomizedStrideParamF32{{2, 16, 8, 8}, true, 1});

const auto NCSP2NSPCparamsFactorIs2 = ::testing::Values(ReorderCustomizedStrideParamS8{{2, 8, 4, 4}, false, 2});
const auto NCSP2NSPCparamsFactorIs5 = ::testing::Values(ReorderCustomizedStrideParamS8{{2, 8, 4, 4}, false, 5});
const auto NCSP2NSPCparamsFactorIs1 = ::testing::Values(ReorderCustomizedStrideParamS8{{2, 8, 4, 4}, false, 1});

INSTANTIATE_TEST_SUITE_P(smoke_ReorderTestCustomNSPC2NCSPtrideWithFactor_2,
                         ReorderCustomizedStrideTestF32,
                         NSPC2NCSPparamsFactorIs2,
                         ReorderCustomizedStrideTestF32::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReorderTestCustomNSPC2NCSPStrideWithFactor_3,
                         ReorderCustomizedStrideTestF32,
                         NSPC2NCSPparamsFactorIs3,
                         ReorderCustomizedStrideTestF32::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReorderTestCustomNSPC2NCSPtrideWithFactor_1,
                         ReorderCustomizedStrideTestF32,
                         NSPC2NCSPparamsFactorIs1,
                         ReorderCustomizedStrideTestF32::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReorderTestCustomNCSP2NSPCFactor_2,
                         ReorderCustomizedStrideTestS8,
                         NCSP2NSPCparamsFactorIs2,
                         ReorderCustomizedStrideTestS8::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReorderTestCustomNCSP2NSPCFactor_5,
                         ReorderCustomizedStrideTestS8,
                         NCSP2NSPCparamsFactorIs5,
                         ReorderCustomizedStrideTestS8::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReorderTestCustomNCSP2NSPCFactor_1,
                         ReorderCustomizedStrideTestS8,
                         NCSP2NSPCparamsFactorIs1,
                         ReorderCustomizedStrideTestS8::getTestCaseName);
/*
 * ReorderCPUTest to test the CPU plugin-in dynamism and RT cache
 */

template <typename T>
struct ReorderCPUTestParamSet {
    ngraph::PartialShape inputPartialShape;
    // logical dimension vector  of input
    std::vector<std::vector<size_t>> inputShapes;
    LayoutType srcLayout;
    LayoutType dstLayout;
    InferenceEngine::Precision prec;
};

using ReorderCPUTestParamSetF32 = ReorderCPUTestParamSet<f32_f32>;
using ReorderCPUTestParamSetS8 = ReorderCPUTestParamSet<s8_s8>;

template <typename T>
class ReorderCPUTest : public ::testing::Test, public ::testing::WithParamInterface<ReorderCPUTestParamSet<T>> {
public:
    using inputType = typename T::first_type;
    using outputType = typename T::second_type;
    static std::string getTestCaseName(const testing::TestParamInfo<ReorderCPUTestParamSet<T>>& obj) {
        ReorderCPUTestParamSet<T> p = obj.param;

        std::ostringstream result;
        result << "IS:(";
        result << "InputPartialShape:" << p.inputPartialShape;
        result << "Shapes:" << p.inputPartialShape;
        for (const auto inputShape : p.inputShapes) {
            result << "[";
            for (const auto s : inputShape)
                result << s << ".";
            result << "],";
        }
        result << "_InputLayoutType:" << static_cast<int>(p.srcLayout) << ".";
        result << "_OutputLayoutType:" << static_cast<int>(p.dstLayout) << ".";
        result << "_InputDataType:" << typeid(inputType).name();
        result << "_OutputDataType:" << typeid(outputType).name();
        result << ")";
        return result.str();
    }

    void Run() {
        for (auto inputshape : inputShapes) {
            generate_inputs(inputshape);
            infer();
            validate();
        }
    }

protected:
    void generate_inputs(const std::vector<size_t> inputShape) {
        DnnlMemoryDescPtr dnnlMdInput = parentEdge->getMemory().GetDescWithType<DnnlMemoryDesc>();

        auto memDesc = inputDesc.cloneWithNewDims(inputShape);
        parentEdge->getMemoryPtr()->redefineDesc(memDesc);
        auto elemNum = std::accumulate(inputShape.begin(), inputShape.end(), size_t(1), std::multiplies<size_t>());

        const auto& inputReorder = parentEdge->getMemory().GetPrimitive();
        auto inputReorderData = map_memory<inputType>(inputReorder);

        dnnlMdInput = parentEdge->getMemory().GetDescWithType<DnnlMemoryDesc>();
        const dnnl::impl::memory_desc_wrapper mdInput{dnnlMdInput->getDnnlDesc().data};

        for (size_t i = 0; i < elemNum; ++i) {
            inputReorderData[mdInput.off_l(i, false)] = inputType(i);
        }
    }
    void infer() {
        reorderNode->executeDynamic(stream);
    }
    void validate() {
        auto reorderInput = parentEdge->getMemory().GetPrimitive();
        auto reorderOutput = childEdge->getMemory().GetPrimitive();
        auto dnnlMdInput = parentEdge->getMemory().GetDescWithType<DnnlMemoryDesc>();
        auto dnnlMdOutput = childEdge->getMemory().GetDescWithType<DnnlMemoryDesc>();
        auto mdInput = dnnlMdInput->getDnnlDesc();
        auto mdOutput = dnnlMdOutput->getDnnlDesc();

        check_reorder<inputType, outputType>(mdInput, mdOutput, reorderInput, reorderOutput);
    }

    void SetUp() override {
        ReorderCPUTestParamSet<T> reorderParam = this->GetParam();
        inputPartialShape = reorderParam.inputPartialShape;
        inputShapes = reorderParam.inputShapes;
        srcLayout = reorderParam.srcLayout;
        dstLayout = reorderParam.dstLayout;
        prec = reorderParam.prec;
        srcDims = ov::intel_cpu::Shape(inputPartialShape);
        dstDims = srcDims;
        blockCreatorMap = BlockedDescCreator::getCommonCreators();
        rtParamsCache = std::make_shared<ov::intel_cpu::MultiCache>(100);
        buildReorderNode();
        reorderNode->setRuntimeCache(rtParamsCache);
    }

    void buildReorderNode() {
        auto srcBlockedDescCreator = blockCreatorMap[srcLayout];
        auto dstBlockedDescCreator = blockCreatorMap[dstLayout];

        const mkldnn::engine cpuEngine(dnnl::engine::kind::cpu, 0);
        ov::intel_cpu::MKLDNNWeightsSharing::Ptr weightsCache;

        inputNode = std::make_shared<ov::intel_cpu::MKLDNNInputNode>(srcDims,
                                                                    prec,
                                                                    "Reorder_Input",
                                                                    "Parameter",
                                                                    cpuEngine,
                                                                    weightsCache);
        reorderNode = std::make_shared<ov::intel_cpu::MKLDNNReorderNode>("Reorder", cpuEngine, weightsCache);
        auto outputNode = std::make_shared<ov::intel_cpu::MKLDNNInputNode>(dstDims,
                                                                          prec,
                                                                          "Reorder_Output",
                                                                          "Output",
                                                                          cpuEngine,
                                                                          weightsCache);

        parentEdge = std::make_shared<ov::intel_cpu::MKLDNNEdge>(inputNode, reorderNode, 0, 0);
        childEdge = std::make_shared<ov::intel_cpu::MKLDNNEdge>(reorderNode, outputNode, 0, 0);
        parentEdge->changeStatus(ov::intel_cpu::MKLDNNEdge::Status::NeedAllocation);
        childEdge->changeStatus(ov::intel_cpu::MKLDNNEdge::Status::NeedAllocation);
        reorderNode->addEdge(parentEdge);
        reorderNode->addEdge(childEdge);
        auto rtParamsCache = std::make_shared<ov::intel_cpu::MultiCache>(100);
        inputDesc = srcBlockedDescCreator->createDesc(prec, srcDims);

        const ov::intel_cpu::CpuBlockedMemoryDesc outputDesc = dstBlockedDescCreator->createDesc(prec, dstDims);

        auto parentMemory = std::make_shared<ov::intel_cpu::MKLDNNMemory>(cpuEngine);
        auto childMemory = std::make_shared<ov::intel_cpu::MKLDNNMemory>(cpuEngine);
        parentMemory->Create(inputDesc, nullptr);
        childMemory->Create(outputDesc, nullptr);
        parentEdge->reuse(parentMemory);
        childEdge->reuse(childMemory);

        reorderNode->setDescs(inputDesc, outputDesc);

        std::vector<std::shared_ptr<ov::intel_cpu::MKLDNNNode>> nodes{inputNode, reorderNode, outputNode};
        for (auto& n : nodes) {
            n->init();
            n->getSupportedDescriptors();
            n->initSupportedPrimitiveDescriptors();
        }
        //Select inputDesc as primitive descriptor for reorder node
        reorderNode->selectPrimitiveDescriptorByIndex(0);
        stream = {cpuEngine};
        return;
    }

private:
    std::shared_ptr<ov::intel_cpu::MultiCache> rtParamsCache;
    mkldnn::stream stream;
    std::shared_ptr<ov::intel_cpu::MKLDNNReorderNode> reorderNode;
    std::shared_ptr<ov::intel_cpu::MKLDNNNode> inputNode;
    std::shared_ptr<ov::intel_cpu::MKLDNNEdge> parentEdge;
    std::shared_ptr<ov::intel_cpu::MKLDNNEdge> childEdge;

    ov::intel_cpu::Shape srcDims;
    ov::intel_cpu::Shape dstDims;
    LayoutType srcLayout;
    LayoutType dstLayout;
    InferenceEngine::Precision prec;
    std::vector<std::vector<size_t>> inputShapes;
    ngraph::PartialShape inputPartialShape;
    BlockedDescCreator::CreatorsMap blockCreatorMap;
    ov::intel_cpu::CpuBlockedMemoryDesc inputDesc{InferenceEngine::Precision::FP32, ov::intel_cpu::Shape{}};
};

using ReorderCPUTestF32 = ReorderCPUTest<f32_f32>;
using ReorderCPUTestS8 = ReorderCPUTest<s8_s8>;

TEST_P(ReorderCPUTestF32, CompareResult) {
    Run();
}

TEST_P(ReorderCPUTestS8, CompareResult) {
    Run();
}

const auto reorderCpuTestParams_1 =
    ::testing::Values(ReorderCPUTestParamSetF32{{2, 16, 8, -1},
                                                {{2, 16, 8, 8}, {2, 16, 8, 16}, {2, 16, 8, 8}},
                                                LayoutType::nspc,
                                                LayoutType::ncsp,
                                                InferenceEngine::Precision::FP32});

const auto reorderCpuTestParams_2 =
    ::testing::Values(ReorderCPUTestParamSetF32{{2, 8, -1, 4},
                                                {{2, 8, 4, 4}, {2, 8, 8, 4}, {2, 8, 4, 4}},
                                                LayoutType::ncsp,
                                                LayoutType::nspc,
                                                InferenceEngine::Precision::FP32});

const auto reorderCpuTestParams_3 =
    ::testing::Values(ReorderCPUTestParamSetF32{{2, 32, -1, 4},
                                                {{2, 32, 3, 4}, {2, 32, 6, 4}, {2, 32, 3, 4}},
                                                LayoutType::ncsp,
                                                LayoutType::nCsp8c,
                                                InferenceEngine::Precision::FP32});

const auto reorderCpuTestParams_4 =
    ::testing::Values(ReorderCPUTestParamSetS8{{2, 32, -1, 4},
                                                {{2, 32, 3, 4}, {2, 32, 6, 4}, {2, 32, 3, 4}},
                                                LayoutType::nCsp16c,
                                                LayoutType::nspc,
                                                InferenceEngine::Precision::U8});


INSTANTIATE_TEST_SUITE_P(smoke_ReorderTest_1,
                         ReorderCPUTestF32,
                         reorderCpuTestParams_1,
                         ReorderCPUTestF32::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReorderTest_2,
                         ReorderCPUTestF32,
                         reorderCpuTestParams_2,
                         ReorderCPUTestF32::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReorderTest_3,
                         ReorderCPUTestF32,
                         reorderCpuTestParams_3,
                         ReorderCPUTestF32::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_ReorderTest_4,
                         ReorderCPUTestS8,
                         reorderCpuTestParams_4,
                         ReorderCPUTestS8::getTestCaseName);