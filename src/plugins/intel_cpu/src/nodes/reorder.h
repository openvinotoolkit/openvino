// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <node.h>
#include <string>
#include <memory>
#include <vector>
#include <utils/general_utils.h>

namespace ov {
namespace intel_cpu {
namespace node {

class Reorder : public Node {
public:
    Reorder(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context);
    Reorder(const std::string& name, const GraphContext::CPtr context);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void execute(dnnl::stream strm) override;
    bool created() const override;
    const std::vector<impl_desc_type>& getDefaultImplPriority() override;

    bool isExecutable() const override;

    void createPrimitive() override;

    void prepareParams() override;

    void executeDynamicImpl(dnnl::stream strm) override;

    void setDescs(const MemoryDesc& in, const MemoryDesc& out) {
        input = in.clone();
        inputShapes.clear();
        inputShapes.push_back(input->getShape());

        this->output = out.clone();
        outputShapes.clear();
        outputShapes.push_back(output->getShape());
    }

    void setSrcPermutation(const std::vector<int>& src_perm) {
        src_permutation = src_perm;
    }

    void setOptimized(bool isOptimized) {
        this->isOptimized = isOptimized;
    }

    bool canBeInPlace() const override {
        return false;
    }

    const MemoryDesc& getInput() { return *input; }
    const MemoryDesc& getOutput() { return *output; }

    static std::string getReorderArgs(const MemoryDesc &parentDesc, const MemoryDesc &childDesc);

    static void reorderData(const IMemory &input, const IMemory &output, MultiCachePtr cache = nullptr);
    static void reorderData2(const IMemory &input, const IMemory &output, MultiCachePtr cache = nullptr);

private:
    class ReorderExecutor {
    public:
        explicit ReorderExecutor(const dnnl::engine& engine,
                                 MultiCachePtr& cache,
                                 const dnnl::memory::desc src_dnnl_desc,
                                 const dnnl::memory::desc dst_dnnl_desc,
                                 const InferenceEngine::Precision src_prc,
                                 const InferenceEngine::Precision dst_prc);
        bool exec(dnnl::stream strm);
        void prepareParams(const dnnl::engine& engine,
                           MultiCachePtr& cache,
                           const ov::intel_cpu::MemoryCPtr& src,
                           const ov::intel_cpu::MemoryCPtr& dst,
                           const std::vector<int> src_permutation);
        void setDescs(MemoryDescPtr in, MemoryDescPtr out) {
            input = in;
            output = out;
        }
        dnnl::reorder::primitive& getPrimitive() {
            return prim;
        }
        virtual ~ReorderExecutor() = default;

        using CombinatedDataType = std::pair<dnnl::memory::data_type, dnnl::memory::data_type>;

    private:
        bool isSupportedDataType(const dnnl::engine& engine,
                                 MultiCachePtr& cache,
                                 const DnnlMemoryDescPtr& src,
                                 const dnnl::memory::data_type& data_type);
        bool isSupportedCombinatedDataType(const dnnl::engine& engine,
                                           MultiCachePtr& cache,
                                           const DnnlMemoryDescPtr& src,
                                           const DnnlMemoryDescPtr& dst,
                                           const CombinatedDataType& combinated_data_type);
        void preConvert();
        void postConvert();

        std::vector<CombinatedDataType> supported_combinated_data_types;

        class IntermConverter {
        public:
            IntermConverter() = default;
            void convert();
            void setInputPrec(const InferenceEngine::Precision prec) {
                src_prec = prec;
            }
            void setOutputPrec(const InferenceEngine::Precision prec) {
                dst_prec = prec;
            }
            void setInputMem(MemoryPtr mem_ptr) {
                src_mem = mem_ptr;
            }
            void setOutputMem(MemoryPtr mem_ptr) {
                dst_mem = mem_ptr;
            }
            InferenceEngine::Precision getInputPrec() const {
                return src_prec;
            }
            InferenceEngine::Precision getOutputPrec() const {
                return dst_prec;
            }
            MemoryPtr& getInputMem() {
                return src_mem;
            }
            MemoryPtr& getOutputMem() {
                return dst_mem;
            }

        private:
            InferenceEngine::Precision src_prec;
            InferenceEngine::Precision dst_prec;
            MemoryPtr src_mem;
            MemoryPtr dst_mem;
        };

        std::shared_ptr<IntermConverter> pre_converter = nullptr;
        std::shared_ptr<IntermConverter> post_converter = nullptr;

        MemoryPtr dst_blocked;
        MemoryPtr src_blocked;

        MemoryDescPtr input;
        MemoryDescPtr output;

        dnnl::reorder::primitive prim;
        std::unordered_map<int, dnnl::memory> primArgs;
    };

    using ExecutorPtr = std::shared_ptr<ReorderExecutor>;
    ExecutorPtr executor_ptr;
    MemoryDescPtr input;
    MemoryDescPtr output;

    bool isOptimized = false;
    std::vector<int> src_permutation;

    bool isNspc2NcspCase = false;
    bool isNcsp2NspcCase = false;
    bool canUseNspc2Ncsp = false;
    bool canUseNcsp2Nspc = false;

    void optimizedNspc2Ncsp();
    void optimizedNcsp2Nspc();
    void createReorderExecutor(const ov::intel_cpu::MemoryCPtr &src, const ov::intel_cpu::MemoryCPtr &dst);
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
