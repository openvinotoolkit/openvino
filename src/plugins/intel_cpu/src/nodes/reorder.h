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
                                 const ov::intel_cpu::MemoryCPtr& src,
                                 const ov::intel_cpu::MemoryCPtr& dst,
                                 const std::vector<int> src_permutation);
        bool exec(dnnl::stream strm);
        dnnl::memory::desc updateSrcDesc(const dnnl::engine& engine, const std::vector<int> src_permutation);
        void prepareParams(const dnnl::engine& engine,
                           MultiCachePtr& cache,
                           const ov::intel_cpu::MemoryCPtr& src,
                           const ov::intel_cpu::MemoryCPtr& dst);
        void setDescs(MemoryDescPtr in, MemoryDescPtr out) {
            input = in;
            output = out;
        }
        void updateMem(const ov::intel_cpu::MemoryPtr& src, const ov::intel_cpu::MemoryPtr& dst);
        dnnl::reorder::primitive& getPrimitive() {
            return prim;
        }
        virtual ~ReorderExecutor() = default;

    private:
        void preConvert();
        void postConvert();

        class IntermConverter {
        public:
            IntermConverter(MemoryPtr in_mem_ptr,
                            const InferenceEngine::Precision in_prec,
                            MemoryPtr out_mem_ptr,
                            const InferenceEngine::Precision out_prec)
                : src_mem(in_mem_ptr),
                  src_prec(in_prec),
                  dst_mem(out_mem_ptr),
                  dst_prec(out_prec) {}
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

        private:
            MemoryPtr src_mem;
            InferenceEngine::Precision src_prec;
            MemoryPtr dst_mem;
            InferenceEngine::Precision dst_prec;
        };

        std::shared_ptr<IntermConverter> pre_converter = nullptr;
        std::shared_ptr<IntermConverter> post_converter = nullptr;
        DnnlScratchPadPtr scratch_ptr;

        MemoryPtr dst_blocked;
        MemoryPtr src_blocked;

        MemoryDescPtr input;
        MemoryDescPtr output;

        bool need_reorder;
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
