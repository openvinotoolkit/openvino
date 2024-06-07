// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "qkv_proj.h"

#include <string>
#include <vector>

#include "common/bfloat16.hpp"
#include "common/cpu_memcpy.h"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "shape_inference/shape_inference_internal_dyn.hpp"
#include "utils/plain_tensor.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

static std::vector<int> allocate_workers(const std::vector<int>& grouped_works, int n_workers) {
    auto n_groups = grouped_works.size();
    // allocate 1 worker for each group
    std::vector<int> g_workers(n_groups, 1);
    auto left_workers = n_workers - n_groups;
    while (left_workers > 0) {
        // which group is working hardest?
        float hardest_works = 0;
        size_t hardest_group = 0;
        for (size_t g = 0; g < n_groups; g++) {
            auto works = static_cast<float>(grouped_works[g]) / g_workers[g];
            if (hardest_works < works) {
                hardest_works = works;
                hardest_group = g;
            }
        }
        g_workers[hardest_group]++;
        left_workers--;
    }

    return g_workers;
}

#if defined(OPENVINO_ARCH_X86_64)
struct QKVProjection::Impl {
    std::vector<Work> works;
    std::vector<PlainTensor> m_tempC;
    QKVProjection * m_node;
    DnnlScratchPadPtr m_scrachPad;
    MemoryPtr m_scratchMem;
    int m_M;

    Impl(QKVProjection * pnode, DnnlScratchPadPtr scrachPad) : m_node(pnode), m_scrachPad(scrachPad) {
        PlainTensor w0(pnode->getSrcMemoryAtPort(1));
        PlainTensor w1(pnode->getSrcMemoryAtPort(2));
        PlainTensor w2(pnode->getSrcMemoryAtPort(3));

        const int blk_K_size = 256;
        auto K = w0.size(1);
        OPENVINO_ASSERT((K % blk_K_size) == 0);
        auto nthr = parallel_get_max_threads();
        auto num_blk_K = K / blk_K_size;
        int stride = K * sizeof(ov::bfloat16);

        works.resize(nthr);

        int cur_work_id = 0;
        auto create_works = [&](ov::bfloat16* pw, int output_id, int N, int valid_nthr) {
            // split task on more cores is better on TBB
            OPENVINO_ASSERT((N % 32) == 0);
            auto num_blk_N = N / 32;
            auto blkN_per_thread = (num_blk_N) / valid_nthr;
            auto blkN_leftover = num_blk_N - (blkN_per_thread * valid_nthr);
            auto start_blkN = 0;

            for (int ithr = 0; ithr < valid_nthr; ithr++) {
                auto blkN = std::min(num_blk_N - start_blkN, blkN_per_thread);
                if (blkN_leftover > 0) {
                    blkN_leftover--;
                    blkN++;
                }
                if (blkN) {
                    auto& work = works[cur_work_id++];
                    work.blk_K_size = blk_K_size;
                    work.n0 = (start_blkN)*32;
                    work.n1 = (start_blkN + blkN) * 32;
                    work.BN = blkN * 32;
                    work.k0 = 0;
                    work.k1 = blk_K_size * num_blk_K;
                    work.output_id = output_id;
                    work.p_raw_weights = pw;
                }
                start_blkN += blkN;
            }
        };
        auto proj_size0 = static_cast<int>(w0.size(0));
        auto proj_size1 = static_cast<int>(w1.size(0));
        auto proj_size2 = static_cast<int>(w2.size(0));
        auto n_group_workers = allocate_workers({proj_size0, proj_size1, proj_size2}, nthr);

        create_works(w0.ptr<ov::bfloat16>(), 0, proj_size0, n_group_workers[0]);
        create_works(w1.ptr<ov::bfloat16>(), 1, proj_size1, n_group_workers[1]);
        create_works(w2.ptr<ov::bfloat16>(), 2, proj_size2, n_group_workers[2]);

        DEBUG_LOG("QKVProj hidden_size=", K, " proj_sizes=",
                    proj_size0, ",", proj_size1, ",", proj_size2,
                    " used_nthr=", cur_work_id);
        ov::parallel_nt_static(0, [&](const size_t ithr, const size_t nthr) {
            auto& work = works[ithr];
            if (work) {
                work.setup(work.p_raw_weights, stride);
            }
        });

        m_tempC.resize(nthr);
    }

    void setM(int M) {
        if (m_M < M) {
            size_t total_scratch_size = 0;
            std::vector<size_t> scratch_offsets;
            std::vector<size_t> scratch_C_sizes;
            for (size_t ithr = 0; ithr < m_tempC.size(); ithr++) {
                scratch_offsets.push_back(total_scratch_size);
                auto max_C_size = works[ithr].get_C_size(M);
                scratch_C_sizes.push_back(max_C_size);
                total_scratch_size += max_C_size * sizeof(float);
            }

            auto newMemDesc = std::make_shared<CpuBlockedMemoryDesc>(ov::element::u8, Shape{total_scratch_size});
            m_scratchMem = m_scrachPad->createScratchPadMem(newMemDesc);

            auto* scratch_base = m_scratchMem->getDataAs<uint8_t>();
            for (size_t ithr = 0; ithr < m_tempC.size(); ithr++) {
                m_tempC[ithr].resize<float>({1, scratch_C_sizes[ithr]}, reinterpret_cast<float*>(scratch_base + scratch_offsets[ithr]));
            }

            m_M = M;
        }
    }

    void execute() {
        static ReduceAdd2bh jit_2bh(false);
        auto input = m_node->getSrcMemoryAtPort(0);
        const auto& ishape = input->getStaticDims();
        uint8_t* pA = input->getDataAs<uint8_t>();
        int M = shape_size(ishape) / ishape[ishape.size() - 1];
        auto* dst0 = m_node->getDstMemoryAtPort(0)->getDataAs<ov::bfloat16>();
        auto* dst1 = m_node->getDstMemoryAtPort(1)->getDataAs<ov::bfloat16>();
        auto* dst2 = m_node->getDstMemoryAtPort(2)->getDataAs<ov::bfloat16>();

        const auto& srcStrides = input->getDescWithType<BlockedMemoryDesc>()->getStrides();
        const auto& dstStrides0 = m_node->getDstMemoryAtPort(0)->getDescWithType<BlockedMemoryDesc>()->getStrides();
        const auto& dstStrides1 = m_node->getDstMemoryAtPort(1)->getDescWithType<BlockedMemoryDesc>()->getStrides();
        const auto& dstStrides2 = m_node->getDstMemoryAtPort(2)->getDescWithType<BlockedMemoryDesc>()->getStrides();

        int strideA = srcStrides[1] * sizeof(ov::bfloat16);
        auto stride_0 = dstStrides0[1];
        auto stride_1 = dstStrides1[1];
        auto stride_2 = dstStrides2[1];

        for (int m = 0; m < M;) {
            int BM = std::min(M - m, 256);

            setM(BM);

            ov::parallel_nt_static(0, [&](const size_t ithr, const size_t nthr) {
                auto& work = works[ithr];
                auto& C = m_tempC[ithr];
                if (work.BN > 0) {
                    work.run(BM, pA, strideA, C);

                    // compress accumulation result into target
                    auto* src = C.ptr<float>();
                    auto stride_src = C.stride(0);
                    ov::bfloat16* dst = nullptr;
                    int stride_dst = 0;
                    if (work.output_id == 0) {
                        dst = dst0 + work.n0;
                        stride_dst = stride_0;
                    }
                    if (work.output_id == 1) {
                        dst = dst1 + work.n0;
                        stride_dst = stride_1;
                    }
                    if (work.output_id == 2) {
                        dst = dst2 + work.n0;
                        stride_dst = stride_2;
                    }

                    for (int mi = 0; mi < BM; mi++, src += stride_src, dst += stride_dst) {
                        // the prefetch distance is increased to ensure by the time store happens
                        // prefetch has done and no HW prefetcher is triggered
                        auto* prefetch_dst = (mi + 2 < BM) ? (dst + 2 * stride_dst) : (dst);
                        jit_2bh(src, dst, prefetch_dst, work.BN);
                    }
                }
            });
            m += BM;
            pA += BM * strideA;
            dst0 += BM * stride_0 / sizeof(ov::bfloat16);
            dst1 += BM * stride_1 / sizeof(ov::bfloat16);
            dst2 += BM * stride_2 / sizeof(ov::bfloat16);
        }
    }
};
#else
struct QKVProjection::Impl {
    Impl(QKVProjection * pnode, DnnlScratchPadPtr scrachPad) {}
    void execute() {}
};
#endif

void QKVProjection::prepareParams() {
    if (!m_pimpl) {
        m_pimpl = std::make_shared<Impl>(this, context->getScratchPad());
    }
}

void QKVProjection::execute(dnnl::stream strm) {
    MAYBE_UNUSED(strm);
    m_pimpl->execute();
}

QKVProjection::QKVProjection(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context)
    : Node(op, context, NgraphShapeInferFactory(op, EMPTY_PORT_MASK)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW("CPU: " + errorMessage);
    }
}

void QKVProjection::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    auto rtPrecision = ov::element::bf16;
    auto weightPrecision = ov::element::bf16;

    // initialize input ports
    std::vector<PortConfigurator> inPortConfigs;
    inPortConfigs.emplace_back(LayoutType::ncsp, rtPrecision, getInputShapeAtPort(0), false, -1);      // input
    inPortConfigs.emplace_back(LayoutType::ncsp, weightPrecision, getInputShapeAtPort(1), false, -1);  // gate
    inPortConfigs.emplace_back(LayoutType::ncsp, weightPrecision, getInputShapeAtPort(2), false, -1);  // up
    inPortConfigs.emplace_back(LayoutType::ncsp, weightPrecision, getInputShapeAtPort(3), false, -1);  // down

    // initialize output port
    std::vector<PortConfigurator> outPortConfigs;
    outPortConfigs.emplace_back(LayoutType::ncsp, rtPrecision, getOutputShapeAtPort(0), false, -1);
    outPortConfigs.emplace_back(LayoutType::ncsp, rtPrecision, getOutputShapeAtPort(1), false, -1);
    outPortConfigs.emplace_back(LayoutType::ncsp, rtPrecision, getOutputShapeAtPort(2), false, -1);

    addSupportedPrimDesc(inPortConfigs, outPortConfigs, impl_desc_type::ref_any);
}

bool QKVProjection::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto node_qkv = std::dynamic_pointer_cast<const QKVProjectionNode>(op);
        if (node_qkv) {
            auto proj_pshape1 = op->input_value(1).get_shape();
            auto proj_pshape2 = op->input_value(2).get_shape();
            auto proj_pshape3 = op->input_value(3).get_shape();
            if ((proj_pshape1[1] % 256) != 0) {
                errorMessage = "QKVProjection input channel size is not multiple of 256";
                return false;
            }
            if ((proj_pshape1[0] % 32) != 0) {
                errorMessage = "QKVProjection 1st proj output channel size is not multiple of 32";
                return false;
            }
            if ((proj_pshape2[0] % 32) != 0) {
                errorMessage = "QKVProjection 2nd proj output channel size is not multiple of 32";
                return false;
            }
            if ((proj_pshape3[0] % 32) != 0) {
                errorMessage = "QKVProjection 3rd proj output channel size is not multiple of 32";
                return false;
            }
        } else {
            errorMessage = "Only QKVProjection operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
