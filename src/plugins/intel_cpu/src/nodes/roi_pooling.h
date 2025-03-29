// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "node.h"

namespace ov {
namespace intel_cpu {
namespace node {

struct jit_roi_pooling_params {
    int mb, c;
    int ih, iw, oh, ow;

    int c_block, nb_c, nb_c_blocking;

    double spatial_scale;
    int pooled_h;
    int pooled_w;

    ov::element::Type src_prc;
    ov::element::Type dst_prc;

    Algorithm alg;

    bool operator==(const jit_roi_pooling_params& rhs) const noexcept;
};

struct jit_roi_pooling_call_args {
    const void* src;
    void* dst;

    size_t kh;
    size_t kw;
    size_t bin_area;

    size_t c_blocks;

    float xf;
    float yf;

    size_t xoff;
    size_t yoff;
};

struct jit_uni_roi_pooling_kernel {
    void (*ker_)(const jit_roi_pooling_call_args*);

    void operator()(const jit_roi_pooling_call_args* args) {
        assert(ker_);
        ker_(args);
    }

    explicit jit_uni_roi_pooling_kernel(jit_roi_pooling_params jpp) : ker_(nullptr), jpp_(jpp) {}
    virtual ~jit_uni_roi_pooling_kernel() {}

    virtual void create_ker() = 0;

    jit_roi_pooling_params jpp_;
};

class ROIPooling : public Node {
public:
    ROIPooling(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(const dnnl::stream& strm) override;
    bool created() const override;

    void executeDynamicImpl(const dnnl::stream& strm) override;
    void prepareParams() override;

private:
    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

    template <typename T>
    void execute();
    template <typename T>
    struct ROIPoolingExecute;

    jit_roi_pooling_params refParams = {};

    class ROIPoolingExecutor {
    public:
        ROIPoolingExecutor() = default;
        virtual void exec(const ov::intel_cpu::IMemory& srcData,
                          const ov::intel_cpu::IMemory& srcRoi,
                          const ov::intel_cpu::IMemory& dst) = 0;
        virtual ~ROIPoolingExecutor() = default;

        static std::shared_ptr<ROIPoolingExecutor> createROIPoolingNewExecutor(const jit_roi_pooling_params& jpp);

    protected:
        std::tuple<int, int, int, int> getBordersForMaxMode(const int roi_start_h,
                                                            const int roi_end_h,
                                                            const int roi_start_w,
                                                            const int roi_end_w,
                                                            const int ih,
                                                            const int oh,
                                                            const int iw,
                                                            const int ow,
                                                            const int pooled_h,
                                                            const int pooled_w);
        std::pair<float, float> getXYForBilinearMode(const float roi_start_h,
                                                     const float roi_end_h,
                                                     const float roi_start_w,
                                                     const float roi_end_w,
                                                     const int ih,
                                                     const int oh,
                                                     const int iw,
                                                     const int ow,
                                                     const int pooled_h,
                                                     const int pooled_w);

    private:
        template <typename T>
        static std::shared_ptr<ROIPoolingExecutor> makeExecutor(const jit_roi_pooling_params& jpp);

        struct ROIPoolingContext {
            std::shared_ptr<ROIPoolingExecutor> executor;
            jit_roi_pooling_params jpp;
        };

        template <typename T>
        struct ROIPoolingExecutorCreation {
            void operator()(ROIPoolingContext& ctx) {
                ctx.executor = ROIPoolingExecutor::makeExecutor<T>(ctx.jpp);
            }
        };
    };

    template <typename T>
    class ROIPoolingJitExecutor;
    template <typename T>
    class ROIPoolingRefExecutor;

    using executorPtr = std::shared_ptr<ROIPoolingExecutor>;
    executorPtr execPtr = nullptr;
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
