// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cassert>
#include <cstddef>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <tuple>
#include <utility>

#include "cpu_memory.h"
#include "cpu_types.h"
#include "graph_context.h"
#include "node.h"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu::node {

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
    void (*ker_)(const jit_roi_pooling_call_args*) = nullptr;

    void operator()(const jit_roi_pooling_call_args* args) const {
        assert(ker_);
        ker_(args);
    }

    explicit jit_uni_roi_pooling_kernel(jit_roi_pooling_params jpp) : jpp_(jpp) {}
    virtual ~jit_uni_roi_pooling_kernel() = default;

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
    [[nodiscard]] bool created() const override;

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
        static std::tuple<int, int, int, int> getBordersForMaxMode(int roi_start_h,
                                                                   int roi_end_h,
                                                                   int roi_start_w,
                                                                   int roi_end_w,
                                                                   int ih,
                                                                   int oh,
                                                                   int iw,
                                                                   int ow,
                                                                   int pooled_h,
                                                                   int pooled_w);
        static std::pair<float, float> getXYForBilinearMode(float roi_start_h,
                                                            float roi_end_h,
                                                            float roi_start_w,
                                                            float roi_end_w,
                                                            int ih,
                                                            int oh,
                                                            int iw,
                                                            int ow,
                                                            int pooled_h,
                                                            int pooled_w);

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

}  // namespace ov::intel_cpu::node
