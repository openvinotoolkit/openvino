// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>

#include <cassert>
#include <common/primitive_attr.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "cpu_types.h"
#include "graph_context.h"
#include "nodes/executors/mvn.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu::node {

class MVN : public Node {
public:
    MVN(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;
    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    bool created() const override;
    void execute(const dnnl::stream& strm) override;
    void executeDynamicImpl(const dnnl::stream& strm) override;
    bool canBeInPlace() const override {
        return false;
    }

    bool getAcrossChannels() const {
        return mvnAttrs.initAcrossChannels_;
    }

    bool getNormalizeVariance() const {
        return mvnAttrs.normalizeVariance_;
    }

    bool canFuse(const NodePtr& node) const override;
    void prepareParams() override;

private:
    void setPostOps(dnnl::primitive_attr& attr, bool initWeights = false);

    void transformTo5DCase(const VectorDims& shape);

    std::vector<const void*> postOpsDataPtrs;

    MVNAttrs mvnAttrs;
    VectorDims shape5D = {0, 0, 0, 0, 0};
    bool onlyUnaryPostOps = true;

    std::shared_ptr<legacy::MVNExecutorBase> execPtr = nullptr;
    bool canUseAclExecutor = false;
    std::shared_ptr<legacy::MVNExecutor> aclExecPtr = nullptr;

    class MVNRefExecutor : public legacy::MVNExecutorBase {
    public:
        MVNRefExecutor(const MVNAttrs& mvnAttrs);

        void exec(const uint8_t* src_data,
                  uint8_t* dst_data,
                  const void* post_ops_data_,
                  const VectorDims& shape5d) override;

    private:
        void mvn_ref(const uint8_t* src_data, uint8_t* dst_data, const VectorDims& shape5d);
    };
};

}  // namespace ov::intel_cpu::node
