// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>
#include <string>
#include <memory>
#include <vector>
#include <dnnl_extension_utils.h>

namespace ov {
namespace intel_cpu {
namespace node {

class Eye : public Node {
public:
    Eye(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, WeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;
    bool needPrepareParams() const override {return false;};
    bool needShapeInfer() const override {return false;};
    void executeDynamicImpl(mkldnn::stream strm) override { execute(strm); }
    std::vector<VectorDims> shapeInfer() const override {
        return {VectorDims{
            // batch_shape
            getRowNum(), getColNum()
        }};
    }

    bool isExecutable() const override;

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    std::string errorPrefix;
    ov::element::Type outType;
    template <typename inputType>
    void executeSpecified();
    template<typename T>
    struct EyeExecute;
    inline const size_t getRowNum() const {
        return reinterpret_cast<const int *>(getParentEdgeAt(ROWS_NUM)->getMemoryPtr()->GetPtr())[0];
    }
    inline const size_t getColNum() const {
        return reinterpret_cast<const int *>(getParentEdgeAt(COLS_NUM)->getMemoryPtr()->GetPtr())[0];
    }
    inline const int getDiagIndex() const {
        return reinterpret_cast<const int *>(getParentEdgeAt(DIAGONAL_INDEX)->getMemoryPtr()->GetPtr())[0];
    }

    static const size_t ROWS_NUM = 0;
    static const size_t COLS_NUM = 1;
    static const size_t DIAGONAL_INDEX = 2;
    static const size_t BATCH_SHAPE = 3;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov