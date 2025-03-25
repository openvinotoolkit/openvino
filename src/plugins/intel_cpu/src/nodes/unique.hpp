// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>

namespace ov::intel_cpu::node {

class Unique : public Node {
public:
    Unique(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);
    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

    void getSupportedDescriptors() override{};
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(const dnnl::stream& strm) override;
    [[nodiscard]] bool created() const override {
        return getType() == Type::Unique;
    }

protected:
    void executeDynamicImpl(const dnnl::stream& strm) override;
    void prepareParams() override;
    [[nodiscard]] bool needShapeInfer() const override {
        return false;
    }

private:
    template <typename T>
    void flattenTensorExec();
    template <typename T>
    void slicedTensorExec();

    template <typename T>
    struct flattenExec;
    template <typename T>
    struct slicedExec;

    std::vector<int32_t> firstUniTmp;
    std::vector<int32_t> inToOutTmp;
    std::vector<int32_t> occurTmp;

    bool sorted = false;
    bool flattened = true;
    int axis = 0;
    bool definedOutputs[4] = {false, false, false, false};
    ov::element::Type dataPrecision;
    int64_t dataTypeSize = 1l;
    size_t uniqueLen = 1lu;

    static constexpr size_t IN_DATA = 0;
    static constexpr size_t AXIS = 1;
    static constexpr size_t UNIQUE_DATA = 0;
    static constexpr size_t FIRST_UNIQUE_IDX = 1;
    static constexpr size_t INPUT_TO_UNIQ_IDX = 2;
    static constexpr size_t OCCURRENCES_NUM = 3;
};

}  // namespace ov::intel_cpu::node
