// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "search_sorted.h"

#include "openvino/op/search_sorted.hpp"
#include "openvino/reference/search_sorted.hpp"

namespace ov::intel_cpu::node {
SearchSorted::SearchSorted(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
    const auto ss_op = ov::as_type_ptr<const ov::op::v15::SearchSorted>(op);
    right_mode = ss_op->get_right_mode();
}

bool SearchSorted::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!ov::is_type<ov::op::v15::SearchSorted>(op)) {
            errorMessage = "Only opset15 SearchSorted operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

void SearchSorted::getSupportedDescriptors() {
    // Validation is already done in the ov::opset15::SearchSorted.
}

void SearchSorted::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    ov::element::Type inputPrec = getOriginalInputPrecisionAtPort(0);
    ov::element::Type outputPrec = getOriginalOutputPrecisionAtPort(0);

    if (!one_of(inputPrec,
                ov::element::f32,
                ov::element::i32,
                ov::element::bf16,
                ov::element::f16,
                ov::element::u8,
                ov::element::i8)) {
        inputPrec = ov::element::f32;
    }

    if (!one_of(outputPrec, ov::element::i32, ov::element::i64)) {
        outputPrec = ov::element::i32;
    }

    addSupportedPrimDesc({{LayoutType::ncsp, inputPrec}, {LayoutType::ncsp, inputPrec}},
                         {{LayoutType::ncsp, outputPrec}},
                         impl_desc_type::ref);
}

bool SearchSorted::created() const {
    return getType() == Type::SearchSorted;
}

bool SearchSorted::needPrepareParams() const {
    return false;
}

void SearchSorted::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

template <typename INPUT_TYPE, typename OUTPUT_TYPE>
void SearchSorted::executeImpl() {
    ov::reference::search_sorted<INPUT_TYPE, OUTPUT_TYPE>(getSrcDataAtPortAs<const INPUT_TYPE>(0),
                                                          getSrcDataAtPortAs<const INPUT_TYPE>(1),
                                                          getDstDataAtPortAs<OUTPUT_TYPE>(0),
                                                          ov::Shape{getSrcMemoryAtPort(0)->getStaticDims()},
                                                          ov::Shape{getSrcMemoryAtPort(1)->getStaticDims()},
                                                          right_mode);
}

namespace {
struct SearchSortedContext {
    SearchSorted& node;
};
}  // namespace

template <typename T>
struct SearchSorted::SearchSortedExecute {
    using TInputType = typename std::tuple_element<0, T>::type;
    using TOutputType = typename std::tuple_element<1, T>::type;

    void operator()(SearchSortedContext& ctx) {
        ctx.node.executeImpl<TInputType, TOutputType>();
    }
};
void SearchSorted::execute(const dnnl::stream& strm) {
    auto inputPrecision = getParentEdgeAt(0)->getMemory().getDesc().getPrecision();
    auto outputPrecision = getChildEdgeAt(0)->getMemory().getDesc().getPrecision();

    SearchSortedContext ctx = {*this};

#define CASE(OV_TYPE)                                                                           \
    OV_CASE2(OV_TYPE, ov::element::i64, ov::element_type_traits<OV_TYPE>::value_type, int64_t), \
        OV_CASE2(OV_TYPE, ov::element::i32, ov::element_type_traits<OV_TYPE>::value_type, int32_t)

    OV_SWITCH(intel_cpu,
              SearchSortedExecute,
              ctx,
              std::tie(inputPrecision, outputPrecision),
              CASE(ov::element::f32),
              CASE(ov::element::f16),
              CASE(ov::element::bf16),
              CASE(ov::element::i32),
              CASE(ov::element::i8),
              CASE(ov::element::u8))

#undef CASE
}
}  // namespace ov::intel_cpu::node
