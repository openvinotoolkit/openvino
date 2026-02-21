// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reorder_transpose.hpp"

#include <oneapi/dnnl/dnnl_types.h>

#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <utility>

#include "cpu_types.h"
#include "dnnl_extension_utils.h"
#include "memory_desc/cpu_memory_desc.h"
#include "memory_desc/dnnl_memory_desc.h"
#include "nodes/common/reorder_prim.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/executors/transpose.hpp"
#include "nodes/executors/transpose_config.hpp"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "thread_pool_imp.hpp"
#if defined(OPENVINO_ARCH_ARM) || defined(OPENVINO_ARCH_ARM64)
#    include "openvino/core/type/element_type.hpp"
#endif

namespace ov::intel_cpu {

ReorderTransposeExecutor::ReorderTransposeExecutor(const TransposeAttrs& attrs, ExecutorContext::CPtr context)
    : TransposeExecutor(attrs, std::move(context)) {}

bool ReorderTransposeExecutor::supports(const TransposeConfig& config) {
    bool result = config.descs.at(ARG_SRC)->hasLayoutType(LayoutType::ncsp) &&
                  config.descs.at(ARG_DST)->hasLayoutType(LayoutType::ncsp) &&
                  config.attrs.permuteParams.order == VectorDims{0, 3, 1, 2};
#if defined(OPENVINO_ARCH_ARM) || defined(OPENVINO_ARCH_ARM64)
    result = result && config.descs.at(ARG_SRC)->getPrecision() == ov::element::f32;
#endif
    return result;
}

ExecutorPtr ReorderTransposeExecutor::create(const TransposeAttrs& attrs,
                                             [[maybe_unused]] const MemoryArgs& memory,
                                             const ExecutorContext::CPtr& context) {
    return std::make_shared<ReorderTransposeExecutor>(attrs, context);
}

bool ReorderTransposeExecutor::init(const MemoryArgs& memory) {
    const auto src = memory.at(ARG_SRC);
    const auto dst = memory.at(ARG_DST);
    OPENVINO_ASSERT(src, "Transpose source memory is undefined");
    OPENVINO_ASSERT(dst, "Transpose destination memory is undefined");

    const auto dstDesc = dst->getDescWithType<DnnlMemoryDesc>()->getDnnlDesc();
    const auto srcDesc =
        dnnl::memory::desc(dstDesc.get_dims(), dstDesc.get_data_type(), dnnl::memory::format_tag::acdb);
    auto primitive = getReorderPrim(context->getRuntimeCache(), context->getEngine(), srcDesc, dstDesc);
    if (!primitive) {
        return false;
    }

    m_primitive = primitive;
    m_implType = parse_impl_name(DnnlExtensionUtils::query_impl_info_str(m_primitive.get_primitive_desc()));

    return true;
}

void ReorderTransposeExecutor::execute(const MemoryArgs& memory) {
    OPENVINO_ASSERT(m_primitive, "Transpose reorder primitive is not initialized");

    const auto src = memory.at(ARG_SRC);
    const auto dst = memory.at(ARG_DST);
    OPENVINO_ASSERT(src, "Transpose source memory is undefined");
    OPENVINO_ASSERT(dst, "Transpose destination memory is undefined");

    auto stream = make_stream(context->getEngine(), context->getThreadPool());
    m_primitive.execute(stream, {{DNNL_ARG_FROM, src->getPrimitive()}, {DNNL_ARG_TO, dst->getPrimitive()}});
}

}  // namespace ov::intel_cpu
