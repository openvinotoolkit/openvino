// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "cpu_shape.h"
#include "cpu_types.h"
#include "graph_context.h"
#include "node.h"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/element_type_traits.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

class OneHot : public Node {
public:
    OneHot(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override{};
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override{};
    void execute(const dnnl::stream& strm) override;
    bool created() const override;

    bool needShapeInfer() const override;
    bool needPrepareParams() const override {
        return false;
    };
    void executeDynamicImpl(const dnnl::stream& strm) override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

private:
    typedef element_type_traits<ov::element::i32>::value_type in_type;

    struct OneHotContext {
        OneHot* nodePtr;
        size_t prefix_size;
        size_t suffix_size;
    };

    template <typename dst_t>
    struct OneHotExecute {
        void operator()(OneHotContext& ctx) {
            ctx.nodePtr->one_hot<dst_t>(ctx.prefix_size, ctx.suffix_size);
        }
    };

    mutable Dim depth = Shape::UNDEFINED_DIM;
    int32_t axis = -1;

    ov::element::Type output_precision;

    static const size_t INDICES_ID = 0;
    static const size_t DEPTH_ID = 1;
    static const size_t ON_VALUE_ID = 2;
    static const size_t OFF_VALUEAXES_ID = 3;

    template <typename out_type>
    void one_hot(size_t prefix_size, size_t suffix_size);
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
