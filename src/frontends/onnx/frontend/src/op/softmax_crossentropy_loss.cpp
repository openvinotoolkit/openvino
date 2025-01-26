// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/log.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/negative.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/softmax.hpp"
#include "utils/common.hpp"
#include "softmax_cross_entropy_loss.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace {
    // softmax cross entropy implementation (Shared helper fn)
    OutputVector impl_softmax_cross_entropy(const Node& node, int64_t axis_default) {
        const auto inputs = node.get_ov_inputs();

        const auto scores = inputs[0];
        const auto labels = inputs[1];

        const auto axis = node.get_attribute_value<int64_t>("axis", axis_default);
        const auto reduction = node.get_attribute_value<std::string>("reduction", "mean");

        // Computing softmax
        const auto softmax = std::make_shared<ov::op::v8::Softmax>(scores, axis);
        const auto log_softmax = std::make_shared<ov::op::v0::Log>(softmax);

        const auto axis_const = ov::op::v0::Constant::create(element::i64, {}, {axis});
        const auto gathered = std::make_shared<ov::op::v8::Gather>(log_softmax, labels, axis_const);


        // Computing loss
        std::shared_ptr<ov::Node> loss = std::make_shared<ov::op::v0::Negative>(gathered);

        // applying reduction as mentioned in https://github.com/onnx/onnx/blob/main/docs/Changelog.md#softmaxcrossentropyloss-12

        if (reduction != "none") {
            const auto reduce_axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
            
            loss = (reduction == "mean")
                       ? static_cast<std::shared_ptr<ov::Node>>(
                             std::make_shared<ov::op::v1::ReduceMean>(loss->output(0), reduce_axis, true))
                       : static_cast<std::shared_ptr<ov::Node>>(
                             std::make_shared<ov::op::v1::ReduceSum>(loss->output(0), reduce_axis, true));
        }

        return {loss};
    }
}
namespace ai_onnx {
    namespace opset_12 {
    OutputVector ov::frontend::onnx::ai_onnx::opset_12::softmax_cross_entropy_loss(const Node& node) {
        return impl_softmax_cross_entropy(node, 1);
    }
    ONNX_OP("SoftmaxCrossEntropyLoss", OPSET_SINCE(12), ai_onnx::opset_12::softmax_cross_entropy_loss);
    }
    namespace opset_13 {
    OutputVector ov::frontend::onnx::ai_onnx::opset_13::softmax_cross_entropy_loss(const Node& node) {
        return impl_softmax_cross_entropy(node, 1);
    }

    ONNX_OP("SoftmaxCrossEntropyLoss", OPSET_SINCE(13), ai_onnx::opset_13::softmax_cross_entropy_loss);
    }
} 
}
}
}