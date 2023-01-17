// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API TransposeSinkingUnaryForward;
class TRANSFORMATIONS_API TransposeSinkingUnaryBackwardSingleConsumer;
class TRANSFORMATIONS_API TransposeSinkingUnaryBackwardMultiConsumers;
class TRANSFORMATIONS_API TransposeSinkingUnaryBackward;

}  // namespace pass
}  // namespace ov

class ov::pass::TransposeSinkingUnaryForward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("TransposeSinkingUnaryForward", "0");
    TransposeSinkingUnaryForward();
};

class ov::pass::TransposeSinkingUnaryBackwardSingleConsumer : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("TransposeSinkingUnaryBackwardSingleConsumer", "0");
    TransposeSinkingUnaryBackwardSingleConsumer();
};

class ov::pass::TransposeSinkingUnaryBackwardMultiConsumers : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("TransposeSinkingUnaryBackwardMultiConsumers", "0");
    TransposeSinkingUnaryBackwardMultiConsumers();
};

class ov::pass::TransposeSinkingUnaryBackward : public ov::pass::GraphRewrite {
public:
    OPENVINO_RTTI("TransposeSinkingUnaryBackward", "0");
    TransposeSinkingUnaryBackward();
};
