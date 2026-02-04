// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Linalg/Passes.h"

#include <openvino/op/scaled_dot_product_attention.hpp>
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include <iostream>

#include "sdpa.hpp"
#include "../convert_common.hpp"

namespace {

using namespace ov::mlir;

struct ConvertSDPA {
    void operator()(ConversionContext& context, NodePtr node) {
        std::cout << "Hello from convertSDPA!\n" << std::endl;
    }
};

}  // namespace

namespace ov {
namespace mlir {

using namespace ov::pass::pattern;
using namespace ov::op;

SDPAPattern::SDPAPattern()
    : MarkPattern(wrap_type<v13::ScaledDotProductAttention>(), ConvertSDPA()) {}

}  // namespace mlir
}  // namespace ov
