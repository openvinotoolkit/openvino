// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/manager.hpp"
#include "partitioning/patterns/opt.hpp"

namespace {

using namespace ov::npuw::patterns::opt;

struct HostGatherChain {
    std::shared_ptr<ov::op::v0::Parameter> weight;
    std::shared_ptr<ov::op::v0::Parameter> scale;
    std::shared_ptr<ov::op::v0::Parameter> ids;
    std::shared_ptr<ov::Model> model;
};

// The dictionary chain HostGatherDQ takes: the weight and its scale are Parameters, so the dequantized table can
// only be built on the host.
//   W:[N, K] -> Convert(f16) -> [Subtract(0.5)] -> Gather -+
//                                                           +-> Multiply -> Result
//   S:[N, 1] -------------------------------> Gather -------^
//   ids:[1, T] -> Convert(i32) ----------------------^ (the Gathers' indices)
HostGatherChain build_host_gather_chain(const ov::element::Type& weight_type,
                                        bool zero_point,
                                        float zero_point_value = 0.5f) {
    const std::size_t vocab = 4;
    const std::size_t hidden = 2048;

    auto weight = std::make_shared<ov::op::v0::Parameter>(weight_type, ov::Shape{vocab, hidden});
    auto scale = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{vocab, 1});
    auto ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1, 2});
    auto axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});

    auto cvt_ids = std::make_shared<ov::op::v0::Convert>(ids, ov::element::i32);
    std::shared_ptr<ov::Node> dequant = std::make_shared<ov::op::v0::Convert>(weight, ov::element::f16);
    if (zero_point) {
        dequant = std::make_shared<ov::op::v1::Subtract>(
            dequant,
            ov::op::v0::Constant::create(ov::element::f16, ov::Shape{}, {zero_point_value}));
    }
    auto gather_w = std::make_shared<ov::op::v8::Gather>(dequant, cvt_ids, axis);
    auto gather_s = std::make_shared<ov::op::v8::Gather>(scale, cvt_ids, axis);
    auto mul = std::make_shared<ov::op::v1::Multiply>(gather_w, gather_s);
    auto result = std::make_shared<ov::op::v0::Result>(mul);

    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{weight, scale, ids});
    return {weight, scale, ids, model};
}

void run_host_gather(std::shared_ptr<ov::Model>& model, Context& ctx) {
    ov::pass::Manager manager;
    // The pattern's result is a new Parameter that the partitioner collects into the subgraph's parameter list,
    // so the model cannot be validated between passes: it would report that parameter as undeclared, which is
    // exactly what the partitioner is about to declare.
    manager.set_per_pass_validation(false);
    manager.register_pass<HostGatherDQ>(std::ref(ctx));
    manager.run_passes(model);
}

// The host gather materialises the whole table in f16. A two-bit weight is not one the device cannot decode:
// the bank presents its codes in `i4`, so the gate leaves it off this route and the stored width stands.
TEST(HostGatherDQ, AnIntegerTwoBitDictionaryIsNotUnpackedOnTheHost) {
    auto chain = build_host_gather_chain(ov::element::i2, /*zero_point=*/false);
    Context ctx;
    run_host_gather(chain.model, ctx);

    EXPECT_TRUE(ctx.params_to_unpack.empty());
    EXPECT_FALSE(ctx.params_to_gather);
}

}  // namespace
