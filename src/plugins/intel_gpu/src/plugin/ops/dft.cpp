// Copyright (C) 2021-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <intel_gpu/plugin/common_utils.hpp>
#include <intel_gpu/plugin/program.hpp>
#include <intel_gpu/primitives/dft.hpp>
#include <ngraph/op/constant.hpp>
#include <ngraph/op/dft.hpp>

namespace ov {
namespace runtime {
namespace intel_gpu {

namespace {

void createDft(Program& p, const std::shared_ptr<ngraph::Node>& op, cldnn::dft_kind kind) {
    auto& outShape = op->get_output_shape(0);
    {
        auto r = outShape.size();
        if (r != 5)  // only tested on 5 and bfzyx is hardcoded below
            throw std::runtime_error{"(i)dft v7 output rank is " + std::to_string(r)};
    }
    auto axes = dynamic_cast<const ngraph::op::Constant&>(*op->get_input_node_shared_ptr(1)).cast_vector<int64_t>();
    {
        auto dataRank = op->get_input_shape(0).size();
        ov::normalize_axes(op.get(), dataRank - 1, axes);
    }
    cldnn::dft prim(layer_type_name_ID(op),
                    p.GetInputPrimitiveIDs(op)[0],
                    std::move(axes),
                    tensor_from_dims(outShape),
                    kind,
                    op->get_friendly_name());
    p.AddPrimitive(prim);
    p.AddPrimitiveToProfiler(op);
}

void CreateDFTOp(Program& p, const std::shared_ptr<ngraph::op::v7::DFT>& op) {
    createDft(p, op, cldnn::dft_kind::forward);
}

void CreateIDFTOp(Program& p, const std::shared_ptr<ngraph::op::v7::IDFT>& op) {
    createDft(p, op, cldnn::dft_kind::inverse);
}

}  // namespace

REGISTER_FACTORY_IMPL(v7, DFT);
REGISTER_FACTORY_IMPL(v7, IDFT);

}  // namespace intel_gpu
}  // namespace runtime
}  // namespace ov
