// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "ngraph/op/split.hpp"
#include "ngraph/op/variadic_split.hpp"

#include "intel_gpu/primitives/crop.hpp"

namespace ov {
namespace intel_gpu {

static void CreateCommonSplitOp(Program& p, const std::shared_ptr<ngraph::Node>& op) {
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    auto input_pshape = op->get_input_partial_shape(0);
    InferenceEngine::SizeVector start_offset(input_pshape.size());

    bool is_single_out_split = op->get_output_size() == 1;

    for (size_t i = 0; i < op->get_output_size(); i++) {
        std::string outLayerName = layerName + (is_single_out_split ? "" : ".out" + std::to_string(i));
        const auto outPatialShape = op->get_output_partial_shape(i);
        NGRAPH_SUPPRESS_DEPRECATED_START
        if (outPatialShape.size() != start_offset.size()) {
            IE_THROW() << "Invalid dimesions in split layer: " << op->get_friendly_name()
                               << " output: " <<  op->get_output_tensor_name(i);
        }
        for (size_t i = 0; i < input_pshape.size(); i++) {
            if ((outPatialShape[i].get_length() + static_cast<ov::Dimension::value_type>(start_offset[i])) > input_pshape[i].get_length()) {
                IE_THROW() << "Invalid dimesions in split layer: " << op->get_friendly_name()
                                   << " output: " <<  op->get_output_tensor_name(i);
            }
        }
        NGRAPH_SUPPRESS_DEPRECATED_END

        auto outTensor = outPatialShape.is_static()? tensor_from_dims(outPatialShape.to_shape(), 1) : cldnn::tensor(1);
        auto offsetTensor = tensor_from_dims(start_offset, 0);
        auto cropPrim = cldnn::crop(outLayerName, inputPrimitives, outTensor, offsetTensor, i);
        if (ngraph::is_type<ngraph::op::v1::Split>(op)) {
            auto split = ngraph::as_type_ptr<ngraph::op::v1::Split>(op);
            cropPrim.num_splits = split->get_num_splits();
        }
        p.add_primitive(*op, cropPrim);

        if (outPatialShape.is_static()) {
            for (size_t i = 0; i < input_pshape.size(); i++) {
                if (outPatialShape[i] != input_pshape[i]) {
                    start_offset[i] += outPatialShape.to_shape()[i];
                }
            }
        }
    }
}

static void CreateSplitOp(Program& p, const std::shared_ptr<ngraph::op::v1::Split>& op) {
    validate_inputs_count(op, {2});
    CreateCommonSplitOp(p, op);
}

static void CreateVariadicSplitOp(Program& p, const std::shared_ptr<ngraph::op::v1::VariadicSplit>& op) {
    validate_inputs_count(op, {3});
    CreateCommonSplitOp(p, op);
}

REGISTER_FACTORY_IMPL(v1, Split);
REGISTER_FACTORY_IMPL(v1, VariadicSplit);

}  // namespace intel_gpu
}  // namespace ov
