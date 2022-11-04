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
    auto get_layer_name = [&](size_t idx)->std::string {
        return layer_type_name_ID(op) + ((op->get_output_size() == 1)? "" : ".out" + std::to_string(idx));
    };

    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    if (p.use_new_shape_infer() || op->is_dynamic()) {
        cldnn::crop_ngraph_op_mode op_mode = cldnn::crop_ngraph_op_mode::variadic_split;
        size_t num_splits = 1;
        if (ngraph::is_type<ngraph::op::v1::Split>(op)) {
            num_splits = ngraph::as_type_ptr<ngraph::op::v1::Split>(op)->get_num_splits();
            op_mode = cldnn::crop_ngraph_op_mode::split;
        }

        for (size_t i = 0; i < op->get_output_size(); i++) {
            auto cropPrim = cldnn::crop(get_layer_name(i), inputPrimitives, cldnn::tensor(1), cldnn::tensor(0), op_mode, i, num_splits);
            p.add_primitive(*op, cropPrim);
        }
    } else {
        auto input_pshape = op->get_input_partial_shape(0);
        InferenceEngine::SizeVector start_offset(input_pshape.size());
        for (size_t i = 0; i < op->get_output_size(); i++) {
            const auto outPartialShape = op->get_output_partial_shape(i);
            NGRAPH_SUPPRESS_DEPRECATED_START
            if (outPartialShape.size() != start_offset.size()) {
                IE_THROW() << "Invalid dimesions in split layer: " << op->get_friendly_name()
                                << " output: " <<  op->get_output_tensor_name(i);
            }
            for (size_t idx = 0; idx < input_pshape.size(); idx++) {
                if ((outPartialShape[idx].get_length() + static_cast<ov::Dimension::value_type>(start_offset[idx])) > input_pshape[idx].get_length()) {
                    IE_THROW() << "Invalid dimesions in split layer: " << op->get_friendly_name()
                                    << " output: " <<  op->get_output_tensor_name(idx);
                }
            }
            NGRAPH_SUPPRESS_DEPRECATED_END

            auto offsetTensor = tensor_from_dims(start_offset, 0);
            auto outTensor = tensor_from_dims(op->get_output_shape(i), 1);
            auto cropPrim = cldnn::crop(get_layer_name(i), inputPrimitives[0], outTensor, offsetTensor);
            p.add_primitive(*op, cropPrim);

            for (size_t idx = 0; idx < input_pshape.size(); idx++) {
                if (outPartialShape[idx] != input_pshape[idx]) {
                    start_offset[idx] += outPartialShape.to_shape()[idx];
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
