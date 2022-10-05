// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "intel_gpu/primitives/matrix_nms.hpp"

#include <memory>

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program.hpp"

// clang-format off
#include <ngraph/opsets/opset8.hpp>
#include "ngraph_ops/nms_static_shape_ie.hpp"
// clang-format on

#include "intel_gpu/primitives/mutable_data.hpp"

namespace ov {
namespace intel_gpu {

namespace {

cldnn::matrix_nms::DecayFunction from(ngraph::op::v8::MatrixNms::DecayFunction decay) {
    switch (decay) {
    case ngraph::op::v8::MatrixNms::DecayFunction::GAUSSIAN:
        return cldnn::matrix_nms::DecayFunction::gaussian;
    case ngraph::op::v8::MatrixNms::DecayFunction::LINEAR:
    default:
        return cldnn::matrix_nms::DecayFunction::linear;
    }
}

cldnn::matrix_nms::SortResultType from(ngraph::op::v8::MatrixNms::SortResultType type) {
    switch (type) {
    case ngraph::op::v8::MatrixNms::SortResultType::CLASSID:
        return cldnn::matrix_nms::SortResultType::class_id;
    case ngraph::op::v8::MatrixNms::SortResultType::SCORE:
        return cldnn::matrix_nms::SortResultType::score;
    case ngraph::op::v8::MatrixNms::SortResultType::NONE:
    default:
        return cldnn::matrix_nms::SortResultType::none;
    }
}

cldnn::matrix_nms::attributes from(const ngraph::op::v8::MatrixNms::Attributes& attrs) {
    return cldnn::matrix_nms::attributes(from(attrs.sort_result_type),
                                         attrs.sort_result_across_batch,
                                         attrs.score_threshold,
                                         attrs.nms_top_k,
                                         attrs.keep_top_k,
                                         attrs.background_class,
                                         from(attrs.decay_function),
                                         attrs.gaussian_sigma,
                                         attrs.post_threshold,
                                         attrs.normalized);
}

void CreateNmsStaticShapeIEMatrixNmsOp(Program& p, const std::shared_ptr<ngraph::op::internal::NmsStaticShapeIE<ngraph::opset8::MatrixNms>>& op) {
    validate_inputs_count(op, {2});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);

    std::vector<cldnn::memory::ptr> shared_memory;

    auto outputIndices = op->get_output_shape(0)[0];
    cldnn::layout mutableLayoutFirst = cldnn::layout(
        cldnn::element_type_to_data_type(ngraph::element::i32),
        cldnn::format::bfyx,
        cldnn::tensor(static_cast<int32_t>(outputIndices), 1, 1, 1));

    shared_memory.emplace_back(p.GetEngine().allocate_memory(mutableLayoutFirst));

    cldnn::primitive_id matrix_nms_mutable_id_w_first = layer_type_name_ID(op) + "_md_write_first";
    auto matrix_nms_mutable_prim_first = cldnn::mutable_data(matrix_nms_mutable_id_w_first,
                                                      shared_memory.back());
    p.add_primitive(*op, matrix_nms_mutable_prim_first);
    inputPrimitives.push_back(matrix_nms_mutable_id_w_first);

    auto batches_num = op->get_output_shape(2)[0];
    cldnn::layout mutableLayoutSecond = cldnn::layout(
        cldnn::element_type_to_data_type(ngraph::element::i32),
        cldnn::format::bfyx,
        cldnn::tensor(static_cast<int32_t>(batches_num), 1, 1, 1));

    shared_memory.emplace_back(p.GetEngine().allocate_memory(mutableLayoutSecond));

    cldnn::primitive_id matrix_nms_mutable_id_w_second = layer_type_name_ID(op) + "_md_write_second";
    auto matrix_nms_mutable_prim_second = cldnn::mutable_data(matrix_nms_mutable_id_w_second, shared_memory.back());
    p.add_primitive(*op, matrix_nms_mutable_prim_second);
    inputPrimitives.push_back(matrix_nms_mutable_id_w_second);

    auto matrixNmsLayerName = layer_type_name_ID(op) + ".out0";

    auto prim = cldnn::matrix_nms(
            matrixNmsLayerName,
            inputPrimitives[0],
            inputPrimitives[1],
            inputPrimitives[inputPrimitives.size() - 2],
            inputPrimitives[inputPrimitives.size() - 1],
            from(op->get_attrs()));

    p.add_primitive(*op, prim);

    cldnn::primitive_id matrix_nms_id_r_first = layer_type_name_ID(op) + ".out1";
    auto matrix_nms_mutable_prim_r_first = cldnn::mutable_data(matrix_nms_id_r_first,
                                                        { matrixNmsLayerName },
                                                        shared_memory.front());
    p.add_primitive(*op, matrix_nms_mutable_prim_r_first);


    cldnn::primitive_id matrix_nms_id_r_second = layer_type_name_ID(op) + ".out2";
    auto matrix_nms_mutable_prim_r_second = cldnn::mutable_data(matrix_nms_id_r_second,
                                                         { matrixNmsLayerName },
                                                         shared_memory.back());
    p.add_primitive(*op, matrix_nms_mutable_prim_r_second);
}

}  // anonymous namespace

REGISTER_FACTORY_IMPL_TYPED(internal, NmsStaticShapeIE, opset8, MatrixNms);

}  // namespace intel_gpu
}  // namespace ov
