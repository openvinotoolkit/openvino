// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <array>

#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "ngraph/op/matmul.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/fake_quantize.hpp"

#include "intel_gpu/primitives/gemm.hpp"
#include "intel_gpu/primitives/fully_connected.hpp"
#include "intel_gpu/primitives/reshape.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/primitives/permute.hpp"

namespace ov {
namespace intel_gpu {

/*
*  get_aligned_shapes function align two input shapes to have the same size and
*  the same batch dimensions (last two dimensions are not comparable).
*  It also checks that dimensions are compatible so in case with two shapes
*  for example: [2, 32, 64] [3, 64, 64] it will raise an exception.
*/

static std::tuple<bool, PartialShape, PartialShape> get_aligned_shapes(const PartialShape& shape_a,
                                                                       const PartialShape& shape_b,
                                                                       const std::shared_ptr<ngraph::op::v0::MatMul>& matmul) {
    PartialShape shape_a_aligned(shape_a), shape_b_aligned(shape_b);
    auto rank_a = shape_a_aligned.rank().get_length();
    auto rank_b = shape_b_aligned.rank().get_length();
    size_t max_size = std::max(rank_a, rank_b);
    if (max_size == 1) {
        return std::make_tuple(false, shape_a_aligned, shape_b_aligned);
    }

    for (size_t i = 0, cnt = max_size - rank_a; i < cnt; ++i) {
        shape_a_aligned.insert(shape_a_aligned.begin(), 1);
    }
    for (size_t i = 0, cnt = max_size - rank_b; i < cnt; ++i) {
        shape_b_aligned.insert(shape_b_aligned.begin(), 1);
    }

    if (matmul->get_transpose_a()) {
        std::swap(*(shape_a_aligned.end() - 1), *(shape_a_aligned.end() - 2));
    }
    if (matmul->get_transpose_b()) {
        std::swap(*(shape_b_aligned.end() - 1), *(shape_b_aligned.end() - 2));
    }

    for (size_t i = 0; i < max_size - 2; ++i) {
        auto a_dim = shape_a_aligned[i], b_dim = shape_b_aligned[i];
        if (a_dim.is_dynamic()) {
            if (b_dim == 1) {
                shape_a_aligned[i] = shape_b_aligned[i] = a_dim;
            } else {
                return std::make_tuple(false, shape_a_aligned, shape_b_aligned);
            }
        } else {
            if (a_dim != b_dim && a_dim.get_length() > 1 && b_dim.get_length() > 1) {
                IE_THROW() << "Shapes can't be aligned: " << shape_a_aligned << " " << shape_b_aligned;
            }
            auto max_value = std::max(a_dim.get_length(), b_dim.get_length());
            shape_a_aligned[i] = shape_b_aligned[i] = max_value;
        }
    }
    return {true, shape_a_aligned, shape_b_aligned};
}

static void CreateMatMulOp(Program& p, const std::shared_ptr<ngraph::op::v0::MatMul>& op) {
    validate_inputs_count(op, {2});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    auto shape_a = op->get_input_partial_shape(0);
    auto shape_b = op->get_input_partial_shape(1);

    auto rank_a = shape_a.rank().get_length();
    auto rank_b = shape_b.rank().get_length();

    bool is_fc = IsNodeOnConstPath(op->get_input_node_shared_ptr(1));
    is_fc &= std::count_if(shape_b.begin(), shape_b.end(), [](Dimension x) { return x != 1; }) <= 2;
    // TODO: This conditions can be relaxed with proper handling in FC path
    is_fc &= rank_a > 1 && rank_b > 1;

    PartialShape shape_a_aligned, shape_b_aligned;
    bool aligned = false;
    if (shape_b.is_static()) {
        std::tie(aligned, shape_a_aligned, shape_b_aligned) = get_aligned_shapes(shape_a, shape_b, op);
    }
    is_fc &= aligned;

    if (is_fc) {
        if (shape_a_aligned.size() < 2 || shape_b_aligned.size() < 2) {
            IE_THROW() << "MatMul " << op->get_friendly_name() << " shapes are inconsistent.";
        }

        auto inputName = inputPrimitives[0];
        auto weightsName = inputPrimitives[1];

        auto create_transpose = [&](const std::string& transposeName, const std::string& transposeInputName, size_t rank) {
            std::vector<uint16_t> transpose_order(rank);
            std::iota(transpose_order.begin(), transpose_order.end(), 0);
            std::swap(*(transpose_order.end() - 1), *(transpose_order.end() - 2));

            auto permutePrim = cldnn::permute(transposeName,
                                              transposeInputName,
                                              transpose_order);
            p.add_primitive(*op, permutePrim);
        };

        // Weights normalization
        if (!op->get_transpose_b()) {
            auto transposeName = op->get_friendly_name() + "/transpose_b";
            create_transpose(transposeName, weightsName, rank_b);
            weightsName = transposeName;
        }

        // Input normalization
        if (op->get_transpose_a()) {
            auto transposeName = op->get_friendly_name() + "/transpose_a";
            create_transpose(transposeName, inputName, rank_a);
            inputName = transposeName;
        }

        auto fcPrim = cldnn::fully_connected(layerName,
                                             inputName,
                                             weightsName,
                                             "",
                                             cldnn::element_type_to_data_type(op->get_output_element_type(0)),
                                             cldnn::padding(),
                                             shape_a.size());

        p.add_primitive(*op, fcPrim);

        if (shape_a_aligned.size() > 3 && !p.use_new_shape_infer()) {
            auto lastLayerName = layerName;
            auto outReshapeName = layerName + "_cldnn_out_reshape";

            // add reorder
            auto outDims = op->get_output_shape(0);
            auto outTensor = tensor_from_dims(outDims);

            if (outDims.size() > 4) {
                cldnn::format outputFormat = cldnn::format::bfyx;
                switch (outDims.size()) {
                    case 5: outputFormat = cldnn::format::bfzyx; break;
                    case 6: outputFormat = cldnn::format::bfwzyx; break;
                    default: break;
                }

                cldnn::primitive_id reorderId = "reorder:" + outReshapeName + "_reorder";
                cldnn::layout outputLayout(cldnn::element_type_to_data_type(op->get_output_element_type(0)), outputFormat, outTensor);
                auto reorder_prim = cldnn::reorder(reorderId, layerName, outputLayout);
                p.add_primitive(*op, reorder_prim);
                lastLayerName = reorderId;
            }

            // add reshape
            auto outReshapePrim = cldnn::reshape(outReshapeName, lastLayerName, outTensor);
            p.add_primitive(*op, outReshapePrim);
        }
    } else {
        // Add actual gemm
        auto alpha = 1.0f;
        auto beta = 0.0f;
        auto transA = op->get_transpose_a();
        auto transB = op->get_transpose_b();

        std::array<ngraph::PartialShape, 2> inputShapes{
            op->get_input_partial_shape(0),
            op->get_input_partial_shape(1)
        };

        auto canTransposeInputs = [] (const std::array<ngraph::PartialShape, 2>& shapes, bool transA, bool transB) -> bool {
            if (!transA && !transB)
                return false;
            if (shapes[0].rank().is_dynamic() ||
                shapes[1].rank().is_dynamic())
                return false;

            // don't transpose inputs if they're aligned to 16
            bool inputsAligned = std::all_of(shapes[0].rbegin(), shapes[0].rbegin() + 2,
                                             [] (const ngraph::Dimension& dim) { return dim.is_static() && dim.get_length() % 16 == 0; }) &&
                                 std::all_of(shapes[1].rbegin(), shapes[1].rbegin() + 2,
                                             [] (const ngraph::Dimension& dim) { return dim.is_static() && dim.get_length() % 16 == 0; });
            if (inputsAligned)
                return false;

            return std::all_of(shapes[0].rbegin(), shapes[0].rbegin() + 2,
                               [] (const ngraph::Dimension& dim) { return dim.is_static() && dim.get_length() >= 64; }) &&
                   std::all_of(shapes[1].rbegin(), shapes[1].rbegin() + 2,
                               [] (const ngraph::Dimension& dim) { return dim.is_static() && dim.get_length() >= 64; });
        };

        auto transposeInput = [&layerName] (Program& p, const std::shared_ptr<ngraph::Node>& op, const ngraph::PartialShape& shape,
                                            const std::string& suffix, const cldnn::primitive_id& primitiveId) -> std::string {
            std::vector<uint16_t> transposeOrder(shape.size());
            std::iota(transposeOrder.begin(), transposeOrder.end(), 0);
            for (auto o = transposeOrder.size(); o < 4; o++)
                transposeOrder.push_back((uint16_t)o);
            std::swap(*(transposeOrder.end() - 1), *(transposeOrder.end() - 2));

            auto permuteName = op->get_friendly_name() + suffix;
            auto permutePrim = cldnn::permute(permuteName,
                                              primitiveId,
                                              transposeOrder);
            p.add_primitive(*op, permutePrim);
            return permuteName;
        };

        if (canTransposeInputs(inputShapes, transA, transB)) {
            if (transA) {
                inputPrimitives[0] = transposeInput(p, op, inputShapes[0], "/transpose_a", inputPrimitives[0]);
                transA = false;
            }

            if (transB) {
                inputPrimitives[1] = transposeInput(p, op, inputShapes[1], "/transpose_b", inputPrimitives[1]);
                transB = false;
            }
        }

        auto gemmPrim = cldnn::gemm(layerName,
                                    inputPrimitives,
                                    cldnn::element_type_to_data_type(op->get_output_element_type(0)),
                                    op->get_transpose_a(),
                                    op->get_transpose_b(),
                                    alpha,
                                    beta,
                                    rank_a,
                                    rank_b);

        p.add_primitive(*op, gemmPrim);

        if (!p.use_new_shape_infer()) {
            auto outDims = op->get_output_shape(0);
            auto outDimsN = outDims.size();
            // Reshape output if gemm specific shape does not match default one
            if (outDimsN < 4) {
                auto outputShape = tensor_from_dims(outDims);
                auto outReshapeName = layerName + "_cldnn_out_reshape";
                auto outReshapePrim = cldnn::reshape(outReshapeName, layerName, outputShape);
                p.add_primitive(*op, outReshapePrim);
            }
        }
    }
}

REGISTER_FACTORY_IMPL(v0, MatMul);

}  // namespace intel_gpu
}  // namespace ov
