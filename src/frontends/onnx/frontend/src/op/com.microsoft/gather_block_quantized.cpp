// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdint>
#include <memory>
#include <vector>

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/decompositions/low_precision_dequantize.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/slice.hpp"
#include "utils/common.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace com_microsoft {
namespace opset_1 {

// com.microsoft.GatherBlockQuantized: a Gather over a block-quantized table.
// https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.GatherBlockQuantized
//
// The op is decomposed into native OpenVINO ops that the plugins' ConvertGatherToGatherCompressed pass folds
// into the optimized internal GatherCompressed op:
//   reinterpret(data)  ->  low_precision_dequantize (Convert [-> Subtract(zp)] -> Multiply(scale))  ->
//   Reshape (back to the logical table)  ->  v8::Gather(gather_axis)
// Dequantization follows ORT semantics: output = (data - zero_point) * scale, one scale per `block_size`
// consecutive elements along `quantize_axis`. Default zero_point is 0 for int4/uint4 and 2^(bits-1) for uint8.
// uint8 data may pack 8/bits sub-byte values per byte along the last dimension (bits 2/4, low-bits-first);
// it is reinterpreted to the matching native u2/u4 type, and its packed zero_points are unpacked the same way.
ov::OutputVector gather_block_quantized(const ov::frontend::onnx::Node& node) {
    using namespace ov::op;
    common::default_op_checks(node, 3);

    const auto inputs = node.get_ov_inputs();
    const auto& data = inputs[0];      // required, quantized table (int4/uint4/uint8)
    const auto& indices = inputs[1];   // required, int32/int64
    const auto& scales = inputs[2];    // required, f32/f16/bf16
    ov::Output<ov::Node> zero_points;  // optional, input[3], same type as data

    const auto gather_axis_attr = node.get_attribute_value<int64_t>("gather_axis", 0);
    const auto quantize_axis_attr = node.get_attribute_value<int64_t>("quantize_axis", 1);
    const auto block_size = node.get_attribute_value<int64_t>("block_size", 128);
    const auto bits = node.get_attribute_value<int64_t>("bits", 4);

    // ---- data: constant of a supported quantized type with a static shape ----
    CHECK_VALID_NODE(node,
                     ov::as_type<v0::Constant>(data.get_node()) != nullptr,
                     "GatherBlockQuantized: 'data' must be a constant");
    const auto data_et = data.get_element_type();
    CHECK_VALID_NODE(node,
                     data_et == ov::element::i4 || data_et == ov::element::u4 || data_et == ov::element::u8,
                     "GatherBlockQuantized: unsupported 'data' type, expected int4/uint4/uint8, got: ",
                     data_et);
    const auto& data_pshape = data.get_partial_shape();
    CHECK_VALID_NODE(node,
                     data_pshape.is_static(),
                     "GatherBlockQuantized: 'data' must have a static shape, got: ",
                     data_pshape);
    const auto data_shape = data_pshape.get_shape();
    const auto r = static_cast<int64_t>(data_shape.size());
    CHECK_VALID_NODE(node, r >= 2, "GatherBlockQuantized: 'data' rank must be >= 2, got: ", r);

    // ---- block_size / bits (mirror ORT contrib_defs.cc + CPU kernel) ----
    CHECK_VALID_NODE(node,
                     block_size >= 16 && (block_size & (block_size - 1)) == 0,
                     "GatherBlockQuantized: 'block_size' must be a power of 2 and >= 16, got: ",
                     block_size);
    const bool is_u8 = data_et == ov::element::u8;
    if (is_u8) {
        CHECK_VALID_NODE(node,
                         bits == 2 || bits == 4 || bits == 8,
                         "GatherBlockQuantized: for uint8 data 'bits' must be 2, 4 or 8, got: ",
                         bits);
    } else {
        CHECK_VALID_NODE(node, bits == 4, "GatherBlockQuantized: for int4/uint4 data 'bits' must be 4, got: ", bits);
    }

    // Native element type the packed 'data' is reinterpreted to, and how many logical values are packed per
    // stored uint8 (1 for whole-byte uint8 and for int4/uint4, which are already native sub-byte constants).
    // For uint8 with bits 2/4, each stored byte holds 8/bits values low-bits-first along the last dim, so we
    // reinterpret to u2/u4 and expand the last dimension by 'components' (mirrors matmulnbits.cpp).
    ov::element::Type reinterp_et = data_et;
    int64_t components = 1;
    if (is_u8 && bits != 8) {
        components = 8 / bits;
        reinterp_et = (bits == 2) ? ov::element::u2 : ov::element::u4;
    }

    // ---- normalize axes ----
    CHECK_VALID_NODE(node,
                     gather_axis_attr >= -r && gather_axis_attr < r,
                     "GatherBlockQuantized: 'gather_axis' out of range [-r, r-1]: ",
                     gather_axis_attr);
    CHECK_VALID_NODE(node,
                     quantize_axis_attr >= -r && quantize_axis_attr < r,
                     "GatherBlockQuantized: 'quantize_axis' out of range [-r, r-1]: ",
                     quantize_axis_attr);
    const auto gather_axis = (gather_axis_attr + r) % r;
    const auto quantize_axis = (quantize_axis_attr + r) % r;
    CHECK_VALID_NODE(node,
                     gather_axis != quantize_axis,
                     "GatherBlockQuantized: 'gather_axis' and 'quantize_axis' must differ");
    // 'quantize_axis' must be the last dimension. This matches the ORT CPU kernel's uint8 constraint and,
    // for int4/uint4, guarantees numerical correctness of the compressed-gather fusion: the plugins'
    // ConvertGatherToGatherCompressed collapses the reinterpreted [.., nb, block_size] table by merging its
    // last two dimensions, which only reconstructs the logical table when the quantized axis is last.
    // A non-last quantize_axis would otherwise silently mis-fuse on CPU/GPU (see the shape check below).
    CHECK_VALID_NODE(node,
                     quantize_axis == r - 1,
                     "GatherBlockQuantized: 'quantize_axis' must be the last dimension, got: ",
                     quantize_axis,
                     " (rank ",
                     r,
                     ")");
    if (is_u8) {
        CHECK_VALID_NODE(node, gather_axis == 0, "GatherBlockQuantized: for uint8 data 'gather_axis' must be 0");
    }

    // Logical (unpacked) size of the quantized axis. For uint8 sub-byte data the stored dim is 'components'
    // times smaller than the logical dim.
    const int64_t quant_dim = static_cast<int64_t>(data_shape[static_cast<size_t>(quantize_axis)]) * components;
    CHECK_VALID_NODE(node,
                     quant_dim % block_size == 0,
                     "GatherBlockQuantized: quantize dimension (",
                     quant_dim,
                     ") must be a multiple of block_size (",
                     block_size,
                     ")");
    const int64_t nb = quant_dim / block_size;

    // ---- scales: constant, matching rank, per-block along quantize_axis ----
    CHECK_VALID_NODE(node,
                     ov::as_type<v0::Constant>(scales.get_node()) != nullptr,
                     "GatherBlockQuantized: 'scales' must be a constant");
    const auto scales_et = scales.get_element_type();
    CHECK_VALID_NODE(node,
                     scales_et == ov::element::f32 || scales_et == ov::element::f16 || scales_et == ov::element::bf16,
                     "GatherBlockQuantized: unsupported 'scales' type, expected f32/f16/bf16, got: ",
                     scales_et);
    const auto& scales_pshape = scales.get_partial_shape();
    CHECK_VALID_NODE(node,
                     scales_pshape.is_static(),
                     "GatherBlockQuantized: 'scales' must have a static shape, got: ",
                     scales_pshape);
    const auto scales_shape = scales_pshape.get_shape();
    CHECK_VALID_NODE(node,
                     static_cast<int64_t>(scales_shape.size()) == r,
                     "GatherBlockQuantized: 'scales' must have the same rank as 'data'");
    for (int64_t i = 0; i < r; ++i) {
        const auto expected = (i == quantize_axis) ? static_cast<size_t>(nb) : data_shape[static_cast<size_t>(i)];
        CHECK_VALID_NODE(node,
                         scales_shape[static_cast<size_t>(i)] == expected,
                         "GatherBlockQuantized: 'scales' shape mismatch at axis ",
                         i,
                         ", expected ",
                         expected,
                         ", got ",
                         scales_shape[static_cast<size_t>(i)]);
    }

    // ---- indices ----
    const auto indices_et = indices.get_element_type();
    CHECK_VALID_NODE(
        node,
        indices_et == ov::element::i32 || indices_et == ov::element::i64 || indices_et == ov::element::dynamic,
        "GatherBlockQuantized: 'indices' must be int32 or int64, got: ",
        indices_et);

    // ---- per-block split shapes ----
    // data:  [.., d_qa, ..]  ->  [.., nb, block_size, ..]   (rank r+1)
    // sc/zp: [.., nb,   ..]  ->  [.., nb, 1,          ..]   (rank r+1, size-1 axis broadcasts over the block)
    ov::Shape data_split_shape;
    ov::Shape param_split_shape;
    data_split_shape.reserve(static_cast<size_t>(r) + 1);
    param_split_shape.reserve(static_cast<size_t>(r) + 1);
    for (int64_t i = 0; i < r; ++i) {
        if (i == quantize_axis) {
            data_split_shape.push_back(static_cast<size_t>(nb));
            data_split_shape.push_back(static_cast<size_t>(block_size));
            param_split_shape.push_back(static_cast<size_t>(nb));
            param_split_shape.push_back(1);
        } else {
            data_split_shape.push_back(data_shape[static_cast<size_t>(i)]);
            param_split_shape.push_back(data_shape[static_cast<size_t>(i)]);
        }
    }

    const auto data_const = ov::as_type_ptr<v0::Constant>(data.get_node_shared_ptr());
    const auto scales_const = ov::as_type_ptr<v0::Constant>(scales.get_node_shared_ptr());

    // Reinterpret 'data' as the per-block-split constant of the native (sub-)byte type. OpenVINO u2/u4/i4
    // packing is low-bits-first, byte-identical to ORT; splitting/unpacking an axis is a pure contiguous
    // reshape over the flat bitstream, so the raw byte buffer is reused verbatim (for uint8 sub-byte,
    // reinterp_et is u2/u4 and the split shape already carries the unpacked, components-expanded size).
    // The buffer is wrapped (not copied) by passing the source constant as the shared owner, which keeps it
    // alive; this avoids duplicating a potentially large embedding table in memory.
    auto data_split =
        std::make_shared<v0::Constant>(reinterp_et, data_split_shape, data_const->get_data_ptr(), data_const);
    // Preserve the original initializer name so downstream weight-sharing by name still works.
    data_split->set_friendly_name(data_const->get_friendly_name());
    data_split->get_output_tensor(0).set_names(data_const->get_output_tensor(0).get_names());

    // Reinterpret 'scales' with an inserted size-1 block axis for per-block broadcast (buffer wrapped, not copied).
    auto scales_split =
        std::make_shared<v0::Constant>(scales_et, param_split_shape, scales_const->get_data_ptr(), scales_const);

    // ---- zero point ----
    ov::Output<ov::Node> zp_for_dequant;
    if (common::is_input_valid(node, 3)) {
        zero_points = inputs[3];
        CHECK_VALID_NODE(node,
                         ov::as_type<v0::Constant>(zero_points.get_node()) != nullptr,
                         "GatherBlockQuantized: 'zero_points' must be a constant");
        CHECK_VALID_NODE(node,
                         zero_points.get_element_type() == data_et,
                         "GatherBlockQuantized: 'zero_points' must have the same type as 'data', got: ",
                         zero_points.get_element_type());
        const auto& zp_pshape = zero_points.get_partial_shape();
        CHECK_VALID_NODE(node,
                         zp_pshape.is_static(),
                         "GatherBlockQuantized: 'zero_points' must have a static shape, got: ",
                         zp_pshape);
        const auto zp_shape = zp_pshape.get_shape();
        CHECK_VALID_NODE(node,
                         static_cast<int64_t>(zp_shape.size()) == r,
                         "GatherBlockQuantized: 'zero_points' must have the same rank as 'data'");
        // zero_points match scales, except that for uint8 sub-byte data the per-block zero_points are
        // themselves packed 'components'-per-byte along the quantize axis (ceil(nb / components) bytes).
        const int64_t zp_qaxis_bytes = (nb + components - 1) / components;
        for (int64_t i = 0; i < r; ++i) {
            const auto expected =
                (i == quantize_axis) ? static_cast<size_t>(zp_qaxis_bytes) : data_shape[static_cast<size_t>(i)];
            CHECK_VALID_NODE(node,
                             zp_shape[static_cast<size_t>(i)] == expected,
                             "GatherBlockQuantized: 'zero_points' shape mismatch at axis ",
                             i,
                             ", expected ",
                             expected,
                             ", got ",
                             zp_shape[static_cast<size_t>(i)]);
        }
        const auto zp_const = ov::as_type_ptr<v0::Constant>(zero_points.get_node_shared_ptr());
        if (components == 1) {
            // int4/uint4/uint8(bits=8): one stored value per block, reinterpret directly to [.., nb, 1, ..].
            zp_for_dequant =
                std::make_shared<v0::Constant>(reinterp_et, param_split_shape, zp_const->get_data_ptr(), zp_const);
        } else {
            // uint8 sub-byte: reinterpret the packed zero_points to u2/u4 at the byte-aligned block count,
            // convert to the dequant precision, then slice off the alignment padding down to nb blocks
            // (mirrors matmulnbits.cpp; convert before slice to avoid slicing a sub-byte constant).
            const int64_t aligned = zp_qaxis_bytes * components;
            ov::Shape zp_aligned_shape = param_split_shape;
            zp_aligned_shape[static_cast<size_t>(quantize_axis)] = static_cast<size_t>(aligned);
            const auto zp_aligned =
                std::make_shared<v0::Constant>(reinterp_et, zp_aligned_shape, zp_const->get_data_ptr(), zp_const);
            ov::Output<ov::Node> zp_converted = std::make_shared<v0::Convert>(zp_aligned, scales_et);
            if (aligned != nb) {
                const auto start = v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
                const auto stop = v0::Constant::create(ov::element::i64, ov::Shape{1}, {nb});
                const auto step = v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
                const auto slice_axis = v0::Constant::create(ov::element::i64, ov::Shape{1}, {quantize_axis});
                zp_converted = std::make_shared<v8::Slice>(zp_converted, start, stop, step, slice_axis);
            }
            zp_for_dequant = zp_converted;
        }
    } else if (is_u8) {
        // Default uint8 zero point is 2^(bits-1); subtract it even in the "symmetric" case.
        const int32_t default_zp = 1 << (static_cast<int>(bits) - 1);
        zp_for_dequant = std::make_shared<v0::Constant>(reinterp_et, ov::Shape{1}, default_zp);
    }
    // int4/uint4 with no zero_points -> default zero point 0 -> leave zp empty so the Subtract is omitted
    // (symmetric dequantization; this is the clean form the compressed-gather fusion prefers).

    // Dequantize: Multiply(Subtract(Convert(data), zp), scale) or, without zp, Multiply(Convert(data), scale).
    // Output element type follows 'scales'.
    auto dequant = ov::decomposition::low_precision_dequantize(data_split, scales_split, zp_for_dequant);

    // Reshape the dequantized blocks back to the logical table shape. For the rank-2 embedding case this is the
    // 3D->2D reshape that ConvertGatherToGatherCompressed folds together with the Gather into GatherCompressed.
    // The (last) quantized axis uses its unpacked size (quant_dim = stored_dim * components).
    std::vector<int64_t> logical_dims(data_shape.begin(), data_shape.end());
    logical_dims[static_cast<size_t>(quantize_axis)] = quant_dim;
    const auto table_shape = v0::Constant::create(ov::element::i64, ov::Shape{logical_dims.size()}, logical_dims);
    const auto table = std::make_shared<v1::Reshape>(dequant, table_shape, false);

    // Gather on gather_axis. v8::Gather natively handles negative indices, matching ORT.
    const auto axis = v0::Constant::create(ov::element::i64, ov::Shape{}, {gather_axis});
    return {std::make_shared<v8::Gather>(table, indices, axis)};
}

ONNX_OP("GatherBlockQuantized", OPSET_SINCE(1), com_microsoft::opset_1::gather_block_quantized, MICROSOFT_DOMAIN);

}  // namespace opset_1
}  // namespace com_microsoft
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
