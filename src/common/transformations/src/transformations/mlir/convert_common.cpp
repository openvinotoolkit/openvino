// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_common.hpp"

namespace {

using namespace mlir;

IntegerType getSInt4Type(MLIRContext* ctx) {
    return IntegerType::get(ctx, 4, IntegerType::Signed);
}

IntegerType getSInt8Type(MLIRContext* ctx) {
    return IntegerType::get(ctx, 8, IntegerType::Signed);
}

IntegerType getSInt16Type(MLIRContext* ctx) {
    return IntegerType::get(ctx, 16, IntegerType::Signed);
}

IntegerType getSInt32Type(MLIRContext* ctx) {
    return IntegerType::get(ctx, 32, IntegerType::Signed);
}

IntegerType getSInt64Type(MLIRContext* ctx) {
    return IntegerType::get(ctx, 64, IntegerType::Signed);
}

IntegerType getUInt4Type(MLIRContext* ctx) {
    return IntegerType::get(ctx, 4, IntegerType::Unsigned);
}

IntegerType getUInt8Type(MLIRContext* ctx) {
    return IntegerType::get(ctx, 8, IntegerType::Unsigned);
}

IntegerType getUInt16Type(MLIRContext* ctx) {
    return IntegerType::get(ctx, 16, IntegerType::Unsigned);
}

IntegerType getUInt32Type(MLIRContext* ctx) {
    return IntegerType::get(ctx, 32, IntegerType::Unsigned);
}

IntegerType getUInt64Type(MLIRContext* ctx) {
    return IntegerType::get(ctx, 64, IntegerType::Unsigned);
}

IntegerType getBool8Type(MLIRContext* ctx) {
    // Signless 8-bit integer use for BOOL, to distinguish it from U8
    return IntegerType::get(ctx, 8, IntegerType::Signless);
}

}

namespace ov {
namespace mlir {


Location createLayerLocation(MLIRContext* ctx, const std::string& layerName, const std::string& layerType) {
    const auto layerNameAttr = StringAttr::get(ctx, layerName);
    const auto nameLoc = NameLoc::get(layerNameAttr);

    SmallVector<NamedAttribute> fields;
    fields.emplace_back(StringAttr::get(ctx, "type"), StringAttr::get(ctx, layerType));
    fields.emplace_back(StringAttr::get(ctx, "name"), layerNameAttr);
    auto metadata = DictionaryAttr::get(ctx, fields);

    return FusedLoc::get(ctx, {nameLoc}, metadata);
}

SmallVector<int64_t> importShape(const ov::PartialShape& shape) {
    SmallVector<int64_t> out(shape.rank().get_length());
    // TODO: Add support for dynamically ranked shapes
    for (size_t i = 0; i < out.size(); ++i) {
        const auto& dim = shape[i];
        out[i] = dim.is_static() ? dim.get_length() : ShapedType::kDynamic;
    }
    return out;
}

Type importPrecision(MLIRContext* ctx, const ov::element::Type& precision) {
    switch (precision) {
    case ov::element::Type_t::f64:
        return Float64Type::get(ctx);
    case ov::element::Type_t::f32:
        return Float32Type::get(ctx);
    case ov::element::Type_t::f16:
        return Float16Type::get(ctx);
    case ov::element::Type_t::bf16:
        return BFloat16Type::get(ctx);
    case ov::element::Type_t::i64:
        return getSInt64Type(ctx);
    case ov::element::Type_t::u64:
        return getUInt64Type(ctx);
    case ov::element::Type_t::i32:
        return getSInt32Type(ctx);
    case ov::element::Type_t::u32:
        return getUInt32Type(ctx);
    case ov::element::Type_t::i16:
        return getSInt16Type(ctx);
    case ov::element::Type_t::u16:
        return getUInt16Type(ctx);
    case ov::element::Type_t::i8:
        return getSInt8Type(ctx);
    case ov::element::Type_t::u8:
        return getUInt8Type(ctx);
    case ov::element::Type_t::i4:
        return getSInt4Type(ctx);
    case ov::element::Type_t::u4:
        return getUInt4Type(ctx);
    case ov::element::Type_t::boolean:
        return getBool8Type(ctx);
    default:
        OPENVINO_THROW("Unsupported element_type: ", precision);
    }
}

RankedTensorType importTensor(MLIRContext* ctx,
                                    const ov::PartialShape& shape,
                                    const ov::element::Type& elemType) {
    return RankedTensorType::get(ArrayRef(importShape(shape)), importPrecision(ctx, elemType));
}

Location createLocation(MLIRContext* ctx, NodePtr node) {
    return createLayerLocation(ctx, node->get_friendly_name(), node->get_type_name());
}

bool elementwise_no_broadcast_predicate_impl(const ov::Output<ov::Node>& output, ov::element::Type type) {
    if (output.get_element_type() != type) {
        return false;
    }
    // Check if implicit broadcast is possible, reject in this case
    // Relies on symbolic information -- register SymbolicPropagation before applying this pattern
    auto inputs = output.get_node_shared_ptr()->inputs();
    auto output_shape = output.get_partial_shape();
    if (output_shape.rank().is_dynamic()) {
        return false;
    }
    if (std::any_of(inputs.begin(), inputs.end(), [&](const ov::Input<ov::Node>& input) {
            auto input_shape = input.get_partial_shape();
            return input_shape.rank().is_dynamic() ||
                   output_shape.rank().get_length() != input_shape.rank().get_length();
        })) {
        return false;
    }

    if (std::any_of(inputs.begin(), inputs.end(), [&](const ov::Input<ov::Node>& input) {
            for (size_t i = 0; i < output_shape.size(); ++i) {
                auto input_shape = input.get_partial_shape();
                if (output_shape[i] != input_shape[i])
                    return true;
                if (output_shape[i].is_static() && input_shape[i].is_static())
                    continue;
                if (!ov::symbol::are_equal(output_shape[i].get_symbol(), input_shape[i].get_symbol()))
                    return true;
            }
            return false;
        })) {
        return false;
    }

    return true;
}

} // namespace mlir
} // namespace ov