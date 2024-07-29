// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_common.hpp"

#include <openvino/util/env_util.hpp>


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

bool is_debug() {
    util::getenv_bool("OV_MLIR_DEBUG", false);
}

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

bool elementwise_no_broadcast_predicate(const ov::Output<ov::Node>& output) {
    if (has_dynamic_rank(output.get_node_shared_ptr())) {
        return false;
    }
    // Check if implicit broadcast is possible, reject in this case
    // Relies on symbolic information -- register SymbolicPropagation before applying this pattern
    auto inputs = output.get_node_shared_ptr()->inputs();
    auto output_shape = output.get_partial_shape();

    if (std::any_of(inputs.begin(), inputs.end(), [&](const ov::Input<ov::Node>& input) {
            auto input_shape = input.get_partial_shape();
            if(output_shape.rank().get_length() != input_shape.rank().get_length()) {
                return true;
            }
            for (size_t i = 0; i < output_shape.size(); ++i) {
                if(!are_equal_dimensions(input_shape[i], output_shape[i]))
                    return true;
            }
            return false;
        })) {
        return false;
    }

    return true;
}

bool has_dynamic_rank(NodePtr node) {
    auto inputs = node->inputs();
    auto outputs = node->outputs();
    if (std::any_of(inputs.begin(), inputs.end(), [&](const ov::Input<ov::Node>& input) {
            return input.get_partial_shape().rank().is_dynamic();
        })) {
        return true;
    }
    if (std::any_of(outputs.begin(), outputs.end(), [&](const ov::Output<ov::Node>& output) {
            return output.get_partial_shape().rank().is_dynamic();
        })) {
        return true;
    }
    return false;
}

bool are_equal_dimensions(Dimension d1, Dimension d2) {
    return
        d1.is_static() && d2.is_static() && d1 == d2
        ||
        ov::symbol::are_equal(d1.get_symbol(), d2.get_symbol());
}

bool has_broadcast(Dimension from, Dimension to) {
    return from.is_static() && from.get_length() == 1 && !are_equal_dimensions(from, to);
}

bool statically_broadcastable(const PartialShape& from, const PartialShape& to) {
    if(from.rank().is_dynamic() || to.rank().is_dynamic()) { // FIXME: `from` can has dynamic rank
        return false;
    }

    auto from_rank = from.rank().get_length();
    auto to_rank = to.rank().get_length();

    if(from_rank > to_rank) { // such cases shouldn't be allowed to this function, but kept to make the function generic
        return false;
    }

    auto offset = to_rank - from_rank;
    for(size_t i = 0; i < from_rank; ++i) {
        auto d_from = from[i];
        auto d_to = to[offset + i];
        if(!are_equal_dimensions(d_from, d_to) && !has_broadcast(d_from, d_to)) {
            // cannot deduce neither dimensions broadcast nor dimensions equality
            return false;
        }
    }

    return true;
}

BroadcastDimensions broadcast_dimensions(const PartialShape& src, const PartialShape& dst) {
    assert(statically_broadcastable(src, dst));

    auto src_rank = src.rank().get_length();
    auto dst_rank = dst.rank().get_length();
    auto offset = dst_rank - src_rank;

    BroadcastDimensions result;
    auto& [collapse_groups, dimensions] = result;
    ReassociationIndices group;
    bool group_bonded = false;  // true if `group` has a non-brodcasted dimension

    size_t dst_i = 0;  // dimension index in the `dst` shape
    for(; dst_i < offset; ++dst_i) {
        dimensions.push_back(dst_i);
    }
    for(; dst_i < dst_rank; ++dst_i) {
        auto src_i = dst_i - offset;
        auto src_d = src[src_i];
        auto dst_d = dst[dst_i];
        if(has_broadcast(src_d, dst_d)) {
            dimensions.push_back(dst_i);
        } else {
            if(group_bonded) {
                collapse_groups.emplace_back(group);
                group = ReassociationIndices();
            } else {
                group_bonded = true;
            }
        }
        group.push_back(src_i);
    }

    if(group_bonded && !group.empty()) {
        collapse_groups.emplace_back(group);
    }

    assert(dst_rank - dimensions.size() == collapse_groups.size());

    return result;
}

bool symbol_ancestor_less (SymbolPtr x, SymbolPtr y) {
    return ov::symbol::ancestor_of(x) < ov::symbol::ancestor_of(y);
}

} // namespace mlir
} // namespace ov