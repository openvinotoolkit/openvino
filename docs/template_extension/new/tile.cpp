// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tile.hpp"

using namespace TemplateExtension;

//! [op:ctor]
Tile::Tile(const ov::Output<ov::Node>& arg, std::vector<int64_t> repeats) : Op({arg}), repeats(std::move(repeats)) {
    constructor_validate_and_infer_types();
}
//! [op:ctor]

//! [op:validate]
void Tile::validate_and_infer_types() {
    auto out_shape = get_input_partial_shape(0);
    if (out_shape.rank().is_static()) {
        OPENVINO_ASSERT(out_shape.rank().get_length() == static_cast<int64_t>(repeats.size()),
                        "The number of Tile repeats should be aligned with input rank.");
        for (int64_t i = 0; i < out_shape.rank().get_length(); i++) {
            if (out_shape[i].is_static()) {
                out_shape[i] *= repeats[i];
            }
        }
    }
    // Operation doesn't change shapes end element type
    set_output_type(0, get_input_element_type(0), out_shape);
}
//! [op:validate]

//! [op:copy]
std::shared_ptr<ov::Node> Tile::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OPENVINO_ASSERT(new_args.size() != 1, "Incorrect number of new arguments");

    return std::make_shared<Tile>(new_args.at(0), repeats);
}
//! [op:copy]

//! [op:visit_attributes]
bool Tile::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("repeats", repeats);
    return true;
}
//! [op:visit_attributes]

//! [op:evaluate]
namespace {
/// \brief For each axis calculates the product of inner axes
/// If dims has shape (2, 3, 4) then for 2 (first axis) the inner axes would be (3, 4)
/// and for 3 (second axis) it would be (4)
/// If dims has shape(2, 3, 4) then the output vector would be (3 * 4, 4, 1)
/// The outermost axis is not used. For innermost axis it is always 1.
/// \param[in] dims Shape of the output
///
/// \return Vector containing calculated values for each axis.
std::vector<int64_t> create_pitches(const ov::Shape& dims) {
    std::vector<int64_t> pitch;
    pitch.resize(dims.size() - 1);
    std::partial_sum(dims.rbegin(), dims.rend() - 1, pitch.rbegin(), std::multiplies<int64_t>());
    pitch.push_back(1);
    return pitch;
}

void tile_implementation(const char* arg,
                         char* out,
                         const ov::Shape& in_shape,
                         const ov::Shape& out_shape,
                         const size_t elem_size,
                         const std::vector<int64_t>& repeats) {
    ov::Shape in_shape_expanded(in_shape);
    in_shape_expanded.insert(in_shape_expanded.begin(), out_shape.size() - in_shape.size(), 1);
    size_t block_size = 0;
    int64_t num_repeats = 0;
    const int input_rank = in_shape_expanded.size();
    const int64_t last_dim = in_shape_expanded[input_rank - 1];
    const std::vector<int64_t> pitches = create_pitches(out_shape);
    const char* copy = nullptr;

    std::vector<size_t> indices(in_shape_expanded.size() - 1, 0);
    size_t axis = indices.size();

    // Copy and repeat data for innermost axis as many times as described in the repeats parameter
    while (axis <= indices.size()) {
        block_size = last_dim * elem_size;
        memcpy(out, arg, block_size);
        out += block_size;
        arg += block_size;

        copy = out - block_size;
        num_repeats = repeats[input_rank - 1] - 1;
        for (int64_t i = 0; i < num_repeats; ++i) {
            memcpy(out, copy, block_size);
            out += block_size;
        }

        // Copy and repeat data for other axes as many times as described in the repeats parameter
        while (axis-- != 0) {
            if (++indices[axis] != in_shape_expanded[axis]) {
                axis = indices.size();
                break;
            }
            indices[axis] = 0;

            ptrdiff_t pitch = pitches[axis] * in_shape_expanded[axis];
            block_size = pitch * elem_size;
            copy = out - block_size;
            num_repeats = repeats[axis] - 1;
            for (int64_t i = 0; i < num_repeats; i++) {
                memcpy(out, copy, block_size);
                out += block_size;
            }
        }
    }
}

bool evaluate_tile(const ov::runtime::Tensor& arg0, ov::runtime::Tensor& out, std::vector<int64_t> repeats_val) {
    const auto in_shape = arg0.get_shape();
    OPENVINO_ASSERT(in_shape.size() == repeats_val.size());
    ov::Shape output_shape(in_shape.size());
    for (size_t i = 0; i < in_shape.size(); i++) {
        output_shape[i] = in_shape[i] * repeats_val[i];
    }
    out.set_shape(output_shape);
    tile_implementation(reinterpret_cast<const char*>(arg0.data()),
                        reinterpret_cast<char*>(out.data()),
                        arg0.get_shape(),
                        output_shape,
                        arg0.get_element_type().size(),
                        repeats_val);
    return true;
}

}  // namespace

bool Tile::evaluate(ov::runtime::TensorVector& outputs, const ov::runtime::TensorVector& inputs) const {
    // Doesn't support LP data types
    switch (inputs[0].get_element_type()) {
    case ov::element::Type_t::i8:
    case ov::element::Type_t::i16:
    case ov::element::Type_t::i32:
    case ov::element::Type_t::i64:
    case ov::element::Type_t::u8:
    case ov::element::Type_t::u16:
    case ov::element::Type_t::u32:
    case ov::element::Type_t::u64:
    case ov::element::Type_t::bf16:
    case ov::element::Type_t::f16:
    case ov::element::Type_t::f32:
        return evaluate_tile(inputs[0], outputs[0], get_repeats());
    default:
        break;
    }
    return false;
}

bool Tile::has_evaluate() const {
    // Doesn't support LP data types
    switch (get_input_element_type(0)) {
    case ov::element::Type_t::i8:
    case ov::element::Type_t::i16:
    case ov::element::Type_t::i32:
    case ov::element::Type_t::i64:
    case ov::element::Type_t::u8:
    case ov::element::Type_t::u16:
    case ov::element::Type_t::u32:
    case ov::element::Type_t::u64:
    case ov::element::Type_t::bf16:
    case ov::element::Type_t::f16:
    case ov::element::Type_t::f32:
        return true;
    default:
        break;
    }
    return false;
}
//! [op:evaluate]
