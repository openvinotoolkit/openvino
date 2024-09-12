// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"
#include <vector>

namespace cldnn {

/// @brief Select mode for the @ref eltwise layer.
enum class eltwise_mode : int32_t {
    /// @brief Eltwise sum.
    sum,
    /// @brief Eltwise subtract.
    sub,
    /// @brief Eltwise max.
    max,
    /// @brief Eltwise product (Hadamard).
    prod,
    /// @brief Eltwise div.
    div,
    /// @brief Eltwise min.
    min,
    /// @brief Eltwise pow.
    pow,
    /// @brief Eltwise squared diff.
    squared_diff,
    /// @brief Eltwise mod.
    mod,
    /// @brief Eltwise equal.
    eq,
    /// @brief Eltwise not equal.
    ne,
    /// @brief Eltwise less.
    lt,
    /// @brief Eltwise less of equal.
    le,
    /// @brief Eltwise greater.
    gt,
    /// @brief Eltwise greater or equal.
    ge,
    /// @brief Eltwise and.
    logic_and,
    /// @brief Eltwise or.
    logic_or,
    /// @brief Eltwise XOR.
    logic_xor,
    /// @brief Eltwise floormod.
    floor_mod,
    /// @brief Eltwise is finite.
    is_finite,
    /// @brief Eltwise is infinite.
    is_inf,
    /// @brief Eltwise is nan.
    is_nan,
    /// @brief Eltwise bitwise right shift.
    right_shift,
    /// @brief Eltwise bitwise left shift.
    left_shift,
    /// @brief Eltwise bitwise and.
    bitwise_and,
    /// @brief Eltwise bitwise or.
    bitwise_or,
    /// @brief Eltwise bitwise xor.
    bitwise_xor
};

/// @brief Performs elementwise operations (sum, subtract, max or product) on two input primitives
/// Also supports built-in Relu @ref activation available by setting it in arguments.
/// @notes
/// - both inputs have to have equal sizes in all dimensions or the input tensors are broadcastable
///   to the same shape in which the size of each dimention is a max. of input sizes on this dimension)
/// - format of both inputs has to be the same
/// - when using integer types, only following eltwise modes are supported: sum, sub, prod, div
struct eltwise : public primitive_base<eltwise> {
    CLDNN_DECLARE_PRIMITIVE(eltwise)

    // Logic and comparison operations should return i8 for any inputs
    static const std::set<eltwise_mode> eltwise_bool_modes;

    eltwise() : primitive_base("", {}) {}

    /// @brief Constructs eltwise primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param input2 Second input primitive id with values needed for eltwise computation.
    /// @param mode Eltwise mode.
    /// @param spec Auto broadcast rule specificiation.
    eltwise(const primitive_id& id,
            const input_info& input,
            const input_info& input2,
            eltwise_mode mode,
            const ov::op::AutoBroadcastSpec& spec = ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::NUMPY))
        : primitive_base(id, {input, input2}),
          mode(mode),
          coefficients(std::vector<float>(0)),
          stride(std::vector<tensor>(0)),
          broadcast_spec(spec.m_type, spec.m_axis),
          m_pythondiv(true) { }

    /// @brief Constructs eltwise primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param input2 Second input primitive id with values needed for eltwise computation.
    /// @param stride Defines shift in input buffers between adjacent calculations of output values.
    /// @param mode Eltwise mode.
    /// @param spec Auto broadcast rule specificiation.
    eltwise(const primitive_id& id,
            const input_info& input,
            const input_info& input2,
            std::vector<tensor> stride,
            eltwise_mode mode,
            const ov::op::AutoBroadcastSpec& spec = ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::NUMPY))
        : primitive_base(id, {input, input2}),
          mode(mode),
          coefficients(std::vector<float>(0)),
          stride(stride),
          broadcast_spec(spec.m_type, spec.m_axis),
          m_pythondiv(true) { }

    /// @brief Constructs eltwise primitive.
    /// @param id This primitive id.
    /// @param inputs Input primitives ids.
    /// @param mode Eltwise mode.
    /// @param data_type Expected output data type.
    /// @param spec Auto broadcast rule specificiation.
    eltwise(const primitive_id& id,
            const std::vector<input_info>& inputs,
            eltwise_mode mode,
            data_types data_type,
            const ov::op::AutoBroadcastSpec& spec = ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::NUMPY))
        : primitive_base(id, inputs, 1, {optional_data_type{data_type}}),
          mode(mode),
          coefficients(std::vector<float>(0)),
          stride(std::vector<tensor>(0)),
          broadcast_spec(spec.m_type, spec.m_axis),
          m_pythondiv(true) { }

    /// @brief Constructs eltwise primitive.
    /// @param id This primitive id.
    /// @param inputs Input primitives ids.
    /// @param mode Eltwise mode.
    /// @param spec Auto broadcast rule specificiation.
    eltwise(const primitive_id& id,
            const std::vector<input_info>& inputs,
            eltwise_mode mode,
            const ov::op::AutoBroadcastSpec& spec = ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::NUMPY))
        : primitive_base(id, inputs),
          mode(mode),
          coefficients(std::vector<float>(0)),
          stride(std::vector<tensor>(0)),
          broadcast_spec(spec.m_type, spec.m_axis),
          m_pythondiv(true) { }

    /// @brief Constructs eltwise primitive.
    /// @param id This primitive id.
    /// @param inputs Input primitives ids.
    /// @param mode Eltwise mode.
    /// @param coefficients Blob-wise coefficient.
    /// @param data_type Expected output data type.
    /// @param spec Auto broadcast rule specificiation.
    /// @param m_pythondiv Specifies if floor division should be calculate. Supported only for integer data types.
    eltwise(const primitive_id& id,
            const std::vector<input_info>& inputs,
            eltwise_mode mode,
            std::vector<float> coeffs,
            data_types data_type,
            const ov::op::AutoBroadcastSpec& spec = ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::NUMPY),
            bool m_pythondiv = true)
        : primitive_base(id, inputs, 1, {optional_data_type{data_type}}),
          mode(mode),
          coefficients(std::move(coeffs)),
          stride(std::vector<tensor>(0)),
          broadcast_spec(spec.m_type, spec.m_axis),
          m_pythondiv(m_pythondiv) {
        if (mode == eltwise_mode::sum && !coefficients.empty() && coefficients.size() != inputs.size()) {
            throw std::invalid_argument("Invalid eltwise sum coefficients count (should be equal to 0 or input.size)");
        }
    }

    /// @param mode Eltwise mode.
    eltwise_mode mode = eltwise_mode::sum;
    /// @param coefficients Blob-wise coefficient.
    std::vector<float> coefficients;
    /// @brief Defines shift in input buffers between adjacent calculations of output values.
    std::vector<tensor> stride;
    /// @brief Define auto broadcast rule specification.
    ov::op::AutoBroadcastSpec broadcast_spec;
    /// @brief Define m_pythondiv.
    bool m_pythondiv = true;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = cldnn::hash_combine(seed, mode);
        seed = cldnn::hash_range(seed, coefficients.begin(), coefficients.end());
        for (auto& s : stride) {
            seed = cldnn::hash_combine(seed, s.hash());
        }
        seed = cldnn::hash_combine(seed, m_pythondiv);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const eltwise>(rhs);

        return mode == rhs_casted.mode &&
               coefficients == rhs_casted.coefficients &&
               broadcast_spec == rhs_casted.broadcast_spec &&
               stride == rhs_casted.stride &&
               m_pythondiv == rhs_casted.m_pythondiv;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<eltwise>::save(ob);
        ob << make_data(&mode, sizeof(eltwise_mode));
        ob << coefficients;
        ob << stride;
        ob << make_data(&broadcast_spec, sizeof(ov::op::AutoBroadcastSpec));
        ob << m_pythondiv;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<eltwise>::load(ib);
        ib >> make_data(&mode, sizeof(eltwise_mode));
        ib >> coefficients;
        ib >> stride;
        ib >> make_data(&broadcast_spec, sizeof(ov::op::AutoBroadcastSpec));
        ib >> m_pythondiv;
    }
};
}  // namespace cldnn
