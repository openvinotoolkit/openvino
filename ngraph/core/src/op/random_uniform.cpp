// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <ngraph/validation_util.hpp>

#include "itt.hpp"
#include "ngraph/op/random_uniform.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::v8::RandomUniform, "RandomUniform", 8);

op::v8::RandomUniform::RandomUniform(const Output<Node>& out_shape, ngraph::element::Type output_type, int64_t seed, int64_t seed2)
        : Op({out_shape}), m_output_type(output_type), m_seed(seed), m_seed2(seed2)
{
    constructor_validate_and_infer_types();
}

void op::v8::RandomUniform::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v8_RandomUniform_validate_and_infer_types);

    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(0).is_integral_number(),
                          "Output shape must be an integral number.");
    PartialShape output_shape = PartialShape::dynamic();
    const auto& input_shape = get_input_partial_shape(0);
    if (input_shape.rank().is_static())
    {
        NGRAPH_CHECK(input_shape.rank() == 1, "The rank of the tensor defining output shape must be equal to 1");
        if (const auto& const_shape = get_constant_from_source(input_value(0)))
        {
            output_shape = PartialShape(const_shape->cast_vector<int64_t>());
        }
    }

    set_output_type(0, get_out_type(), output_shape);
}

bool op::v8::RandomUniform::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v8_RandomUniform_visit_attributes);
    visitor.on_attribute("output_type", m_output_type);
    visitor.on_attribute("seed", m_seed);
    visitor.on_attribute("seed2", m_seed2);
    return true;
}

shared_ptr<Node> op::v8::RandomUniform::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v8_Roll_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<v8::RandomUniform>(new_args[0], m_output_type, m_seed, m_seed2);
}

std::pair<uint32_t, uint32_t> split_high_low(uint64_t value)
{
    uint32_t low = static_cast<uint32_t>(value);
    uint32_t high = static_cast<uint32_t>(value >> 32);
    return {low, high};
}

uint64_t unite_high_low(uint32_t high, uint32_t low)
{
    return (static_cast<uint64_t>(high) << 32) + low;
}

void calculate_round(const uint64_t& key, uint64_t& counter, uint64_t& n)
{
    auto counter_lr = split_high_low(counter);
    auto key_lr = split_high_low(key);
    auto n_lr = split_high_low(n);
    auto prod0 = split_high_low(static_cast<uint64_t>(0xD2511F53) * n_lr.first);
    auto prod1 = split_high_low(static_cast<uint64_t>(0xCD9E8D57) * counter_lr.first);

    n_lr.first = prod1.second ^ n_lr.second ^ key_lr.first;
    n_lr.second = prod1.first;
    counter_lr.first = prod0.second ^ counter_lr.second ^ key_lr.second;
    counter_lr.second = prod0.first;

    counter = unite_high_low(counter_lr.second, counter_lr.first);
    n = unite_high_low(n_lr.second, n_lr.first);
}

void raise_key(uint64_t& key)
{
    auto key_lr = split_high_low(key);
    key_lr.first += 0x9E3779B9;
    key_lr.second += 0xBB67AE85;
    key = unite_high_low(key_lr.second, key_lr.first);
}

float Uint32ToFloat(uint32_t x) {
    const uint32_t man = x & 0x7fffffu;  // 23 bit mantissa
    const uint32_t exp = static_cast<uint32_t>(127);
    const uint32_t val = (exp << 23) | man;

    // Assumes that endian-ness is same for float and uint32.
    float result;
    memcpy(&result, &val, sizeof(val));
    return result - 1.0f;
}

float16 Uint16ToHalf(uint16_t x) {
// IEEE754 halfs are formatted as follows (MSB first):
//    sign(1) exponent(5) mantissa(10)
// Conceptually construct the following:
//    sign == 0
//    exponent == 15  -- an excess 15 representation of a zero exponent
//    mantissa == 10 random bits
const uint16_t man = x & 0x3ffu;  // 10 bit mantissa
const uint16_t exp = static_cast<uint16_t>(15);
const uint16_t val = (exp << 10) | man;

float16 result;
memcpy(&result, &val, sizeof(val));
return result - float16(1.0);
}

void run_philox(uint64_t key, uint64_t counter, uint64_t n, size_t n_rounds, uint32_t* res)
{
    for (int i = 0; i < n_rounds; i++) {
        calculate_round(key, counter, n);
        if (i < n_rounds-1)
            raise_key(key);
    }
    auto res1 = split_high_low(n);
    auto res2 = split_high_low(counter);
    res[0] = res1.first;
    res[1] = res1.second;
    res[2] = res2.first;
    res[3] = res2.second;
}

void random_uniform(const uint64_t* out_shape,
                    char* out,
                    const Shape& out_shape_shape,
                    ngraph::element::Type elem_type,
                    uint64_t seed,
                    uint64_t seed2)
{
    if (seed == 0 && seed2 == 0)
    {
        std::srand(std::time(nullptr));
        seed = std::rand();
    }
    uint64_t key = seed;
    uint64_t counter = seed2;
    uint64_t n = 0;
    size_t shape_count = shape_size(out_shape_shape);
    size_t elem_count = 1;
    for (size_t i = 0; i < shape_count; i++) {
        elem_count *= out_shape[i];
    }
    size_t step = elem_type.size() > 4 ? 2 : 4;

    for (size_t k = 0; k < elem_count; k+=step) {
        uint32_t res[4];
        run_philox(key, counter, n, 10, res);

        if (elem_type == ngraph::element::Type_t::f32) {
            float res_float[4];
            res_float[0] = Uint32ToFloat(res[0]);
            res_float[1] = Uint32ToFloat(res[1]);
            res_float[2] = Uint32ToFloat(res[2]);
            res_float[3] = Uint32ToFloat(res[3]);

            memcpy(out + k * elem_type.size(), res_float, std::min((size_t) 4, elem_count - k) * elem_type.size());
        } else
        {
            float16 res_float[4];
            res_float[0] = Uint16ToHalf(res[0]);
            res_float[1] = Uint16ToHalf(res[1]);
            res_float[2] = Uint16ToHalf(res[2]);
            res_float[3] = Uint16ToHalf(res[3]);

            memcpy(out + k * elem_type.size(), res_float, std::min((size_t) 4, elem_count - k) * elem_type.size());
        }
        if (++n == 0)
            ++counter;
    }
}

bool op::v8::RandomUniform::evaluate(const HostTensorVector& outputs,
              const HostTensorVector& inputs) const
{
    element::Type_t t_out = get_out_type();
        char* out;
        switch (t_out) {
            case element::Type_t::i32:
                out = (char *) outputs[0]->get_data_ptr<const int32_t>();
                break;
            case element::Type_t::i64:
                out = (char *) outputs[0]->get_data_ptr<const int64_t>();
                break;
            case element::Type_t::f16:
                out = (char *) outputs[0]->get_data_ptr<const float16>();
                break;
            case element::Type_t::f32:
                out = (char *) outputs[0]->get_data_ptr<const float>();
                break;
            case element::Type_t::f64:
                out = (char *) outputs[0]->get_data_ptr<const double >();
                break;
            default:
                out = nullptr;
        }

        uint64_t* out_shape;
        std::vector<uint64_t> out_shape_uint64;
        out_shape_uint64.resize(shape_size(inputs[0]->get_shape()));

        if (inputs[0]->get_element_type() == element::Type_t::u64)
        {
            out_shape = (uint64_t*)inputs[0]->get_data_ptr<const char>();
        }
        else if (inputs[0]->get_element_type() == element::Type_t::i32)
        {
            auto out_shape_i32 = inputs[0]->get_data_ptr<const int32_t>();
            std::transform(out_shape_i32,
                           out_shape_i32 + shape_size(inputs[0]->get_shape()),
                           out_shape_uint64.begin(),
                           [](const int32_t& elem) { return static_cast<uint64_t>(elem); });
            out_shape = out_shape_uint64.data();
        }
        else if (inputs[0]->get_element_type() == element::Type_t::i64)
        {
            auto out_shape_i64 = inputs[0]->get_data_ptr<const int64_t>();
            std::transform(out_shape_i64,
                           out_shape_i64 + shape_size(inputs[0]->get_shape()),
                           out_shape_uint64.begin(),
                           [](const int64_t& elem) { return static_cast<uint64_t>(elem); });
            out_shape = out_shape_uint64.data();
        }

        random_uniform(out_shape,
                       out,
                       inputs[0]->get_shape(),
                                           get_out_type(),
                                           get_seed(),
                                           get_seed2());
        return true;
}