// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <ngraph/validation_util.hpp>

#include "itt.hpp"
#include "ngraph/op/random_uniform.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::v8::RandomUniform, "RandomUniform", 8);

op::v8::RandomUniform::RandomUniform(const Output<Node>& out_shape, const Output<Node>& min_val, const Output<Node>& max_val, ngraph::element::Type output_type, int64_t seed, int64_t seed2)
        : Op({out_shape, min_val, max_val}), m_output_type(output_type), m_seed(seed), m_seed2(seed2)
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

    NODE_VALIDATION_CHECK(
            this, get_input_partial_shape(1).compatible(Shape{}), "'min_val' input is not a scalar");
    NODE_VALIDATION_CHECK(
            this, get_input_partial_shape(2).compatible(Shape{}), "'max_val' input is not a scalar");

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
    return make_shared<v8::RandomUniform>(new_args[0], new_args[1], new_args[2], m_output_type, m_seed, m_seed2);
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

float uint32_to_float(uint32_t x) {
    uint32_t x_uint32 = (static_cast<uint32_t>(127) << 23) | x & 0x7fffffu;

    float x_float;
    memcpy(&x_float, &x_uint32, sizeof(x_uint32));
    return x_float - 1.0f;
}

float16 uint32_to_float16(uint32_t x) {
    uint16_t x_uint16 = static_cast<uint16_t>(x);
    x_uint16 = (static_cast<uint16_t>(15) << 10) | x_uint16 & 0x3ffu;

    float16 x_float16;
    memcpy(&x_float16, &x_uint16, sizeof(x_uint16));
    return x_float16 - static_cast<float16>(1);
}

bfloat16 uint32_to_bfloat16(uint32_t x) {
    uint16_t x_uint16 = static_cast<uint16_t>(x);
    x_uint16 = (static_cast<uint16_t>(127) << 7) | x_uint16 & 0x7fu;

    bfloat16 x_bfloat16;
    memcpy(&x_bfloat16, &x_uint16, sizeof(x_uint16));
    return x_bfloat16 - static_cast<bfloat16>(1);
}

double uint32_to_double(uint32_t x1, uint32_t x2) {
    uint64_t mantissa = ((static_cast<uint64_t>(x1) & 0xfffffu) << 32) | static_cast<uint64_t>(x2);
    uint64_t x_uint64 = ((static_cast<uint64_t>(1023) << 52) | mantissa);

    double x_double;
    memcpy(&x_double, &x_uint64, sizeof(x_uint64));
    return x_double - 1.0;
}

uint64_t uint32_to_uint64(uint32_t x1, uint32_t x2) {
    return (static_cast<uint64_t>(x2) << 32) | static_cast<uint64_t>(x1);
}

// Helper function to convert an 16-bit integer to a bfloat16 between [0..1).
// This can create a uniform distribution of values between [0..1).
bfloat16 Uint16ToGfloat16(uint16_t x) {
// bfloat are formatted as follows (MSB first):
//    sign(1) exponent(8) mantissa(7)
// Conceptually construct the following:
//    sign == 0
//    exponent == 127  -- an excess 127 representation of a zero exponent
//    mantissa == 7 random bits
    uint16_t man = x & 0x7fu;  // 7 bit mantissa
    uint16_t exp = static_cast<uint16_t>(127);
    uint16_t val = (exp << 7) | man;

bfloat16 result;
memcpy(&result, &val, sizeof(val));
// The mantissa has an implicit leading 1, so the above code creates a value
// in [1, 2). The minus will not cause a rounding that makes the result 1.
// Instead it will just be close to 1.
return result - bfloat16(1.0);
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
                    const char* min_val,
                    const char* max_val,
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

        switch (elem_type) {
            case ngraph::element::Type_t::f32: {
                float res_float[4];
                float mn[1];
                float mx[1];
                memcpy(mn, min_val, elem_type.size());
                memcpy(mx, max_val, elem_type.size());
                std::transform(res,
                               res + 4,
                               res_float,
                               [&mn, &mx](const uint32_t& elem) { return uint32_to_float(elem) * (mx[0] - mn[0]) + mn[0]; });

                memcpy(out + k * elem_type.size(), res_float, std::min((size_t) 4, elem_count - k) * elem_type.size());
                break;
            }
            case ngraph::element::Type_t::f16: {
                float16 res_float16[4];
                float16 mn[1];
                float16 mx[1];
                memcpy(mn, min_val, elem_type.size());
                memcpy(mx, max_val, elem_type.size());
                std::transform(res,
                               res + 4,
                               res_float16,
                               [&mn, &mx](const uint32_t& elem) { return uint32_to_float16(elem) * (mx[0] - mn[0]) + mn[0]; });
                memcpy(out + k * elem_type.size(), res_float16,
                       std::min((size_t) 4, elem_count - k) * elem_type.size());
                break;
            }
            case ngraph::element::Type_t::bf16: {
                bfloat16 res_bfloat16[4];
                bfloat16 mn[1];
                bfloat16 mx[1];
                memcpy(mn, min_val, elem_type.size());
                memcpy(mx, max_val, elem_type.size());
//                bfloat16 range = mx[0] - mn[0];
//                bfloat16 val3 =Uint16ToGfloat16(res[3]);
//                float v1 = static_cast<float>(val3);
//                float v2 = static_cast<float>(range);
//                float mul_f = v1 * v2;
//                bfloat16 =
//
//                uint32_t input = {mul_f};
//                uint32_t lsb = (input >> 16) & 1;
//                uint32_t rounding_bias = 0x7fff + lsb;
//                input += rounding_bias;
//                output.value = static_cast<uint16_t>(input >> 16);
//
//                bfloat16 mul_bf = {mul_f};
//
//                bfloat16 mul = val3 * range;


                std::transform(res,
                               res + 4,
                               res_bfloat16,
                               [&mn, &mx](const uint32_t& elem) { return Uint16ToGfloat16(elem) * (mx[0] - mn[0]) + mn[0]; });
                memcpy(out + k * elem_type.size(), res_bfloat16,
                       std::min((size_t) 4, elem_count - k) * elem_type.size());
                break;
            }
            case ngraph::element::Type_t::f64: {
                double res_double[2];
                double mn[1];
                double mx[1];
                memcpy(mn, min_val, elem_type.size());
                memcpy(mx, max_val, elem_type.size());
                res_double[0] = uint32_to_double(res[0], res[1]) * (mx[0] - mn[0]) + mn[0];
                res_double[1] = uint32_to_double(res[2], res[3]) * (mx[0] - mn[0]) + mn[0];
                memcpy(out + k * elem_type.size(), res_double,
                       std::min((size_t) 2, elem_count - k) * elem_type.size());
                break;
            }
            case ngraph::element::Type_t::i32: {
                int res_int[4];
                int mn[1];
                int mx[1];
                memcpy(mn, min_val, elem_type.size());
                memcpy(mx, max_val, elem_type.size());
                std::transform(res,
                               res + 4,
                               res_int,
                               [&mn, &mx](const uint32_t& elem) { return elem % (mx[0] - mn[0]) + mn[0]; });
                memcpy(out + k * elem_type.size(), res_int, std::min((size_t) 4, elem_count - k) * elem_type.size());
                break;
            }
            case ngraph::element::Type_t::i64: {
                int64_t res_int64[2];
                int64_t mn[1];
                int64_t mx[1];
                memcpy(mn, min_val, elem_type.size());
                memcpy(mx, max_val, elem_type.size());
                res_int64[0] = uint32_to_uint64(res[0], res[1]) % (mx[0] - mn[0]) + mn[0];
                res_int64[1] = uint32_to_uint64(res[2], res[3]) % (mx[0] - mn[0]) + mn[0];
                memcpy(out + k * elem_type.size(), res_int64,
                       std::min((size_t) 2, elem_count - k) * elem_type.size());
                break;
            }
            default:
                throw ngraph_error("Unsupported type of RandomUniform: " + elem_type.get_type_name());

        }

        if (++n == 0)
            ++counter;
    }
}

bool op::v8::RandomUniform::evaluate(const HostTensorVector& outputs,
              const HostTensorVector& inputs) const
{
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
        case element::Type_t::bf16:
            out = (char *) outputs[0]->get_data_ptr<const bfloat16>();
            break;
        case element::Type_t::f32:
            out = (char *) outputs[0]->get_data_ptr<const float>();
            break;
        case element::Type_t::f64:
            out = (char *) outputs[0]->get_data_ptr<const double >();
            break;
        default:
            throw ngraph_error("Unsupported type of RandomUniform: " + get_out_type().get_type_name());
    }

    random_uniform(out_shape,
                   inputs[1]->get_data_ptr<const char>(),
                   inputs[2]->get_data_ptr<const char>(),
                   out,
                   inputs[0]->get_shape(),
                                       get_out_type(),
                                       get_seed(),
                                       get_seed2());
    return true;
}