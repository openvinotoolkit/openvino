// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/convert_fp8.hpp"

#include "ngraph/op/equal.hpp"
#include "ngraph/op/select.hpp"
#include "ngraph/runtime/reference/convert.hpp"

using namespace ngraph;

BWDCMP_RTTI_DEFINITION(op::v1::ConvertFP8);

op::v1::ConvertFP8::ConvertFP8() : Op(), m_destination_type("hf8_ext"), m_apply_scale(false) {}

//! [op:ctor]
op::v1::ConvertFP8::ConvertFP8(const ov::Output<ov::Node>& arg,
                               const ov::Output<ov::Node>& scale,
                               const std::string& destination_type,
                               bool apply_scale)
    : Op({arg, scale}),
      m_destination_type(destination_type),
      m_apply_scale(apply_scale) {
    validate();
    constructor_validate_and_infer_types();
}
//! [op:ctor]

const std::vector<std::string> op::v1::ConvertFP8::m_valid_types({"hf8", "hf8_ext", "bf8", "hf8_libxsmm"});

//! [op:validate]
void op::v1::ConvertFP8::validate_and_infer_types() {
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}
//! [op:validate]

//! [op:copy]
std::shared_ptr<ov::Node> op::v1::ConvertFP8::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OPENVINO_ASSERT(new_args.size() == 2, "Incorrect number of new arguments");

    return std::make_shared<ConvertFP8>(new_args.at(0), new_args.at(1), m_destination_type, m_apply_scale);
}
//! [op:copy]

//! [op:visit_attributes]
bool op::v1::ConvertFP8::visit_attributes(ov::AttributeVisitor& visitor) {
    validate();
    visitor.on_attribute("destination_type", m_destination_type);
    visitor.on_attribute("apply_scale", m_apply_scale);

    return true;
}
//! [op:visit_attributes]

void op::v1::ConvertFP8::validate() const {
    OPENVINO_ASSERT(std::find(m_valid_types.begin(), m_valid_types.end(), m_destination_type) != m_valid_types.end(),
                    "Bad format for f8 conversion type: " + m_destination_type);
}

bool op::v1::ConvertFP8::has_evaluate() const {
    return true;
}

namespace convert_fp8 {
namespace {
void print_tensor(const ov::Tensor& t, std::string s) {
    std::cout << "Tensor " << s << ": ";
    auto shape = t.get_shape();
    int len = shape_size(shape);

    if (t.get_element_type() == ov::element::f16) {
        auto ptr = static_cast<ov::float16*>(t.data());
        for (int i = 0; i < len; i++) {
            std::cout << ptr[i] << " ";
        }
    }
    if (t.get_element_type() == ov::element::f32) {
        auto ptr = static_cast<float*>(t.data());
        for (int i = 0; i < len; i++) {
            std::cout << ptr[i] << " ";
        }
    }
    std::cout << std::endl;
}
/// <summary>
/// emulation of convertation fp16 value to bf8 1s-5e-2m format, Brain Float
/// </summary>
/// <typeparam name="T">Every possible type with 16 bit size</typeparam>
/// <param name="arg"></param>
/// <param name="out"></param>
/// <param name="count"></param>
template <typename T>
void convertfp16_bf8(const T* const arg, T* out, size_t count, int exp_bits = 5, int mbits = 8) {
    typedef union half_t {
        unsigned short u;
        T f;
    } __half_t;

    int non_mant_bits = exp_bits + 1;           /* exponent + sign */
    int lshift = 10 - (mbits - non_mant_bits);  // 10 - (8 - 6) == 8 ???

    unsigned short mask_mant = (unsigned short)(0xffff << lshift);  // 1111111111111111 -> 1 11111 1100000000
    unsigned short grs_bitmask = 0x00ff;                            // 0 00000 0011111111 - guard, round, sticky bits
    unsigned short rne_tie = 0x0180;                                // 0 00000 0110000000

    __half_t h;
    for (size_t i = 0; i < count; ++i) {
        /// converts float number to half precision in round-to-nearest-even mode and returns half with converted value.
        h.f = arg[i];
        unsigned short is_normal = 1;
        /// 0x7c00 = 0111110000000000 - exponent mask
        /// s 11111 xxx xxxx xxxx - is nan (if some x is 1) or inf (if all x is 0)
        /// 0x7800 is 0111100000000000 and 0x400 is 0000010000000000
        /// number is not normal if all exponent is 1 or 0
        is_normal = (((h.u & 0x7c00) <= 0x7800) && ((h.u & 0x7c00) >= 0x0400)) ? 1 : 0;
        /// 0x7f00 is 0 11111 1100000000
        /// 0x7b00 is 0 11110 1100000000
        unsigned short can_round = ((h.u & 0x7f00) < 0x7b00) ? 1 : 0;
        /// s 11111 xxx xxxx xxxx - is nan (if some x is 1) or inf (if all x is 0)
        unsigned short is_naninf = ((h.u & 0x7c00) == 0x7c00) ? 1 : 0;
        /* nearest rounding masks */
        /// grs_bitmask - grs_bitmask is 0 00000 0011111111 or 0 00000 00grs11111
        unsigned short rnmask = (h.u & grs_bitmask);
        /// rne_tie - 0x180 is      0 00000 0110000000 or 384.0 ???
        unsigned short rnmask_tie = (h.u & rne_tie);

        if (!is_naninf && can_round) {
            /* round to nearest even, if rne_mask is enabled */
            /* 0 00000 0010000000, find grs patterns */
            // 0xx - do nothing
            // 100 - this is a tie : round up if the mantissa's bit just before G is 1, else do nothing
            // 101, 110, 111 - round up > 0x0080
            h.u += (((rnmask > 0x0080) || (rnmask_tie == rne_tie)) << lshift);
        }
        h.u = (h.u & mask_mant); /* truncation */
        out[i] = h.f;
    }
}

/// <summary>
/// emulation of convertation fp16 value to hf8 1s-4e-3m format, Hybrid Float
/// </summary>
/// <typeparam name="T">Every possible type with 16 bit size</typeparam>
/// <param name="arg"></param>
/// <param name="out"></param>
/// <param name="count"></param>
// Exponent denormal values 0 -11
// Exponent normal values 1..14 -10..3 (11 - exponent)
// Exponent NaN values 15 4
template <typename T>
void convertfp16_hf8(const T* arg, T* out, size_t count, int exp_bits = 5, int mbits = 9, bool use_clamp = true) {
    typedef union half_t {
        unsigned short u;
        T f;
    } __half_t;

    int non_mant_bits = exp_bits + 1; /* exponent + sign */         ///  6 - ?
    int lshift = 10 - (mbits - non_mant_bits);                      /// 10 - (9 - 6) == 7 - ???
    unsigned short rne_mask = 1;                                    /* round to nearest even mask */
    unsigned short mask_mant = (unsigned short)(0xFFFF << lshift);  // 1111111111111111 -> 1 11111 1111000000
    unsigned short grs_bitmask = 0x007F;                            /// 0 00000 0001111111
    unsigned short rne_tie = 0x00C0;                                /// 0 00000 0011000000

    __half_t h;
    for (size_t i = 0; i < count; ++i) {
        h.f = arg[i];
        float inval = ngraph::float16(arg[i]);
        /* flush values below 1-4-3 (offset=4) subnormal range to zero */
        if (fabs(inval) < 1.2207031e-4)
            h.f = 0;

        short exp_h =
            (short)((h.u & 0x7C00) >> 10) - 15;  /// 0111110000000000 -> 0000000000011111 - 15 - biased exponent
        short sign_h = (h.u & 0x8000);           /// 1 00000 0000000000
        short mantissa_h = (h.u & 0x03FF);       /// 0 00000 1111111111
        ///(h.u && 0111111111111111) < 0 10010 1110000000 (19326) - ????
        unsigned short can_round = ((h.u & 0x7FFF) < 0x4B80) ? 1 : 0;
        unsigned short is_normal = 1;

        is_normal = (((h.u & 0x7C00) <= 0x7800) && ((h.u & 0x7C00) >= 0x0400)) ? 1 : 0;
        unsigned short is_naninf = ((h.u & 0x7C00) == 0x7C00) ? 1 : 0;

        int dshift = 0;
        if (exp_h > 3) {  // too large, set it to NaN or inf
            if (use_clamp) {
                exp_h = 3;
                mantissa_h = 0b0000001110000000;
            } else {
                mantissa_h = 0;
                exp_h = 16;
                is_naninf = 1;
            }
        } else if (exp_h < -13) {  /// -13, -12, -11 for rounding
            /* flush values below 1-4-3 (offset=4) subnormal range to zero */
            exp_h = -15;
            mantissa_h = 0;
        }
        /* nearest rounding masks, & 0 00000 000 111 1111 - mantissa bits below hf8 (grs) */
        unsigned short rnmask = (mantissa_h & grs_bitmask);
        /* & 0 00000 0011000000 - edge between hf8 and fp16 mantissa */
        unsigned short rnmask_tie = (mantissa_h & rne_tie);
        if (!is_naninf && can_round && rne_mask) {
            /* round to nearest even, if rne_mask is enabled */
            /// rnmask > 0 00000 0001000000(64) or 0 00000 0011000000 - edge bits is 1
            /// += 0 00000 0010000000
            mantissa_h += (((rnmask > 0x0040) || (rnmask_tie == rne_tie)) << lshift);
        }
        if (exp_h < -10) { /* handle denormals -13, -12, -11, dshift 1, 2, 3 */
            dshift = (-10 - exp_h);
            mantissa_h = mantissa_h >> dshift;
        }
        mantissa_h &= mask_mant; /* truncation */
        mantissa_h <<= dshift;
        mantissa_h += ((exp_h + 15) << 10);
        h.u = mantissa_h | sign_h;
        out[i] = h.f;
    }
}

/// <summary>
/// emulation of convertation fp16 value to hf8 1s-4e-3m format, Extended Hybrid Float
/// </summary>
/// <typeparam name="T">Every possible type with 16 bit size</typeparam>
/// <param name="arg"></param>
/// <param name="out"></param>
/// <param name="count"></param>
// Exponent denormal values 0 -11
// Exponent normal values 1..14 -10..3 (11 - exponent)
// Exponent NaN values 15 4
template <typename T>
void convertfp16_hf8_ext(const T* arg, T* out, size_t count, int exp_bits = 5, int mbits = 9, bool use_clamp = true) {
    typedef union half_t {
        unsigned short u;
        T f;
    } __half_t;

    int non_mant_bits = exp_bits + 1; /* exponent + sign */         ///  6 - ?
    int lshift = 10 - (mbits - non_mant_bits);                      /// 10 - (9 - 6) == 7 - ???
    unsigned short rne_mask = 1;                                    /* round to nearest even mask */
    unsigned short mask_mant = (unsigned short)(0xFFFF << lshift);  // 1111111111111111 -> 1 11111 1111000000
    unsigned short grs_bitmask = 0x007F;                            /// 0 00000 0001111111
    unsigned short rne_tie = 0x00C0;                                /// 0 00000 0011000000

    __half_t h;
    for (size_t i = 0; i < count; ++i) {
        h.f = arg[i];
        float inval = ngraph::float16(arg[i]);
        /* flush values below 1-4-3 (offset=4) subnormal range to zero */
        if (fabs(inval) < 1.2207031e-4)
            h.f = 0;

        short exp_h =
            (short)((h.u & 0x7C00) >> 10) - 15;  /// 0111110000000000 -> 0000000000011111 - 15 - biased exponent
        short sign_h = (h.u & 0x8000);           /// 1 00000 0000000000
        short mantissa_h = (h.u & 0x03FF);       /// 0 00000 1111111111
        ///(h.u && 0111111111111111) < 0 10010 1110000000 (19326) - ????
        unsigned short can_round = ((h.u & 0x7FFF) < 0b100111110000000) ? 1 : 0;
        unsigned short is_normal = 1;

        is_normal = (((h.u & 0x7C00) <= 0x7800) && ((h.u & 0x7C00) >= 0x0400)) ? 1 : 0;
        unsigned short is_naninf = ((h.u & 0x7C00) == 0x7C00) ? 1 : 0;

        int dshift = 0;
        if (exp_h > 4) {  // too large, set it to NaN or inf
            if (use_clamp) {
                exp_h = 4;
                mantissa_h = 0b0000001100000000;
            } else {
                mantissa_h = 0;
                exp_h = 16;
                is_naninf = 1;
            }
        } else if (exp_h < -13) {  /// -13, -12, -11 for rounding
            /* flush values below 1-4-3 (offset=4) subnormal range to zero */
            exp_h = -15;
            mantissa_h = 0;
        }

        if (exp_h == 4 && mantissa_h > 0b0000001100000000) {
            mantissa_h = 0b0000001100000000;
        }
        /* nearest rounding masks, & 0 00000 000 111 1111 - mantissa bits below hf8 (grs) */
        unsigned short rnmask = (mantissa_h & grs_bitmask);
        /* & 0 00000 0011000000 - edge between hf8 and fp16 mantissa */
        unsigned short rnmask_tie = (mantissa_h & rne_tie);
        if (!is_naninf && can_round && rne_mask) {
            /* round to nearest even, if rne_mask is enabled */
            /// rnmask > 0 00000 0001000000(64) or 0 00000 0011000000 - edge bits is 1
            /// += 0 00000 0010000000
            mantissa_h += (((rnmask > 0x0040) || (rnmask_tie == rne_tie)) << lshift);
        }
        if (exp_h < -10) { /* handle denormals -13, -12, -11, dshift 1, 2, 3 */
            dshift = (-10 - exp_h);
            mantissa_h = mantissa_h >> dshift;
        }
        mantissa_h &= mask_mant; /* truncation */
        mantissa_h <<= dshift;
        mantissa_h += ((exp_h + 15) << 10);
        h.u = mantissa_h | sign_h;
        out[i] = h.f;
    }
}

/// not original
#define LIBXSMM_CAST_USHORT(VALUE) ((unsigned short)((VALUE)))

unsigned char convert_fp16_hf8_libxsmm(ov::float16 inp) {
    unsigned int f16_bias = 15;
    unsigned int f8_bias = 7;  // [-7..8], -7 is denormal
    unsigned char res = 0;
    unsigned short s, e, m, e_f16, m_f16;
    unsigned int fixup;
    unsigned short in = inp.to_bits();

    s = (in & 0x8000) >> 8;
    e_f16 = (in & 0x7c00) >> 10;  /// & 0b0111110000000000
    m_f16 = (in & 0x03ff);        /// & 0b0000001111111111

    /* special value --> make it max */
    if (e_f16 == 0x1f) {  // == 31 or 0000000000011111
        e = 0xf;          // 0000000000001111
        m = 0x6;          // 0000000000000110
        /* overflow --> make it max */
    } else if ((e_f16 > (f16_bias - f8_bias + 15)) ||  // 23 (exp - bias > 8 or 1111)
               ((e_f16 == (f16_bias - f8_bias + 15)) &&
                (m_f16 > 0x0300))) {  // (exp - bias == 8 or 1111) and mantissa > 0s00000e1100000000m
        e = 0xf;
        m = 0x6;
        /* smaller than denormal f8 + eps */
    } else if (e_f16 < f16_bias - f8_bias - 3) {  // < 5
        e = 0x0;
        m = 0x0;
        /* denormal */
    } else if (e_f16 <= f16_bias - f8_bias) {  // <= 8
        /* RNE */
        /* denormalized mantissa */
        m = m_f16 | 0x0400;  // 0000010000000000
        /* addtionally subnormal shift */
        m = m >> ((f16_bias - f8_bias) + 1 - e_f16);
        /* preserve sticky bit (some sticky bits are lost when denormalizing) */
        m |= (((m_f16 & 0x007f) + 0x007f) >> 7);
        /* RNE Round */
        fixup = (m >> 7) & 0x1;
        m = m + LIBXSMM_CAST_USHORT(0x003f + fixup);
        m = m >> 7;
        e = 0x0;
        /* normal */
    } else {
        /* RNE round */
        fixup = (m_f16 >> 7) & 0x1;
        in = in + LIBXSMM_CAST_USHORT(0x003f + fixup);
        e = (in & 0x7c00) >> 10;
        m = (in & 0x03ff);
        OPENVINO_ASSERT(e >= LIBXSMM_CAST_USHORT(f16_bias - f8_bias), "");
        e -= LIBXSMM_CAST_USHORT(f16_bias - f8_bias);
        m = m >> 7;
    }

    /* set result to 0 */
    res = 0x0;
    /* set exp and mant */
    res |= e << 3;
    res |= m;
    /* sign it */
    res |= s;

    return res;
}

unsigned short convert_hf8_fp16_libxsmm(unsigned char inp) {
    unsigned int f16_bias = 15;
    unsigned int f8_bias = 7;
    unsigned short s = (inp & 0x80) << 8;
    unsigned short e = (inp & 0x78) >> 3;
    unsigned short m = (inp & 0x07);
    unsigned short e_norm = e + (f16_bias - f8_bias);
    unsigned short res = 0;
    /* convert denormal fp8 number into a normal fp16 number */
    if ((e == 0) && (m != 0)) {
        unsigned int lz_cnt = 2;
        lz_cnt = (m > 0x1) ? 1 : lz_cnt;
        lz_cnt = (m > 0x3) ? 0 : lz_cnt;
        OPENVINO_ASSERT(e_norm >= lz_cnt, "e_norm >= lz_cnt");
        e_norm -= lz_cnt;
        m = (m << (lz_cnt + 1)) & 0x07;
    } else if ((e == 0) && (m == 0)) {
        e_norm = 0;
    } else if ((e == 0xf) && (m == 0x7)) {
        e_norm = 0xff;
        m = 0x4; /* making first mantissa bit 1 */
    }

    /* set exp and mant */
    res |= (e_norm << 10);
    res |= (m << 7);
    /* sign it */
    res |= s;
    return res;
}

unsigned short convert_fp16_hf8_fp16_libxsmm(ov::float16 inp) {
    return convert_hf8_fp16_libxsmm(convert_fp16_hf8_libxsmm(inp));
}

/// <summary>
/// emulation of convertation fp16 value to hf8 1s-4e-3m format, Extended Hybrid Float
/// exponent bias is 7
/// </summary>
/// <typeparam name="T">Every possible type with 16 bit size</typeparam>
/// <param name="arg"></param>
/// <param name="out"></param>
/// <param name="count"></param>
template <typename T>
void convertfp16_hf8_libxsmm(const T* arg, T* out, size_t count, bool use_clamp = true) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = convert_fp16_hf8_fp16_libxsmm(ov::float16::from_bits(arg[i]));
    }
}

template <typename ET>
bool evaluate(ov::Tensor& arg, ov::Tensor& out, const std::string& destination_type) {
    out.set_shape(arg.get_shape());
    size_t element_count = shape_size(out.get_shape());

    if ((ov::element::f16 != arg.get_element_type()) || ov::element::f16 != out.get_element_type()) {
        std::cout << "Bad arg or out types: " << arg.get_element_type() << " " << out.get_element_type() << std::endl;
        return false;
    }

    auto inPtr = static_cast<ET*>(arg.data());
    auto outPtr = static_cast<ET*>(out.data());

    if (destination_type == "bf8") {
        convertfp16_bf8(inPtr, outPtr, element_count);
    } else if (destination_type == "hf8") {
        convertfp16_hf8(inPtr, outPtr, element_count);
    } else if (destination_type == "hf8_ext") {
        convertfp16_hf8_ext(inPtr, outPtr, element_count);
    } else if (destination_type == "hf8_libxsmm") {
        convertfp16_hf8_libxsmm(inPtr, outPtr, element_count);
    } else {
        std::cout << "Bad destination_type: " << destination_type << std::endl;
    }

    return true;
}

template <typename ET>
bool evaluate_mixed(ov::Tensor& arg, ov::Tensor& out, const ov::Tensor& scale) {
    out.set_shape(arg.get_shape());
    memcpy(out.data(), arg.data(), arg.get_byte_size());
    return true;
    size_t element_count = shape_size(out.get_shape());

    if ((ov::element::f16 != arg.get_element_type()) || ov::element::f16 != out.get_element_type()) {
        std::cout << "Bad arg or out types: " << arg.get_element_type() << " " << out.get_element_type() << std::endl;
        return false;
    }

    if (ov::element::f32 != scale.get_element_type()) {
        std::cout << "Bad type of scale: " << scale.get_element_type() << std::endl;
        return false;
    }

    auto dataShape = arg.get_shape();
    auto scaleSize = scale.get_size();

    OPENVINO_ASSERT(dataShape[0] == scaleSize, "Shape mismatch in scale");

    size_t step = 1;
    for (size_t i = 1; i < dataShape.size(); i++) {
        step *= dataShape[i];
    }

    const float* scalePtr = static_cast<float*>(scale.data());
    for (size_t i = 0; i < scaleSize; i++) {
        auto inPtr = static_cast<ET*>(arg.data()) + i * step;
        auto outPtr = static_cast<ET*>(out.data()) + i * step;
        if (scalePtr[i] > 1.0) {
            convertfp16_bf8(inPtr, outPtr, step);
        } else {
            convertfp16_hf8_ext(inPtr, outPtr, step);
        }
    }

    return true;
}

template <typename T, typename S>
void apply_scale(T* data, int sz, S scale) {
    for (int i = 0; i < sz; i++) {
        data[i] = scale * data[i];
    }
}

template <typename T>
void apply_per_channel_scale(ov::Tensor& data, const ov::Tensor& scale, bool invert = false) {
    auto dataShape = data.get_shape();
    auto scaleShape = scale.get_shape();
    auto scaleSize = scale.get_size();

    T* dataPtr = static_cast<T*>(data.data());
    float* scalePtr = static_cast<float*>(scale.data());

    if (scaleSize == 1) {  // per tensor scale, probably for activation
        auto dataSize = data.get_size();
        float s = scalePtr[0];
        if (invert)
            s = 1.0 / s;

        for (size_t j = 0; j < dataSize; j++) {
            dataPtr[j] *= s;
        }
        return;
    }

    if (scaleShape[0] == 1 && dataShape[1] == scaleShape[1]) {  // per channel scale for DW activations
        size_t step = 1;
        for (size_t i = 2; i < dataShape.size(); i++) {
            step *= dataShape[i];
        }

        for (size_t bs = 0; bs < dataShape[0]; bs++) {
            for (size_t i = 0; i < scaleSize; i++) {
                float s = scalePtr[i];
                if (invert)
                    s = 1.0 / s;
                for (size_t j = 0; j < step; j++) {
                    dataPtr[j] *= s;
                }
                dataPtr += step;
            }
        }
        return;
    }

    OPENVINO_ASSERT(dataShape[0] == scaleSize, "Shape mismatch in scale ");

    size_t step = 1;
    for (size_t i = 1; i < dataShape.size(); i++) {
        step *= dataShape[i];
    }

    for (size_t i = 0; i < scaleSize; i++) {
        T s = static_cast<T>(scalePtr[i]);
        if (invert) {
            for (size_t j = 0; j < step; j++) {
                dataPtr[j] /= s;
            }
        } else {
            for (size_t j = 0; j < step; j++) {
                dataPtr[j] *= s;
            }
        }
        dataPtr += step;
    }
}

}  // namespace
}  // namespace convert_fp8

//! [op:evaluate]
bool op::v1::ConvertFP8::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    ov::TensorVector fp16;

    OPENVINO_ASSERT(
        outputs[0].get_element_type() == ov::element::f32 && inputs[0].get_element_type() == ov::element::f32,
        "Wrong input or output type for ConvertFP8::evaluate");

    outputs[0].set_shape(inputs[0].get_shape());
    fp16.emplace_back(ov::Tensor(ov::element::f16, inputs[0].get_shape()));
    int element_count = inputs[0].get_size();

    ngraph::runtime::reference::convert(inputs[0].data<float>(), fp16[0].data<ov::float16>(), element_count);

    if (m_apply_scale) {
        convert_fp8::apply_per_channel_scale<ov::float16>(fp16[0], inputs[1]);
    }

    if (outputs[0].get_element_type() == ov::element::f16)
        convert_fp8::evaluate<unsigned short>(fp16[0], outputs[0], m_destination_type);
    else if (outputs[0].get_element_type() == ov::element::f32) {
        convert_fp8::evaluate<unsigned short>(fp16[0], fp16[0], m_destination_type);
        ngraph::runtime::reference::convert(fp16[0].data<ov::float16>(), outputs[0].data<float>(), element_count);
    }

    if (m_apply_scale) {
        convert_fp8::apply_per_channel_scale<float>(outputs[0], inputs[1], true);
    }

    return true;
}
//! [op:evaluate]