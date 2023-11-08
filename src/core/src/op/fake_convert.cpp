// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/fake_convert.hpp"

#include "element_visitor.hpp"
#include "itt.hpp"
#include "openvino/reference/convert.hpp"

namespace ov {
namespace op {
namespace v13 {
namespace fake_convert {
static const std::vector<std::string>& get_valid_types() {
    static const std::vector<std::string> valid_types{"f8e4m3", "f8e5m2"};
    return valid_types;
}

// struct Evaluate : element::NoAction<bool> {
//     using element::NoAction<bool>::visit;
// };

}  // namespace fake_convert
FakeConvert::FakeConvert(const ov::Output<ov::Node>& arg,
                         const ov::Output<ov::Node>& scale,
                         const ov::Output<ov::Node>& shift,
                         std::string destination_type,
                         bool apply_scale)
    : Op({arg, scale, shift}),
      m_destination_type(std::move(destination_type)),
      m_apply_scale(apply_scale) {
    constructor_validate_and_infer_types();
}

bool FakeConvert::get_apply_scale() const {
    return m_apply_scale;
}

const std::string& FakeConvert::get_destination_type() const {
    return m_destination_type;
}

void FakeConvert::validate_and_infer_types() {
    OV_OP_SCOPE(v13_FakeConvert_validate_and_infer_types);
    validate_type();
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

std::shared_ptr<ov::Node> FakeConvert::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OV_OP_SCOPE(v13_FakeConvert_clone_with_new_inputs);
    OPENVINO_ASSERT(new_args.size() == 3, "Incorrect number of new arguments");

    return std::make_shared<FakeConvert>(new_args.at(0),
                                         new_args.at(1),
                                         new_args.at(2),
                                         m_destination_type,
                                         m_apply_scale);
}

bool FakeConvert::visit_attributes(ov::AttributeVisitor& visitor) {
    OV_OP_SCOPE(v13_FakeConvert_visit_attributes);
    visitor.on_attribute("destination_type", m_destination_type);
    visitor.on_attribute("apply_scale", m_apply_scale);

    return true;
}

void FakeConvert::validate_type() const {
    const auto& valid_types = fake_convert::get_valid_types();
    OPENVINO_ASSERT(std::find(valid_types.begin(), valid_types.end(), m_destination_type) != valid_types.end(),
                    "Bad format for f8 conversion type: " + m_destination_type);
}

// bool FakeConvert::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
//     OV_OP_SCOPE(v13_FakeConvert_evaluate);
//     OPENVINO_ASSERT(outputs.size() == 1);
//     OPENVINO_ASSERT(inputs.size() == 3);

//     const auto& data_shape = inputs[0].get_shape();
//     outputs[0].set_shape(data_shape);

//     return true;
// }

bool FakeConvert::has_evaluate() const {
    OV_OP_SCOPE(v13_FakeConvert_has_evaluate);
    switch (get_input_element_type(0)) {
    case element::f16:
    case element::f32:
        return true;
    default:
        return false;
    }
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
        unsigned short can_round = ((h.u & 0x7FFF) < 0x4B80) ? 1 : 0;  // 0b100111110000000
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

/// <summary>
/// emulation of convertation fp16 value to hf8 1s-4e-3m format, Extended Hybrid Float
/// </summary>
/// <typeparam name="T">Every possible type with 16 bit size</typeparam>
/// <param name="arg"></param>
/// <param name="out"></param>
/// <param name="count"></param>
// Exponent denormal values 0 -7
// Exponent normal values 1..15 -6..8 (7 - exponent)
// Exponent NaN values 15 8
template <typename T>
void convertfp16_hf8_bias7(const T* arg, T* out, size_t count, int exp_bits = 5, int mbits = 9, bool use_clamp = true) {
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
        if (fabs(inval) < 0.001953125)  // 2**-9
            h.f = 0;

        short exp_h = (short)((h.u & 0x7C00) >> 10) -
                      15;                   /// 0111110000000000 -> 0000000000011111 - 15 - biased exponent for fp16
        short sign_h = (h.u & 0x8000);      /// & 1 00000 0000000000
        short mantissa_h = (h.u & 0x03FF);  /// & 0 00000 1111111111
        ///(h.u && 0111111111111111) < 0 10010 1110000000 (19326) - ????
        unsigned short can_round = ((h.u & 0x7FFF) < 0b101111110000000) ? 1 : 0;

        unsigned short is_naninf = ((h.u & 0x7C00) == 0x7C00) ? 1 : 0;

        int dshift = 0;
        if (exp_h > 8) {  // too large, set it to NaN or inf
            is_naninf = 1;
            if (use_clamp) {
                exp_h = 8;
                mantissa_h = 0b0000001100000000;
            } else {
                mantissa_h = 0;
                exp_h = 16;
            }
        } else if (exp_h < -9) {  /// -13, -12, -11 for rounding
            /* flush values below 1-4-3 (offset=4) subnormal range to zero */
            exp_h = -15;
            mantissa_h = 0;
        }

        if (exp_h == 8 && mantissa_h >= 0b0000001100000000) {
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
        if (exp_h < -6) { /* handle denormals -11, -10, -9, dshift 1, 2, 3 */
            dshift = (-6 - exp_h);
            mantissa_h = mantissa_h >> dshift;
        }
        mantissa_h &= mask_mant; /* truncation */
        mantissa_h <<= dshift;
        mantissa_h += ((exp_h + 15) << 10);
        h.u = mantissa_h | sign_h;
        out[i] = h.f;
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

    if (destination_type == "BH8") {
        convertfp16_bf8(inPtr, outPtr, element_count);
    } else if (destination_type == "HF8") {
        convertfp16_hf8_bias7(inPtr, outPtr, element_count);
    } else {
        std::cout << "Bad destination_type: " << destination_type << std::endl;
    }

    return true;
}

template <typename T>
void apply_per_channel_scale(ov::Tensor& data, const ov::Tensor& scale, const ov::Tensor& offset, bool invert = false) {
    auto dataShape = data.get_shape();
    auto scaleShape = scale.get_shape();
    auto scaleSize = scale.get_size();

    T* dataPtr = static_cast<T*>(data.data());
    float* scalePtr = static_cast<float*>(scale.data());
    float* offsetPtr = static_cast<float*>(offset.data());

    if (scaleSize == 1) {  // per tensor scale, probably for activation
        auto dataSize = data.get_size();
        float s = scalePtr[0];
        float o = offsetPtr[0];
        if (invert) {
            for (size_t j = 0; j < dataSize; j++) {
                dataPtr[j] = (dataPtr[j] + o) / s;
            }
        } else {
            for (size_t j = 0; j < dataSize; j++) {
                dataPtr[j] = dataPtr[j] * s - o;  // o = quntized(o * s)
            }
        }
        return;
    }

    if (scaleShape[0] == 1 && dataShape[1] == scaleShape[1]) {  // per channel scale for DW activations
        size_t step = 1;
        for (size_t i = 2; i < dataShape.size(); i++) {  // <batch_size, out_channels, in_channels, H, W> or
            step *= dataShape[i];
        }

        for (size_t bs = 0; bs < dataShape[0]; bs++) {
            for (size_t i = 0; i < scaleSize; i++) {
                float s = scalePtr[i];
                float o = offsetPtr[i];
                if (invert) {
                    for (size_t j = 0; j < step; j++) {
                        dataPtr[j] = (dataPtr[j] + o) / s;
                    }
                } else {
                    for (size_t j = 0; j < step; j++) {
                        dataPtr[j] = dataPtr[j] * s - o;  // o = quntized(o * s)
                    }
                }
                dataPtr += step;
            }
        }
        return;
    }

    OPENVINO_ASSERT(dataShape[0] == scaleSize, "Shape mismatch in scale ");

    // per channel scale for weights
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
bool FakeConvert::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    ov::TensorVector fp16;

    OPENVINO_ASSERT(
        outputs[0].get_element_type() == ov::element::f32 && inputs[0].get_element_type() == ov::element::f32,
        "Wrong input or output type for FakeConvertFP::evaluate");

    outputs[0].set_shape(inputs[0].get_shape());
    fp16.emplace_back(ov::Tensor(ov::element::f16, inputs[0].get_shape()));
    int element_count = inputs[0].get_size();

    ov::reference::convert(inputs[0].data<float>(), fp16[0].data<ov::float16>(), element_count);

    if (m_apply_scale) {
        convert_fp8::apply_per_channel_scale<ov::float16>(fp16[0], inputs[1], inputs[2]);
    }

    if (outputs[0].get_element_type() == ov::element::f16)
        convert_fp8::evaluate<unsigned short>(fp16[0], outputs[0], m_destination_type);
    else if (outputs[0].get_element_type() == ov::element::f32) {
        convert_fp8::evaluate<unsigned short>(fp16[0], fp16[0], m_destination_type);
        ov::reference::convert(fp16[0].data<ov::float16>(), outputs[0].data<float>(), element_count);
    }

    if (m_apply_scale) {
        convert_fp8::apply_per_channel_scale<float>(outputs[0], inputs[1], inputs[2], true);
    }

    return true;
}

}  // namespace v13
}  // namespace op
}  // namespace ov
