// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/extractimagepatches.hpp"

namespace ov {
namespace reference {
template <typename T>
void extract_image_patches(const std::shared_ptr<op::v3::ExtractImagePatches> extImgPatches,
                           const T* input,
                           T* out,
                           const Shape& inShape,
                           const Shape& outShape) {
    const size_t dimsSize = inShape.size();
    const size_t BATCH = 0, CHANNEL = 1, HIGHT = 0, WIDTH = 1;

    const int64_t KH = extImgPatches->get_sizes()[HIGHT];
    const int64_t KW = extImgPatches->get_sizes()[WIDTH];
    const int64_t SH = extImgPatches->get_strides()[HIGHT];
    const int64_t SW = extImgPatches->get_strides()[WIDTH];
    const int64_t RH = extImgPatches->get_rates()[HIGHT];
    const int64_t RW = extImgPatches->get_rates()[WIDTH];
    const auto auto_pad = extImgPatches->get_auto_pad();

    const int64_t IB = inShape[BATCH];
    const int64_t IC = inShape[CHANNEL];
    const int64_t IH = inShape[dimsSize - 2];
    const int64_t IW = inShape[dimsSize - 1];

    const int64_t OB = outShape[BATCH];
    const int64_t OC = outShape[CHANNEL];
    const int64_t OH = outShape[dimsSize - 2];
    const int64_t OW = outShape[dimsSize - 1];

    int64_t iwStep = KW + (RW - 1) * (KW - 1);
    int64_t ihStep = KH + (RH - 1) * (KH - 1);

    const int64_t OH_OW = OH * OW;
    const int64_t OC_OH_OW = OC * OH_OW;
    const int64_t OB_OC_OH_OW = OC_OH_OW * OB;
    const int64_t IH_IW = IH * IW;
    const int64_t IC_IH_IW = IC * IH_IW;
    const int64_t IB_IC_IH_IW = IC_IH_IW * IB;

    int64_t PL = 0, PT = 0;

    if (auto_pad != op::PadType::VALID) {
        int64_t PW = static_cast<int64_t>(std::ceil(1.f * IW / SW) - 1) * SW + iwStep - IW;
        int64_t PH = static_cast<int64_t>(std::ceil(1.f * IH / SH) - 1) * SH + ihStep - IH;

        if ((PW > 0) && (PW < iwStep)) {
            if (PW % 2 == 1) {
                if (auto_pad == op::PadType::SAME_LOWER) {
                    PL = (PW + 1) / 2;
                } else if (auto_pad == op::PadType::SAME_UPPER) {
                    PL = (PW - 1) / 2;
                }
            } else {
                PL = PW / 2;
            }
        }
        if ((PH > 0) && (PH < ihStep)) {
            if (PH % 2 == 1) {
                if (auto_pad == op::PadType::SAME_LOWER) {
                    PT = (PH + 1) / 2;
                } else if (auto_pad == op::PadType::SAME_UPPER) {
                    PT = (PH - 1) / 2;
                }
            } else {
                PT = PH / 2;
            }
        }
    }

    for (int64_t ob = 0; ob < OB; ob++) {
        const int64_t ib_ICIHIW = ob * IC_IH_IW;
        const int64_t ob_OCOHOW = ob * OC_OH_OW;
        for (int64_t oh = 0; oh < OH; oh++) {
            const int64_t ob_OCOHOW_ohOW = ob_OCOHOW + oh * OW;
            int64_t ih0 = oh * SH - PT;
            for (int64_t ow = 0; ow < OW; ow++) {
                const int64_t ob_OCOHOW_ohOW_ow = ob_OCOHOW_ohOW + ow;
                int64_t iw0 = ow * SW - PL;
                int64_t oc = 0;

                for (int64_t kh = 0; kh < KH; kh++) {
                    int64_t ihKH = ih0 + kh * RH;
                    int64_t ib_ICIHIW_ihKH_IW = ib_ICIHIW + ihKH * IW;
                    for (int64_t kw = 0; kw < KW; kw++) {
                        for (int64_t ic = 0; ic < IC; ic++, oc++) {
                            int64_t iwKW = iw0 + kw * RW;
                            int64_t dst_idx = ob_OCOHOW_ohOW_ow + oc * OH_OW;
                            if (dst_idx >= OB_OC_OH_OW)
                                OPENVINO_THROW("ExtractImagePatches. Destination index is out of "
                                               "bounds.");
                            if (ihKH < 0 || ihKH >= IH || iwKW < 0 || iwKW >= IW) {
                                out[dst_idx] = T(0);
                            } else {
                                int64_t src_idx = ib_ICIHIW_ihKH_IW + ic * IH_IW + iwKW;
                                if (src_idx >= IB_IC_IH_IW)
                                    OPENVINO_THROW("ExtractImagePatches. Source index is out of "
                                                   "bounds.");
                                out[dst_idx] = input[src_idx];
                            }
                        }
                    }
                }
            }
        }
    }
}  // extractImagePatches

}  // namespace reference
}  // namespace ov
