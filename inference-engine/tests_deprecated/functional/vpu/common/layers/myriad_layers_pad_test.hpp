// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_tests.hpp"
#include <algorithm>

using std::tuple;
using std::get;

using namespace InferenceEngine;

struct pad_parameters {
    size_t padb_begin;
    size_t padc_begin;
    size_t padh_begin;
    size_t padw_begin;

    size_t padb_end;
    size_t padc_end;
    size_t padh_end;
    size_t padw_end;

    friend std::ostream& operator<<(std::ostream& os, pad_parameters const& tst)
    {
        return os << " pads_begin=" << tst.padb_begin << ", " << tst.padc_begin << ", " << tst.padh_begin << ", " << tst.padw_begin << "; "
                  << " pads_end=" << tst.padb_end << ", " << tst.padc_end << ", " << tst.padh_end << ", " << tst.padw_end;
    };
};

PRETTY_PARAM(layoutPreference, vpu::LayoutPreference);
PRETTY_PARAM(pad_mode, std::string);

typedef myriadLayerTestBaseWithParam<std::tuple<DimsInput, pad_parameters, layoutPreference, pad_mode, IRVersion>> myriadLayerPad;

const float pad_value = 42.0f;

void ref_pad(const Blob::Ptr src,
             Blob::Ptr dst,
             pad_parameters pad_params,
             const std::string mode) {
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);

    ie_fp16 *src_data = static_cast<ie_fp16*>(src->buffer());
    ie_fp16 *dst_data = static_cast<ie_fp16*>(dst->buffer());
    ASSERT_NE(src_data, nullptr);
    ASSERT_NE(dst_data, nullptr);

    int32_t padb_begin = pad_params.padb_begin;
    int32_t padb_end = pad_params.padb_end;

    int32_t padc_begin = pad_params.padc_begin;
    int32_t padc_end = pad_params.padc_end;

    int32_t padh_begin = pad_params.padh_begin;
    int32_t padh_end = pad_params.padh_end;

    int32_t padw_begin = pad_params.padw_begin;
    int32_t padw_end = pad_params.padw_end;

    int32_t IW = 0;
    int32_t IH = 0;
    int32_t IC = 0;
    int32_t OW = 0;
    int32_t OH = 0;
    int32_t OC = 0;

    get_dims(src, IW, IH, IC);
    get_dims(dst, OW, OH, OC);

    for (int32_t oc = 0; oc < OC; oc++) {
        for (int32_t oh = 0; oh < OH; oh++) {
            for (int32_t ow = 0; ow < OW; ow++) {
                int32_t ic = oc - padc_begin;
                int32_t iw = ow - padw_begin;
                int32_t ih = oh - padh_begin;

                float v = 0.0f;
                if (ic >= 0 && ic < IC && iw >= 0 && iw < IW && ih >= 0 && ih < IH)
                {
                    int32_t iidx = ic + iw * IC + ih * IC * IW;
                    ASSERT_LT(iidx, src->size());
                    v = PrecisionUtils::f16tof32(src_data[iidx]);
                }
                else
                {
                    if (mode == std::string("constant"))
                    {
                        v = pad_value;
                    }
                    else if (mode == std::string("edge"))
                    {
                        iw = std::min(std::max(iw, 0), IW - 1);
                        ih = std::min(std::max(ih, 0), IH - 1);
                        ic = std::min(std::max(ic, 0), IC - 1);

                        int32_t iidx = ic + iw * IC + ih * IC * IW;
                        ASSERT_LT(iidx, src->size());
                        v = PrecisionUtils::f16tof32(src_data[iidx]);
                    }
                    else if (mode == std::string("reflect") || mode == std::string("symmetric"))
                    {
                        int mode_offset = (mode == std::string("symmetric")) ? 1 : 0;

                        if (iw > IW - 1) iw = IW-1 - (iw - (IW-1)) + mode_offset;
                        if (iw < 0) iw = -iw - mode_offset;

                        if (ih > IH - 1) ih = IH-1 - (ih - (IH-1)) + mode_offset;
                        if (ih < 0) ih = -ih - mode_offset;

                        if (ic > IC - 1) ic = IC-1 - (ic - (IC-1)) + mode_offset;
                        if (ic < 0) ic = -ic - mode_offset;

                        int32_t iidx = ic + iw * IC + ih * IC * IW;
                        ASSERT_LT(iidx, src->size());
                        v = PrecisionUtils::f16tof32(src_data[iidx]);
                    }
                }

                int32_t oidx = oc + ow * OC + oh * OC * OW;
                ASSERT_LT(oidx, dst->size());
                dst_data[oidx] = PrecisionUtils::f32tof16(v);
            }
        }
    }
}

TEST_P(myriadLayerPad, Pad) {
    tensor_test_params input_dims = get<0>(GetParam());
    pad_parameters pad_parameter = get<1>(GetParam());
    auto layoutPreference = get<2>(GetParam());
    std::string pad_mode = get<3>(GetParam());
    _irVersion           = get<4>(GetParam());

    int padb_begin = pad_parameter.padb_begin;
    int padb_end = pad_parameter.padb_end;

    int padc_begin = pad_parameter.padc_begin;
    int padc_end = pad_parameter.padc_end;

    int padh_begin = pad_parameter.padh_begin;
    int padh_end = pad_parameter.padh_end;

    int padw_begin = pad_parameter.padw_begin;
    int padw_end = pad_parameter.padw_end;

    tensor_test_params output_dims = {1, input_dims.c + padc_begin + padc_end, input_dims.h + padh_begin + padh_end, input_dims.w + padw_begin + padw_end};

    SetInputTensor(input_dims);
    SetOutputTensor(output_dims);

    std::map<std::string, std::string> params;
    params["pads_begin"] = std::to_string(padb_begin)+","+std::to_string(padc_begin)+","+std::to_string(padh_begin)+","+std::to_string(padw_begin);
    params["pads_end"] = std::to_string(padb_end)+","+std::to_string(padc_end)+","+std::to_string(padh_end)+","+std::to_string(padw_end);
    params["pad_mode"] = pad_mode;
    if (pad_mode == std::string("constant"))
        params["pad_value"] = std::to_string(pad_value);

    ASSERT_NO_FATAL_FAILURE(makeSingleLayerNetwork(LayerInitParams("Pad").params(params), NetworkInitParams().layoutPreference(layoutPreference)));
    SetFirstInputToRange(1.0f, 100.0f);

    ASSERT_TRUE(Infer());
    auto inputBlob = _inputMap.begin()->second;
    auto outputBlob = _outputMap.begin()->second;

    ref_pad(inputBlob, _refBlob, pad_parameter, pad_mode);

    CompareCommonAbsolute(outputBlob, _refBlob, 0);
}
