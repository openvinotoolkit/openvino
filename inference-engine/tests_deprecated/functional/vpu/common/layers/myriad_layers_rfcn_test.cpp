// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "myriad_layers_tests.hpp"

#define ERROR_BOUND (1.2e-3f)

using namespace InferenceEngine;

static const int numCLASSes = 21;
static const int numROIs = 300;
static const std::string model_to_psroipooling = R"V0G0N(
    <net name="RFCN_TEST" version="2" batch="1">
        <layers>
            <layer name="input0" type="Input" precision="FP16" id="0">
                <output>
                     <port id="0">
                         <dim>1</dim>
                         <dim>1029</dim>
                         <dim>14</dim>
                         <dim>14</dim>
                     </port>
                </output>
            </layer>
            <layer name="input1" type="Input" precision="FP16" id="1">
                        <output>
                     <port id="1">
                         <dim>
)V0G0N"
                            + std::to_string(numROIs) +
R"V0G0N(
                         </dim>
                         <dim>5</dim>
                     </port>
                </output>
            </layer>
            <layer name="PSROIPooling" type="PSROIPooling" precision="FP16" id="2">
                <data group_size="7" spatial_scale="0.062500" output_dim="
)V0G0N"
                            + std::to_string(numCLASSes) +
R"V0G0N(
                " />
                <input>
                     <port id="2">
                         <dim>1</dim>
                         <dim>1029</dim>
                         <dim>14</dim>
                         <dim>14</dim>
                     </port>
                     <port id="3">
                         <dim>
)V0G0N"
                            + std::to_string(numROIs) +
R"V0G0N(
                         </dim>
                         <dim>5</dim>
                     </port>
                </input>
                <output>
                     <port id="4">
                         <dim>1</dim>
                         <dim>
)V0G0N"
                            + std::to_string(numCLASSes * numROIs) +
R"V0G0N(
                         </dim>
                         <dim>7</dim>
                         <dim>7</dim>
                     </port>
                </output>
            </layer>
        </layers>
        <edges>
           <edge from-layer="0" from-port="0" to-layer="2" to-port="2"/>
           <edge from-layer="1" from-port="1" to-layer="2" to-port="3"/>
        </edges>
    </net>
)V0G0N";

static const std::string model_to_pooling = R"V0G0N(
    <net name="RFCN_TEST" version="2" batch="1">
        <layers>
            <layer name="input0" type="Input" precision="FP16" id="0">
                <output>
                     <port id="0">
                         <dim>1</dim>
                         <dim>1029</dim>
                         <dim>14</dim>
                         <dim>14</dim>
                     </port>
                </output>
            </layer>
            <layer name="input1" type="Input" precision="FP16" id="1">
                        <output>
                     <port id="1">
                         <dim>
)V0G0N"
                            + std::to_string(numROIs) +
R"V0G0N(
                         </dim>
                         <dim>5</dim>
                     </port>
                </output>
            </layer>
            <layer name="PSROIPooling" type="PSROIPooling" precision="FP16" id="2">
                <data group_size="7" spatial_scale="0.062500" output_dim="
)V0G0N"
                            + std::to_string(numCLASSes) +
R"V0G0N(
                " />
                <input>
                     <port id="2">
                         <dim>1</dim>
                         <dim>1029</dim>
                         <dim>14</dim>
                         <dim>14</dim>
                     </port>
                     <port id="3">
                         <dim>
)V0G0N"
                            + std::to_string(numROIs) +
R"V0G0N(
                         </dim>
                         <dim>5</dim>
                     </port>
                </input>
                <output>
                     <port id="4">
                         <dim>
)V0G0N"
                            + std::to_string(numROIs) +
R"V0G0N(
                         </dim>
                         <dim>
)V0G0N"
                            + std::to_string(numCLASSes) +
R"V0G0N(
                         </dim>
                         <dim>7</dim>
                         <dim>7</dim>
                     </port>
                </output>
            </layer>
            <layer name="ave_cls_score_rois" type="Pooling" precision="FP16" id="3">
                <data exclude-pad="false" kernel-x="7" kernel-y="7" pad-x="0" pad-y="0" pool-method="avg" rounding_type="ceil" stride="1,1,7,7" stride-x="7" stride-y="7"/>
                <input>
                    <port id="5">
                         <dim>
)V0G0N"
                            + std::to_string(numROIs) +
R"V0G0N(
                         </dim>
                         <dim>
)V0G0N"
                            + std::to_string(numCLASSes) +
R"V0G0N(
                         </dim>
                        <dim>7</dim>
                        <dim>7</dim>
                    </port>
                </input>
                <output>
                    <port id="6">
                         <dim>
)V0G0N"
                            + std::to_string(numROIs) +
R"V0G0N(
                         </dim>
                         <dim>
)V0G0N"
                            + std::to_string(numCLASSes) +
R"V0G0N(
                         </dim>
                        <dim>1</dim>
                        <dim>1</dim>
                    </port>
                </output>
            </layer>
        </layers>
        <edges>
           <edge from-layer="0" from-port="0" to-layer="2" to-port="2"/>
           <edge from-layer="1" from-port="1" to-layer="2" to-port="3"/>
           <edge from-layer="2" from-port="4" to-layer="3" to-port="5"/>
        </edges>
    </net>
)V0G0N";

static const std::string model_to_softmax = R"V0G0N(
    <net name="RFCN_TEST" version="2" batch="1">
        <layers>
            <layer name="input0" type="Input" precision="FP16" id="0">
                <output>
                     <port id="0">
                         <dim>1</dim>
                         <dim>1029</dim>
                         <dim>14</dim>
                         <dim>14</dim>
                     </port>
                </output>
            </layer>
            <layer name="input1" type="Input" precision="FP16" id="1">
                        <output>
                     <port id="1">
                         <dim>
)V0G0N"
                            + std::to_string(numROIs) +
R"V0G0N(
                         </dim>
                         <dim>5</dim>
                     </port>
                </output>
            </layer>
            <layer name="PSROIPooling" type="PSROIPooling" precision="FP16" id="2">
                <data group_size="7" spatial_scale="0.062500" output_dim="
)V0G0N"
                            + std::to_string(numCLASSes) +
R"V0G0N(
                " />
                <input>
                     <port id="2">
                         <dim>1</dim>
                         <dim>1029</dim>
                         <dim>14</dim>
                         <dim>14</dim>
                     </port>
                     <port id="3">
                         <dim>
)V0G0N"
                            + std::to_string(numROIs) +
R"V0G0N(
                         </dim>
                         <dim>5</dim>
                     </port>
                </input>
                <output>
                     <port id="4">
                         <dim>
)V0G0N"
                            + std::to_string(numROIs) +
R"V0G0N(
                         </dim>
                         <dim>
)V0G0N"
                            + std::to_string(numCLASSes) +
R"V0G0N(
                         </dim>
                         <dim>7</dim>
                         <dim>7</dim>
                     </port>
                </output>
            </layer>
            <layer name="ave_cls_score_rois" type="Pooling" precision="FP16" id="3">
                <data exclude-pad="false" kernel-x="7" kernel-y="7" pad-x="0" pad-y="0" pool-method="avg" rounding_type="ceil" stride="1,1,7,7" stride-x="7" stride-y="7"/>
                <input>
                    <port id="5">
                         <dim>
)V0G0N"
                            + std::to_string(numROIs) +
R"V0G0N(
                         </dim>
                         <dim>
)V0G0N"
                            + std::to_string(numCLASSes) +
R"V0G0N(
                         </dim>
                        <dim>7</dim>
                        <dim>7</dim>
                    </port>
                </input>
                <output>
                    <port id="6">
                         <dim>
)V0G0N"
                            + std::to_string(numROIs) +
R"V0G0N(
                         </dim>
                         <dim>
)V0G0N"
                            + std::to_string(numCLASSes) +
R"V0G0N(
                         </dim>
                        <dim>1</dim>
                        <dim>1</dim>
                    </port>
                </output>
            </layer>
            <layer name="cls_prob" type="SoftMax" precision="FP16" id="4">
                <data axis="1"/>
                <input>
                    <port id="7">
                         <dim>
)V0G0N"
                            + std::to_string(numROIs) +
R"V0G0N(
                         </dim>
                         <dim>
)V0G0N"
                            + std::to_string(numCLASSes) +
R"V0G0N(
                         </dim>
                        <dim>1</dim>
                        <dim>1</dim>
                    </port>
                </input>
                <output>
                    <port id="8">
                         <dim>
)V0G0N"
                            + std::to_string(numROIs) +
R"V0G0N(
                         </dim>
                         <dim>
)V0G0N"
                            + std::to_string(numCLASSes) +
R"V0G0N(
                         </dim>
                        <dim>1</dim>
                        <dim>1</dim>
                    </port>
                </output>
            </layer>
        </layers>
        <edges>
           <edge from-layer="0" from-port="0" to-layer="2" to-port="2"/>
           <edge from-layer="1" from-port="1" to-layer="2" to-port="3"/>
           <edge from-layer="2" from-port="4" to-layer="3" to-port="5"/>
           <edge from-layer="3" from-port="6" to-layer="4" to-port="7"/>
        </edges>
    </net>
)V0G0N";

static const std::string model_to_reshape = R"V0G0N(
    <net name="RFCN_TEST" version="2" batch="1">
        <layers>
            <layer name="input0" type="Input" precision="FP16" id="0">
                <output>
                     <port id="0">
                         <dim>1</dim>
                         <dim>1029</dim>
                         <dim>14</dim>
                         <dim>14</dim>
                     </port>
                </output>
            </layer>
            <layer name="input1" type="Input" precision="FP16" id="1">
                        <output>
                     <port id="1">
                         <dim>
)V0G0N"
                            + std::to_string(numROIs) +
R"V0G0N(
                         </dim>
                         <dim>5</dim>
                     </port>
                </output>
            </layer>
            <layer name="PSROIPooling" type="PSROIPooling" precision="FP16" id="2">
                <data group_size="7" spatial_scale="0.062500" output_dim="
)V0G0N"
                            + std::to_string(numCLASSes) +
R"V0G0N(
                " />
                <input>
                     <port id="2">
                         <dim>1</dim>
                         <dim>1029</dim>
                         <dim>14</dim>
                         <dim>14</dim>
                     </port>
                     <port id="3">
                         <dim>
)V0G0N"
                            + std::to_string(numROIs) +
R"V0G0N(
                         </dim>
                         <dim>5</dim>
                     </port>
                </input>
                <output>
                     <port id="4">
                         <dim>
)V0G0N"
                            + std::to_string(numROIs) +
R"V0G0N(
                         </dim>
                         <dim>
)V0G0N"
                            + std::to_string(numCLASSes) +
R"V0G0N(
                         </dim>
                         <dim>7</dim>
                         <dim>7</dim>
                     </port>
                </output>
            </layer>
            <layer name="ave_cls_score_rois" type="Pooling" precision="FP16" id="3">
                <data exclude-pad="false" kernel-x="7" kernel-y="7" pad-x="0" pad-y="0" pool-method="avg" rounding_type="ceil" stride="1,1,7,7" stride-x="7" stride-y="7"/>
                <input>
                    <port id="5">
                         <dim>
)V0G0N"
                            + std::to_string(numROIs) +
R"V0G0N(
                         </dim>
                         <dim>
)V0G0N"
                            + std::to_string(numCLASSes) +
R"V0G0N(
                         </dim>
                        <dim>7</dim>
                        <dim>7</dim>
                    </port>
                </input>
                <output>
                    <port id="6">
                         <dim>
)V0G0N"
                            + std::to_string(numROIs) +
R"V0G0N(
                         </dim>
                         <dim>
)V0G0N"
                            + std::to_string(numCLASSes) +
R"V0G0N(
                         </dim>
                        <dim>1</dim>
                        <dim>1</dim>
                    </port>
                </output>
            </layer>
            <layer name="cls_prob" type="SoftMax" precision="FP16" id="4">
                <data axis="1"/>
                <input>
                    <port id="7">
                         <dim>
)V0G0N"
                            + std::to_string(numROIs) +
R"V0G0N(
                         </dim>
                         <dim>
)V0G0N"
                            + std::to_string(numCLASSes) +
R"V0G0N(
                         </dim>
                        <dim>1</dim>
                        <dim>1</dim>
                    </port>
                </input>
                <output>
                    <port id="8">
                         <dim>
)V0G0N"
                            + std::to_string(numROIs) +
R"V0G0N(
                         </dim>
                         <dim>
)V0G0N"
                            + std::to_string(numCLASSes) +
R"V0G0N(
                         </dim>
                        <dim>1</dim>
                        <dim>1</dim>
                    </port>
                </output>
            </layer>
            <layer name="cls_prob_reshape" type="Reshape" precision="FP16" id="5">
                <data axis="0" num_axes="-1" dim="-1,
)V0G0N"
                            + std::to_string(numCLASSes) +
R"V0G0N(
                                                       "/>
                <input>
                    <port id="9">
                        <dim>
)V0G0N"
                            + std::to_string(numROIs) +
R"V0G0N(
                        </dim>
                        <dim>
)V0G0N"
                            + std::to_string(numCLASSes) +
R"V0G0N(
                        </dim>
                        <dim>1</dim>
                        <dim>1</dim>
                    </port>
                </input>
                <output>
                    <port id="10">
                         <dim>
)V0G0N"
                            + std::to_string(numROIs) +
R"V0G0N(
                         </dim>
                         <dim>
)V0G0N"
                            + std::to_string(numCLASSes) +
R"V0G0N(
                         </dim>
                    </port>
                </output>
            </layer>
        </layers>
        <edges>
           <edge from-layer="0" from-port="0" to-layer="2" to-port="2"/>
           <edge from-layer="1" from-port="1" to-layer="2" to-port="3"/>
           <edge from-layer="2" from-port="4" to-layer="3" to-port="5"/>
           <edge from-layer="3" from-port="6" to-layer="4" to-port="7"/>
           <edge from-layer="4" from-port="8" to-layer="5" to-port="9"/>
        </edges>
    </net>
)V0G0N";

// This ref function is modified version from ref_soft_max @ myriad_layers_softmax_test.cpp
static void ref_soft_max(const Blob::Ptr& src, Blob::Ptr& dst, int axis) {
    const ie_fp16 *src_data = src->cbuffer().as<const ie_fp16*>();
    ie_fp16 *dst_data = dst->buffer().as<ie_fp16*>();

    const auto& dims = src->getTensorDesc().getDims();
    int32_t dimx, dimy, dimz;
    dimy = dims[2]; // H:1
    dimz = dims[1]; // C:numCLASS
    dimx = dims[0]; // N:numROI
    // arg axis = 1:  axis for numCLASS(dimz=channels)

    switch (src->getTensorDesc().getDims().size()) {
    case 2:
        axis += 2;
        break;
    case 3:
        axis++;
        break;
    }

    int dim0, dim1, dim2;
    int stride0, stride1, stride2;
    switch (axis) {
    case 1:
        /* channels */
        dim0 = dimy; stride0 = dimx * dimz;
        dim1 = dimx; stride1 = dimz;
        dim2 = dimz; stride2 = 1;
        break;
    case 2:
        /* height */
        dim0 = dimx; stride0 = dimz;
        dim1 = dimz; stride1 = 1;
        dim2 = dimy; stride2 = dimx * dimz;
        break;
    case 3:
        /* width */
        dim0 = dimy; stride0 = dimx * dimz;
        dim1 = dimz; stride1 = 1;
        dim2 = dimx; stride2 = dimz;
        break;
    default:
        FAIL() << "Unsupported axis value = " << axis;
    }

    std::vector<float> temp(dim2);
    for (int i0 = 0; i0 < dim0; ++i0) {
        for (int i1 = 0; i1 < dim1; ++i1) {
            float largest = std::numeric_limits<float>::lowest();
            for (int i2 = 0; i2 < dim2; ++i2) {
                int ind = i0 * stride0 + i1 * stride1 + i2 * stride2;
                float val = PrecisionUtils::f16tof32(src_data[ind]);
                largest = std::max(val, largest);
            }

            float sum = 0.0f;
            for (int i2 = 0; i2 < dim2; ++i2) {
                int ind = i0 * stride0 + i1 * stride1 + i2 * stride2;
                float val = PrecisionUtils::f16tof32(src_data[ind]);
                temp[i2] = std::exp(val - largest);
                sum += temp[i2];
            }

            for (int i2 = 0; i2 < dim2; ++i2) {
                int ind = i0 * stride0 + i1 * stride1 + i2 * stride2;
                dst_data[ind] = PrecisionUtils::f32tof16(temp[i2] / sum);
            }
        }
    }
}

static void refGlobalAvgPooling7x7Rfcn(const Blob::Ptr src,
                                Blob::Ptr dst) {
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);
    const uint16_t *src_data = src->buffer();
    uint16_t *dst_data = dst->buffer();
    ASSERT_NE(src_data, nullptr);
    ASSERT_NE(dst_data, nullptr);

    param_size kernel = {7, 7};
    param_size stride = {1, 1};
    param_size pad    = {0, 0};

    auto src_dims = src->getTensorDesc().getDims();
    int32_t IW =   src_dims[3];   // 7
    int32_t IH =   src_dims[2];   // 7
    int32_t IC =   src_dims[1];   // numROI*numCLASS
    int32_t INUM = src_dims[0]; // 1

    auto dst_dims = dst->getTensorDesc().getDims();
    int32_t OW =   dst_dims[3];   // 1
    int32_t OH =   dst_dims[2];   // 1
    int32_t OC =   dst_dims[1];   // numCLASS
    int32_t ONUM = dst_dims[0]; // numROI

    // CompareCommonAbsolute multiplied value since input shape might be 3D
    ASSERT_EQ(IC * INUM, OC * ONUM);

    for (size_t n = 0; n < ONUM; n++) {
        for (size_t c = 0; c < OC; c++) {
            for (size_t oh = 0; oh < OH; oh++) {
                for (size_t ow = 0; ow < OW; ow++) {
                    size_t oidx = c + ow * OC + oh * OC * OW + n * OC * OW * OH; // Default layout is NHWC
                    float out_ref = 0.0f;
                    size_t count = 0;

                    for (uint32_t kh = 0; kh < kernel.y; kh++) {
                        for (uint32_t kw = 0; kw < kernel.x; kw++) {
                            int32_t iw = ow * stride.x - pad.x + kw;
                            int32_t ih = oh * stride.y - pad.y + kh;
                            if (iw < 0 || iw >= IW || ih < 0 || ih >= IH)
                                continue;

                            // If PSROIPooling is output of network, its layout is ZYX(NCHW). Use OC instead of IC since IC is actually INUM*IC
                            size_t iidx = iw + IW * (ih + c * IH) + n * OC * IW * IH;

                            float d = PrecisionUtils::f16tof32(src_data[iidx]);
                            out_ref += d;
                            count++;
                        }
                    }
                    if (pad.x || pad.y) {
                        dst_data[oidx] = PrecisionUtils::f32tof16(out_ref /(kernel.y * kernel.x));
                    }
                    else
                        dst_data[oidx] = PrecisionUtils::f32tof16(out_ref /count);
                }
            }
        }
    }
}

class myriadLayersRfcnTests_smoke: public myriadLayersTests_nightly {
public:
    void GenROIs(InferenceEngine::Blob::Ptr rois,
                 const uint32_t in_width, const uint32_t in_height,
                 const uint32_t num_rois) {
        ie_fp16 *roisBlob_data = rois->buffer().as<ie_fp16*>();
        const int max_range = in_width * 4 / 5;
        std::srand(std::time(nullptr));
        for (int i = 0; i < num_rois; i++)
        {
            int x0 = std::rand() % max_range;
            int x1 = x0 + (std::rand() % (in_width - x0 - 1)) + 1;
            int y0 = std::rand() % max_range;
            int y1 = y0 + (std::rand() % (in_height - y0 - 1)) + 1;

            roisBlob_data[i * 5 + 0] = PrecisionUtils::f32tof16(0);
            roisBlob_data[i * 5 + 1] = PrecisionUtils::f32tof16(x0);
            roisBlob_data[i * 5 + 2] = PrecisionUtils::f32tof16(y0);
            roisBlob_data[i * 5 + 3] = PrecisionUtils::f32tof16(x1);
            roisBlob_data[i * 5 + 4] = PrecisionUtils::f32tof16(y1);
        }
    }

    void PrepareInputAndReference(const std::string& model_prior_network, const std::string& output_layer)
    {
        SetSeed(DEFAULT_SEED_VALUE);

        // Prior-part of network to generate reference
        Core ie;
        auto network_part = ie.ReadNetwork(model_prior_network, Blob::CPtr());

        auto inputsInfo = network_part.getInputsInfo();
        inputsInfo["input0"]->setPrecision(Precision::FP16);
        inputsInfo["input0"]->setLayout(NCHW); // Input layout for PSROIPooling should be NCHW order, it's same as psroipooling test
        inputsInfo["input1"]->setPrecision(Precision::FP16);

        auto outputsInfo = network_part.getOutputsInfo();
        outputsInfo[output_layer]->setPrecision(Precision::FP16);
        if (output_layer == "PSROIPooling")
            outputsInfo[output_layer]->setLayout(NCHW);

        // Disable HW pooling
        std::map<std::string, std::string> networkConfig;
        networkConfig["VPU_HW_STAGES_OPTIMIZATION"] = "NO";

        ExecutableNetwork exeNetwork;
        ASSERT_NO_THROW(exeNetwork = _vpuPluginPtr->LoadNetwork(network_part, networkConfig));

        InferRequest inferRequest;
        ASSERT_NO_THROW(inferRequest = exeNetwork.CreateInferRequest());

        Blob::Ptr input0;
        ASSERT_NO_THROW(input0 = inferRequest.GetBlob("input0"));

        Blob::Ptr input1;
        ASSERT_NO_THROW(input1 = inferRequest.GetBlob("input1"));

        // Allocate buffer
        input0_share = make_shared_blob<ie_fp16>({Precision::FP16, input0->getTensorDesc().getDims(), ANY});
        input0_share->allocate();
        input1_share = make_shared_blob<ie_fp16>({Precision::FP16, input1->getTensorDesc().getDims(), ANY});
        input1_share->allocate();

        // Generate random input
        GenRandomData(input0_share);
        GenROIs(input1_share, 224, 224, numROIs);

        ASSERT_EQ(input0->size(), input0_share->size());
        ASSERT_EQ(input1->size(), input1_share->size());

        ie_fp16 *input0_data = static_cast<ie_fp16*>(input0->buffer());
        ie_fp16 *input0_share_data = static_cast<ie_fp16*>(input0_share->buffer());
        ie_fp16 *input1_data = static_cast<ie_fp16*>(input1->buffer());
        ie_fp16 *input1_share_data = static_cast<ie_fp16*>(input1_share->buffer());
        std::copy(input0_share_data, input0_share_data + input0_share->size(), input0_data);
        std::copy(input1_share_data, input1_share_data + input1_share->size(), input1_data);

        ASSERT_NO_THROW(inferRequest.Infer());
        ASSERT_NO_THROW(prior_network_output = inferRequest.GetBlob(output_layer.c_str()));
    }

    void RunNetwork(const std::string& model, const std::string& output_layer)
    {
        ASSERT_NO_THROW(readNetwork(model));

        const auto& network = _cnnNetwork;

        _inputsInfo = network.getInputsInfo();
        _inputsInfo["input0"]->setPrecision(Precision::FP16);
        _inputsInfo["input0"]->setLayout(NCHW); // Input layout for PSROIPooling should be NCHW order, it's same as psroipooling test
        _inputsInfo["input1"]->setPrecision(Precision::FP16);

        _outputsInfo = network.getOutputsInfo();
        _outputsInfo[output_layer]->setPrecision(Precision::FP16);

        // Disable HW pooling
        std::map<std::string, std::string> networkConfig;
        networkConfig["VPU_HW_STAGES_OPTIMIZATION"] = "NO";

        ASSERT_NO_THROW(_exeNetwork = _vpuPluginPtr->LoadNetwork(network, networkConfig));
        ASSERT_NO_THROW(_inferRequest = _exeNetwork.CreateInferRequest());

        Blob::Ptr input0;
        ASSERT_NO_THROW(input0 = _inferRequest.GetBlob("input0"));

        Blob::Ptr input1;
        ASSERT_NO_THROW(input1 = _inferRequest.GetBlob("input1"));

        ASSERT_NO_THROW(outputBlob = _inferRequest.GetBlob(output_layer.c_str()));

        _refBlob = make_shared_blob<ie_fp16>({Precision::FP16, outputBlob->getTensorDesc().getDims(), ANY});
        _refBlob->allocate();

        // Set input to run test
        ASSERT_EQ(input0->size(), input0_share->size());
        ASSERT_EQ(input1->size(), input1_share->size());

        ie_fp16 *input0_data = static_cast<ie_fp16*>(input0->buffer());
        ie_fp16 *input0_share_data = static_cast<ie_fp16*>(input0_share->buffer());
        ie_fp16 *input1_data = static_cast<ie_fp16*>(input1->buffer());
        ie_fp16 *input1_share_data = static_cast<ie_fp16*>(input1_share->buffer());
        std::copy(input0_share_data, input0_share_data + input0_share->size(), input0_data);
        std::copy(input1_share_data, input1_share_data + input1_share->size(), input1_data);

        ASSERT_NO_THROW(_inferRequest.Infer());
    }

    Blob::Ptr input0_share;
    Blob::Ptr input1_share;
    Blob::Ptr prior_network_output;
    Blob::Ptr outputBlob;
};

TEST_F(myriadLayersRfcnTests_smoke, ReshapeRfcn)
{
    std::string prior_network_output_layer = "cls_prob";
    std::string test_network_output_layer = "cls_prob_reshape";

    ASSERT_NO_THROW(PrepareInputAndReference(model_to_softmax, prior_network_output_layer));
    ASSERT_NO_THROW(RunNetwork(model_to_reshape, test_network_output_layer));

    ASSERT_TRUE(prior_network_output);

    ASSERT_EQ(outputBlob->size(), prior_network_output->size());
    CompareCommonAbsolute(outputBlob, prior_network_output, 0.0f);
}

TEST_F(myriadLayersRfcnTests_smoke, SoftmaxRfcn)
{
    std::string prior_network_output_layer = "ave_cls_score_rois";
    std::string test_network_output_layer = "cls_prob";

    ASSERT_NO_THROW(PrepareInputAndReference(model_to_pooling, prior_network_output_layer));
    ASSERT_NO_THROW(RunNetwork(model_to_softmax, test_network_output_layer));

    int param_axis = 1;
    ASSERT_TRUE(prior_network_output);

    ref_soft_max(prior_network_output, _refBlob, param_axis);

    CompareCommonAbsolute(outputBlob, _refBlob, ERROR_BOUND);
}

TEST_F(myriadLayersRfcnTests_smoke, GlobalAvgPooling7x7Rfcn)
{
    std::string prior_network_output_layer = "PSROIPooling";
    std::string test_network_output_layer = "ave_cls_score_rois";

    ASSERT_NO_THROW(PrepareInputAndReference(model_to_psroipooling, prior_network_output_layer));
    ASSERT_NO_THROW(RunNetwork(model_to_pooling, test_network_output_layer));

    ASSERT_TRUE(prior_network_output);

    refGlobalAvgPooling7x7Rfcn(prior_network_output, _refBlob);

    CompareCommonAbsolute(outputBlob, _refBlob, ERROR_BOUND);
}
