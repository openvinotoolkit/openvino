// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_tests.hpp"

using namespace InferenceEngine;

struct PriorBoxParams {
    tensor_test_params in1 = {1, 512, 38, 38};
    tensor_test_params in2 = {1, 3, 300, 300};

    std::vector<float> min_size = {21.0};
    std::vector<float> max_size = {45.0};
    std::vector<float> aspect_ratio = {2.0};
    int flip = 1;
    int clip = 0;
    std::vector<float> variance = {0.1f, 0.1f, 0.2f, 0.2f};
    int img_size = 0;
    int img_h = 0;
    int img_w = 0;
    float step_ = 8.0;
    float step_h = 0.0;
    float step_w = 0.0;
    float offset = 0.5;
    int scale_all_sizes = 1;
    std::vector<float> fixed_sizes = {};
    std::vector<float> fixed_ratios = {};
    std::vector<float> density = {};
};

// The code was taken from caffe and adopted to InferenceEngine reality
void refPriorBox(Blob::Ptr dst, const PriorBoxParams &p) {
    std::vector<float> aspect_ratios_;
    aspect_ratios_.reserve(p.aspect_ratio.size() + 1);
    aspect_ratios_.push_back(1.0f);
    for (const auto& aspect_ratio : p.aspect_ratio) {
        bool already_exist = false;
        for (const auto& aspect_ratio_ : aspect_ratios_) {
            if (fabsf(aspect_ratio - aspect_ratio_) < 1e-6) {
                already_exist = true;
                break;
            }
        }
        if (!already_exist) {
            aspect_ratios_.push_back(aspect_ratio);
            if (p.flip) {
                aspect_ratios_.push_back(1.0 / aspect_ratio);
            }
        }
    }

    int num_priors_ = 0;
    if (p.scale_all_sizes) {
        num_priors_ = static_cast<int>(aspect_ratios_.size() * p.min_size.size());
    } else {
        num_priors_ = static_cast<int>(aspect_ratios_.size() + p.min_size.size() - 1);
    }

    if (!p.fixed_sizes.empty()) {
        num_priors_ = static_cast<int>(aspect_ratios_.size() * p.fixed_sizes.size());
    }

    if (!p.density.empty()) {
        for (const auto& _density : p.density) {
            if (!p.fixed_ratios.empty()) {
                num_priors_ += (p.fixed_ratios.size()) * (static_cast<int>(pow(_density, 2)) - 1);
            } else {
                num_priors_ += (aspect_ratios_.size()) * (static_cast<int>(pow(_density, 2)) - 1);
            }
        }
    }

    num_priors_ += p.max_size.size();

    const auto layer_width  = p.in1.w;
    const auto layer_height = p.in1.h;

    const auto img_width  = p.img_w == 0 ? p.in2.w : p.img_w;
    const auto img_height = p.img_h == 0 ? p.in2.h : p.img_h;
    const auto img_width_inv = 1.f / static_cast<float>(img_width);
    const auto img_height_inv = 1.f / static_cast<float>(img_height);

    auto step_w = p.step_w == 0 ? p.step_ : p.step_w;
    auto step_h = p.step_h == 0 ? p.step_ : p.step_h;

    if (step_w == 0 || step_h == 0) {
        step_w = static_cast<float>(img_width) / static_cast<float>(layer_width);
        step_h = static_cast<float>(img_height) / static_cast<float>(layer_height);
    }

    std::vector<float> top_data(dst->size());
    int dim = layer_height * layer_width * num_priors_ * 4;

    float center_x = 0.f;
    float center_y = 0.f;
    float box_width = 0.f;
    float box_height = 0.f;

    size_t idx = 0;
    for (int h = 0; h < layer_height; ++h) {
        for (int w = 0; w < layer_width;  ++w) {
            if (p.step_ == 0) {
                center_x = (static_cast<float>(w) + 0.5f) * p.step_w;
                center_y = (static_cast<float>(h) + 0.5f) * p.step_h;
            } else {
                center_x = (p.offset + static_cast<float>(w)) * p.step_;
                center_y = (p.offset + static_cast<float>(h)) * p.step_;
            }

            for (size_t s = 0; s < p.fixed_sizes.size(); ++s) {
                auto fixed_size_ = static_cast<size_t>(p.fixed_sizes[s]);
                box_width = box_height = fixed_size_ * 0.5f;

                int density_ = 0;
                int shift = 0;
                if (s < p.density.size()) {
                    density_ = static_cast<size_t>(p.density[s]);
                    shift = static_cast<int>(p.fixed_sizes[s] / density_);
                }

                if (!p.fixed_ratios.empty()) {
                    for (const auto& fr : p.fixed_ratios) {
                        const auto box_width_ratio = p.fixed_sizes[s] * 0.5f * std::sqrt(fr);
                        const auto box_height_ratio = p.fixed_sizes[s] * 0.5f / std::sqrt(fr);

                        for (size_t r = 0; r < density_; ++r) {
                            for (size_t c = 0; c < density_; ++c) {
                                const auto center_x_temp = center_x - fixed_size_ / 2 + shift / 2.f + c * shift;
                                const auto center_y_temp = center_y - fixed_size_ / 2 + shift / 2.f + r * shift;

                                top_data[idx++] = std::fmax((center_x_temp - box_width_ratio) * img_width_inv, 0);
                                top_data[idx++] = std::fmax((center_y_temp - box_height_ratio) * img_height_inv, 0.f);
                                top_data[idx++] = std::fmin((center_x_temp + box_width_ratio) * img_width_inv, 1.f);
                                top_data[idx++] = std::fmin((center_y_temp + box_height_ratio) * img_height_inv, 1.f);
                            }
                        }
                    }
                } else {
                    if (!p.density.empty()) {
                        for (int r = 0; r < density_; ++r) {
                            for (int c = 0; c < density_; ++c) {
                                const auto center_x_temp = center_x - fixed_size_ / 2 + shift / 2.f + c * shift;
                                const auto center_y_temp = center_y - fixed_size_ / 2 + shift / 2.f + r * shift;

                                top_data[idx++] = std::fmax((center_x_temp - box_width) * img_width_inv, 0);
                                top_data[idx++] = std::fmax((center_y_temp - box_height) * img_height_inv, 0);
                                top_data[idx++] = std::fmin((center_x_temp + box_width) * img_width_inv, 1);
                                top_data[idx++] = std::fmin((center_y_temp + box_height) * img_height_inv, 1);
                            }
                        }
                    }
                    // Rest of priors
                    for (const auto& ar : p.aspect_ratio) {
                        if (fabs(ar - 1.) < 1e-6) {
                            continue;
                        }

                        const auto box_width_ratio = p.fixed_sizes[s] * 0.5f * std::sqrt(ar);
                        const auto box_height_ratio = p.fixed_sizes[s] * 0.5f / std::sqrt(ar);
                        for (int r = 0; r < density_; ++r) {
                            for (int c = 0; c < density_; ++c) {
                                const auto center_x_temp = center_x - fixed_size_ / 2 + shift / 2.f + c * shift;
                                const auto center_y_temp = center_y - fixed_size_ / 2 + shift / 2.f + r * shift;

                                top_data[idx++] = std::fmax((center_x_temp - box_width_ratio) * img_width_inv, 0);
                                top_data[idx++] = std::fmax((center_y_temp - box_height_ratio) * img_height_inv, 0);
                                top_data[idx++] = std::fmin((center_x_temp + box_width_ratio) * img_width_inv, 1);
                                top_data[idx++] = std::fmin((center_y_temp + box_height_ratio) * img_height_inv, 1);
                            }
                        }
                    }
                }
            }

            for (size_t s = 0; s < p.min_size.size(); ++s) {
                const auto min_size_ = p.min_size[s];

                // first prior: aspect_ratio = 1, size = min_size
                box_width = box_height = min_size_;
                // xmin
                top_data[idx++] = (center_x - box_width / 2.) * img_width_inv;
                // ymin
                top_data[idx++] = (center_y - box_height / 2.) * img_height_inv;
                // xmax
                top_data[idx++] = (center_x + box_width / 2.) * img_width_inv;
                // ymax
                top_data[idx++] = (center_y + box_height / 2.) * img_height_inv;

                if (!p.max_size.empty()) {
                    const auto max_size_ = p.max_size[s];

                    // second prior: aspect_ratio = 1, size = sqrt(min_size * max_size)
                    box_width = box_height = std::sqrt(min_size_ * max_size_);
                    // xmin
                    top_data[idx++] = (center_x - box_width / 2.) * img_width_inv;
                    // ymin
                    top_data[idx++] = (center_y - box_height / 2.) * img_height_inv;
                    // xmax
                    top_data[idx++] = (center_x + box_width / 2.) * img_width_inv;
                    // ymax
                    top_data[idx++] = (center_y + box_height / 2.) * img_height_inv;
                }

                // rest of priors
                for (const auto& ar : aspect_ratios_) {
                    if (fabs(ar - 1.) < 1e-6) {
                        continue;
                    }

                    box_width = min_size_ * std::sqrt(ar);
                    box_height = min_size_ / std::sqrt(ar);

                    // xmin
                    top_data[idx++] = (center_x - box_width / 2.) * img_width_inv;
                    // ymin
                    top_data[idx++] = (center_y - box_height / 2.) * img_height_inv;
                    // xmax
                    top_data[idx++] = (center_x + box_width / 2.) * img_width_inv;
                    // ymax
                    top_data[idx++] = (center_y + box_height / 2.) * img_height_inv;
                }
            }
        }
    }

    auto output_data = static_cast<ie_fp16*>(dst->buffer());

    // clip the prior's coordidate such that it is within [0, 1]
    if (p.clip) {
        for (int d = 0; d < dim; ++d) {
            float val = std::min(std::max(top_data[d], 0.0f), 1.0f);
            output_data[d] = PrecisionUtils::f32tof16(val);
        }
    } else {
        for (int d = 0; d < dim; ++d) {
            output_data[d] = PrecisionUtils::f32tof16(top_data[d]);
        }
    }

    output_data += dst->getTensorDesc().getDims().back();

    // set the variance.
    if (p.variance.empty()) {
        // Set default to 0.1.
        for (int d = 0; d < dim; ++d) {
            output_data[d] = PrecisionUtils::f32tof16(0.1f);
        }
    } else if (p.variance.size() == 1) {
        for (int d = 0; d < dim; ++d) {
            output_data[d] = PrecisionUtils::f32tof16(p.variance[0]);
        }
    } else {
        // Must and only provide 4 variance.
        ASSERT_EQ(4u, p.variance.size());

        idx = 0;
        for (int h = 0; h < layer_height; ++h) {
            for (int w = 0; w < layer_width; ++w) {
                for (int i = 0; i < num_priors_; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        output_data[idx++] = PrecisionUtils::f32tof16(p.variance[j]);
                    }
                }
            }
        }
    }
}

class myriadLayersPriorBoxTests_smoke : public myriadLayersTests_nightly {
public:
    Blob::Ptr getFp16Blob(const Blob::Ptr& in) {
        if (in->getTensorDesc().getPrecision() == Precision::FP16)
            return in;

        auto out = make_shared_blob<ie_fp16>(TensorDesc(Precision::FP16, in->getTensorDesc().getDims(), in->getTensorDesc().getLayout()));
        out->allocate();

        if (in->getTensorDesc().getPrecision() == Precision::FP32) {
            PrecisionUtils::f32tof16Arrays(out->buffer().as<ie_fp16 *>(), in->cbuffer().as<float *>(), in->size());
        } else {
            ADD_FAILURE() << "Unsupported precision " << in->getTensorDesc().getPrecision();
        }

        return out;
    }

    void RunOnModelWithParams(const std::string& model, const std::string& outputName,
                              const PriorBoxParams& params, Precision outPrec = Precision::FP16) {
        SetSeed(DEFAULT_SEED_VALUE + 5);

        StatusCode st;

        ASSERT_NO_THROW(readNetwork(model));

        const auto& network = _cnnNetwork;

        _inputsInfo = network.getInputsInfo();
        _inputsInfo["data1"]->setPrecision(Precision::FP16);
        _inputsInfo["data2"]->setPrecision(Precision::FP16);

        _outputsInfo = network.getOutputsInfo();
        _outputsInfo["data1_copy"]->setPrecision(Precision::FP16);
        _outputsInfo["data2_copy"]->setPrecision(Precision::FP16);
        _outputsInfo[outputName]->setPrecision(outPrec);

        ASSERT_NO_THROW(st = _vpuPluginPtr->LoadNetwork(_exeNetwork, network, {}, &_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
        ASSERT_NE(_exeNetwork, nullptr) << _resp.msg;

        ASSERT_NO_THROW(st = _exeNetwork->CreateInferRequest(_inferRequest, &_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

        Blob::Ptr data1;
        ASSERT_NO_THROW(st = _inferRequest->GetBlob("data1", data1, &_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

        Blob::Ptr data2;
        ASSERT_NO_THROW(st = _inferRequest->GetBlob("data2", data2, &_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

        GenRandomData(data1);
        GenRandomData(data2);

        ASSERT_NO_THROW(st = _inferRequest->Infer(&_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

        Blob::Ptr outputBlob;
        ASSERT_NO_THROW(_inferRequest->GetBlob(outputName.c_str(), outputBlob, &_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

        _refBlob = make_shared_blob<ie_fp16>(TensorDesc(Precision::FP16, outputBlob->getTensorDesc().getDims(), ANY));
        _refBlob->allocate();

        refPriorBox(_refBlob, params);

        CompareCommonAbsolute(getFp16Blob(outputBlob), _refBlob, 0.0);
    }

    void RunOnModel(const std::string& model, const std::string& outputName, Precision outPrec = Precision::FP16) {
        RunOnModelWithParams(model, outputName, PriorBoxParams(), outPrec);
    }
};

TEST_F(myriadLayersPriorBoxTests_smoke, NotLastLayer)
{
    std::string model = R"V0G0N(
        <net name="PriorBox" version="2" batch="1">
            <layers>
                <layer name="data1" type="Input" precision="FP16" id="1">
                    <output>
                        <port id="11">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>300</dim>
                            <dim>300</dim>
                        </port>
                    </output>
                </layer>
                <layer name="data1_copy" type="Power" precision="FP16" id="2">
                    <power_data power="1" scale="1" shift="0"/>
                    <input>
                        <port id="21">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>300</dim>
                            <dim>300</dim>
                        </port>
                    </input>
                    <output>
                        <port id="22">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>300</dim>
                            <dim>300</dim>
                        </port>
                    </output>
                </layer>
                <layer name="data2" type="Input" precision="FP16" id="3">
                    <output>
                        <port id="31">
                            <dim>1</dim>
                            <dim>512</dim>
                            <dim>38</dim>
                            <dim>38</dim>
                        </port>
                    </output>
                </layer>
                <layer name="data2_copy" type="Power" precision="FP16" id="4">
                    <power_data power="1" scale="1" shift="0"/>
                    <input>
                        <port id="41">
                            <dim>1</dim>
                            <dim>512</dim>
                            <dim>38</dim>
                            <dim>38</dim>
                        </port>
                    </input>
                    <output>
                        <port id="42">
                            <dim>1</dim>
                            <dim>512</dim>
                            <dim>38</dim>
                            <dim>38</dim>
                        </port>
                    </output>
                </layer>
                <layer name="priorbox" type="PriorBox" precision="FP16" id="5">
                    <data
                        min_size="21.000000"
                        max_size="45.000000"
                        aspect_ratio="2.000000"
                        flip="1"
                        clip="0"
                        variance="0.100000,0.100000,0.200000,0.200000"
                        img_size="0"
                        img_h="0"
                        img_w="0"
                        step="8.000000"
                        step_h="0.000000"
                        step_w="0.000000"
                        offset="0.500000"
                        width="#"
                        height="#"
                    />
                    <input>
                        <port id="51">
                            <dim>1</dim>
                            <dim>512</dim>
                            <dim>38</dim>
                            <dim>38</dim>
                        </port>
                        <port id="52">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>300</dim>
                            <dim>300</dim>
                        </port>
                    </input>
                    <output>
                        <port id="53">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>23104</dim>
                        </port>
                    </output>
                </layer>
                <layer name="priorbox_copy" type="Power" precision="FP16" id="6">
                    <power_data power="1" scale="1" shift="0"/>
                    <input>
                        <port id="61">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>23104</dim>
                        </port>
                    </input>
                    <output>
                        <port id="62">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>23104</dim>
                        </port>
                    </output>
                </layer>
            </layers>
            <edges>
                <edge from-layer="1" from-port="11" to-layer="2" to-port="21"/>
                <edge from-layer="3" from-port="31" to-layer="4" to-port="41"/>
                <edge from-layer="3" from-port="31" to-layer="5" to-port="51"/>
                <edge from-layer="1" from-port="11" to-layer="5" to-port="52"/>
                <edge from-layer="5" from-port="53" to-layer="6" to-port="61"/>
            </edges>
        </net>
    )V0G0N";

    RunOnModel(model, "priorbox_copy");
}

TEST_F(myriadLayersPriorBoxTests_smoke, LastLayer_FP16)
{
    std::string model = R"V0G0N(
        <net name="PriorBox" version="2" batch="1">
            <layers>
                <layer name="data1" type="Input" precision="FP16" id="1">
                    <output>
                        <port id="11">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>300</dim>
                            <dim>300</dim>
                        </port>
                    </output>
                </layer>
                <layer name="data1_copy" type="Power" precision="FP16" id="2">
                    <power_data power="1" scale="1" shift="0"/>
                    <input>
                        <port id="21">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>300</dim>
                            <dim>300</dim>
                        </port>
                    </input>
                    <output>
                        <port id="22">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>300</dim>
                            <dim>300</dim>
                        </port>
                    </output>
                </layer>
                <layer name="data2" type="Input" precision="FP16" id="3">
                    <output>
                        <port id="31">
                            <dim>1</dim>
                            <dim>512</dim>
                            <dim>38</dim>
                            <dim>38</dim>
                        </port>
                    </output>
                </layer>
                <layer name="data2_copy" type="Power" precision="FP16" id="4">
                    <power_data power="1" scale="1" shift="0"/>
                    <input>
                        <port id="41">
                            <dim>1</dim>
                            <dim>512</dim>
                            <dim>38</dim>
                            <dim>38</dim>
                        </port>
                    </input>
                    <output>
                        <port id="42">
                            <dim>1</dim>
                            <dim>512</dim>
                            <dim>38</dim>
                            <dim>38</dim>
                        </port>
                    </output>
                </layer>
                <layer name="priorbox" type="PriorBox" precision="FP16" id="5">
                    <data
                        min_size="21.000000"
                        max_size="45.000000"
                        aspect_ratio="2.000000"
                        flip="1"
                        clip="0"
                        variance="0.100000,0.100000,0.200000,0.200000"
                        img_size="0"
                        img_h="0"
                        img_w="0"
                        step="8.000000"
                        step_h="0.000000"
                        step_w="0.000000"
                        offset="0.500000"
                        width="#"
                        height="#"
                    />
                    <input>
                        <port id="51">
                            <dim>1</dim>
                            <dim>512</dim>
                            <dim>38</dim>
                            <dim>38</dim>
                        </port>
                        <port id="52">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>300</dim>
                            <dim>300</dim>
                        </port>
                    </input>
                    <output>
                        <port id="53">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>23104</dim>
                        </port>
                    </output>
                </layer>
            </layers>
            <edges>
                <edge from-layer="1" from-port="11" to-layer="2" to-port="21"/>
                <edge from-layer="3" from-port="31" to-layer="4" to-port="41"/>
                <edge from-layer="3" from-port="31" to-layer="5" to-port="51"/>
                <edge from-layer="1" from-port="11" to-layer="5" to-port="52"/>
            </edges>
        </net>
    )V0G0N";

    RunOnModel(model, "priorbox", Precision::FP16);
}

TEST_F(myriadLayersPriorBoxTests_smoke, LastLayer_FP32)
{
    std::string model = R"V0G0N(
        <net name="PriorBox" version="2" batch="1">
            <layers>
                <layer name="data1" type="Input" precision="FP16" id="1">
                    <output>
                        <port id="11">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>300</dim>
                            <dim>300</dim>
                        </port>
                    </output>
                </layer>
                <layer name="data1_copy" type="Power" precision="FP16" id="2">
                    <power_data power="1" scale="1" shift="0"/>
                    <input>
                        <port id="21">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>300</dim>
                            <dim>300</dim>
                        </port>
                    </input>
                    <output>
                        <port id="22">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>300</dim>
                            <dim>300</dim>
                        </port>
                    </output>
                </layer>
                <layer name="data2" type="Input" precision="FP16" id="3">
                    <output>
                        <port id="31">
                            <dim>1</dim>
                            <dim>512</dim>
                            <dim>38</dim>
                            <dim>38</dim>
                        </port>
                    </output>
                </layer>
                <layer name="data2_copy" type="Power" precision="FP16" id="4">
                    <power_data power="1" scale="1" shift="0"/>
                    <input>
                        <port id="41">
                            <dim>1</dim>
                            <dim>512</dim>
                            <dim>38</dim>
                            <dim>38</dim>
                        </port>
                    </input>
                    <output>
                        <port id="42">
                            <dim>1</dim>
                            <dim>512</dim>
                            <dim>38</dim>
                            <dim>38</dim>
                        </port>
                    </output>
                </layer>
                <layer name="priorbox" type="PriorBox" precision="FP16" id="5">
                    <data
                        min_size="21.000000"
                        max_size="45.000000"
                        aspect_ratio="2.000000"
                        flip="1"
                        clip="0"
                        variance="0.100000,0.100000,0.200000,0.200000"
                        img_size="0"
                        img_h="0"
                        img_w="0"
                        step="8.000000"
                        step_h="0.000000"
                        step_w="0.000000"
                        offset="0.500000"
                        width="#"
                        height="#"
                    />
                    <input>
                        <port id="51">
                            <dim>1</dim>
                            <dim>512</dim>
                            <dim>38</dim>
                            <dim>38</dim>
                        </port>
                        <port id="52">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>300</dim>
                            <dim>300</dim>
                        </port>
                    </input>
                    <output>
                        <port id="53">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>23104</dim>
                        </port>
                    </output>
                </layer>
            </layers>
            <edges>
                <edge from-layer="1" from-port="11" to-layer="2" to-port="21"/>
                <edge from-layer="3" from-port="31" to-layer="4" to-port="41"/>
                <edge from-layer="3" from-port="31" to-layer="5" to-port="51"/>
                <edge from-layer="1" from-port="11" to-layer="5" to-port="52"/>
            </edges>
        </net>
    )V0G0N";

    RunOnModel(model, "priorbox", Precision::FP32);
}

TEST_F(myriadLayersTests_nightly, PriorBox_WithConcat)
{
    std::string model = R"V0G0N(
        <net name="PriorBox_WithConcat" version="2" batch="1">
            <layers>
                <layer name="input" type="Input" precision="FP16" id="1">
                    <output>
                        <port id="1">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>300</dim>
                            <dim>300</dim>
                        </port>
                    </output>
                </layer>
                <layer name="input_copy" type="Power" precision="FP16" id="2">
                    <power_data power="1" scale="1" shift="0"/>
                    <input>
                        <port id="2">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>300</dim>
                            <dim>300</dim>
                        </port>
                    </input>
                    <output>
                        <port id="3">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>300</dim>
                            <dim>300</dim>
                        </port>
                    </output>
                </layer>

                <layer name="conv4_3_norm" type="Input" precision="FP16" id="3">
                    <output>
                        <port id="4">
                            <dim>1</dim>
                            <dim>512</dim>
                            <dim>38</dim>
                            <dim>38</dim>
                        </port>
                    </output>
                </layer>
                <layer name="conv4_3_norm_copy" type="Power" precision="FP16" id="4">
                    <power_data power="1" scale="1" shift="0"/>
                    <input>
                        <port id="5">
                            <dim>1</dim>
                            <dim>512</dim>
                            <dim>38</dim>
                            <dim>38</dim>
                        </port>
                    </input>
                    <output>
                        <port id="6">
                            <dim>1</dim>
                            <dim>512</dim>
                            <dim>38</dim>
                            <dim>38</dim>
                        </port>
                    </output>
                </layer>
                <layer name="conv4_3_norm_mbox_priorbox" type="PriorBox" precision="FP16" id="5">
                    <data
                        min_size="21.000000"
                        max_size="45.000000"
                        aspect_ratio="2.000000"
                        flip="1"
                        clip="0"
                        variance="0.100000,0.100000,0.200000,0.200000"
                        img_size="0"
                        img_h="0"
                        img_w="0"
                        step="8.000000"
                        step_h="0.000000"
                        step_w="0.000000"
                        offset="0.500000"
                        width="#"
                        height="#"
                    />
                    <input>
                        <port id="7">
                            <dim>1</dim>
                            <dim>512</dim>
                            <dim>38</dim>
                            <dim>38</dim>
                        </port>
                        <port id="8">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>300</dim>
                            <dim>300</dim>
                        </port>
                    </input>
                    <output>
                        <port id="9">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>23104</dim>
                        </port>
                    </output>
                </layer>

                <layer name="fc7" type="Input" precision="FP16" id="6">
                    <output>
                        <port id="10">
                            <dim>1</dim>
                            <dim>1024</dim>
                            <dim>19</dim>
                            <dim>19</dim>
                        </port>
                    </output>
                </layer>
                <layer name="fc7_copy" type="Power" precision="FP16" id="7">
                    <power_data power="1" scale="1" shift="0"/>
                    <input>
                        <port id="11">
                            <dim>1</dim>
                            <dim>1024</dim>
                            <dim>19</dim>
                            <dim>19</dim>
                        </port>
                    </input>
                    <output>
                        <port id="12">
                            <dim>1</dim>
                            <dim>1024</dim>
                            <dim>19</dim>
                            <dim>19</dim>
                        </port>
                    </output>
                </layer>
                <layer name="fc7_mbox_priorbox" type="PriorBox" precision="FP16" id="8">
                    <data
                        min_size="45.000000"
                        max_size="99.000000"
                        aspect_ratio="2.000000,3.000000"
                        flip="1"
                        clip="0"
                        variance="0.100000,0.100000,0.200000,0.200000"
                        img_size="0"
                        img_h="0"
                        img_w="0"
                        step="16.000000"
                        step_h="0.000000"
                        step_w="0.000000"
                        offset="0.500000"
                        width="#"
                        height="#"
                    />
                    <input>
                        <port id="13">
                            <dim>1</dim>
                            <dim>1024</dim>
                            <dim>19</dim>
                            <dim>19</dim>
                        </port>
                        <port id="14">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>300</dim>
                            <dim>300</dim>
                        </port>
                    </input>
                    <output>
                        <port id="15">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>8664</dim>
                        </port>
                    </output>
                </layer>

                <layer name="conv6_2" type="Input" precision="FP16" id="9">
                    <output>
                        <port id="16">
                            <dim>1</dim>
                            <dim>512</dim>
                            <dim>10</dim>
                            <dim>10</dim>
                        </port>
                    </output>
                </layer>
                <layer name="conv6_2_copy" type="Power" precision="FP16" id="10">
                    <power_data power="1" scale="1" shift="0"/>
                    <input>
                        <port id="17">
                            <dim>1</dim>
                            <dim>512</dim>
                            <dim>10</dim>
                            <dim>10</dim>
                        </port>
                    </input>
                    <output>
                        <port id="18">
                            <dim>1</dim>
                            <dim>512</dim>
                            <dim>10</dim>
                            <dim>10</dim>
                        </port>
                    </output>
                </layer>
                <layer name="conv6_2_mbox_priorbox" type="PriorBox" precision="FP16" id="11">
                    <data
                        min_size="99.000000"
                        max_size="153.000000"
                        aspect_ratio="2.000000,3.000000"
                        flip="1"
                        clip="0"
                        variance="0.100000,0.100000,0.200000,0.200000"
                        img_size="0"
                        img_h="0"
                        img_w="0"
                        step="32.000000"
                        step_h="0.000000"
                        step_w="0.000000"
                        offset="0.500000"
                        width="#"
                        height="#"
                    />
                    <input>
                        <port id="19">
                            <dim>1</dim>
                            <dim>512</dim>
                            <dim>10</dim>
                            <dim>10</dim>
                        </port>
                        <port id="20">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>300</dim>
                            <dim>300</dim>
                        </port>
                    </input>
                    <output>
                        <port id="21">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>2400</dim>
                        </port>
                    </output>
                </layer>

                <layer name="conv7_2" type="Input" precision="FP16" id="12">
                    <output>
                        <port id="22">
                            <dim>1</dim>
                            <dim>256</dim>
                            <dim>5</dim>
                            <dim>5</dim>
                        </port>
                    </output>
                </layer>
                <layer name="conv7_2_copy" type="Power" precision="FP16" id="13">
                    <power_data power="1" scale="1" shift="0"/>
                    <input>
                        <port id="23">
                            <dim>1</dim>
                            <dim>256</dim>
                            <dim>5</dim>
                            <dim>5</dim>
                        </port>
                    </input>
                    <output>
                        <port id="24">
                            <dim>1</dim>
                            <dim>256</dim>
                            <dim>5</dim>
                            <dim>5</dim>
                        </port>
                    </output>
                </layer>
                <layer name="conv7_2_mbox_priorbox" type="PriorBox" precision="FP16" id="14">
                    <data
                        min_size="153.000000"
                        max_size="207.000000"
                        aspect_ratio="2.000000,3.000000"
                        flip="1"
                        clip="0"
                        variance="0.100000,0.100000,0.200000,0.200000"
                        img_size="0"
                        img_h="0"
                        img_w="0"
                        step="64.000000"
                        step_h="0.000000"
                        step_w="0.000000"
                        offset="0.500000"
                        width="#"
                        height="#"
                    />
                    <input>
                        <port id="25">
                            <dim>1</dim>
                            <dim>256</dim>
                            <dim>5</dim>
                            <dim>5</dim>
                        </port>
                        <port id="26">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>300</dim>
                            <dim>300</dim>
                        </port>
                    </input>
                    <output>
                        <port id="27">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>600</dim>
                        </port>
                    </output>
                </layer>

                <layer name="conv8_2" type="Input" precision="FP16" id="15">
                    <output>
                        <port id="28">
                            <dim>1</dim>
                            <dim>256</dim>
                            <dim>3</dim>
                            <dim>3</dim>
                        </port>
                    </output>
                </layer>
                <layer name="conv8_2_copy" type="Power" precision="FP16" id="16">
                    <power_data power="1" scale="1" shift="0"/>
                    <input>
                        <port id="29">
                            <dim>1</dim>
                            <dim>256</dim>
                            <dim>3</dim>
                            <dim>3</dim>
                        </port>
                    </input>
                    <output>
                        <port id="30">
                            <dim>1</dim>
                            <dim>256</dim>
                            <dim>3</dim>
                            <dim>3</dim>
                        </port>
                    </output>
                </layer>
                <layer name="conv8_2_mbox_priorbox" type="PriorBox" precision="FP16" id="17">
                    <data
                        min_size="207.000000"
                        max_size="261.000000"
                        aspect_ratio="2.000000"
                        flip="1"
                        clip="0"
                        variance="0.100000,0.100000,0.200000,0.200000"
                        img_size="0"
                        img_h="0"
                        img_w="0"
                        step="100.000000"
                        step_h="0.000000"
                        step_w="0.000000"
                        offset="0.500000"
                        width="#"
                        height="#"
                    />
                    <input>
                        <port id="31">
                            <dim>1</dim>
                            <dim>256</dim>
                            <dim>3</dim>
                            <dim>3</dim>
                        </port>
                        <port id="32">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>300</dim>
                            <dim>300</dim>
                        </port>
                    </input>
                    <output>
                        <port id="33">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>144</dim>
                        </port>
                    </output>
                </layer>)V0G0N";

    model += R"V0G0N(
                <layer name="conv9_2" type="Input" precision="FP16" id="18">
                    <output>
                        <port id="34">
                            <dim>1</dim>
                            <dim>256</dim>
                            <dim>1</dim>
                            <dim>1</dim>
                        </port>
                    </output>
                </layer>
                <layer name="conv9_2_copy" type="Power" precision="FP16" id="19">
                    <power_data power="1" scale="1" shift="0"/>
                    <input>
                        <port id="35">
                            <dim>1</dim>
                            <dim>256</dim>
                            <dim>1</dim>
                            <dim>1</dim>
                        </port>
                    </input>
                    <output>
                        <port id="36">
                            <dim>1</dim>
                            <dim>256</dim>
                            <dim>1</dim>
                            <dim>1</dim>
                        </port>
                    </output>
                </layer>
                <layer name="conv9_2_mbox_priorbox" type="PriorBox" precision="FP16" id="20">
                    <data
                        min_size="261.000000"
                        max_size="315.000000"
                        aspect_ratio="2.000000"
                        flip="1"
                        clip="0"
                        variance="0.100000,0.100000,0.200000,0.200000"
                        img_size="0"
                        img_h="0"
                        img_w="0"
                        step="300.000000"
                        step_h="0.000000"
                        step_w="0.000000"
                        offset="0.500000"
                        width="#"
                        height="#"
                    />
                    <input>
                        <port id="37">
                            <dim>1</dim>
                            <dim>256</dim>
                            <dim>1</dim>
                            <dim>1</dim>
                        </port>
                        <port id="38">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>300</dim>
                            <dim>300</dim>
                        </port>
                    </input>
                    <output>
                        <port id="39">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>16</dim>
                        </port>
                    </output>
                </layer>

                <layer name="mbox_priorbox" type="Concat" precision="FP16" id="21">
                    <concat_data axis="2"/>
                    <input>
                        <port id="40">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>23104</dim>
                        </port>
                        <port id="41">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>8664</dim>
                        </port>
                        <port id="42">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>2400</dim>
                        </port>
                        <port id="43">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>600</dim>
                        </port>
                        <port id="44">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>144</dim>
                        </port>
                        <port id="45">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>16</dim>
                        </port>
                    </input>
                    <output>
                        <port id="46">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>34928</dim>
                        </port>
                    </output>
                </layer>
                <layer name="mbox_priorbox_copy" type="Power" precision="FP16" id="22">
                    <power_data power="1" scale="1" shift="0"/>
                    <input>
                        <port id="47">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>34928</dim>
                        </port>
                    </input>
                    <output>
                        <port id="48">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>34928</dim>
                        </port>
                    </output>
                </layer>
            </layers>

            <edges>
                <!-- input > input_copy -->
                <edge from-layer="1" from-port="1" to-layer="2" to-port="2"/>

                <!-- conv4_3_norm > conv4_3_norm_copy -->
                <edge from-layer="3" from-port="4" to-layer="4" to-port="5"/>

                <!-- conv4_3_norm > conv4_3_norm_mbox_priorbox -->
                <edge from-layer="3" from-port="4" to-layer="5" to-port="7"/>
                <!-- input > conv4_3_norm_mbox_priorbox -->
                <edge from-layer="1" from-port="1" to-layer="5" to-port="8"/>

                <!-- fc7 > fc7_copy -->
                <edge from-layer="6" from-port="10" to-layer="7" to-port="11"/>

                <!-- fc7 > fc7_mbox_priorbox -->
                <edge from-layer="6" from-port="10" to-layer="8" to-port="13"/>
                <!-- input > fc7_mbox_priorbox -->
                <edge from-layer="1" from-port="1" to-layer="8" to-port="14"/>

                <!-- conv6_2 > conv6_2_copy -->
                <edge from-layer="9" from-port="16" to-layer="10" to-port="17"/>

                <!-- conv6_2 > conv6_2_mbox_priorbox -->
                <edge from-layer="9" from-port="16" to-layer="11" to-port="19"/>
                <!-- input > conv6_2_mbox_priorbox -->
                <edge from-layer="1" from-port="1" to-layer="11" to-port="20"/>

                <!-- conv7_2 > conv7_2_copy -->
                <edge from-layer="12" from-port="22" to-layer="13" to-port="23"/>

                <!-- conv7_2 > conv7_2_mbox_priorbox -->
                <edge from-layer="12" from-port="22" to-layer="14" to-port="25"/>
                <!-- input > conv7_2_mbox_priorbox -->
                <edge from-layer="1" from-port="1" to-layer="14" to-port="26"/>

                <!-- conv8_2 > conv8_2_copy -->
                <edge from-layer="15" from-port="28" to-layer="16" to-port="29"/>

                <!-- conv8_2 > conv8_2_mbox_priorbox -->
                <edge from-layer="15" from-port="28" to-layer="17" to-port="31"/>
                <!-- input > conv8_2_mbox_priorbox -->
                <edge from-layer="1" from-port="1" to-layer="17" to-port="32"/>

                <!-- conv9_2 > conv9_2_copy -->
                <edge from-layer="18" from-port="34" to-layer="19" to-port="35"/>

                <!-- conv9_2 > conv9_2_mbox_priorbox -->
                <edge from-layer="18" from-port="34" to-layer="20" to-port="37"/>
                <!-- input > conv9_2_mbox_priorbox -->
                <edge from-layer="1" from-port="1" to-layer="20" to-port="38"/>

                <!-- conv4_3_norm_mbox_priorbox > mbox_priorbox -->
                <edge from-layer="5" from-port="9" to-layer="21" to-port="40"/>
                <!-- fc7_mbox_priorbox > mbox_priorbox -->
                <edge from-layer="8" from-port="15" to-layer="21" to-port="41"/>
                <!-- conv6_2_mbox_priorbox > mbox_priorbox -->
                <edge from-layer="11" from-port="21" to-layer="21" to-port="42"/>
                <!-- conv7_2_mbox_priorbox > mbox_priorbox -->
                <edge from-layer="14" from-port="27" to-layer="21" to-port="43"/>
                <!-- conv8_2_mbox_priorbox > mbox_priorbox -->
                <edge from-layer="17" from-port="33" to-layer="21" to-port="44"/>
                <!-- conv9_2_mbox_priorbox > mbox_priorbox -->
                <edge from-layer="20" from-port="39" to-layer="21" to-port="45"/>

                <!-- mbox_priorbox > mbox_priorbox_copy -->
                <edge from-layer="21" from-port="46" to-layer="22" to-port="47"/>
            </edges>
        </net>
    )V0G0N";

    StatusCode st;

    ASSERT_NO_THROW(readNetwork(model));

    const auto& network = _cnnNetwork;

    _inputsInfo = network.getInputsInfo();
    _inputsInfo["input"]->setPrecision(Precision::FP16);
    _inputsInfo["conv4_3_norm"]->setPrecision(Precision::FP16);
    _inputsInfo["fc7"]->setPrecision(Precision::FP16);
    _inputsInfo["conv6_2"]->setPrecision(Precision::FP16);
    _inputsInfo["conv7_2"]->setPrecision(Precision::FP16);
    _inputsInfo["conv8_2"]->setPrecision(Precision::FP16);
    _inputsInfo["conv9_2"]->setPrecision(Precision::FP16);

    _outputsInfo = network.getOutputsInfo();
    _outputsInfo["input_copy"]->setPrecision(Precision::FP16);
    _outputsInfo["conv4_3_norm_copy"]->setPrecision(Precision::FP16);
    _outputsInfo["fc7_copy"]->setPrecision(Precision::FP16);
    _outputsInfo["conv6_2_copy"]->setPrecision(Precision::FP16);
    _outputsInfo["conv7_2_copy"]->setPrecision(Precision::FP16);
    _outputsInfo["conv8_2_copy"]->setPrecision(Precision::FP16);
    _outputsInfo["conv9_2_copy"]->setPrecision(Precision::FP16);
    _outputsInfo["mbox_priorbox_copy"]->setPrecision(Precision::FP16);

    ASSERT_NO_THROW(st = _vpuPluginPtr->LoadNetwork(_exeNetwork, network, {}, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
    ASSERT_NE(_exeNetwork, nullptr) << _resp.msg;

    ASSERT_NO_THROW(st = _exeNetwork->CreateInferRequest(_inferRequest, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    ASSERT_NO_THROW(st = _inferRequest->Infer(&_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    // TODO: uncomment this code when GraphTransformer will be updated
    // to optimize out extra copies in case of PriorBox+Concat pair.
#if 0
    {
        std::map<std::string, InferenceEngineProfileInfo> perfMap;
        ASSERT_NO_THROW(st = _inferRequest->GetPerformanceCounts(perfMap, &_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

        int count = 0;
        for (auto p : perfMap) {
            auto layerName = p.first;
            auto status = p.second.status;
            if (layerName.find("mbox_priorbox@copy") == 0) {
                EXPECT_EQ(InferenceEngineProfileInfo::OPTIMIZED_OUT, status) << layerName;
                ++count;
            }
        }
        EXPECT_EQ(6, count);
    }
#endif

    Blob::Ptr outputBlob;
    ASSERT_NO_THROW(_inferRequest->GetBlob("mbox_priorbox_copy", outputBlob, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    auto conv4_3_norm_mbox_priorbox = make_shared_blob<ie_fp16>(TensorDesc(Precision::FP16, {1, 2, 23104}, Layout::ANY));
    {
        conv4_3_norm_mbox_priorbox->allocate();

        PriorBoxParams params;
        params.in1 = {1, 512, 38, 38};
        params.in2 = {1, 3, 300, 300};
        params.min_size = {21.0};
        params.max_size = {45.0};
        params.aspect_ratio = {2.0};
        params.flip = 1;
        params.clip = 0;
        params.variance = {0.1f, 0.1f, 0.2f, 0.2f};
        params.img_size = 0;
        params.img_h =  0;
        params.img_w = 0;
        params.step_ = 8.0;
        params.step_h = 0.0;
        params.step_w = 0.0;
        params.offset = 0.5;

        refPriorBox(conv4_3_norm_mbox_priorbox, params);
    }

    auto fc7_mbox_priorbox = make_shared_blob<ie_fp16>(TensorDesc(Precision::FP16, {1, 2, 8664}, Layout::ANY));
    {
        fc7_mbox_priorbox->allocate();

        PriorBoxParams params;
        params.in1 = {1, 1024, 19, 19};
        params.in2 = {1, 3, 300, 300};
        params.min_size = {45.0};
        params.max_size = {99.0};
        params.aspect_ratio = {2.0, 3.0};
        params.flip = 1;
        params.clip = 0;
        params.variance = {0.1f, 0.1f, 0.2f, 0.2f};
        params.img_size = 0;
        params.img_h =  0;
        params.img_w = 0;
        params.step_ = 16.0;
        params.step_h = 0.0;
        params.step_w = 0.0;
        params.offset = 0.5;

        refPriorBox(fc7_mbox_priorbox, params);
    }

    auto conv6_2_mbox_priorbox = make_shared_blob<ie_fp16>(TensorDesc(Precision::FP16, {1, 2, 2400}, Layout::ANY));
    {
        conv6_2_mbox_priorbox->allocate();

        PriorBoxParams params;
        params.in1 = {1, 512, 10, 10};
        params.in2 = {1, 3, 300, 300};
        params.min_size = {99.0};
        params.max_size = {153.0};
        params.aspect_ratio = {2.0, 3.0};
        params.flip = 1;
        params.clip = 0;
        params.variance = {0.1f, 0.1f, 0.2f, 0.2f};
        params.img_size = 0;
        params.img_h =  0;
        params.img_w = 0;
        params.step_ = 32.0;
        params.step_h = 0.0;
        params.step_w = 0.0;
        params.offset = 0.5;

        refPriorBox(conv6_2_mbox_priorbox, params);
    }

    auto conv7_2_mbox_priorbox = make_shared_blob<ie_fp16>(TensorDesc(Precision::FP16, {1, 2, 600}, Layout::ANY));
    {
        conv7_2_mbox_priorbox->allocate();

        PriorBoxParams params;
        params.in1 = {1, 256, 5, 5};
        params.in2 = {1, 3, 300, 300};
        params.min_size = {153.0};
        params.max_size = {207.0};
        params.aspect_ratio = {2.0, 3.0};
        params.flip = 1;
        params.clip = 0;
        params.variance = {0.1f, 0.1f, 0.2f, 0.2f};
        params.img_size = 0;
        params.img_h =  0;
        params.img_w = 0;
        params.step_ = 64.0;
        params.step_h = 0.0;
        params.step_w = 0.0;
        params.offset = 0.5;

        refPriorBox(conv7_2_mbox_priorbox, params);
    }

    auto conv8_2_mbox_priorbox = make_shared_blob<ie_fp16>(TensorDesc(Precision::FP16, {1, 2, 144}, Layout::ANY));
    {
        conv8_2_mbox_priorbox->allocate();

        PriorBoxParams params;
        params.in1 = {1, 256, 3, 3};
        params.in2 = {1, 3, 300, 300};
        params.min_size = {207.0};
        params.max_size = {261.0};
        params.aspect_ratio = {2.0};
        params.flip = 1;
        params.clip = 0;
        params.variance = {0.1f, 0.1f, 0.2f, 0.2f};
        params.img_size = 0;
        params.img_h =  0;
        params.img_w = 0;
        params.step_ = 100.0;
        params.step_h = 0.0;
        params.step_w = 0.0;
        params.offset = 0.5;

        refPriorBox(conv8_2_mbox_priorbox, params);
    }

    auto conv9_2_mbox_priorbox = make_shared_blob<ie_fp16>(TensorDesc(Precision::FP16, {1, 2, 16}, Layout::ANY));
    {
        conv9_2_mbox_priorbox->allocate();

        PriorBoxParams params;
        params.in1 = {1, 256, 1, 1};
        params.in2 = {1, 3, 300, 300};
        params.min_size = {261.0};
        params.max_size = {315.0};
        params.aspect_ratio = {2.0};
        params.flip = 1;
        params.clip = 0;
        params.variance = {0.1f, 0.1f, 0.2f, 0.2f};
        params.img_size = 0;
        params.img_h =  0;
        params.img_w = 0;
        params.step_ = 300.0;
        params.step_h = 0.0;
        params.step_w = 0.0;
        params.offset = 0.5;

        refPriorBox(conv9_2_mbox_priorbox, params);
    }

    _refBlob = make_shared_blob<ie_fp16>(TensorDesc(Precision::FP16, {1, 2, 34928}, ANY));
    _refBlob->allocate();
    {
        ie_fp16* dst_ptr = _refBlob->buffer().as<ie_fp16*>();
        int dst_stride = _refBlob->getTensorDesc().getDims().back();

        int dst_offset = 0;

        auto concat = [&](const Blob::Ptr& src) {
            const ie_fp16* src_ptr = src->cbuffer().as<const ie_fp16*>();
            int num = src->getTensorDesc().getDims().back();

            for (int y = 0; y < 2; ++y) {
                for (int x = 0; x < num; ++x) {
                    dst_ptr[dst_offset + x + y * dst_stride] = src_ptr[x + y * num];
                }
            }

            dst_offset += num;
        };

        concat(conv4_3_norm_mbox_priorbox);
        concat(fc7_mbox_priorbox);
        concat(conv6_2_mbox_priorbox);
        concat(conv7_2_mbox_priorbox);
        concat(conv8_2_mbox_priorbox);
        concat(conv9_2_mbox_priorbox);
    }

    CompareCommonAbsolute(_refBlob, outputBlob, 0.0);
}

TEST_F(myriadLayersPriorBoxTests_smoke, FaceBoxLayer)
{
    std::string model = R"V0G0N(
        <net name="PriorBox" version="2" batch="1">
            <layers>
                <layer name="data1" type="Input" precision="FP16" id="1">
                    <output>
                        <port id="11">
                            <dim>1</dim>
                            <dim>128</dim>
                            <dim>32</dim>
                            <dim>32</dim>
                        </port>
                    </output>
                </layer>
                <layer name="data1_copy" type="Power" precision="FP16" id="2">
                    <power_data power="1" scale="1" shift="0"/>
                    <input>
                        <port id="21">
                            <dim>1</dim>
                            <dim>128</dim>
                            <dim>32</dim>
                            <dim>32</dim>
                        </port>
                    </input>
                    <output>
                        <port id="22">
                            <dim>1</dim>
                            <dim>128</dim>
                            <dim>32</dim>
                            <dim>32</dim>
                        </port>
                    </output>
                </layer>
                <layer name="data2" type="Input" precision="FP16" id="3">
                    <output>
                        <port id="31">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>1024</dim>
                            <dim>1024</dim>
                        </port>
                    </output>
                </layer>
                <layer name="data2_copy" type="Power" precision="FP16" id="4">
                    <power_data power="1" scale="1" shift="0"/>
                    <input>
                        <port id="41">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>1024</dim>
                            <dim>1024</dim>
                        </port>
                    </input>
                    <output>
                        <port id="42">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>1024</dim>
                            <dim>1024</dim>
                        </port>
                    </output>
                </layer>
                <layer name="priorbox" type="PriorBox" precision="FP16" id="5">
                    <data
                        aspect_ratio=""
                        clip="0"
                        density="4.0,2.0,1.0"
                        fixed_ratio=""
                        fixed_size="32.0,64.0,128.0"
                        flip="1"
                        max_size=""
                        min_size=""
                        offset="0.5"
                        step="32.0"
                        variance="0.10000000149011612,0.10000000149011612,0.20000000298023224,0.20000000298023224"
                    />
                    <input>
                        <port id="51">
                            <dim>1</dim>
                            <dim>128</dim>
                            <dim>32</dim>
                            <dim>32</dim>
                        </port>
                        <port id="52">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>1024</dim>
                            <dim>1024</dim>
                        </port>
                    </input>
                    <output>
                        <port id="53">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>86016</dim>
                        </port>
                    </output>
                </layer>
            </layers>
            <edges>
                <edge from-layer="1" from-port="11" to-layer="2" to-port="21"/>
                <edge from-layer="3" from-port="31" to-layer="4" to-port="41"/>
                <edge from-layer="1" from-port="11" to-layer="5" to-port="51"/>
                <edge from-layer="3" from-port="31" to-layer="5" to-port="52"/>
            </edges>
        </net>
    )V0G0N";

    PriorBoxParams params;
    params.in1 = {1, 128, 32, 32};
    params.in2 = {1, 3, 1024, 1024};
    params.min_size = {};
    params.max_size = {};
    params.aspect_ratio = {};
    params.flip = 1;
    params.clip = 0;
    params.variance = {0.10000000149011612, 0.10000000149011612, 0.20000000298023224, 0.20000000298023224};
    params.img_size = 0;
    params.img_h =  0;
    params.img_w = 0;
    params.step_ = 32.0;
    params.step_h = 0.0;
    params.step_w = 0.0;
    params.offset = 0.5;
    params.density = {4.0, 2.0, 1.0};
    params.fixed_sizes = {32.0, 64.0, 128.0};
    params.fixed_ratios = {};

    RunOnModelWithParams(model, "priorbox", params, Precision::FP16);
}

TEST_F(myriadLayersPriorBoxTests_smoke, TwoPriorBoxLayersWithUnusedInput)
{
    std::string model = R"V0G0N(
        <net name="PriorBox" version="2" batch="1">
            <layers>
                <layer name="data1" type="Input" precision="FP16" id="1">
                    <output>
                        <port id="11">
                            <dim>3</dim>
                            <dim>300</dim>
                            <dim>300</dim>
                        </port>
                    </output>
                </layer>
                <layer name="data1_reshaped" type="Reshape" precision="FP16" id="2">
                    <input>
                        <port id="21">
                            <dim>3</dim>
                            <dim>300</dim>
                            <dim>300</dim>
                        </port>
                    </input>
                    <output>
                        <port id="22">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>300</dim>
                            <dim>300</dim>
                        </port>
                    </output>
                </layer>
                <layer name="data1_copy" type="Power" precision="FP16" id="3">
                    <power_data power="1" scale="1" shift="0"/>
                    <input>
                        <port id="31">
                            <dim>3</dim>
                            <dim>300</dim>
                            <dim>300</dim>
                        </port>
                    </input>
                    <output>
                        <port id="32">
                            <dim>3</dim>
                            <dim>300</dim>
                            <dim>300</dim>
                        </port>
                    </output>
                </layer>
                <layer name="data2" type="Input" precision="FP16" id="4">
                    <output>
                        <port id="41">
                            <dim>1</dim>
                            <dim>512</dim>
                            <dim>38</dim>
                            <dim>38</dim>
                        </port>
                    </output>
                </layer>
                <layer name="data2_copy" type="Power" precision="FP16" id="5">
                    <power_data power="1" scale="1" shift="0"/>
                    <input>
                        <port id="51">
                            <dim>1</dim>
                            <dim>512</dim>
                            <dim>38</dim>
                            <dim>38</dim>
                        </port>
                    </input>
                    <output>
                        <port id="52">
                            <dim>1</dim>
                            <dim>512</dim>
                            <dim>38</dim>
                            <dim>38</dim>
                        </port>
                    </output>
                </layer>
                <layer name="priorbox1" type="PriorBox" precision="FP16" id="6">
                    <data
                        min_size="21.000000"
                        max_size="45.000000"
                        aspect_ratio="2.000000"
                        flip="1"
                        clip="0"
                        variance="0.100000,0.100000,0.200000,0.200000"
                        img_size="0"
                        img_h="0"
                        img_w="0"
                        step="8.000000"
                        step_h="0.000000"
                        step_w="0.000000"
                        offset="0.500000"
                        width="#"
                        height="#"
                    />
                    <input>
                        <port id="61">
                            <dim>1</dim>
                            <dim>512</dim>
                            <dim>38</dim>
                            <dim>38</dim>
                        </port>
                        <port id="62">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>300</dim>
                            <dim>300</dim>
                        </port>
                    </input>
                    <output>
                        <port id="63">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>23104</dim>
                        </port>
                    </output>
                </layer>
                <layer name="priorbox2" type="PriorBox" precision="FP16" id="7">
                    <data
                        min_size="21.000000"
                        max_size="45.000000"
                        aspect_ratio="2.000000"
                        flip="1"
                        clip="0"
                        variance="0.100000,0.100000,0.200000,0.200000"
                        img_size="0"
                        img_h="0"
                        img_w="0"
                        step="8.000000"
                        step_h="0.000000"
                        step_w="0.000000"
                        offset="0.500000"
                        width="#"
                        height="#"
                    />
                    <input>
                        <port id="71">
                            <dim>1</dim>
                            <dim>512</dim>
                            <dim>38</dim>
                            <dim>38</dim>
                        </port>
                        <port id="72">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>300</dim>
                            <dim>300</dim>
                        </port>
                    </input>
                    <output>
                        <port id="73">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>23104</dim>
                        </port>
                    </output>
                </layer>
            </layers>
            <edges>
                <edge from-layer="1" from-port="11" to-layer="2" to-port="21"/>
                <edge from-layer="1" from-port="11" to-layer="3" to-port="31"/>
                <edge from-layer="4" from-port="41" to-layer="5" to-port="51"/>
                <edge from-layer="4" from-port="41" to-layer="6" to-port="61"/>
                <edge from-layer="4" from-port="41" to-layer="7" to-port="71"/>
                <edge from-layer="2" from-port="22" to-layer="6" to-port="62"/>
                <edge from-layer="2" from-port="22" to-layer="7" to-port="72"/>
            </edges>
        </net>
    )V0G0N";

    SetSeed(DEFAULT_SEED_VALUE + 5);

    StatusCode st;

    ASSERT_NO_THROW(readNetwork(model));

    const auto& network = _cnnNetwork;

    _inputsInfo = network.getInputsInfo();
    _inputsInfo["data1"]->setPrecision(Precision::FP16);
    _inputsInfo["data2"]->setPrecision(Precision::FP16);

    _outputsInfo = network.getOutputsInfo();
    _outputsInfo["data1_copy"]->setPrecision(Precision::FP16);
    _outputsInfo["data2_copy"]->setPrecision(Precision::FP16);
    _outputsInfo["priorbox1"]->setPrecision(Precision::FP16);
    _outputsInfo["priorbox2"]->setPrecision(Precision::FP16);

    ASSERT_NO_THROW(st = _vpuPluginPtr->LoadNetwork(_exeNetwork, network, {}, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
    ASSERT_NE(_exeNetwork, nullptr) << _resp.msg;

    ASSERT_NO_THROW(st = _exeNetwork->CreateInferRequest(_inferRequest, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    Blob::Ptr data1;
    ASSERT_NO_THROW(st = _inferRequest->GetBlob("data1", data1, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    Blob::Ptr data2;
    ASSERT_NO_THROW(st = _inferRequest->GetBlob("data2", data2, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    GenRandomData(data1);
    GenRandomData(data2);

    ASSERT_NO_THROW(st = _inferRequest->Infer(&_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    Blob::Ptr outputBlob1;
    Blob::Ptr outputBlob2;
    ASSERT_NO_THROW(_inferRequest->GetBlob("priorbox1", outputBlob1, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
    ASSERT_NO_THROW(_inferRequest->GetBlob("priorbox2", outputBlob2, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    _refBlob = make_shared_blob<ie_fp16>(TensorDesc(Precision::FP16, outputBlob1->getTensorDesc().getDims(), ANY));
    _refBlob->allocate();

    refPriorBox(_refBlob, PriorBoxParams());

    CompareCommonAbsolute(getFp16Blob(outputBlob1), _refBlob, 0.0);
    CompareCommonAbsolute(getFp16Blob(outputBlob2), _refBlob, 0.0);
}
