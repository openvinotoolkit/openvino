// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "list.hpp"
#include "base.hpp"
#include "caseless.hpp"

#include <cmath>
#include <string>
#include <vector>
#include <cassert>
#include <set>
#include "ie_parallel.hpp"
#include <ngraph/opsets/opset3.hpp>
#include <utils/general_utils.h>

using namespace MKLDNNPlugin;

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class ExtractImagePatchesImpl : public ExtLayerBase {
    enum class ExtImgPatcherPadType {
        VALID,
        SAME_LOWER,
        SAME_UPPER
    };

    bool isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept {
        try {
            const auto extImgPatcher = std::dynamic_pointer_cast<const ngraph::opset3::ExtractImagePatches>(op);
            if (!extImgPatcher) {
                errorMessage = "Only opset3 ExtractImagePatches operation is supported";
                return false;
            }
            const auto padValue = extImgPatcher->get_auto_pad();
            if (!one_of(padValue, ngraph::op::PadType::VALID, ngraph::op::PadType::SAME_LOWER, ngraph::op::PadType::SAME_UPPER)) {
                errorMessage = "Does not support pad type: " + ngraph::as_string(padValue);
                return false;
            }
            if (!everyone_is(2, extImgPatcher->get_sizes().size(), extImgPatcher->get_strides().size(), extImgPatcher->get_rates().size())) {
                errorMessage = "Doesn't support 'sizes', 'strides', 'rates', attributes with rank != 2";
                return false;
            }
        } catch (...) {
            return false;
        }
        return true;
    }

    std::string errorPrefix;

public:
    explicit ExtractImagePatchesImpl(const std::shared_ptr<ngraph::Node>& op) {
        try {
            std::string errorMessage;
            if (!isSupportedOperation(op, errorMessage)) {
                IE_THROW(NotImplemented) << errorMessage;
            }

            errorPrefix = "ExtractImagePatches layer with name '" + op->get_friendly_name() + "' ";
            const auto extImgPatcher = std::dynamic_pointer_cast<const ngraph::opset3::ExtractImagePatches>(op);

            if (op->get_input_size() != 1 || op->get_output_size() != 1)
                IE_THROW() << errorPrefix << "has incorrect number of input or output edges!"
                    << " Input: " << op->get_input_size() << "; Output: " << op->get_output_size();

            if (op->get_input_shape(0).size() != 4)
                IE_THROW() << errorPrefix << "must have 4D input tensor. Actual: " << op->get_input_shape(0).size();

            if (op->get_output_shape(0).size() != 4)
                IE_THROW() << errorPrefix << "must have 4D output tensor. Actual: " << op->get_output_shape(0).size();

            const auto precision = details::convertPrecision(op->get_input_element_type(0));
            if (_supported_precisions_sizes.find(precision.size()) == _supported_precisions_sizes.end())
                IE_THROW() << errorPrefix << "has unsupported precision: " << precision.name();

            auto ksizes = extImgPatcher->get_sizes();
            auto strides = extImgPatcher->get_strides();
            auto rates = extImgPatcher->get_rates();
            if (extImgPatcher->get_auto_pad() == ngraph::op::PadType::VALID) {
                _auto_pad = ExtImgPatcherPadType::VALID;
            } else if (extImgPatcher->get_auto_pad() == ngraph::op::PadType::SAME_LOWER) {
                _auto_pad = ExtImgPatcherPadType::SAME_LOWER;
            } else if (extImgPatcher->get_auto_pad() == ngraph::op::PadType::SAME_UPPER) {
                _auto_pad = ExtImgPatcherPadType::SAME_UPPER;
            } else {
                IE_THROW() << errorPrefix << "has unsupported pad type: " << extImgPatcher->get_auto_pad();
            }

            _ksizes.clear();
            _strides.clear();
            _rates.clear();
            for (size_t i = 0; i < ksizes.size(); i++)
                _ksizes.push_back((int64_t)ksizes[i]);
            for (size_t i = 0; i < strides.size(); i++)
                _strides.push_back((int64_t)strides[i]);
            for (size_t i = 0; i < rates.size(); i++)
                _rates.push_back((int64_t)rates[i]);

            addConfig(op, {{TensorDescCreatorTypes::ncsp, precision}},
                          {{TensorDescCreatorTypes::ncsp, precision}});
        } catch (InferenceEngine::Exception &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) noexcept override {
        switch (inputs[0]->getTensorDesc().getPrecision().size()) {
            case 1: {
                process_data<PrecisionTrait<Precision::U8>::value_type>(inputs, outputs);
                break;
            }
            case 2: {
                process_data<PrecisionTrait<Precision::U16>::value_type>(inputs, outputs);
                break;
            }
            case 4: {
                process_data<PrecisionTrait<Precision::I32>::value_type>(inputs, outputs);
                break;
            }
            case 8: {
                process_data<PrecisionTrait<Precision::U64>::value_type>(inputs, outputs);
                break;
            }
            default: {
                if (resp) {
                    std::string errorMsg = "ExtractImagePatches layer does not support precision '"
                            + std::string(inputs[0]->getTensorDesc().getPrecision().name()) + "'";
                    errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
                }
                return GENERAL_ERROR;
            }
        }

        return OK;
    }

    template<typename T>
    void process_data(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs) noexcept {
        const T* src_data = inputs[0]->cbuffer().as<const T*>() +
            inputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        T* dst_data = outputs[0]->buffer().as<T*>() +
            outputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();

        const auto& inDims = inputs[0]->getTensorDesc().getDims();
        const size_t inDimsSize = inDims.size();

        const size_t BATCH = 0, CHANNEL = 1, HIGHT = 0, WIDTH = 1;

        const int64_t IC = inDims[CHANNEL];
        const int64_t IH = inDims[inDimsSize - 2];
        const int64_t IW = inDims[inDimsSize - 1];

        const auto& outDims = outputs[0]->getTensorDesc().getDims();
        const size_t outDimsSize = outDims.size();

        const int64_t OB = outDims[BATCH];
        const int64_t OC = outDims[CHANNEL];
        const int64_t OH = outDims[outDimsSize - 2];
        const int64_t OW = outDims[outDimsSize - 1];

        const int64_t KH = _ksizes[HIGHT];
        const int64_t KW = _ksizes[WIDTH];
        const int64_t SH = _strides[HIGHT];
        const int64_t SW = _strides[WIDTH];
        const int64_t RH = _rates[HIGHT];
        const int64_t RW = _rates[WIDTH];

        int64_t iwStep = KW + (RW - 1) * (KW - 1);
        int64_t ihStep = KH + (RH - 1) * (KH - 1);

        int64_t PL = 0, PT = 0;
        if (_auto_pad != ExtImgPatcherPadType::VALID) {
            int64_t PW = (std::ceil(1.f * IW/SW) - 1) * SW + iwStep - IW;
            int64_t PH = (std::ceil(1.f * IH/SH) - 1) * SH + ihStep - IH;

            if ((PW > 0) && (PW < iwStep)) {
                if (PW % 2 == 1) {
                    if (_auto_pad == ExtImgPatcherPadType::SAME_LOWER) {
                        PL = (PW + 1) / 2;
                    } else if (_auto_pad == ExtImgPatcherPadType::SAME_UPPER) {
                        PL = (PW - 1) / 2;
                    }
                } else {
                    PL = PW / 2;
                }
            }
            if ((PH > 0) && (PH < ihStep)) {
                if (PH % 2 == 1) {
                    if (_auto_pad == ExtImgPatcherPadType::SAME_LOWER) {
                        PT = (PH + 1) / 2;
                    } else if (_auto_pad == ExtImgPatcherPadType::SAME_UPPER) {
                        PT = (PH - 1) / 2;
                    }
                } else {
                    PT = PH / 2;
                }
            }
        }

        const int64_t OH_OW = OH * OW;
        const int64_t OC_OH_OW = OC * OH_OW;
        const int64_t IH_IW = IH * IW;
        const int64_t IC_IH_IW = IC * IH_IW;

        const int64_t work_amount = OB;

        auto thread_body = [&](const int ithr, const int nthr) {
            int64_t start(0lu), end(0lu);
            splitter(work_amount, nthr, ithr, start, end);
            if (start >= end)
                return;

            for (int64_t ob = start; ob < end; ob++) {
                const int64_t ibICIHIW = ob * IC_IH_IW;
                const int64_t obOCOHOW = ob * OC_OH_OW;
                for (int64_t oh = 0; oh < OH; oh++) {
                    const int64_t obOCOHOWohOW = obOCOHOW + oh * OW;
                    int64_t ih0 = oh * SH - PT;
                    for (int64_t ow = 0; ow < OW; ow++) {
                        const int64_t obOCOHOWohOWow = obOCOHOWohOW + ow;
                        int64_t iw0 = ow * SW - PL;
                        int64_t oc = 0;

                        for (int64_t kh = 0; kh < KH; kh++) {
                            int64_t ihKH = ih0 + kh * RH;
                            int64_t ibICIHIWihFHIW = ibICIHIW + ihKH * IW;
                            for (int64_t kw = 0; kw < KW; kw++) {
                                for (int64_t ic = 0; ic < IC; ic++, oc++) {
                                    int64_t iwKW = iw0 + kw * RW;
                                    int64_t dst_idx = obOCOHOWohOWow + oc * OH_OW;
                                    if (ihKH < 0 || ihKH >= IH || iwKW < 0 || iwKW >= IW) {
                                        dst_data[dst_idx] = T(0);
                                    } else {
                                        int64_t src_idx = ibICIHIWihFHIW + ic * IH_IW + iwKW;
                                        dst_data[dst_idx] = src_data[src_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        };

        parallel_nt(0, thread_body);
    }

private:
    std::vector<int64_t> _ksizes;
    std::vector<int64_t> _strides;
    std::vector<int64_t> _rates;
    ExtImgPatcherPadType _auto_pad;

    static const std::set<size_t> _supported_precisions_sizes;
};

const std::set<size_t> ExtractImagePatchesImpl::_supported_precisions_sizes = {1, 2, 4, 8};

REG_FACTORY_FOR(ExtractImagePatchesImpl, ExtractImagePatches);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
