// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base.hpp"

#include <cmath>
#include <string>
#include <vector>
#include <cassert>
#include "ie_parallel.hpp"
#include <ngraph/opsets/opset1.hpp>
#include <utils/general_utils.h>

using namespace MKLDNNPlugin;

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {



class RangeImpl: public ExtLayerBase {
    bool isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept {
        try {
            if (!MKLDNNPlugin::one_of(op->get_type_info(), ngraph::op::v0::Range::type_info, ngraph::op::v4::Range::type_info)) {
                errorMessage = "Only opset1 and opset4 Range operation is supported";
                return false;
            }
            if (std::dynamic_pointer_cast<const ngraph::opset1::Constant>(op->get_input_node_shared_ptr(RANGE_START)) == nullptr ||
                std::dynamic_pointer_cast<const ngraph::opset1::Constant>(op->get_input_node_shared_ptr(RANGE_LIMIT)) == nullptr ||
                    std::dynamic_pointer_cast<const ngraph::opset1::Constant>(op->get_input_node_shared_ptr(RANGE_DELTA)) == nullptr) {
                errorMessage = "Only const inputs for Range operation is supported";
                return false;
            }
        } catch (...) {
            return false;
        }
        return true;
    }

    std::string errorPrefix;

public:
    explicit RangeImpl(const std::shared_ptr<ngraph::Node>& op) {
        try {
            std::string errorMessage;
            if (!isSupportedOperation(op, errorMessage)) {
                IE_THROW(NotImplemented) << errorMessage;
            }

            errorPrefix = "Range layer with name '" + op->get_friendly_name() + "'";

            if (op->get_input_size() != 3 || op->get_output_size() != 1)
                IE_THROW() << errorPrefix << " has incorrect number of input/output edges!";

            SizeVector start_dims = op->get_input_shape(RANGE_START);
            if (ngraph::shape_size(start_dims) != 1)
                IE_THROW() << errorPrefix << " has start scalar with more than 1 value";

            SizeVector limit_dims = op->get_input_shape(RANGE_LIMIT);
            if (ngraph::shape_size(limit_dims) != 1)
                IE_THROW() << errorPrefix << " has limit scalar with more than 1 value";

            SizeVector delta_dims = op->get_input_shape(RANGE_DELTA);
            if (ngraph::shape_size(delta_dims) != 1)
                IE_THROW() << errorPrefix << " has delta scalar with more than 1 value";

            SizeVector dst_dims = op->get_output_shape(0);
            if (dst_dims.size() > 1)
                IE_THROW() << errorPrefix << " has unsupported rank for output: " << dst_dims.size();

            if (!(details::convertPrecision(op->get_input_element_type(RANGE_START)) == Precision::I32 &&
                  details::convertPrecision(op->get_input_element_type(RANGE_LIMIT)) == Precision::I32 &&
                  details::convertPrecision(op->get_input_element_type(RANGE_DELTA)) == Precision::I32 &&
                  details::convertPrecision(op->get_output_element_type(0)) == Precision::I32) &&
                !(details::convertPrecision(op->get_input_element_type(RANGE_START)) == Precision::FP32 &&
                  details::convertPrecision(op->get_input_element_type(RANGE_LIMIT)) == Precision::FP32 &&
                  details::convertPrecision(op->get_input_element_type(RANGE_DELTA)) == Precision::FP32 &&
                  details::convertPrecision(op->get_output_element_type(0)) == Precision::FP32)) {
                      addConfig(op, {{TensorDescCreatorTypes::ncsp, Precision::FP32},
                                     {TensorDescCreatorTypes::ncsp, Precision::FP32},
                                     {TensorDescCreatorTypes::ncsp, Precision::FP32}},
                                    {{TensorDescCreatorTypes::ncsp, Precision::FP32}});
            } else {
                addConfig(op, {{TensorDescCreatorTypes::ncsp},
                               {TensorDescCreatorTypes::ncsp},
                               {TensorDescCreatorTypes::ncsp}},
                              {{TensorDescCreatorTypes::ncsp}});
            }
        } catch (InferenceEngine::Exception &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) noexcept override {
        StatusCode retcode = OK;
        switch (outputs[0]->getTensorDesc().getPrecision()) {
        case Precision::FP32: {
            retcode = range((inputs[RANGE_START]->cbuffer().as<float *>() +
                             inputs[RANGE_START]->getTensorDesc().getBlockingDesc().getOffsetPadding())[0],
                            (inputs[RANGE_LIMIT]->cbuffer().as<float *>() +
                             inputs[RANGE_LIMIT]->getTensorDesc().getBlockingDesc().getOffsetPadding())[0],
                            (inputs[RANGE_DELTA]->cbuffer().as<float *>() +
                             inputs[RANGE_DELTA]->getTensorDesc().getBlockingDesc().getOffsetPadding())[0], outputs[0]);
        }
        break;
        case Precision::I32: {
            retcode = range((inputs[RANGE_START]->cbuffer().as<int32_t *>() +
                             inputs[RANGE_START]->getTensorDesc().getBlockingDesc().getOffsetPadding())[0],
                            (inputs[RANGE_LIMIT]->cbuffer().as<int32_t *>() +
                             inputs[RANGE_LIMIT]->getTensorDesc().getBlockingDesc().getOffsetPadding())[0],
                            (inputs[RANGE_DELTA]->cbuffer().as<int32_t *>() +
                             inputs[RANGE_DELTA]->getTensorDesc().getBlockingDesc().getOffsetPadding())[0], outputs[0]);
        }
        break;
        default:
            if (resp) {
                std::string errorMsg = "Incorrect output precision. Only FP32 and I32 are supported!";
                errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
            }
            retcode = GENERAL_ERROR;
        }
        if (resp && retcode == PARAMETER_MISMATCH) {
            std::string errorMsg = "Range indexes exceeds data tensor dimension";
            errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
        }
        return retcode;
    }

private:
    static const size_t RANGE_START = 0;
    static const size_t RANGE_LIMIT = 1;
    static const size_t RANGE_DELTA = 2;

    template <typename data_t>
    StatusCode range(data_t start, data_t limit, data_t delta, Blob::Ptr output);
};

template <typename data_t>
StatusCode RangeImpl::range(data_t start, data_t limit, data_t delta, Blob::Ptr output) {
    size_t dst_size = (output->getTensorDesc().getDims())[0];
    data_t* dst_data = output->cbuffer().as<data_t *>() +
                       output->getTensorDesc().getBlockingDesc().getOffsetPadding();
    size_t work_amount_dst = static_cast<size_t>(std::floor(std::abs((limit - start) / delta)));
    if (work_amount_dst != dst_size)
        return PARAMETER_MISMATCH;

    parallel_nt(0, [&](const int ithr, const int nthr) {
        size_t iwork = 0, end = 0;
        splitter(work_amount_dst, nthr, ithr, iwork, end);
        data_t dst_value = start + iwork * delta;

        for (; iwork < end; ++iwork, dst_value += delta) {
            dst_data[iwork] = dst_value;
        }
    });
    return OK;
}
REG_FACTORY_FOR(RangeImpl, Range);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
