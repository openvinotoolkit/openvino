// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base.hpp"
#include <ngraph/op/gather_elements.hpp>
#include <nodes/common/tensor_desc_creator.h>
#include <utils/general_utils.h>

#include <string>
#include <vector>
#include "ie_parallel.hpp"

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

using MKLDNNPlugin::TensorDescCreatorTypes;

class GatherElementsImpl: public ExtLayerBase {
public:
    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
        try {
            auto gatherElementsOp = ngraph::as_type_ptr<const ngraph::op::v6::GatherElements>(op);
            if (!gatherElementsOp) {
                errorMessage = "Node is not an instance of the GatherElements operation from operation set v6.";
                return false;
            }
        } catch (...) {
            return false;
        }

        return true;
    }

    explicit GatherElementsImpl(const std::shared_ptr<ngraph::Node>& op) : strideAx1Diff_(0) {
        try {
            std::string errorMessage;
            if (!isSupportedOperation(op, errorMessage)) {
                IE_THROW(NotImplemented) << errorMessage;
            }
            errorPrefix_ = std::string("Layer GatherElements with name '") + op->get_friendly_name() + "'";

            if (op->get_input_size() != 2 || op->get_output_size() != 1)
                IE_THROW() << errorPrefix_ << " has invalid number of input/output edges.";

            const auto& dataDims = op->get_input_shape(dataIndex_);
            const auto& indicesDims = op->get_input_shape(indicesIndex_);
            if (dataDims.size() != indicesDims.size())
                IE_THROW() << errorPrefix_ << " has invalid input shapes. Inputs 'Data' and 'Indices' must have equal ranks.";

            Precision inDataPrecision = details::convertPrecision(op->get_input_element_type(dataIndex_));
            if (!MKLDNNPlugin::one_of(inDataPrecision.size(),
                    sizeof(PrecisionTrait<Precision::I32>::value_type),
                    sizeof(PrecisionTrait<Precision::I16>::value_type),
                    sizeof(PrecisionTrait<Precision::I8>::value_type))) {
                IE_THROW() << errorPrefix_ << " has unsupported 'inputData' input precision: " << inDataPrecision;
            }

            Precision indicesPrecision = details::convertPrecision(op->get_input_element_type(indicesIndex_));
            if (!MKLDNNPlugin::one_of(indicesPrecision, Precision::I32, Precision::I64)) {
                IE_THROW() << errorPrefix_ << " has unsupported 'indices' input precision: " << indicesPrecision;
            }

            dataTypeSize_ = inDataPrecision.size();

            auto gatherElementsOp = ngraph::as_type_ptr<const ngraph::op::v6::GatherElements>(op);
            auto axis = gatherElementsOp->get_axis();
            if (axis < 0)
                axis += dataDims.size();
            if (axis < 0 || axis >= static_cast<int>(dataDims.size()))
                IE_THROW() << errorPrefix_ << " has invalid axis attribute: " << axis;
            axis_ = axis;

            auto outputShape = op->get_output_shape(0);
            strideAxDst_ = 1;
            for (int i = outputShape.size() - 1; i > axis_; i--)
                strideAxDst_ *= outputShape[i];
            dstAxDim_ = op->get_output_shape(0)[axis_];
            if (axis_ > 0) {
                strideAx1Diff_ = 1;
                for (int i = dataDims.size() - 1; i >= axis_; i--)
                    strideAx1Diff_ *= dataDims[i];
                strideAx1Diff_ -= strideAxDst_ * outputShape[axis_];
            }

            addConfig(op, {{TensorDescCreatorTypes::ncsp, inDataPrecision},
                           {TensorDescCreatorTypes::ncsp, Precision::I32}},
                          {{TensorDescCreatorTypes::ncsp, inDataPrecision}});
        } catch (InferenceEngine::Exception &ex) {
            errorMsg = ex.what();
            throw;
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) noexcept override {
        switch (dataTypeSize_) {
            case sizeof(PrecisionTrait<Precision::I32>::value_type):
                return directExecution<PrecisionTrait<Precision::I32>::value_type>(inputs, outputs, resp);
            case sizeof(PrecisionTrait<Precision::I16>::value_type):
                return directExecution<PrecisionTrait<Precision::I16>::value_type>(inputs, outputs, resp);
            case sizeof(PrecisionTrait<Precision::I8>::value_type):
                return directExecution<PrecisionTrait<Precision::I8>::value_type>(inputs, outputs, resp);
            default:
                std::string errMsg = errorPrefix_ + " has inputData input with unsupported precision: " +
                    inputs[dataIndex_]->getTensorDesc().getPrecision().name();
                errMsg.copy(resp->msg, sizeof(resp->msg) - 1);
                return GENERAL_ERROR;
        }
    }

protected:
    template <typename dataType>
    StatusCode directExecution(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) noexcept {
        const dataType* srcData = inputs[dataIndex_]->cbuffer().as<const dataType*>() +
            inputs[dataIndex_]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        const int* indices = inputs[indicesIndex_]->cbuffer().as<const int*>() +
            inputs[indicesIndex_]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        dataType* dstData = outputs[0]->buffer().as<dataType*>() +
            outputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();

        const int outSize = outputs[0]->size();
        auto threadBody = [&](const int ithr, const int nthr) {
            int start(0lu), end(0lu);
            splitter(outSize, nthr, ithr, start, end);
            if (start >= end)
                return;

            int axStrideIt = start % strideAxDst_;
            int dstAxIdx = (start / strideAxDst_) % dstAxDim_;
            int dstShift0 = (start / strideAxDst_ / dstAxDim_) * strideAx1Diff_;

            for (size_t o = start; o < end; o++, axStrideIt++) {
                if (axStrideIt == strideAxDst_) {
                    axStrideIt = 0;
                    dstAxIdx++;
                    if (dstAxIdx == dstAxDim_) {
                        dstAxIdx = 0;
                        dstShift0 += strideAx1Diff_;
                    }
                }
                dstData[o] = srcData[o + dstShift0 + (indices[o] - dstAxIdx) * strideAxDst_];
            }
        };
        parallel_nt(0, threadBody);

        return OK;
    }

    const size_t dataIndex_ = 0;
    const size_t indicesIndex_ = 1;

    size_t axis_;
    size_t dataTypeSize_;
    int strideAxDst_;
    int dstAxDim_;
    int strideAx1Diff_;
    std::string errorPrefix_;
};

REG_FACTORY_FOR(GatherElementsImpl, GatherElements);
}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
