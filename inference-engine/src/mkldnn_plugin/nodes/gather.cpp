// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base.hpp"

#include <cmath>
#include <string>
#include <vector>
#include <cassert>
#include <algorithm>
#include <limits>
#include "ie_parallel.hpp"
#include "common/cpu_memcpy.h"
#include "common/fp16_utils.h"
#include <ngraph/op/gather.hpp>
#include <ie_ngraph_utils.hpp>
#include <ngraph/op/constant.hpp>

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

using MKLDNNPlugin::TensorDescCreatorTypes;

class GatherImpl: public ExtLayerBase {
public:
    static bool isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept {
        try {
            auto gatherOp = ngraph::as_type_ptr<const ngraph::op::v1::Gather>(op);
            if (!gatherOp) {
                errorMessage = "Only opset1 Gather operation is supported";
                return false;
            }

            auto axesOp = gatherOp->get_input_node_shared_ptr(GATHER_AXIS);
            if (!ngraph::as_type_ptr<const ngraph::op::Constant>(axesOp)) {
                errorMessage = "Only Constant operation on 'axis' input is supported";
                return false;
            }
        } catch (...) {
            return false;
        }

        return true;
    }

    explicit GatherImpl(const std::shared_ptr<ngraph::Node>& op) {
        try {
            errorPrefix_ = std::string("Layer Gather with name '") + op->get_friendly_name() + "' ";

            std::string errorMessage;
            if (!isSupportedOperation(op, errorMessage)) {
                IE_THROW(NotImplemented) << errorMessage;
            }

            auto gatherOp = ngraph::as_type_ptr<ngraph::op::v1::Gather>(op);
            if (gatherOp->get_input_size() != 3 || gatherOp->get_output_size() != 1)
                IE_THROW() << errorPrefix_ << "has incorrect number of input/output edges!";

            Precision inIdxPrecision = details::convertPrecision(gatherOp->get_input_element_type(GATHER_INDEXES));
            if (inIdxPrecision != Precision::FP32 && inIdxPrecision != Precision::I32 && inIdxPrecision != Precision::FP16)
                inIdxPrecision = Precision::I32;

            const SizeVector& dictionary_dims = gatherOp->get_input_shape(GATHER_DICTIONARY);
            if (dictionary_dims.size() == 0)
                IE_THROW() << errorPrefix_ << "has incorrect input parameters dimension!";

            axis = static_cast<int>(gatherOp->get_axis());
            if (axis < 0)
                axis += dictionary_dims.size();
            // Dictionary must be at least rank axis + 1
            if (!(-static_cast<int>(dictionary_dims.size()) <= axis && axis < static_cast<int>(dictionary_dims.size())))
                IE_THROW() << errorPrefix_ << "has incorrect input parameters dimensions and axis number!";

            //  Find number of dictionaries, index range and data length
            for (int i = 0; i < axis; i++)
                numDictionaries *= dictionary_dims[i];
            indexRange = dictionary_dims[axis];
            for (size_t i = axis + 1; i < dictionary_dims.size(); i++)
                dataLength *= dictionary_dims[i];

            if (dataLength == 0)
                IE_THROW() << errorPrefix_ << "had incorrect input parameters dimension!";

            Precision dataPrecision = details::convertPrecision(gatherOp->get_input_element_type(GATHER_DICTIONARY));

            addConfig(op, {{TensorDescCreatorTypes::ncsp, dataPrecision},
                           {TensorDescCreatorTypes::ncsp, inIdxPrecision},
                           {TensorDescCreatorTypes::ncsp, Precision::I32}},
                          {{TensorDescCreatorTypes::ncsp, dataPrecision}});
        } catch (InferenceEngine::Exception &ex) {
            errorMsg = ex.what();
            throw;
        }
    }

    struct f32toUi32 {
        inline unsigned int operator()(const float value) {
            return static_cast<unsigned int>(value);
        }
    };

    struct f16toUi32 {
        inline unsigned int operator()(const ie_fp16 value) {
            return static_cast<unsigned int>(f16tof32(value));
        }
    };

    struct i32toUi32 {
        inline unsigned int operator()(const int32_t value) {
            return static_cast<unsigned int>(value);
        }
    };

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) noexcept override {
        switch (inputs[GATHER_INDEXES]->getTensorDesc().getPrecision()) {
            case Precision::FP32:
                gather<float, f32toUi32>(inputs[GATHER_INDEXES], inputs[GATHER_DICTIONARY], outputs[0]);
                break;
            case Precision::FP16:
                gather<ie_fp16, f16toUi32>(inputs[GATHER_INDEXES], inputs[GATHER_DICTIONARY], outputs[0]);
                break;
            case Precision::I32:
                gather<int32_t, i32toUi32>(inputs[GATHER_INDEXES], inputs[GATHER_DICTIONARY], outputs[0]);
                break;
            default:
                return GENERAL_ERROR;
        }

        return OK;
    }

private:
    template <typename index_t, class Conversion>
    void gather(Blob::Ptr indexes, Blob::Ptr dictionary, Blob::Ptr output) {
        size_t src_indexSize = indexes->size();
        const index_t *src_index = indexes->cbuffer().as<const index_t *>() + indexes->getTensorDesc().getBlockingDesc().getOffsetPadding();
        const uint8_t *src_dataDict = dictionary->cbuffer().as<const uint8_t *>() + dictionary->getTensorDesc().getBlockingDesc().getOffsetPadding();
        uint8_t *dst_data = output->cbuffer().as<uint8_t*>() + output->getTensorDesc().getBlockingDesc().getOffsetPadding();
        size_t len = dataLength * dictionary->getTensorDesc().getPrecision().size();

        parallel_for(src_indexSize, [&](size_t i) {
            unsigned int idx = Conversion()(src_index[i]);

            //  Index clipping
            if (idx < indexRange) {
                //  Copying data to destination from Dictionary
                for (size_t j = 0; j < numDictionaries; j++) {
                    cpu_memcpy_s(&dst_data[len * (i + j * src_indexSize)],
                                output->byteSize() - (len * (i + j * src_indexSize)),
                                &src_dataDict[len * (idx + j * indexRange)],
                                len);
                }
            } else {
                for (size_t j = 0; j < numDictionaries; j++) {
                    memset(&dst_data[len * (i + j * src_indexSize)], 0, len);
                }
            }
        });
    }

    int axis = 0;
    size_t numDictionaries = 1;
    size_t indexRange = 0;
    size_t dataLength = 1;
    static const size_t GATHER_DICTIONARY = 0;
    static const size_t GATHER_INDEXES = 1;
    static const size_t GATHER_AXIS = 2;

    std::string errorPrefix_;
};


REG_FACTORY_FOR(GatherImpl, Gather);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
