// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base.hpp"

#include <cmath>
#include <string>
#include <vector>
#include <set>
#include <cassert>
#include "ie_parallel.hpp"
#include "common/cpu_memcpy.h"
#include <ngraph/op/shuffle_channels.hpp>
#include "common/tensor_desc_creator.h"

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

using MKLDNNPlugin::TensorDescCreatorTypes;

class ShuffleChannelsImpl: public ExtLayerBase {
#define CNTR_SIZE 3

__inline size_t initter(size_t start, size_t size, size_t* counters, size_t* own_dims, size_t* ownStrides) {
    size_t i = start;
    size_t idx = 0;
    for (int j = size - 1; j >= 0; j--) {
        counters[j] = i % own_dims[j];
        idx += counters[j] * ownStrides[j];
        i /= own_dims[j];
    }
    return idx;
}

__inline size_t updater(size_t idx, size_t size, size_t* counters, size_t* own_dims, size_t* ownStrides) {
    size_t i = 1;
    for (int j = size - 1; j >= 0; j--) {
        counters[j]++;
        if (counters[j] < own_dims[j]) {
            idx += ownStrides[j];
            break;
        } else {
            counters[j] = 0;
            i = 0;
        }
    }
    if (!i) {
        for (idx = 0; i < CNTR_SIZE; ++i)
            idx += counters[i] * ownStrides[i];
    }
    return idx;
}

public:
    bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
        try {
            auto scOp = ngraph::as_type_ptr<const ngraph::op::v0::ShuffleChannels>(op);
            if (!scOp) {
                errorMessage = "Node is not an instance of the TopK from the operations set v1.";
                return false;
            }

            if (_supported_precisions_sizes.find(op->get_input_element_type(0).size()) == _supported_precisions_sizes.end()) {
                errorMessage = "Unsupported precision: " + op->get_input_element_type(0).get_type_name();
                return false;
            }
        } catch (...) {
            return false;
        }
        return true;
    }

    explicit ShuffleChannelsImpl(const std::shared_ptr<ngraph::Node>& op) {
        try {
            std::string errorMessage;
            if (!isSupportedOperation(op, errorMessage)) {
                IE_THROW(NotImplemented) << errorMessage;
            }
            auto scOp = ngraph::as_type_ptr<const ngraph::op::v0::ShuffleChannels>(op);
            auto& dstDims = op->get_output_shape(0);

            int64_t axis = scOp->get_axis();
            if (axis < 0)
                axis += dstDims.size();

            if (axis < 0 || axis >= static_cast<int64_t>(dstDims.size()))
                IE_THROW() << op->get_friendly_name() << " Incorrect input parameters dimensions and axis number!";

            size_t group = scOp->get_group();
            if (group == 0 || dstDims[axis] % group)
                IE_THROW() << op->get_friendly_name() << " Group parameter must evenly divide the channel dimension!";

            //  Find number of dictionaries, index range and data length
            own_dims[0] = 1;
            for (int i = 0; i < axis; i++)
                own_dims[0] *= dstDims[i];

            for (size_t i = axis + 1; i < dstDims.size(); i++)
                dataLength *= dstDims[i];

            if (dataLength == 0)
                IE_THROW() << op->get_friendly_name() << " Incorrect input parameters dimension!";

            own_dims[1] = dstDims[axis] / group;
            own_dims[2] = group;
            ownStrides[0] = dstDims[axis];
            ownStrides[1] = 1;
            ownStrides[2] = own_dims[1];
            work_amount_dst = ownStrides[0] * own_dims[0];

            addConfig(op, {{TensorDescCreatorTypes::ncsp, details::convertPrecision(op->get_input_element_type(0))}},
                          {{TensorDescCreatorTypes::ncsp, details::convertPrecision(op->get_input_element_type(0))}});
        } catch (InferenceEngine::Exception &ex) {
            errorMsg = ex.what();
            throw;
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
                    std::string errorMsg = "ShuffleChannels layer does not support precision '"
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
        T* dst_data = outputs[0]->cbuffer().as<T*>() +
                          outputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();

        if (dataLength > 1) {
            //  Vectorized & Parallel
            parallel_nt(0, [&](const int ithr, const int nthr) {
                size_t start = 0, end = 0, src_idx = 0;
                size_t counters[CNTR_SIZE] = { 0 };
                splitter(work_amount_dst, nthr, ithr, start, end);
                src_idx = initter(start, CNTR_SIZE, counters, own_dims, ownStrides);
                for (size_t iwork = start, dst_idx = start * dataLength; iwork < end; ++iwork, dst_idx += dataLength) {
                    cpu_memcpy(&dst_data[dst_idx], &src_data[dataLength * src_idx], sizeof(T) * dataLength);
                    src_idx = updater(src_idx, CNTR_SIZE, counters, own_dims, ownStrides);
                }
            });
        } else {
            //  Parallel
            parallel_nt(0, [&](const int ithr, const int nthr) {
                size_t start = 0, end = 0, src_idx = 0;
                size_t counters[CNTR_SIZE] = { 0 };
                splitter(work_amount_dst, nthr, ithr, start, end);
                src_idx = initter(start, CNTR_SIZE, counters, own_dims, ownStrides);
                for (size_t iwork = start; iwork < end; ++iwork) {
                    dst_data[iwork] = src_data[src_idx];
                    src_idx = updater(src_idx, CNTR_SIZE, counters, own_dims, ownStrides);
                }
            });
        }
    }

private:
    size_t dataLength = 1;
    size_t work_amount_dst;
    size_t own_dims[CNTR_SIZE];
    size_t ownStrides[CNTR_SIZE];

    static const std::set<size_t> _supported_precisions_sizes;
};

const std::set<size_t> ShuffleChannelsImpl::_supported_precisions_sizes = {1, 2, 4, 8};

REG_FACTORY_FOR(ShuffleChannelsImpl, ShuffleChannels);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
