// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base.hpp"

#include <cmath>
#include <limits>
#include <cfloat>
#include <string>
#include <vector>
#include <cassert>
#include <legacy/ie_util_internal.hpp>
#include "ie_parallel.hpp"

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class ReduceImpl: public ExtLayerBase {
public:
    explicit ReduceImpl(const CNNLayer* layer) {
        try {
            if (layer->insData.empty() || layer->outData.empty())
                THROW_IE_EXCEPTION << layer->name << " Incorrect number of input/output edges!";

            if (layer->insData.size() != 2)
                THROW_IE_EXCEPTION << layer->name << " Incorrect number of input edges!";

            idx_dims = layer->insData[REDUCE_INDEXES].lock()->getTensorDesc().getDims();
            if (idx_dims.size() > 1)
                THROW_IE_EXCEPTION << layer->name << " Index vector should be 1 dimension";

            if (layer->insData[REDUCE_DATA].lock()->getTensorDesc().getPrecision() != Precision::FP32 &&
                layer->insData[REDUCE_DATA].lock()->getTensorDesc().getPrecision() != Precision::I32 &&
                layer->insData[REDUCE_DATA].lock()->getTensorDesc().getPrecision() != Precision::U8)
                THROW_IE_EXCEPTION << layer->name << " Incorrect input data tensor precision. Only FP32/I32/U8 are supported!";

            if (layer->insData[REDUCE_INDEXES].lock()->getTensorDesc().getPrecision() != Precision::I32)
                THROW_IE_EXCEPTION << layer->name << " Incorrect 'axes_to_reduction' input precision. Only I32 is supported!";

            data_dims = layer->insData[REDUCE_DATA].lock()->getTensorDesc().getDims();
            SizeVector dst_dims = layer->outData[0]->getTensorDesc().getDims();

            keep_dims = layer->GetParamAsBool("keep_dims", true);
            if (keep_dims) {
                if (data_dims.size() != dst_dims.size())
                    THROW_IE_EXCEPTION << layer->name << " Incorrect number of input/output dimensions!";
            } else {
                if (data_dims.size() <= dst_dims.size())
                    THROW_IE_EXCEPTION << layer->name << " Incorrect number of input/output dimensions!";
            }

            std::string reduce_mode = layer->type;
            if (reduce_mode == "ReduceAnd") reduceMode = Reduce::And;
            else if (reduce_mode == "ReduceL1") reduceMode = Reduce::L1;
            else if (reduce_mode == "ReduceL2") reduceMode = Reduce::L2;
            else if (reduce_mode == "ReduceLogSum") reduceMode = Reduce::LogSum;
            else if (reduce_mode == "ReduceLogSumExp") reduceMode = Reduce::LogSumExp;
            else if (reduce_mode == "ReduceMax") reduceMode = Reduce::Max;
            else if (reduce_mode == "ReduceMean") reduceMode = Reduce::Mean;
            else if (reduce_mode == "ReduceMin") reduceMode = Reduce::Min;
            else if (reduce_mode == "ReduceOr") reduceMode = Reduce::Or;
            else if (reduce_mode == "ReduceProd") reduceMode = Reduce::Prod;
            else if (reduce_mode == "ReduceSum") reduceMode = Reduce::Sum;
            else if (reduce_mode == "ReduceSumSquare") reduceMode = Reduce::SumSquare;
            else
                THROW_IE_EXCEPTION << layer->name << " Incorrect Reduce layer type!";

            src_dims = layer->insData[REDUCE_DATA].lock()->getTensorDesc().getDims();
            srcStrides = layer->insData[REDUCE_DATA].lock()->getTensorDesc().getBlockingDesc().getStrides();

            addConfig(layer, { { ConfLayout::PLN, false }, { ConfLayout::PLN, false } }, { { ConfLayout::PLN, false } });
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) noexcept override {
        int32_t *idx_data = inputs[REDUCE_INDEXES]->cbuffer().as<int32_t *>() +
                            inputs[REDUCE_INDEXES]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        SizeVector axes;
        const size_t axesIter = idx_dims.empty() ? 1 : idx_dims[0];
        for (size_t i = 0; i < axesIter; i++) {
            int32_t axis = idx_data[i];
            if (axis < 0)
                axis += data_dims.size();

            if (static_cast<size_t>(axis) > data_dims.size()) {
                if (resp) {
                    std::string errorMsg = "Index to reduce exceeds data tensor dimension";
                    errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
                }
                return PARAMETER_MISMATCH;
            }
            axes.push_back(static_cast<size_t>(axis));
        }

        size_t reduced_dims_work_amount = 1;
        InferenceEngine::SizeVector our_dims, out_dims, axes_for_reduction;
        for (size_t i = 0; i < src_dims.size(); i++) {
            bool found = false;
            for (size_t axis : axes)
                if (i == axis) found = true;

            if (found) {
                axes_for_reduction.push_back(i);
                reduced_dims_work_amount *= src_dims[i];
                if (keep_dims) out_dims.push_back(1);
                our_dims.push_back(1);
            } else {
                out_dims.push_back(src_dims[i]);
                our_dims.push_back(src_dims[i]);
            }
        }

        if (!our_dims.size())
            our_dims = InferenceEngine::SizeVector(1, 1);

        InferenceEngine::SizeVector dst_dims = outputs[0]->getTensorDesc().getDims();
        for (size_t i = 0; i < (std::min)(out_dims.size(), dst_dims.size()); i++) {
            if (out_dims[i] != dst_dims[i]) {
                if (resp) {
                    std::string errorMsg = "Incorrect number of output dimensions!";
                    errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
                }
                return PARAMETER_MISMATCH;
            }
        }

        size_t work_amount_dst;
        if (!dst_dims.size()) {
            work_amount_dst = 1;
        } else {
            size_t stride = !outputs[0]->getTensorDesc().getBlockingDesc().getStrides().empty()
                    ? outputs[0]->getTensorDesc().getBlockingDesc().getStrides()[0]
                    : 1;
            work_amount_dst = stride * dst_dims[0];
        }

        auto compare = getPrecisionMask(inputs[REDUCE_DATA]->getTensorDesc().getPrecision(), outputs[0]->getTensorDesc().getPrecision());
        switch (compare) {
            case getPrecisionMask(Precision::FP32, Precision::FP32):
                return reduce_type<float , float>(inputs, outputs, work_amount_dst, reduced_dims_work_amount, axes_for_reduction, our_dims);
            case getPrecisionMask(Precision::I32, Precision::I64):
                return reduce_type<int32_t , int64_t>(inputs, outputs, work_amount_dst, reduced_dims_work_amount, axes_for_reduction, our_dims);
            case getPrecisionMask(Precision::I32, Precision::U64):
                return reduce_type<int32_t , uint64_t>(inputs, outputs, work_amount_dst, reduced_dims_work_amount, axes_for_reduction, our_dims);
            case getPrecisionMask(Precision::I32, Precision::FP32):
                return reduce_type<int32_t , float>(inputs, outputs, work_amount_dst, reduced_dims_work_amount, axes_for_reduction, our_dims);
            case getPrecisionMask(Precision::I32, Precision::I32):
                return reduce_type<int32_t , int32_t>(inputs, outputs, work_amount_dst, reduced_dims_work_amount, axes_for_reduction, our_dims);
            case getPrecisionMask(Precision::U8, Precision::U8):
                return reduce_type<int8_t , int8_t>(inputs, outputs, work_amount_dst, reduced_dims_work_amount, axes_for_reduction, our_dims);
            case getPrecisionMask(Precision::FP32, Precision::U8):
                return reduce_type<float , uint8_t>(inputs, outputs, work_amount_dst, reduced_dims_work_amount, axes_for_reduction, our_dims);
            default:
                if (resp) {
                    std::string errorMsg = "Incorrect Reduce layer type";
                    errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
                }
                return GENERAL_ERROR;
        }
    }

private:
    template <typename src_d, typename dst_t, typename F1, typename F2>
    void reduce(const src_d *src_data, dst_t* dst_data, size_t work_amount_dst, size_t reduced_dims_work_amount,
        SizeVector axes_for_reduction, SizeVector dst_dims, dst_t init_value, F1 func1, F2 func2);
    template <typename src_d, typename dst_t>
    StatusCode reduce_type(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, size_t work_amount_dst, size_t reduced_dims_work_amount,
                SizeVector axes_for_reduction, SizeVector dst_dims);
    enum class Reduce { And, L1, L2, LogSum, LogSumExp, Max, Mean, Min, Or, Prod, Sum, SumSquare };

    const size_t REDUCE_DATA = 0;
    const size_t REDUCE_INDEXES = 1;
    bool keep_dims = true;
    Reduce reduceMode = Reduce::Sum;
    SizeVector data_dims;
    SizeVector idx_dims;
    SizeVector src_dims;
    SizeVector srcStrides;
};

template <typename src_d, typename dst_t>
StatusCode ReduceImpl::reduce_type(
        std::vector<Blob::Ptr>& inputs,
        std::vector<Blob::Ptr>& outputs,
        size_t       work_amount_dst,
        size_t       reduced_dims_work_amount,
        SizeVector   axes_for_reduction,
        SizeVector   our_dims
) {
    const src_d *src_data = inputs[REDUCE_DATA]->cbuffer().as<src_d *>() +
                            inputs[REDUCE_DATA]->getTensorDesc().getBlockingDesc().getOffsetPadding();
    dst_t* dst_data = outputs[0]->cbuffer().as<dst_t *>() +
                      outputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();

    switch (reduceMode) {
        case Reduce::And:
            reduce<src_d, dst_t>(src_data, dst_data, work_amount_dst, reduced_dims_work_amount, axes_for_reduction, our_dims, static_cast<dst_t>(1),
                   [](dst_t x, src_d y)->dst_t { return x && y; },
                   [](dst_t x, src_d y)->dst_t { return x && y; });
            break;
        case Reduce::L1:
            reduce<src_d, dst_t>(src_data, dst_data, work_amount_dst, reduced_dims_work_amount, axes_for_reduction, our_dims, static_cast<dst_t>(0),
                   [](dst_t old, src_d y)->dst_t { return old + (std::abs)(y); },
                   [](dst_t x, src_d y)->dst_t { return x + y; });
            break;
        case Reduce::L2:
            reduce<src_d, dst_t>(src_data, dst_data, work_amount_dst, reduced_dims_work_amount, axes_for_reduction, our_dims, static_cast<dst_t>(0),
                   [](dst_t old, src_d y)->dst_t { return old + y * y;},
                   [](dst_t x, src_d y)->dst_t { return x + y; });

            parallel_for(work_amount_dst, [&](size_t i) {
                dst_data[i] = sqrt(dst_data[i]);
            });
            break;
        case Reduce::LogSum:
            reduce<src_d, dst_t>(src_data, dst_data, work_amount_dst, reduced_dims_work_amount, axes_for_reduction, our_dims, static_cast<dst_t>(0),
                   [](dst_t x, src_d y)->dst_t { return x + y; },
                   [](dst_t x, src_d y)->dst_t { return x + y; });

            parallel_for(work_amount_dst, [&](size_t i) {
                dst_data[i] = logf(dst_data[i]);
            });
            break;
        case Reduce::LogSumExp:
            reduce<src_d, dst_t>(src_data, dst_data, work_amount_dst, reduced_dims_work_amount, axes_for_reduction, our_dims, static_cast<dst_t>(0),
                   [](dst_t old, src_d y)->dst_t { return old + expf(y); },
                   [](dst_t x, src_d y)->dst_t { return x + y; });

            parallel_for(work_amount_dst, [&](size_t i) {
                dst_data[i] = logf(dst_data[i]);
            });
            break;
        case Reduce::Max:
            reduce<src_d, dst_t>(src_data, dst_data, work_amount_dst, reduced_dims_work_amount, axes_for_reduction, our_dims,
                                 (std::numeric_limits<dst_t>::min)(),
                   [](dst_t x, src_d y)->dst_t { return x > y ? x : y; },
                   [](dst_t x, src_d y)->dst_t { return x > y ? x : y; });
            break;
        case Reduce::Mean:
            reduce<src_d, dst_t>(src_data, dst_data, work_amount_dst, reduced_dims_work_amount, axes_for_reduction, our_dims, static_cast<dst_t>(0),
                   [](dst_t x, src_d y)->dst_t { return x + y; },
                   [](dst_t x, src_d y)->dst_t { return x + y; });

            parallel_for(work_amount_dst, [&](size_t i) {
                dst_data[i] /= static_cast<dst_t>(reduced_dims_work_amount);
            });
            break;
        case Reduce::Min:
            reduce<src_d, dst_t>(src_data, dst_data, work_amount_dst, reduced_dims_work_amount, axes_for_reduction, our_dims,
                                 (std::numeric_limits<dst_t>::max)(),
                   [](dst_t x, src_d y)->dst_t { return x < y ? x : y; },
                   [](dst_t x, src_d y)->dst_t { return x < y ? x : y; });
            break;
        case Reduce::Or:
            reduce<src_d, dst_t>(src_data, dst_data, work_amount_dst, reduced_dims_work_amount, axes_for_reduction, our_dims, static_cast<dst_t>(0),
                   [](dst_t x, src_d y)->dst_t { return x || y; },
                   [](dst_t x, src_d y)->dst_t { return x || y; });
            break;
        case Reduce::Prod:
            reduce<src_d, dst_t>(src_data, dst_data, work_amount_dst, reduced_dims_work_amount, axes_for_reduction, our_dims, static_cast<dst_t>(1),
                   [](dst_t x, src_d y)->dst_t { return x * y; },
                   [](dst_t x, src_d y)->dst_t { return x * y; });
            break;
        case Reduce::Sum:
            reduce(src_data, dst_data, work_amount_dst, reduced_dims_work_amount, axes_for_reduction, our_dims, static_cast<dst_t>(0),
                   [](dst_t x, src_d y)->dst_t { return x + y; },
                   [](dst_t x, src_d y)->dst_t { return x + y; });
            break;
        case Reduce::SumSquare:
            reduce<src_d, dst_t>(src_data, dst_data, work_amount_dst, reduced_dims_work_amount, axes_for_reduction, our_dims, static_cast<dst_t>(0),
                   [](dst_t old, src_d y)->dst_t { return old + y * y; },
                   [](dst_t x, src_d y)->dst_t { return x + y; });
            break;
        default:
            return GENERAL_ERROR;
    }
    return OK;
}

template <typename src_d, typename dst_t, typename F1, typename F2>
void ReduceImpl::reduce(
    const src_d *src_data,
    dst_t       *dst_data,
    size_t       work_amount_dst,
    size_t       reduced_dims_work_amount,
    SizeVector   axes_for_reduction,
    SizeVector   dst_dims,
    dst_t        init_value,
    F1           func1,
    F2           func2
) {
    unsigned int nthr = parallel_get_max_threads();
    if ((work_amount_dst + 1) >= nthr) {
        parallel_nt(0, [&](const int ithr, const int nthr) {
            int j;
            size_t i, start = 0, end = 0;
            SizeVector dst_counters(dst_dims.size(), 0);
            splitter(work_amount_dst, nthr, ithr, start, end);
            for (j = dst_dims.size() - 1, i = start; j >= 0; j--) {
                dst_counters[j] = i % dst_dims[j];
                i /= dst_dims[j];
            }
            for (size_t src_idx = 0, dst_idx = start; dst_idx < end; ++dst_idx) {
                dst_t reduce_prod = init_value;
                bool update_idx = true;
                SizeVector src_counters = dst_counters;
                for (i = 0; i < reduced_dims_work_amount; ++i) {
                    if (update_idx) {
                        src_idx = 0;
                        for (j = 0; j < static_cast<int>(src_dims.size()); ++j)
                            src_idx += (src_counters[j] % src_dims[j]) * srcStrides[j];
                        update_idx = false;
                    }
                    reduce_prod = func1(reduce_prod, src_data[src_idx]);
                    for (j = axes_for_reduction.size() - 1; j >= 0; j--) {
                        src_counters[axes_for_reduction[j]]++;
                        if (src_counters[axes_for_reduction[j]] < src_dims[axes_for_reduction[j]]) {
                            src_idx += srcStrides[axes_for_reduction[j]];
                            break;
                        } else {
                            src_counters[axes_for_reduction[j]] = 0;
                            update_idx = true;
                        }
                    }
                }
                dst_data[dst_idx] = reduce_prod;
                for (j = dst_dims.size() - 1; j >= 0; j--) {
                    dst_counters[j]++;
                    if (dst_counters[j] < dst_dims[j])
                        break;
                    else
                        dst_counters[j] = 0;
                }
            }
        });
    } else {
        std::vector<dst_t> reduce_prod((nthr * work_amount_dst), init_value);
        if (work_amount_dst == 1) {
            parallel_nt(nthr, [&](const int ithr, const int nthr) {
                size_t i, start = 0, end = 0;
                splitter((srcStrides[0] * src_dims[0]), nthr, ithr, start, end);
                for (i = start; i < end; ++i)
                    reduce_prod[ithr] = func1(reduce_prod[ithr], src_data[i]);
            });
        } else {
            SizeVector dstStrides(dst_dims.size(), 1);
            for (int j = dst_dims.size() - 1; j >= 1; --j)
                dstStrides[j - 1] = dstStrides[j] * dst_dims[j];
            parallel_nt(nthr, [&](const int ithr, const int nthr) {
                int j;
                bool update_idx = true;
                size_t i, src_idx, dst_idx = 0, start = 0, end = 0;
                splitter((srcStrides[0] * src_dims[0]), nthr, ithr, start, end);
                SizeVector src_counters(src_dims.size(), 0);
                for (j = src_dims.size() - 1, src_idx = start; j >= 0; j--) {
                    src_counters[j] = src_idx % src_dims[j];
                    src_idx /= src_dims[j];
                }
                for (src_idx = start; src_idx < end; ++src_idx) {
                    if (update_idx) {
                        for (i = 0, dst_idx = 0; i < dst_dims.size(); ++i)
                            dst_idx += (src_counters[i] % dst_dims[i]) * dstStrides[i];
                        update_idx = false;
                    }
                    reduce_prod[ithr * work_amount_dst + dst_idx] = func1(reduce_prod[ithr * work_amount_dst + dst_idx], src_data[src_idx]);
                    for (j = src_dims.size() - 1; j >= 0; j--) {
                        src_counters[j]++;
                        if (src_counters[j] < src_dims[j]) {
                            if (dst_dims[j] > 1) dst_idx += dstStrides[j];
                            break;
                        } else {
                            src_counters[j] = 0;
                            update_idx = true;
                        }
                    }
                }
            });
        }
        for (size_t dst_idx = 0; dst_idx < work_amount_dst; dst_idx++) {
            for (size_t ithr = work_amount_dst; ithr < (nthr * work_amount_dst); ithr += work_amount_dst)
                reduce_prod[dst_idx] = func2(reduce_prod[dst_idx], reduce_prod[dst_idx + ithr]);
            dst_data[dst_idx] = reduce_prod[dst_idx];
        }
    }
}

REG_FACTORY_FOR(ReduceImpl, ReduceAnd);
REG_FACTORY_FOR(ReduceImpl, ReduceL1);
REG_FACTORY_FOR(ReduceImpl, ReduceL2);
REG_FACTORY_FOR(ReduceImpl, ReduceLogSum);
REG_FACTORY_FOR(ReduceImpl, ReduceLogSumExp);
REG_FACTORY_FOR(ReduceImpl, ReduceMax);
REG_FACTORY_FOR(ReduceImpl, ReduceMean);
REG_FACTORY_FOR(ReduceImpl, ReduceMin);
REG_FACTORY_FOR(ReduceImpl, ReduceOr);
REG_FACTORY_FOR(ReduceImpl, ReduceProd);
REG_FACTORY_FOR(ReduceImpl, ReduceSum);
REG_FACTORY_FOR(ReduceImpl, ReduceSumSquare);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
