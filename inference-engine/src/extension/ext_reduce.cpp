// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ext_list.hpp"
#include "ext_base.hpp"

#include <cmath>
#include <limits>
#include <cfloat>
#include <string>
#include <vector>
#include <cassert>
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

            if (layer->insData[REDUCE_DATA].lock()->getTensorDesc().getPrecision() != Precision::FP32)
                THROW_IE_EXCEPTION << layer->name << " Incorrect input data tensor precision. Only FP32 is supported!";

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
        for (size_t i = 0; i < idx_dims[0]; i++) {
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

        const float *src_data = inputs[REDUCE_DATA]->cbuffer().as<float *>() +
            inputs[REDUCE_DATA]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        float* dst_data = outputs[0]->cbuffer().as<float *>() +
            outputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();

        size_t work_amount_dst;
        if (!dst_dims.size())
            work_amount_dst = 1;
        else
            work_amount_dst = outputs[0]->getTensorDesc().getBlockingDesc().getStrides()[0] * dst_dims[0];

        switch (reduceMode) {
        case Reduce::And:
            reduce(src_data, dst_data, work_amount_dst, reduced_dims_work_amount, axes_for_reduction, our_dims, 1.0f,
                   [](float x, float y)->float { return x && y; },
                   [](float x, float y)->float { return x && y; });
            break;
        case Reduce::L1:
            reduce(src_data, dst_data, work_amount_dst, reduced_dims_work_amount, axes_for_reduction, our_dims, 0.0f,
                [](float old, float y)->float { return old + (std::abs)(y); },
                [](float x, float y)->float { return x + y; });
            break;
        case Reduce::L2:
            reduce(src_data, dst_data, work_amount_dst, reduced_dims_work_amount, axes_for_reduction, our_dims, 0.0f,
                [](float old, float y)->float { return old + y * y;},
                [](float x, float y)->float { return x + y; });

            parallel_for(work_amount_dst, [&](size_t i) {
                dst_data[i] = sqrt(dst_data[i]);
            });
            break;
        case Reduce::LogSum:
            reduce(src_data, dst_data, work_amount_dst, reduced_dims_work_amount, axes_for_reduction, our_dims, 0.0f,
                [](float x, float y)->float { return x + y; },
                [](float x, float y)->float { return x + y; });

            parallel_for(work_amount_dst, [&](size_t i) {
                dst_data[i] = logf(dst_data[i]);
            });
            break;
        case Reduce::LogSumExp:
            reduce(src_data, dst_data, work_amount_dst, reduced_dims_work_amount, axes_for_reduction, our_dims, 0.0f,
                [](float old, float y)->float { return old + expf(y); },
                [](float x, float y)->float { return x + y; });

            parallel_for(work_amount_dst, [&](size_t i) {
                dst_data[i] = logf(dst_data[i]);
            });
            break;
        case Reduce::Max:
            reduce(src_data, dst_data, work_amount_dst, reduced_dims_work_amount, axes_for_reduction, our_dims, FLT_MIN,
                [](float x, float y)->float { return x > y ? x : y; },
                [](float x, float y)->float { return x > y ? x : y; });
            break;
        case Reduce::Mean:
            reduce(src_data, dst_data, work_amount_dst, reduced_dims_work_amount, axes_for_reduction, our_dims, 0.0f,
                [](float x, float y)->float { return x + y; },
                [](float x, float y)->float { return x + y; });

            parallel_for(work_amount_dst, [&](size_t i) {
                dst_data[i] /= static_cast<float>(reduced_dims_work_amount);
            });
            break;
        case Reduce::Min:
            reduce(src_data, dst_data, work_amount_dst, reduced_dims_work_amount, axes_for_reduction, our_dims, FLT_MAX,
                [](float x, float y)->float { return x < y ? x : y; },
                [](float x, float y)->float { return x < y ? x : y; });
            break;
        case Reduce::Or:
            reduce(src_data, dst_data, work_amount_dst, reduced_dims_work_amount, axes_for_reduction, our_dims, 0.0f,
                   [](float x, float y)->float { return x || y; },
                   [](float x, float y)->float { return x || y; });
            break;
        case Reduce::Prod:
            reduce(src_data, dst_data, work_amount_dst, reduced_dims_work_amount, axes_for_reduction, our_dims, 1.0f,
                [](float x, float y)->float { return x * y; },
                [](float x, float y)->float { return x * y; });
            break;
        case Reduce::Sum:
            reduce(src_data, dst_data, work_amount_dst, reduced_dims_work_amount, axes_for_reduction, our_dims, 0.0f,
                [](float x, float y)->float { return x + y; },
                [](float x, float y)->float { return x + y; });
            break;
        case Reduce::SumSquare:
            reduce(src_data, dst_data, work_amount_dst, reduced_dims_work_amount, axes_for_reduction, our_dims, 0.0f,
                [](float old, float y)->float { return old + y * y; },
                [](float x, float y)->float { return x + y; });
            break;
        default:
            if (resp) {
                std::string errorMsg = "Incorrect Reduce layer type";
                errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
            }
            return GENERAL_ERROR;
        }
        return OK;
    }

private:
    template <typename F1, typename F2>
    void reduce(const float *src_data, float* dst_data, size_t work_amount_dst, size_t reduced_dims_work_amount,
        SizeVector axes_for_reduction, SizeVector dst_dims, float init_value, F1 func1, F2 func2);

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

template <typename F1, typename F2>
void ReduceImpl::reduce(
    const float *src_data,
    float       *dst_data,
    size_t       work_amount_dst,
    size_t       reduced_dims_work_amount,
    SizeVector   axes_for_reduction,
    SizeVector   dst_dims,
    float        init_value,
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
            for (size_t src_idx, dst_idx = start; dst_idx < end; ++dst_idx) {
                float reduce_prod = init_value;
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
        std::vector<float> reduce_prod((nthr * work_amount_dst), init_value);
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

REG_FACTORY_FOR(ImplFactory<ReduceImpl>, ReduceAnd);
REG_FACTORY_FOR(ImplFactory<ReduceImpl>, ReduceL1);
REG_FACTORY_FOR(ImplFactory<ReduceImpl>, ReduceL2);
REG_FACTORY_FOR(ImplFactory<ReduceImpl>, ReduceLogSum);
REG_FACTORY_FOR(ImplFactory<ReduceImpl>, ReduceLogSumExp);
REG_FACTORY_FOR(ImplFactory<ReduceImpl>, ReduceMax);
REG_FACTORY_FOR(ImplFactory<ReduceImpl>, ReduceMean);
REG_FACTORY_FOR(ImplFactory<ReduceImpl>, ReduceMin);
REG_FACTORY_FOR(ImplFactory<ReduceImpl>, ReduceOr);
REG_FACTORY_FOR(ImplFactory<ReduceImpl>, ReduceProd);
REG_FACTORY_FOR(ImplFactory<ReduceImpl>, ReduceSum);
REG_FACTORY_FOR(ImplFactory<ReduceImpl>, ReduceSumSquare);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
