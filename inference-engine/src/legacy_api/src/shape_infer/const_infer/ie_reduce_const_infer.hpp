// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_blob.h>
#include <ie_layers.h>

#include <cfloat>
#include <cmath>
#include <ie_algorithm.hpp>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ie_const_infer_impl.hpp"
#include "ie_parallel.hpp"

namespace InferenceEngine {
namespace ShapeInfer {

/**
 *@brief Implementation of Const inference for Reduce layer
 */
class ReduceConstInfer : public ConstInferImpl {
private:
    const size_t REDUCE_DATA = 0;
    const size_t REDUCE_INDEXES = 1;

    template <typename src_t, typename dst_t>
    void reduce(SizeVector src_dims, SizeVector srcStrides, const src_t* src_data, dst_t* dst_data,
                size_t work_amount_dst, size_t reduced_dims_work_amount, SizeVector axes_for_reduction,
                SizeVector dst_dims, dst_t init_value, std::string reduceType) {
        // I don't know why func 2 is necessary!
        std::function<dst_t(dst_t, src_t)> func1;
        std::function<dst_t(dst_t, src_t)> func2;
        if (reduceType == "ReduceAnd") {
            func1 = [](dst_t x, src_t y) -> dst_t {
                return x && y;
            };
            func2 = [](dst_t x, src_t y) -> dst_t {
                return x && y;
            };
        } else if (reduceType == "ReduceL1") {
            func1 = [](dst_t x, src_t y) -> dst_t {
                return x + (std::abs)(y);
            };
            func2 = [](dst_t x, src_t y) -> dst_t {
                return x + y;
            };
        } else if (reduceType == "ReduceL2") {
            func1 = [](dst_t x, src_t y) -> dst_t {
                return x + y * y;
            };
            func2 = [](dst_t x, src_t y) -> dst_t {
                return x + y;
            };
        } else if (reduceType == "ReduceLogSum") {
            func1 = [](dst_t x, src_t y) -> dst_t {
                return x + y;
            };
            func2 = [](dst_t x, src_t y) -> dst_t {
                return x + y;
            };
        } else if (reduceType == "ReduceLogSumExp") {
            func1 = [](dst_t x, src_t y) -> dst_t {
                return x + expf(y);
            };
            func2 = [](dst_t x, src_t y) -> dst_t {
                return x + y;
            };
        } else if (reduceType == "ReduceMax") {
            func1 = [](dst_t x, src_t y) -> dst_t {
                return x > y ? x : y;
            };
            func2 = [](dst_t x, src_t y) -> dst_t {
                return x > y ? x : y;
            };
        } else if (reduceType == "ReduceMean") {
            func1 = [](dst_t x, src_t y) -> dst_t {
                return (x + y);
            };
            func2 = [](dst_t x, src_t y) -> dst_t {
                return (x + y);
            };
        } else if (reduceType == "ReduceMin") {
            func1 = [](dst_t x, src_t y) -> dst_t {
                return x < y ? x : y;
            };
            func2 = [](dst_t x, src_t y) -> dst_t {
                return x < y ? x : y;
            };
        } else if (reduceType == "ReduceOr") {
            func1 = [](dst_t x, src_t y) -> dst_t {
                return x || y;
            };
            func2 = [](dst_t x, src_t y) -> dst_t {
                return x || y;
            };
        } else if (reduceType == "ReduceProd") {
            func1 = [](dst_t x, src_t y) -> dst_t {
                return x * y;
            };
            func2 = [](dst_t x, src_t y) -> dst_t {
                return x * y;
            };
        } else if (reduceType == "ReduceSum") {
            func1 = [](dst_t x, src_t y) -> dst_t {
                return x + y;
            };
            func2 = [](dst_t x, src_t y) -> dst_t {
                return x + y;
            };
        } else if (reduceType == "ReduceSumSquare") {
            func1 = [](dst_t x, src_t y) -> dst_t {
                return x + y * y;
            };
            func2 = [](dst_t x, src_t y) -> dst_t {
                return x + y;
            };
        }

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
                    for (i = start; i < end; ++i) reduce_prod[ithr] = func1(reduce_prod[ithr], src_data[i]);
                });
            } else {
                SizeVector dstStrides(dst_dims.size(), 1);
                for (int j = dst_dims.size() - 1; j >= 1; --j) dstStrides[j - 1] = dstStrides[j] * dst_dims[j];
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
                        reduce_prod[ithr * work_amount_dst + dst_idx] =
                            func1(reduce_prod[ithr * work_amount_dst + dst_idx], src_data[src_idx]);
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

    template <typename src_d, typename dst_d>
    void exec_reduce(const std::vector<Blob::CPtr>& insData, std::vector<Blob::Ptr>& outData, std::string reduce_mode,
                     SizeVector src_dims, SizeVector srcStrides, size_t work_amount_dst,
                     size_t reduced_dims_work_amount, SizeVector axes_for_reduction, SizeVector our_dims, dst_d min_val,
                     dst_d max_val) {
        const src_d* src_data = insData[REDUCE_DATA]->cbuffer().as<src_d*>() +
                                insData[REDUCE_DATA]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        dst_d* dst_data =
            outData[0]->cbuffer().as<dst_d*>() + outData[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        if (reduce_mode == "ReduceAnd") {
            reduce<src_d, dst_d>(src_dims, srcStrides, src_data, dst_data, work_amount_dst, reduced_dims_work_amount,
                                 axes_for_reduction, our_dims, 1, reduce_mode);
        } else if (reduce_mode == "ReduceL1") {
            reduce<src_d, dst_d>(src_dims, srcStrides, src_data, dst_data, work_amount_dst, reduced_dims_work_amount,
                                 axes_for_reduction, our_dims, 0, reduce_mode);
        } else if (reduce_mode == "ReduceL2") {
            reduce<src_d, dst_d>(src_dims, srcStrides, src_data, dst_data, work_amount_dst, reduced_dims_work_amount,
                                 axes_for_reduction, our_dims, 0, reduce_mode);

            parallel_for(work_amount_dst, [&](size_t i) {
                dst_data[i] = sqrt(dst_data[i]);
            });
        } else if (reduce_mode == "ReduceLogSum") {
            reduce<src_d, dst_d>(src_dims, srcStrides, src_data, dst_data, work_amount_dst, reduced_dims_work_amount,
                                 axes_for_reduction, our_dims, 0, reduce_mode);

            parallel_for(work_amount_dst, [&](size_t i) {
                dst_data[i] = logf(dst_data[i]);
            });
        } else if (reduce_mode == "ReduceLogSumExp") {
            reduce<src_d, dst_d>(src_dims, srcStrides, src_data, dst_data, work_amount_dst, reduced_dims_work_amount,
                                 axes_for_reduction, our_dims, 0, reduce_mode);

            parallel_for(work_amount_dst, [&](size_t i) {
                dst_data[i] = logf(dst_data[i]);
            });
        } else if (reduce_mode == "ReduceMax") {
            reduce<src_d, dst_d>(src_dims, srcStrides, src_data, dst_data, work_amount_dst, reduced_dims_work_amount,
                                 axes_for_reduction, our_dims, min_val, reduce_mode);
        } else if (reduce_mode == "ReduceMean") {
            reduce<src_d, dst_d>(src_dims, srcStrides, src_data, dst_data, work_amount_dst, reduced_dims_work_amount,
                                 axes_for_reduction, our_dims, 0, reduce_mode);

            parallel_for(work_amount_dst, [&](size_t i) {
                dst_data[i] /= static_cast<float>(reduced_dims_work_amount);
            });
        } else if (reduce_mode == "ReduceMin") {
            reduce<src_d, dst_d>(src_dims, srcStrides, src_data, dst_data, work_amount_dst, reduced_dims_work_amount,
                                 axes_for_reduction, our_dims, max_val, reduce_mode);
        } else if (reduce_mode == "ReduceOr") {
            reduce<src_d, dst_d>(src_dims, srcStrides, src_data, dst_data, work_amount_dst, reduced_dims_work_amount,
                                 axes_for_reduction, our_dims, 0, reduce_mode);
        } else if (reduce_mode == "ReduceProd") {
            reduce<src_d, dst_d>(src_dims, srcStrides, src_data, dst_data, work_amount_dst, reduced_dims_work_amount,
                                 axes_for_reduction, our_dims, 1, reduce_mode);
        } else if (reduce_mode == "ReduceSum") {
            reduce<src_d, dst_d>(src_dims, srcStrides, src_data, dst_data, work_amount_dst, reduced_dims_work_amount,
                                 axes_for_reduction, our_dims, 0, reduce_mode);
        } else if (reduce_mode == "ReduceSumSquare") {
            reduce<src_d, dst_d>(src_dims, srcStrides, src_data, dst_data, work_amount_dst, reduced_dims_work_amount,
                                 axes_for_reduction, our_dims, 0, reduce_mode);
        } else {
            THROW_IE_EXCEPTION << " Incorrect Reduce layer type!";
        }
    }

public:
    explicit ReduceConstInfer(const std::string& type): ConstInferImpl(type) {}

    void inferImpl(const std::vector<Blob::CPtr>& insData, const std::map<std::string, std::string>& params,
                   const std::map<std::string, Blob::Ptr>& blobs, std::vector<Blob::Ptr>& outData) override {
        LayerParams lp {"", _type, Precision::UNSPECIFIED};
        CNNLayer layer(lp);
        layer.params = params;

        if (insData.empty() || outData.empty())
            THROW_IE_EXCEPTION << " Reduce constant inference error: empty input or output data!";

        if (insData.size() != 2)
            THROW_IE_EXCEPTION
                << " Reduce constant inference error: Incorrect number of input edges! Should be 2 edges, got "
                << insData.size();

        SizeVector idx_dims = insData[REDUCE_INDEXES]->getTensorDesc().getDims();
        if (idx_dims.size() > 1)
            THROW_IE_EXCEPTION << " Reduce constant inference error: Index vector should be 1 dimension, got "
                               << idx_dims.size() << " dimensions";

        if (insData[REDUCE_INDEXES]->getTensorDesc().getPrecision() != Precision::I32)
            THROW_IE_EXCEPTION << " Reduce constant inference error: Incorrect 'axes_to_reduction' input precision. "
                                  "Only I32 is supported! Current precision: "
                               << insData[REDUCE_INDEXES]->getTensorDesc().getPrecision();

        SizeVector data_dims = insData[REDUCE_DATA]->getTensorDesc().getDims();
        SizeVector dst_dims = outData[0]->getTensorDesc().getDims();

        bool keep_dims = layer.GetParamAsBool("keep_dims", true);
        if (keep_dims) {
            if (data_dims.size() != dst_dims.size())
                THROW_IE_EXCEPTION << " Reduce constant inference error: Incorrect number of input/output dimensions!";
        } else {
            if (data_dims.size() <= dst_dims.size())
                THROW_IE_EXCEPTION << " Reduce constant inference error: Incorrect number of input/output dimensions!";
        }

        SizeVector src_dims = insData[REDUCE_DATA]->getTensorDesc().getDims();
        SizeVector srcStrides = insData[REDUCE_DATA]->getTensorDesc().getBlockingDesc().getStrides();

        int32_t* idx_data = insData[REDUCE_INDEXES]->cbuffer().as<int32_t*>() +
                            insData[REDUCE_INDEXES]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        SizeVector axes;
        for (size_t i = 0; i < idx_dims[0]; i++) {
            int32_t axis = idx_data[i];
            if (axis < 0) axis += data_dims.size();

            if (static_cast<size_t>(axis) > data_dims.size())
                THROW_IE_EXCEPTION << " Reduce constant inference error: Index to reduce exceeds data tensor dimension";
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

        if (!our_dims.size()) our_dims = SizeVector(1, 1);

        for (size_t i = 0; i < (std::min)(out_dims.size(), dst_dims.size()); i++)
            if (out_dims[i] != dst_dims[i])
                THROW_IE_EXCEPTION << " Reduce constant inference error: Incorrect number of output dimensions!";

        size_t work_amount_dst;
        if (!dst_dims.size())
            work_amount_dst = 1;
        else
            work_amount_dst = outData[0]->getTensorDesc().getBlockingDesc().getStrides()[0] * dst_dims[0];

        std::string reduce_mode = layer.type;

        auto compare = getPrecisionMask(insData[REDUCE_DATA]->getTensorDesc().getPrecision(),
                                        outData[0]->getTensorDesc().getPrecision());
        switch (compare) {
        case getPrecisionMask(Precision::FP32, Precision::FP32):
            exec_reduce<PrecisionTrait<Precision::FP32>::value_type, PrecisionTrait<Precision::FP32>::value_type>(
                insData, outData, reduce_mode, src_dims, srcStrides, work_amount_dst, reduced_dims_work_amount,
                axes_for_reduction, dst_dims, (std::numeric_limits<PrecisionTrait<Precision::FP32>::value_type>::min)(),
                (std::numeric_limits<PrecisionTrait<Precision::FP32>::value_type>::max)());
            break;

        case getPrecisionMask(Precision::I32, Precision::I64):
            exec_reduce<PrecisionTrait<Precision::I32>::value_type, PrecisionTrait<Precision::I64>::value_type>(
                insData, outData, reduce_mode, src_dims, srcStrides, work_amount_dst, reduced_dims_work_amount,
                axes_for_reduction, dst_dims, (std::numeric_limits<PrecisionTrait<Precision::I64>::value_type>::min)(),
                (std::numeric_limits<PrecisionTrait<Precision::I64>::value_type>::max)());
            break;
        case getPrecisionMask(Precision::I32, Precision::U64):
            exec_reduce<PrecisionTrait<Precision::I32>::value_type, PrecisionTrait<Precision::U64>::value_type>(
                insData, outData, reduce_mode, src_dims, srcStrides, work_amount_dst, reduced_dims_work_amount,
                axes_for_reduction, dst_dims, (std::numeric_limits<PrecisionTrait<Precision::U64>::value_type>::min)(),
                (std::numeric_limits<PrecisionTrait<Precision::U64>::value_type>::max)());
            break;
        case getPrecisionMask(Precision::I32, Precision::FP32):
            exec_reduce<PrecisionTrait<Precision::I32>::value_type, PrecisionTrait<Precision::FP32>::value_type>(
                insData, outData, reduce_mode, src_dims, srcStrides, work_amount_dst, reduced_dims_work_amount,
                axes_for_reduction, dst_dims, (std::numeric_limits<PrecisionTrait<Precision::FP32>::value_type>::min)(),
                (std::numeric_limits<PrecisionTrait<Precision::FP32>::value_type>::max)());
            break;
        case getPrecisionMask(Precision::I32, Precision::I32):
            exec_reduce<PrecisionTrait<Precision::I32>::value_type, PrecisionTrait<Precision::I32>::value_type>(
                insData, outData, reduce_mode, src_dims, srcStrides, work_amount_dst, reduced_dims_work_amount,
                axes_for_reduction, dst_dims, (std::numeric_limits<PrecisionTrait<Precision::I32>::value_type>::min)(),
                (std::numeric_limits<PrecisionTrait<Precision::I32>::value_type>::max)());
            break;
        default:
            THROW_IE_EXCEPTION
                << "Reduce constant inference error: Incorrect data tensor precisions. REDUCE_DATA precision: "
                << insData[REDUCE_DATA]->getTensorDesc().getPrecision()
                << " Output precision: " << outData[0]->getTensorDesc().getPrecision();
        }
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
