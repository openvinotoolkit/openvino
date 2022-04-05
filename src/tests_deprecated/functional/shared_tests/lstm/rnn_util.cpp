// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rnn_util.hpp"

#include <string>
#include <cmath>

using namespace InferenceEngine;

#define T_LOOP_RANK 5

/**
 * @brief Iterate through tensor values and do action for each
 * elements.
 *
 * Signature of action is : (data_t &x, int *i) -> void
 *      x - is reference on tensor element
 *      i - array of logical indexes
 *
 * @tparam T action functor type. Generally is lambda
 * @param blob to iterate through
 * @param act functor to apply for each value in tensor
 */
template <typename T>
void T_LOOP(Blob::Ptr &blob, const T &act) {

    const auto &td = blob->getTensorDesc();
    const auto &dims = td.getDims();

    const auto &blk_d = td.getBlockingDesc();
    const auto &strides = blk_d.getStrides();

    int D[] = {1, 1, 1, 1, 1};
    std::copy(dims.begin(), dims.end(), std::end(D) - dims.size() );

    int i[] = {0, 0, 0, 0, 0};
    int &i0 = i[0], &i1 = i[1], &i2 = i[2], &i3 = i[3], &i4 = i[4];

    int s[] = {0, 0, 0, 0, 0};
    std::copy(strides.begin(), strides.end(), std::end(s) - dims.size());                                                        \
    int &s0 = s[0], &s1 = s[1], &s2 = s[2], &s3 = s[3], &s4 = s[4];

    size_t off_ = blk_d.getOffsetPadding();

    auto *ptr = blob->buffer().as<float*>();

    for (i0 = 0; i0 < D[0]; i0++) { auto off0 = off_ + i0 * s0;
    for (i1 = 0; i1 < D[1]; i1++) { auto off1 = off0 + i1 * s1;
    for (i2 = 0; i2 < D[2]; i2++) { auto off2 = off1 + i2 * s2;
    for (i3 = 0; i3 < D[3]; i3++) { auto off3 = off2 + i3 * s3;
    for (i4 = 0; i4 < D[4]; i4++) { auto off4 = off3 + i4 * s4; auto &off = off4;
        act(ptr[off], i);
    }}}}}
}

Checker negative(Checker checker) {
    return [=] (Blob::Ptr blob) -> bool {
        auto dims = blob->getTensorDesc().getDims();
        auto layout = blob->getTensorDesc().getLayout();
        auto new_blob = make_shared_blob<float>({Precision::FP32, dims, layout});
        new_blob->allocate();

        float *new_blob_ptr = new_blob->buffer().as<float*>();
        float *blob_ptr = blob->buffer().as<float*>();
        int size = blob->size();
        for (int i = 0; i < size; i++)
            *new_blob_ptr++ = -(*blob_ptr++);

        return checker(new_blob);
    };
}

static void copy_with_reverse(Blob::Ptr &src, Blob::Ptr &dst, int axis) {
    IE_ASSERT(src->getTensorDesc().getDims() == dst->getTensorDesc().getDims());

    const auto &td = src->getTensorDesc();
    const auto &dims = td.getDims();

    const auto &blk_d = td.getBlockingDesc();
    const auto &strides = blk_d.getStrides();

    int D[] = {1, 1, 1, 1, 1};
    std::copy(dims.begin(), dims.end(), std::end(D) - dims.size() );

    int s[] = {0, 0, 0, 0, 0};
    std::copy(strides.begin(), strides.end(), std::end(s) - dims.size());                                                        \
    int &s0 = s[0], &s1 = s[1], &s2 = s[2], &s3 = s[3], &s4 = s[4];

    size_t off_ = blk_d.getOffsetPadding();

    axis += T_LOOP_RANK - dims.size();

    // to iterate through tensor with reversed one dimension we need to
    // make stride negative and update offset.
    int reverse_str = s[axis];
    s[axis] = -reverse_str;
    off_ += (D[axis] - 1)*reverse_str;

    auto src_off = [=] (const int *i) {
        return off_ + i[0]*s0 + i[1]*s1 + i[2]*s2 + i[3]*s3 + i[4]*s4;
    };

    const auto *src_ptr = src->buffer().as<float*>();

    T_LOOP( dst, [&](float &x, const int *i) {
        x = src_ptr[ src_off(i) ];
    });
}

/** Make view blob (ROI) on parent blob. Doesn't hold parent blob */
static Blob::Ptr make_view(const Blob::Ptr &src, const SizeVector dims, const SizeVector offsets) {
    auto src_dims = src->getTensorDesc().getDims();
    IE_ASSERT(dims.size() == src_dims.size());
    IE_ASSERT(dims.size() == offsets.size());

    for (size_t i = 0; i < dims.size(); i++)
        IE_ASSERT(dims[i] + offsets[i] <= src_dims[i]);

    auto desc = src->getTensorDesc();
    auto b_desc = desc.getBlockingDesc();

    // move T desc to specified offset
    const auto new_off = desc.offset(offsets);
    TensorDesc new_desc { desc.getPrecision(), dims,
                          BlockingDesc { dims,
                                         b_desc.getOrder(), new_off,
                                         b_desc.getOffsetPaddingToData(),
                                         b_desc.getStrides() }
    };

    // TODO: Only FP32 supported here
    IE_ASSERT(desc.getPrecision() == Precision::FP32) << "Current limitation. Only FP32 is supported";
    return make_shared_blob<float>(new_desc, src->buffer());
}

Checker reverse(const Checker checker, int axis) {
    return [=] (Blob::Ptr blob) -> bool {
        auto dims = blob->getTensorDesc().getDims();
        auto layout = blob->getTensorDesc().getLayout();
        Blob::Ptr new_blob = make_shared_blob<float>({Precision::FP32, dims, layout});
        new_blob->allocate();

        copy_with_reverse(blob, new_blob, axis);
        return checker(new_blob);
    };
}

Filler reverse(const Filler filler, int axis) {
    return [=] (Blob::Ptr blob) {
        auto dims = blob->getTensorDesc().getDims();
        auto layout = blob->getTensorDesc().getLayout();
        Blob::Ptr new_blob = make_shared_blob<float>({Precision::FP32, dims, layout});
        new_blob->allocate();

        filler(new_blob);
        copy_with_reverse(new_blob, blob, axis);
    };
}

static void copy_with_permute(Blob::Ptr &src, Blob::Ptr &dst, const std::vector<int> order) {
    IE_ASSERT(order == std::vector<int>({1,0,2}));
    IE_ASSERT(src->getTensorDesc().getDims().size() == order.size());

    SizeVector prm_dims, dims = src->getTensorDesc().getDims();
    for (int i : order) prm_dims.push_back(dims[i]);

    IE_ASSERT(prm_dims == dst->getTensorDesc().getDims());

    size_t stride_2 = 1;
    size_t stride_1 = prm_dims[2] * stride_2;
    size_t stride_0 = prm_dims[1] * stride_1;

    float *src_ptr = src->buffer().as<float*>();
    float *dst_ptr = dst->buffer().as<float*>();

    for (int i0 = 0; i0 < dims[0]; i0++)
    for (int i1 = 0; i1 < dims[1]; i1++)
    for (int i2 = 0; i2 < dims[2]; i2++)
        dst_ptr[i1*stride_0 + i0*stride_1 + i2*stride_2] = *src_ptr++;
}

Filler permute(const Filler filler, const std::vector<int> order) {
    return [=] (Blob::Ptr blob) {
        SizeVector perm_dims, dims = blob->getTensorDesc().getDims();
        for (int i : order) perm_dims.push_back(dims[i]);

        Blob::Ptr new_blob = make_shared_blob<float>({Precision::FP32, perm_dims, blob->getTensorDesc().getLayout()});
        new_blob->allocate();

        filler(new_blob);
        copy_with_permute(new_blob, blob, order);
    };
}

Checker permute(const Checker checker, const std::vector<int> order) {
    return [=] (Blob::Ptr blob) -> bool {
        SizeVector perm_dims, dims = blob->getTensorDesc().getDims();
        for (int i : order) perm_dims.push_back(dims[i]);

        Blob::Ptr new_blob = make_shared_blob<float>({Precision::FP32, perm_dims, blob->getTensorDesc().getLayout()});
        new_blob->allocate();

        copy_with_permute(blob, new_blob, order);
        return checker(new_blob);
    };
}

Checker concat(const Checker checker1, const Checker checker2, int axis) {
    return [=] (Blob::Ptr blob) -> bool {
        auto dims = blob->getTensorDesc().getDims();

        const size_t split_size = 1;  // counting from end

        SizeVector dims1(dims);
        SizeVector offs1(dims.size(), 0);
        dims1[axis] -= split_size;

        SizeVector dims2 = dims;
        SizeVector offs2(dims.size(), 0);
        dims2[axis] = split_size;
        offs2[axis] = dims1[axis];

        auto blob1 = make_view(blob, dims1, offs1);
        auto blob2 = make_view(blob, dims2, offs2);

        return checker1(blob1) && checker2(blob2);
    };
}

Filler concat(const Filler filler1, const Filler filler2, int axis) {
    return [=] (Blob::Ptr blob) {
        auto dims = blob->getTensorDesc().getDims();

        const size_t split_size = 1;  // counting from end

        SizeVector dims1(dims);
        SizeVector offs1(dims.size(), 0);
        dims1[axis] -= split_size;

        SizeVector dims2 = dims;
        SizeVector offs2(dims.size(), 0);
        dims2[axis] = split_size;
        offs2[axis] = dims1[axis];

        auto blob1 = make_view(blob, dims1, offs1);
        auto blob2 = make_view(blob, dims2, offs2);

        filler1(blob1);
        filler2(blob2);
    };
}

static inline bool cmp_near(float res, float ref) {
    constexpr float eps = 1e-4;
    auto ref_abs = std::abs(ref);
    if (ref_abs > eps)
        return std::abs(res-ref)/ref_abs < eps;
    else
        return std::abs(res-ref) < eps;
}

bool scalar_checker(Blob::Ptr blob, SizeVector dims, float val) {
    IE_ASSERT(blob->getTensorDesc().getDims() == dims);

    bool res = true;
    T_LOOP(blob, [&](float x, int *i) {
        if (!cmp_near(x, val))
            res = false;
    });
    return res;
}

bool vector_checker(Blob::Ptr blob, SizeVector dims, std::vector<float> val, int axis) {
    IE_ASSERT(blob->getTensorDesc().getDims() == dims);
    IE_ASSERT(dims[axis] == val.size());

    axis += T_LOOP_RANK - dims.size();
    bool res = true;

    T_LOOP( blob, [&](float &x, int *i) {
        if (!cmp_near(x, val[ i[axis] ]))
            res = false;
    });

    return res;
}

void scalar_filler (Blob::Ptr blob, SizeVector dims, float val) {
    IE_ASSERT(blob->getTensorDesc().getDims() == dims);

    T_LOOP( blob, [&](float &x, int *i) {
        x = val;
    });
}

void vector_filler (Blob::Ptr blob, SizeVector dims, std::vector<float> val, int axis) {
    IE_ASSERT(blob->getTensorDesc().getDims() == dims);
    IE_ASSERT(dims[axis] == val.size());

    axis += T_LOOP_RANK - dims.size();

    T_LOOP( blob, [&](float &x, int *i) {
        x = val[ i[axis] ];
    });
}
