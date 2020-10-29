/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef PRIMITIVE_ATTR_HPP
#define PRIMITIVE_ATTR_HPP

#include <mkldnn.hpp>
#include "mkldnn.h"

#include "c_types_map.hpp"
#include "nstl.hpp"
#include "utils.hpp"

namespace mkldnn {
namespace impl {

struct rnn_data_qparams_t : public c_compatible {
    rnn_data_qparams_t() : scale_(1.), shift_(0.) {}
    bool has_default_values() const { return (scale_ == 1. && shift_ == 0.); }

    status_t set(float scale, float shift) {
        scale_ = scale;
        shift_ = shift;
        return status::success;
    }

    float scale_;
    float shift_;
};

struct scales_t: public c_compatible {
    scales_t(): count_(1), mask_(0), scales_(scales_buf_)
    { set(1.); }

    scales_t(const scales_t &rhs): scales_t()
    { set(rhs.count_, rhs.mask_, rhs.scales_); }

    ~scales_t() { cleanup(); }

    scales_t &operator=(const scales_t &rhs) {
        if (&rhs == this)
            return *this;
        status_t status = set(rhs.count_, rhs.mask_, rhs.scales_);
        assert(status == status::success);
        (void)status;
        return *this;
    }

    bool has_default_values() const {
        for (int c = 0; c < count_; ++c) {
            if(scales_[c] != 1.) return false;
        }
        return true;
    }

    status_t set(int count, int mask, const float *scales);
    status_t set(float single_scale) { return this->set(1, 0, &single_scale); }

    int count_;
    int mask_;
    float *scales_;

private:
    enum { scales_buf_size = 16 };
    float scales_buf_[scales_buf_size];

    void cleanup() {
        if (scales_ != scales_buf_ && scales_ != nullptr)
            impl::free(scales_);

        count_ = 1;
        mask_ = 0;
        scales_ = scales_buf_;
    }
};

template <typename T>
struct shifts_t: public c_compatible {
    shifts_t(): count_(1), mask_(0), shifts_(shifts_buf_)
    { set(0); }

    shifts_t(const shifts_t &rhs): shifts_t()
    { set(rhs.count_, rhs.mask_, rhs.shifts_); }

    ~shifts_t() { cleanup(); }

    shifts_t &operator=(const shifts_t &rhs) {
        if (&rhs == this)
            return *this;
        status_t status = set(rhs.count_, rhs.mask_, rhs.shifts_);
        assert(status == status::success);
        (void)status;
        return *this;
    }

    bool has_default_values() const {
        for (int c = 0; c < count_; ++c) {
            if(shifts_[c] != 0) return false;
        }
        return true;
    }

    status_t set(int count, int mask, const T *zero_points);
    status_t set(T single_zero_point) { return this->set(1, 0, &single_zero_point); }

    int count_;
    int mask_;
    T *shifts_;

private:
    enum { shifts_buf_size = 16 };
    T shifts_buf_[shifts_buf_size];

    void cleanup() {
        if (shifts_ != shifts_buf_ && shifts_ != nullptr)
            impl::free(shifts_);

        count_ = 1;
        mask_ = 0;
        shifts_ = shifts_buf_;
    }
};

}
}

struct mkldnn_post_ops: public mkldnn::impl::c_compatible {
    struct entry_t {
        struct eltwise_t {
            mkldnn::impl::alg_kind_t alg;
            float scale, alpha, beta;
        };

        mkldnn::impl::primitive_kind_t kind;
        union {
            struct {
                float scale;
                mkldnn::impl::data_type_t data_type;
            } sum;
            eltwise_t eltwise;
            struct {
                mkldnn::impl::alg_kind_t alg;
                const float* weights_data;
                const float* biases_data;
            } depthwise;
            struct {
                int in_h;
                int in_w;
                int ker_h;
                int ker_w;
                int str_h;
                int str_w;
                mkldnn::impl::data_type_t in_dt;
                const float* weights_data;
                const float* biases_data;
            } dw_conv;
            struct {
                mkldnn::impl::alg_kind_t alg;
                const float* thresholds_data;
                const float* output_mask_data;
            } binarization;
            struct {
                mkldnn::impl::alg_kind_t alg;
                mkldnn::impl::shifts_t<float>* crop_low_data;
                mkldnn::impl::shifts_t<float>* crop_high_data;
                mkldnn::impl::scales_t* input_scale_data;
                mkldnn::impl::shifts_t<float>* input_shift_data;
                mkldnn::impl::scales_t* output_scale_data;
                mkldnn::impl::shifts_t<float>* output_shift_data;
            } quantization;
        };

        bool is_eltwise(bool require_scale_one = true) const {
            using namespace mkldnn::impl;
            return kind == primitive_kind::eltwise
                && IMPLICATION(require_scale_one, eltwise.scale == 1.f);
        }

        bool is_relu(bool require_scale_one = true,
                bool require_nslope_zero = true) const {
            using namespace mkldnn::impl;
            return is_eltwise(require_scale_one)
                && eltwise.alg == alg_kind::eltwise_relu
                && IMPLICATION(require_nslope_zero, eltwise.alpha == 0.f);
        }

        bool is_sum(bool require_scale_one = true) const {
            using namespace mkldnn::impl;
            return kind == primitive_kind::sum
                && IMPLICATION(require_scale_one, sum.scale == 1.f);
        }

        bool is_depthwise() const {
            using namespace mkldnn::impl;
            return kind == primitive_kind::depthwise;
        }

        bool is_dw_conv() const {
            using namespace mkldnn::impl;
            return kind == primitive_kind::convolution;
        }
        bool is_binarization() const {
            using namespace mkldnn::impl;
            return kind == primitive_kind::binarization;
        }
        bool is_quantization() const {
            using namespace mkldnn::impl;
            return kind == primitive_kind::quantization;
        }
    };

    mkldnn_post_ops(): len_(0) {}

//    ~mkldnn_post_ops() {
//        for (int i = 0; i < len_; i++) {
//            auto &post_op = entry_[i];
//            if (post_op.is_quantization()) {
//                delete post_op.quantization.crop_low_data;
//                delete post_op.quantization.crop_high_data;
//                delete post_op.quantization.input_scale_data;
//                delete post_op.quantization.input_shift_data;
//                delete post_op.quantization.output_scale_data;
//                delete post_op.quantization.output_shift_data;
//            }
//        }
//    }

    mkldnn::impl::status_t append_sum(float scale, mkldnn::impl::data_type_t data_type);
    mkldnn::impl::status_t append_eltwise(float scale,
            mkldnn::impl::alg_kind_t alg, float alpha, float beta);
    mkldnn::impl::status_t append_depthwise(mkldnn::impl::alg_kind_t alg,
            const float* weights_data, const float* biases_data);
    mkldnn::impl::status_t append_dw_conv(int in_h, int in_w, int ker_h, int ker_w, int str_h, int str_w,
                                          mkldnn::impl::data_type_t in_dt,
                                          const float* weights_data,
                                          const float* biases_data);
    mkldnn::impl::status_t append_binarization(mkldnn::impl::alg_kind_t alg, const float* weights_data,
                                               const float* output_mask_data);
    mkldnn::impl::status_t append_quantization(mkldnn::impl::alg_kind_t alg,
                                               int crop_low_count, const float* crop_low, int crop_high_count, const float* crop_high,
                                               int input_scale_count, const float* input_scale, int input_shift_count, const float* input_shift,
                                               int output_scale_count, const float* output_scale, int output_shift_count, const float* output_shif);

    int find(mkldnn::impl::primitive_kind_t kind, int start = 0,
            int stop = -1) const {
        if (stop == -1) stop = len_;
        stop = mkldnn::impl::nstl::min(stop, len_);
        for (int idx = start; idx < stop; ++idx)
            if (entry_[idx].kind == kind) return idx;
        return -1;
    }

    int count(mkldnn::impl::primitive_kind_t kind, int start = 0,
             int stop = -1) const {
        if (stop == -1) stop = len_;
        stop = mkldnn::impl::nstl::min(stop, len_);
        int cnt = 0;
        for (int idx = start; idx < stop; ++idx)
            if (entry_[idx].kind == kind) cnt++;
        return cnt;
    }

    bool has_default_values() const { return len_ == 0; }

    bool contain(mkldnn::impl::primitive_kind_t kind, int index) const
    { return find(kind, index, index + 1) == index; }

    enum { capacity = 10 };

    int len_;
    entry_t entry_[capacity];
};

struct mkldnn_primitive_attr: public mkldnn::impl::c_compatible {
    mkldnn_primitive_attr()
        : round_mode_(mkldnn::impl::round_mode::nearest) {}

    mkldnn_primitive_attr *clone() const
    { return new mkldnn_primitive_attr(*this); }

    bool has_default_values() const {
       return true
            && round_mode_ == mkldnn::impl::round_mode::nearest
            && output_scales_.has_default_values()
            && post_ops_.has_default_values()
            && rnn_data_qparams_.has_default_values()
            && rnn_weights_qparams_.has_default_values()
            && input_zero_points_.has_default_values()
            && weights_zero_points_.has_default_values()
            && output_compensations_.has_default_values();
    }

    bool has_asymmetric_quantization() const {
        return true
               && round_mode_ == mkldnn::impl::round_mode::nearest
               && output_scales_.has_default_values()
               && rnn_data_qparams_.has_default_values()
               && rnn_weights_qparams_.has_default_values()
               && (!input_zero_points_.has_default_values() || !weights_zero_points_.has_default_values());
    }

    mkldnn::impl::status_t set_round_mode(
            mkldnn::impl::round_mode_t round_mode);
    mkldnn::impl::status_t set_post_ops(
            const mkldnn::impl::post_ops_t &post_ops);

    mkldnn::impl::round_mode_t round_mode_;
    mkldnn::impl::scales_t output_scales_;
    mkldnn::impl::post_ops_t post_ops_;
    mkldnn::impl::rnn_data_qparams_t rnn_data_qparams_;
    mkldnn::impl::scales_t rnn_weights_qparams_;

    mkldnn::impl::shifts_t<uint8_t> input_zero_points_;
    mkldnn::impl::shifts_t<float> weights_zero_points_;
    mkldnn::impl::shifts_t<int32_t> output_compensations_;
};


#endif
