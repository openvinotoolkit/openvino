/*******************************************************************************
* Copyright 2017-2021 Intel Corporation
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

#ifndef DNNL_MEMORY_HPP
#define DNNL_MEMORY_HPP

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_DPCPP
#include "oneapi/dnnl/dnnl_sycl.h"
#endif

#include "common.hpp"
#include "dnnl_common.hpp"

#define dnnl_mem_default_value 0xFF

struct dnn_mem_t {
    struct handle_info_t {
        bool is_host_ptr;
        void *ptr;

        bool is_allocate() const { return ptr == DNNL_MEMORY_ALLOCATE; }

        static handle_info_t allocate() {
            return {false, DNNL_MEMORY_ALLOCATE};
        }
    };

    dnn_mem_t() { map(); }

    dnn_mem_t(const dnnl_memory_desc_t &md, dnnl_engine_t engine) {
        active_ = (initialize(md, engine) == OK);
    }

    dnn_mem_t(const dnnl_memory_desc_t &md, dnnl_engine_t engine,
            const handle_info_t &handle_info) {
        active_ = (initialize(md, engine, handle_info) == OK);
    }

    dnn_mem_t(int ndims, const dnnl_dims_t dims, dnnl_data_type_t dt,
            const std::string &tag, dnnl_engine_t engine) {
        active_ = (initialize(ndims, dims, dt, tag, engine) == OK);
    }

    dnn_mem_t(int ndims, const dnnl_dims_t dims, dnnl_data_type_t dt,
            const dnnl_dims_t strides, dnnl_engine_t engine) {
        active_ = (initialize(ndims, dims, dt, strides, engine) == OK);
    }

    dnn_mem_t(const dnnl_memory_desc_t &md, dnnl_data_type_t dt,
            const std::string &tag, dnnl_engine_t engine) {
        active_ = (initialize(md, dt, tag, engine) == OK);
    }

    dnn_mem_t(const dnnl_memory_desc_t &md, dnnl_data_type_t dt,
            dnnl_engine_t engine) {
        active_ = (initialize(md, dt, tag::undef, engine) == OK);
    }

    dnn_mem_t(const dnn_mem_t &rhs, dnnl_data_type_t dt, const std::string &tag,
            dnnl_engine_t engine)
        : dnn_mem_t(rhs.md_, dt, tag, engine) {
        if (active_) reorder(rhs);
    }

    dnn_mem_t(const dnn_mem_t &rhs) = delete;
    dnn_mem_t &operator=(const dnn_mem_t &rhs) = delete;

    dnn_mem_t &operator=(dnn_mem_t &&rhs) {
        if (&rhs == this) return *this;
        cleanup();

        md_ = rhs.md_;
        m_ = rhs.m_;
        data_ = rhs.data_;
        is_data_owner_ = rhs.is_data_owner_;
        active_ = rhs.active_;
        engine_kind_ = rhs.engine_kind_;
        engine_ = rhs.engine_;
        is_mapped_ = (bool)rhs.is_mapped_;
        mapped_ptr_ = rhs.mapped_ptr_;

        rhs.active_ = false;
        return *this;
    }
    dnn_mem_t(dnn_mem_t &&rhs) : dnn_mem_t() { *this = std::move(rhs); }

    ~dnn_mem_t() { cleanup(); }

    int reorder(const dnn_mem_t &rhs, const_dnnl_primitive_attr_t attr);
    int reorder(const dnn_mem_t &rhs) { return reorder(rhs, nullptr); }

    size_t size() const { return dnnl_memory_desc_get_size(&md_); }

    int64_t nelems(bool with_padded_dims = false) const {
        auto dims = with_padded_dims ? md_.padded_dims : md_.dims;
        if (md_.ndims == 0) return 0;

        int64_t n = 1;
        for (int i = 0; i < md_.ndims; ++i)
            n *= dims[i];
        return n;
    }

    dnnl_data_type_t dt() const { return md_.data_type; }
    size_t sizeof_dt() const { return ::sizeof_dt(dt()); }

    void set_dt(dnnl_data_type_t dt) { md_.data_type = dt; }

    template <typename T>
    explicit operator T *() const {
        assert(is_mapped_);
        return static_cast<T *>(mapped_ptr_);
    }

    float get_elem(int64_t idx) const {
        void *data = (void *)*this;
        float elem = 0.0;
        switch (dt()) {
            case dnnl_s8: elem = static_cast<int8_t *>(data)[idx]; break;
            case dnnl_u8: elem = static_cast<uint8_t *>(data)[idx]; break;
            case dnnl_s32: elem = static_cast<int32_t *>(data)[idx]; break;
            case dnnl_f32: elem = static_cast<float *>(data)[idx]; break;
            case dnnl_f16: elem = static_cast<float16_t *>(data)[idx]; break;
            case dnnl_bf16: elem = static_cast<bfloat16_t *>(data)[idx]; break;
            default: assert(!"bad data type");
        }
        return elem;
    }

    void set_elem(int64_t idx, float value) {
        void *data = (void *)*this;
        switch (dt()) {
            case dnnl_s8: ((int8_t *)data)[idx] = value; break;
            case dnnl_u8: ((uint8_t *)data)[idx] = value; break;
            case dnnl_s32: ((int32_t *)data)[idx] = value; break;
            case dnnl_f32: ((float *)data)[idx] = value; break;
            case dnnl_f16: ((float16_t *)data)[idx] = value; break;
            case dnnl_bf16: ((bfloat16_t *)data)[idx] = value; break;
            default: assert(!"bad data type");
        }
    }

    int64_t get_scale_idx(
            int64_t data_idx, int scale_mask, const int ndims) const {
        const auto &dims = md_.dims;
        int64_t stride = 1;
        int64_t offset = 0;

        if (scale_mask != 0) {
            for (int i = 0; i < ndims; ++i) {
                int d = ndims - 1 - i;
                auto pos = data_idx % dims[d];
                data_idx /= dims[d];
                if (scale_mask & (1 << d)) {
                    offset += pos * stride;
                    stride *= dims[d];
                }
            }
        }

        return offset;
    }

    int64_t get_scale_idx(int64_t data_idx, int scale_mask) const {
        return get_scale_idx(data_idx, scale_mask, md_.ndims);
    }

    dnnl_engine_t engine() const { return engine_; }
    dnnl_engine_kind_t engine_kind() const { return engine_kind_; }

    bool is_mapped() const { return is_mapped_; }

    void map() const {
        assert(!is_mapped_ && "memory is already mapped");
        is_mapped_ = true;

        if (!m_) return;
        DNN_SAFE_V(dnnl_memory_map_data(m_, &mapped_ptr_));
    }

    void unmap() const {
        assert(is_mapped_ && "memory is not mapped");
        is_mapped_ = false;

        if (!m_) return;
        DNN_SAFE_V(dnnl_memory_unmap_data(m_, mapped_ptr_));
        mapped_ptr_ = NULL;
    }

    static dnn_mem_t create_from_host_ptr(
            const dnnl_memory_desc_t &md, dnnl_engine_t engine, void *host_ptr);

    /* fields */
    dnnl_memory_desc_t md_ {};
    dnnl_memory_t m_ {};

private:
    void *data_ = NULL;
    bool is_data_owner_ = false;
    bool active_ = false;

    dnnl_engine_kind_t engine_kind_ = dnnl_any_engine;
    dnnl_engine_t engine_ = NULL;

    mutable bool is_mapped_ = false;
    mutable void *mapped_ptr_ = NULL;

    int initialize_memory_create_sycl(const handle_info_t &handle_info);
    int initialize_memory_create_opencl(const handle_info_t &handle_info);

    int initialize_memory_create(const handle_info_t &handle_info);

    int initialize(const dnnl_memory_desc_t &md, dnnl_data_type_t dt,
            const std::string &tag, dnnl_engine_t engine,
            const handle_info_t &handle_info = handle_info_t::allocate()) {
        is_mapped_ = false;

        if (tag == tag::undef) {
            md_ = md;
            md_.data_type = dt;
        } else {
            SAFE(init_md(&md_, md.ndims, md.dims, dt, tag), CRIT);
        }
        engine_ = engine;
        DNN_SAFE(dnnl_engine_get_kind(engine_, &engine_kind_), CRIT);

        SAFE(initialize_memory_create(handle_info), CRIT);

        size_t sz = dnnl_memory_desc_get_size(&md_);
        // Do not fill a memory if its size is zero. Moreover, memset expects
        // defined pointer, nullptr is not allowed.
        if (sz != 0 && handle_info.is_allocate()) {
            // Fill memory with a magic number (NAN for fp data types) to catch
            // possible uninitialized access.
            map();
            memset(mapped_ptr_, dnnl_mem_default_value, sz);
            unmap();
        }

        // Keep memory mapped and unmap only before execution
        map();

        return OK;
    }

    int initialize(const dnnl_memory_desc_t &md, dnnl_engine_t engine,
            const handle_info_t &handle_info = handle_info_t::allocate()) {
        return initialize(md, md.data_type, tag::undef, engine, handle_info);
    }

    int initialize(int ndims, const dnnl_dims_t dims, dnnl_data_type_t dt,
            const std::string &tag, dnnl_engine_t engine) {
        dnnl_memory_desc_t xmd;
        SAFE(init_md(&xmd, ndims, dims, dt, tag), CRIT);
        SAFE(initialize(xmd, engine), CRIT);
        return OK;
    }

    int initialize(int ndims, const dnnl_dims_t dims, dnnl_data_type_t dt,
            const dnnl_dims_t strides, dnnl_engine_t engine) {
        dnnl_memory_desc_t xmd;
        DNN_SAFE(dnnl_memory_desc_init_by_strides(
                         &xmd, ndims, dims, dt, strides),
                CRIT);
        SAFE(initialize(xmd, engine), CRIT);
        return OK;
    }

    int cleanup_sycl();
    int cleanup_opencl();

    int cleanup() {
        if (!active_) return OK;
        unmap();
        DNN_SAFE(dnnl_memory_destroy(m_), CRIT);
        if (is_data_owner_) {
            if (is_sycl_engine(engine_)) {
                SAFE(cleanup_sycl(), CRIT);
            } else if (is_opencl_engine(engine_)) {
                SAFE(cleanup_opencl(), CRIT);
            } else {
                zfree(data_);
            }
        }
        return OK;
    }
};

// Check that zero padding is preserved.
int check_zero_padding(
        const dnn_mem_t &mem, int arg, int *error_count = nullptr);

// Returns physical offset by logical one. Logical offset is represented by an
// array pos. If is_pos_padded is true pos represents the position in already
// padded area.
dnnl_dim_t md_off_v(const dnnl_memory_desc_t &md, const dnnl_dims_t pos,
        bool is_pos_padded = false);

#endif
