// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/serialization/binary_buffer.hpp"
#include "intel_gpu/graph/serialization/cl_kernel_data_serializer.hpp"
#include "intel_gpu/graph/serialization/helpers.hpp"
#include "intel_gpu/graph/serialization/set_serializer.hpp"
#include "intel_gpu/graph/serialization/string_serializer.hpp"
#include "intel_gpu/graph/serialization/vector_serializer.hpp"
#include "intel_gpu/graph/program.hpp"

#include "kernel_selector_common.h"
#include "openvino/core/except.hpp"
#include "primitive_inst.h"
#include "kernel_selector_helper.h"
#include "register.hpp"
#include "registry/implementation_map.hpp"
#include "concatenation_inst.h"
#include "gather_inst.h"
#include "permute_inst.h"

#include <vector>
#include <list>
#include <utility>

namespace cldnn {
namespace ocl {

/*
Base class for GPU implementations which require multiple kernel selectors to be used and multiple kernel scheduled.
*/
template <class PType>
struct multi_stage_primitive : public typed_primitive_impl<PType> {
    std::vector<kernel_selector::kernel_data> _kernels_data;
    std::vector<kernel::ptr> _kernels;

    // a pair of batch program hash and kernel entry hash of each ocl impl.
    std::pair<std::string, std::string> kernel_dump_info;

    multi_stage_primitive() : _kernels_data({}), _kernels({}) {}

    multi_stage_primitive(const multi_stage_primitive<PType>& other)
        : typed_primitive_impl<PType>()
        , _kernels_data(other._kernels_data)
        , _kernels({}) {
        _kernels.reserve(other._kernels.size());
        for (size_t k = 0; k < other._kernels.size(); ++k) {
            _kernels.emplace_back(other._kernels[k]->clone(other.can_share_kernels));
        }
        this->can_reuse_memory = other.can_reuse_memory;
        this->can_share_kernels = other.can_share_kernels;
        this->_kernel_name = other._kernel_name;
        this->can_reuse_memory = other.can_reuse_memory;
        this->_is_dynamic = other._is_dynamic;
        this->m_manager = other.m_manager;
    }

    multi_stage_primitive(const std::vector<kernel_selector::kernel_data>& kd)
        : typed_primitive_impl<PType>()
        , _kernels_data(kd) {
        this->can_reuse_memory = false;
        this->_kernel_name = kd[0].kernelName;
    }

    bool is_cpu() const final { return false; }

    // Cache blob format:
    //     [ kernel_selector::kernel_data ]
    //     [ kernel_ids ]
    void save(BinaryOutputBuffer& ob) const override {
        primitive_impl::save(ob);
        ob << _kernels_data.size();
        for (auto& kd : _kernels_data) {
            ob << make_data(&kd.internalBufferDataType, sizeof(kernel_selector::Datatype));
            ob << kd.internalBuffers;
            ob << kd.kernels;
            ob << kd.kernelName;
        }
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_impl::load(ib);
        size_t kernels_size;
        ib >> kernels_size;
        _kernels_data.resize(kernels_size);
        for (size_t i = 0; i < kernels_size; i++) {
            kernel_selector::kernel_data kd;
            ib >> make_data(&kd.internalBufferDataType, sizeof(kernel_selector::Datatype));
            ib >> kd.internalBuffers;
            ib >> kd.kernels;
            ib >> kd.kernelName;
            _kernels_data[i] = kd;
        }
    }

    void update(primitive_inst& inst, const kernel_impl_params& impl_params) override {
        auto new_impl_params = this->canonicalize_shapes(impl_params);
        update_dispatch_data(new_impl_params);
        inst.update_shape_info_tensor(new_impl_params);
    }

protected:
    virtual kernel_arguments_data get_arguments(const typed_primitive_inst<PType>& instance, size_t stage) const = 0;

    void init_kernels(const kernels_cache& kernels_cache, const kernel_impl_params& params) override {
        _kernels.clear();
        if (!_kernels_data.empty() && !_kernels_data[0].kernels.empty()) {
            auto compiled_kernels = kernels_cache.get_kernels(params);
            size_t total_kernels = std::accumulate(_kernels_data.begin(), _kernels_data.end(), (size_t)0,
                [](size_t acc, const kernel_selector::kernel_data& kd) {
                    return acc + kd.kernels.size();
                });
            OPENVINO_ASSERT(total_kernels == compiled_kernels.size(), "[GPU] Mismatch between number of expected and actually compiled kernels.\n",
                                                                      "Expected: ", total_kernels, "\n"
                                                                      "Got: ", compiled_kernels.size());
            _kernels.insert(_kernels.begin(), compiled_kernels.begin(), compiled_kernels.end());
            // batch program hash and kernel entry point to find corresponding cl source code
            kernel_dump_info = std::make_pair(std::to_string(kernels_cache.get_kernel_batch_hash(params)),
                                              _kernels_data[0].kernels[0].code.kernelString->entry_point);
            for (size_t i = 1; i < _kernels_data[0].kernels.size(); ++i)
                kernel_dump_info.second += " " + _kernels_data[0].kernels[i].code.kernelString->entry_point;
        }
        this->can_share_kernels = kernels_cache.get_kernels_reuse();
    }

    void init_by_cached_kernels(const kernels_cache& kernels_cache, std::vector<std::string>& cached_kernel_ids) override {
        _kernels.clear();

        _kernels.reserve(cached_kernel_ids.size());
        for (size_t k = 0; k < cached_kernel_ids.size(); ++k) {
            _kernels.emplace_back(kernels_cache.get_kernel_from_cached_kernels(cached_kernel_ids[k]));
        }
        this->can_share_kernels = kernels_cache.get_kernels_reuse();
    }

    std::vector<std::string> get_cached_kernel_ids(const kernels_cache& kernels_cache) override {
        return {kernels_cache.get_cached_kernel_ids(_kernels)};
    }

    template<typename ImplType, typename KernelParamsType>
    static std::unique_ptr<primitive_impl> make_deep_copy(const ImplType& impl_ocl) {
        auto prim_impl = std::make_unique<ImplType>(impl_ocl);
        for (auto& _kernel_data : (*prim_impl)._kernels_data) {
            KernelParamsType* params_ptr = dynamic_cast<KernelParamsType*>(_kernel_data.params.get());
            if (params_ptr != nullptr) {
                _kernel_data.params = std::make_unique<KernelParamsType>(*params_ptr);
            }
        }
        return prim_impl;
    }

    std::vector<kernel::ptr> get_kernels() const override {
        return _kernels;
    }

    std::vector<BufferDescriptor> get_internal_buffer_descs(const kernel_impl_params&) const override {
        std::vector<BufferDescriptor> internal_buffers;
        for (auto& kd : _kernels_data) {
            if (kd.internalBuffers.empty())
                continue;

            auto dtype = from_data_type(kd.internalBufferDataType);
            const auto bpp = data_type_traits::size_of(dtype);
            for (const auto& buffer : kd.internalBuffers) {
                internal_buffers.emplace_back(buffer.byte_count / bpp, dtype, buffer.lockable);
            }
        }
        return internal_buffers;
    }

    void set_arguments_impl(typed_primitive_inst<PType>& instance) override {
        if (instance.can_be_optimized()) {
            return;
        }

        for (size_t stage = 0; stage < _kernels_data.size(); stage++) {
            auto& kd = _kernels_data[stage];
            stream& stream = instance.get_network().get_stream();
            for (size_t kd_idx = 0; kd_idx < kd.kernels.size(); ++kd_idx) {
                if (kd.kernels[kd_idx].skip_execution) {
                    continue;
                }

                auto args = get_arguments(instance, stage);
                args.scalars = &kd.kernels[kd_idx].params.scalars;

                for (const auto& m : instance.get_intermediates_memories()) {
                    args.intermediates.push_back(m);
                }

                stream.set_arguments(*_kernels[kd_idx], kd.kernels[kd_idx].params, args);
            }
        }
    }

    void set_arguments_impl(typed_primitive_inst<PType>& instance, kernel_arguments_data& args) override {
        OPENVINO_NOT_IMPLEMENTED;
    }

    std::vector<std::shared_ptr<cldnn::kernel_string>> get_kernels_source() override {
        std::vector<std::shared_ptr<cldnn::kernel_string>> kernel_strings;
        for (auto& kd : _kernels_data) {
            for (size_t i = 0; i < kd.kernels.size(); ++i) {
                kernel_strings.push_back(kd.kernels[i].code.kernelString);
            }
        }
        return kernel_strings;
    }

    void reset_kernels_source() override {
        for (auto& kd : _kernels_data) {
            for (size_t i = 0; i < kd.kernels.size(); ++i) {
                kd.kernels[i].code.kernelString.reset();
            }
        }
    }

    void set_kernels(cldnn::kernels_cache::compiled_kernels kernels) override {
        OPENVINO_ASSERT(kernels.size() == 1, "Only the kernels of the single primitive should be allowed.");
        auto& kernel_vec = kernels.begin()->second;
        _kernels.clear();
        _kernels.resize(kernel_vec.size());
        for (auto& k : kernel_vec) {
            auto sub_kernel_idx = k.second;
            _kernels[sub_kernel_idx] = k.first;
        }
    }

    std::pair<std::string, std::string> get_kernels_dump_info() const override {
        return kernel_dump_info;
    }

    virtual void update_dispatch_data(const kernel_impl_params& impl_params) {
        OPENVINO_ASSERT(this->_is_dynamic, "[GPU] update_dispatch_data() is called for static shape implementation ", this-> _kernel_name);
        OPENVINO_ASSERT(false, "[GPU] update_dispatch_data() is not implemented for dynamic implemenation ", this->_kernel_name);
    }
};

}  // namespace ocl
}  // namespace cldnn
