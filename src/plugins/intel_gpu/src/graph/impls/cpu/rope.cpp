// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "register.hpp"
#include "rope_inst.h"
#include "implementation_map.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
#include "intel_gpu/op/rope.hpp"
#include "openvino/core/parallel.hpp"

namespace cldnn {
namespace cpu {

using RoPE = ov::intel_gpu::op::RoPE;

class RoPEExecutor {
public:
    void execute(const RoPE::Config& config, const ov::TensorVector& inputs, const ov::TensorVector& outputs);
    void selectExecutor(const RoPE::Config& config, ov::element::Type dt);

private:
    struct Executor {
        virtual void execute(const RoPE::Config& config,
                             const ov::TensorVector& inputs,
                             const ov::TensorVector& outputs) = 0;
    };

    template <typename T>
    struct RoPEExecutorRotateHalf;
    template <typename T>
    struct RoPEExecutorInterleaved;
    template <typename T>
    struct RoPEExecutorChatGLM;
    template <typename T>
    struct RoPEExecutorQwen;

    std::shared_ptr<Executor> m_executor;
};

static ov::Tensor slice(ov::Tensor& tensor, int axis, int start, int end, int step = 1) {
    ov::Shape shape = tensor.get_shape();
    ov::Shape new_shape;

    if (end > start) {
        new_shape = shape;
        new_shape[axis] = (end - start - 1) / step + 1;
    } else {
        // squeeze if end == start
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i != static_cast<size_t>(axis)) {
                new_shape.emplace_back(shape[i]);
            }
        }
    }

    auto off = start * tensor.get_strides()[axis]; // strides calc in bytes
    auto* data = reinterpret_cast<uint8_t*>(tensor.data()) + off;

    ov::Tensor new_tensor(tensor.get_element_type(), new_shape, reinterpret_cast<void*>(data));

    return new_tensor;
}

// static ov::Tensor permute(ov::Tensor& tensor, const std::vector<size_t>& order) {
//     auto& orig_shape = tensor.get_shape();
//     size_t rank = orig_shape.size();
//     assert(order.size() == rank);

//     ov::Shape new_shape;
//     for (size_t i = 0; i < rank; i++) {
//         new_shape.emplace_back(orig_shape[order[i]]);
//     }
//     tensor.set_shape(new_shape);
//     return tensor;
//     // return ov::Tensor(tensor.get_element_type(), new_shape, tensor.data());
// }

template <typename DT>
DT& get_data(const ov::Tensor& tensor, const std::initializer_list<size_t>& index, bool allow_broadcast = false, ov::Strides old_strides = {}) {
    const auto& shape = tensor.get_shape();
    // const auto& strides = tensor.get_strides();
    if (old_strides.empty()) {
        old_strides = tensor.get_strides();
    }
    size_t off = 0;
    auto it = index.begin();
    for (size_t i = 0; i < shape.size(); ++i) {
        size_t coordinate = (it != index.end()) ? (*it++) : 0;
        if (allow_broadcast && shape[i] == 1) {
            // allow_broadcast only works when the dimension is really 1
            coordinate = 0;
        } else {
            assert(coordinate < shape[i]);
        }
        off += old_strides[i] * coordinate;
    }
    return (reinterpret_cast<DT*>(reinterpret_cast<uint8_t*>(tensor.data()) + off))[0];
}

template <typename T>
struct RoPEExecutor::RoPEExecutorRotateHalf : public RoPEExecutor::Executor {
    void execute(const RoPE::Config& config,
                 const ov::TensorVector& inputs,
                 const ov::TensorVector& outputs) override {
        auto t_src = inputs[0];
        auto& t_cos = inputs[1];
        auto& t_sin = inputs[2];
        auto& t_dst = outputs[0];
        const ov::Tensor* gather = nullptr;

        if (config.slice_stop - config.slice_start > 0) {
            t_src = slice(t_src, 3, config.slice_start, config.slice_stop);
        }
        // if (config.input_trans0213) {
            // t_src = permute(t_src, {0, 2, 1, 3});
        // }

        if (config.gather_position_arg_id > 0) {
            gather = &inputs[config.gather_position_arg_id];
        }

        auto batch_size = t_src.get_shape()[0];
        auto head_cnt = t_src.get_shape()[1];
        auto seq_len = t_src.get_shape()[2];
        auto feature_size = t_src.get_shape()[3];

        auto rotary_dims = config.rotary_ndims;
        auto half_rotary_dims = rotary_dims / 2;

        ov::parallel_for3d(batch_size, head_cnt, seq_len, [&](size_t b, size_t h, size_t p) {
            auto cos_pos = p;
            if (gather != nullptr) {
                if (gather->get_shape().size() == 4)
                    cos_pos = get_data<int32_t>(*gather, {b, h, p, 0}, true);
                else
                    cos_pos = get_data<int32_t>(*gather, {b, p}, true);
            }
            T* src = &get_data<T>(t_src, {b, h, p, 0});
            float* cos = &get_data<float>(t_cos, {b, h, cos_pos, 0}, true);
            float* sin = &get_data<float>(t_sin, {b, h, cos_pos, 0}, true);
            T* dst = &get_data<T>(t_dst, {b, h, p, 0});

            size_t i = 0;
            for (; i < half_rotary_dims; ++i) {
                dst[i] = cos[i] * src[i] + sin[i] * (-src[i + half_rotary_dims]);
            }
            for (; i < rotary_dims; ++i) {
                dst[i] = cos[i] * src[i] + sin[i] * (src[i - half_rotary_dims]);
            }
            for (; i < feature_size; ++i) {
                dst[i] = src[i];
            }
        });
    }
};

template <typename T>
struct RoPEExecutor::RoPEExecutorInterleaved : public RoPEExecutor::Executor {
    void execute(const RoPE::Config& config,
                 const ov::TensorVector& inputs,
                 const ov::TensorVector& outputs) override {
        auto t_src(inputs[0]);
        auto t_sin_cos(inputs[1]);
        auto t_dst(outputs[0]);

        auto batch_size = t_src.get_shape()[0];
        auto seq_len = t_src.get_shape()[1];
        auto head_cnt = t_src.get_shape()[2];
        auto head_dims = t_src.get_shape()[3];

        auto rotary_dims = config.rotary_ndims;
        auto half_rotary_dims = rotary_dims / 2;
        ov::parallel_for3d(batch_size, seq_len, head_cnt, [&](size_t b, size_t p, size_t h) {
            T* x = &get_data<T>(t_src, {b, p, h, 0});
            float* sin = &get_data<float>(t_sin_cos, {b, p, 0}, true);
            float* cos = &get_data<float>(t_sin_cos, {b, p, half_rotary_dims}, true);
            T* dst = &get_data<T>(t_dst, {b, h, p, 0});

            size_t i = 0;
            for (size_t j = 0; i < rotary_dims; i += 2, j++) {
                dst[i] = cos[j] * x[i] - sin[j] * x[i + 1];
                dst[i + 1] = cos[j] * x[i + 1] + sin[j] * x[i];
            }
            for (; i < head_dims; i++) {
                dst[i] = x[i];
            }
        });
    }
};

template <typename T>
struct RoPEExecutor::RoPEExecutorChatGLM : public RoPEExecutor::Executor {
    void execute(const RoPE::Config& config,
                 const ov::TensorVector& inputs,
                 const ov::TensorVector& outputs) override {
        auto t_src(inputs[0]);
        auto t_cos_sin(inputs[1]);
        auto t_dst(outputs[0]);

        auto old_strides = t_src.get_strides();

        // [seq_len, batch_size, (hidden_states_q + hidden_states_k + hidden_states_v)]
        if (config.slice_stop - config.slice_start > 0) {
            t_src = slice(t_src, 2, config.slice_start, config.slice_stop);
        }

        auto seq_len = t_src.get_shape()[0];
        auto batch_size = t_src.get_shape()[1];

        auto head_cnt = config.head_cnt;
        auto head_size = config.head_size;

        auto rotary_dims = config.rotary_ndims;

        ov::parallel_for3d(seq_len, batch_size, head_cnt, [&](size_t p, size_t b, size_t h) {
            T* src = &get_data<T>(t_src, {p, b, h * head_size}, false, old_strides);
            // [length, batch_size, ndims//2, 2]
            T* cos_sin = &get_data<T>(t_cos_sin, {p, b, 0, 0}, true);
            T* dst = &get_data<T>(t_dst, {p, b, h, 0});

            size_t i = 0;
            for (; i < rotary_dims; i += 2) {
                auto cosv = cos_sin[i];
                auto sinv = cos_sin[i + 1];
                dst[i] = cosv * src[i] - sinv * src[i + 1];
                dst[i + 1] = sinv * src[i] + cosv * src[i + 1];
            }
            for (; i < head_size; i++) {
                dst[i] = src[i];
            }
        });
    }
};

template <typename T>
struct RoPEExecutor::RoPEExecutorQwen : public RoPEExecutor::Executor {
    void execute(const RoPE::Config& config,
                 const ov::TensorVector& inputs,
                 const ov::TensorVector& outputs) override {
        auto t_src(inputs[0]);    // [batch, length, head_cnt*head_size * 3]
        auto t_cos(inputs[1]);    // [1, present-kv-length, 1, rotary_dims]
        auto t_sin(inputs[2]);    // [1, present-kv-length, 1, rotary_dims]
        auto t_dst(outputs[0]);   // [batch, length, head_cnt, head_size]>

        if (config.slice_stop - config.slice_start > 0) {
            t_src = slice(t_src, 2, config.slice_start, config.slice_stop);
        }

        auto batch_size = t_src.get_shape()[0];
        auto seq_len = t_src.get_shape()[1];
        auto head_cnt = config.head_cnt;
        auto head_size = config.head_size;
        auto present_kv_len = t_cos.get_shape()[1];

        auto rotary_dims = t_cos.get_shape()[3];
        auto half_rotary_dims = rotary_dims / 2;

        ov::parallel_for3d(batch_size, seq_len, head_cnt, [&](size_t b, size_t p, size_t h) {
            T* src = &get_data<T>(t_src, {b, p, h * head_size});
            float* cos = &get_data<float>(t_cos, {b, present_kv_len - seq_len + p, h, 0}, true);
            float* sin = &get_data<float>(t_sin, {b, present_kv_len - seq_len + p, h, 0}, true);
            T* dst = &get_data<T>(t_dst, {b, p, h, 0});

            size_t i = 0;
            for (; i < half_rotary_dims; i++) {
                dst[i] = cos[i] * src[i] + sin[i] * (-src[i + half_rotary_dims]);
            }
            for (; i < rotary_dims; i++) {
                dst[i] = cos[i] * src[i] + sin[i] * (src[i - half_rotary_dims]);
            }
            for (; i < head_size; i++) {
                dst[i] = src[i];
            }
        });
    }
};

void RoPEExecutor::selectExecutor(const RoPE::Config& config, ov::element::Type data_type) {
    if (config.is_qwen) {
        if (data_type == ov::element::f16) {
            m_executor = std::make_shared<RoPEExecutorQwen<ov::float16>>();
        } else {
            m_executor = std::make_shared<RoPEExecutorQwen<float>>();
        }
    } else if (config.is_chatglm) {
        if (data_type == ov::element::f16) {
            m_executor = std::make_shared<RoPEExecutorChatGLM<ov::float16>>();
        } else {
            m_executor = std::make_shared<RoPEExecutorChatGLM<float>>();
        }
    } else if (config.is_interleaved) {
        OPENVINO_ASSERT(config.input_trans0213 == false);
        OPENVINO_ASSERT(config.slice_start == 0);
        OPENVINO_ASSERT(config.slice_stop == 0);
        OPENVINO_ASSERT(config.gather_position_arg_id == 0);
        if (data_type == ov::element::f16) {
            m_executor = std::make_shared<RoPEExecutorInterleaved<ov::float16>>();
        } else {
            m_executor = std::make_shared<RoPEExecutorInterleaved<float>>();
        }
    } else {
        if (data_type == ov::element::f16) {
            m_executor = std::make_shared<RoPEExecutorRotateHalf<ov::float16>>();
        } else {
            m_executor = std::make_shared<RoPEExecutorRotateHalf<float>>();
        }
    }
}

void RoPEExecutor::execute(const RoPE::Config& config,
                           const ov::TensorVector& inputs,
                           const ov::TensorVector& outputs) {
    OPENVINO_ASSERT(m_executor != nullptr);
    m_executor->execute(config, inputs, outputs);
}

struct rope_impl : public typed_primitive_impl<rope> {
    using parent = typed_primitive_impl<rope>;
    using parent::parent;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::cpu::rope_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<rope_impl>(*this);
    }

    rope_impl() : parent("rope_cpu_impl") {}

    // void save(BinaryOutputBuffer& ob) const override {
    //     parent::save(ob);
    //     // ob << make_data();
    // }

    // void load(BinaryInputBuffer& ib) override {
    //     parent::load(ib);
    //     // ib >> make_data(&, sizeof(ov::op::));
    // }

    event::ptr execute_impl(const std::vector<event::ptr>& events, rope_inst& instance) override {
        OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "rope::execute_impl");

        for (auto e : events) {
            e->wait();
        }

        auto& stream = instance.get_network().get_stream();
        auto ev = stream.create_user_event(false);

        auto params = instance.get_impl_params();
        const auto& primitive = params->typed_desc<cldnn::rope>();
        const auto& config = primitive->config;

        ov::TensorVector input_host_tensors;
        ov::TensorVector output_host_tensors;

        std::vector<memory::ptr> input_mem_ptrs;
        for (size_t i = 0; i < instance.dependencies().size(); ++i) {
            input_mem_ptrs.push_back(instance.dep_memory_ptr(i));
        }

        for (size_t i = 0; i < input_mem_ptrs.size(); ++i) {
            void* mem_ptr = input_mem_ptrs[i]->lock(stream, mem_lock_type::read);
            input_host_tensors.push_back(make_tensor(params->input_layouts[i], mem_ptr));
        }

        auto output_mem_ptr = instance.output_memory_ptr();

        cldnn::mem_lock<uint8_t, mem_lock_type::write> output_lock(output_mem_ptr, stream);
        output_host_tensors.push_back(make_tensor(params->output_layouts[0], output_lock.data()));

        RoPEExecutor rope;
        rope.selectExecutor(config, params->get_input_layout().data_type);
        rope.execute(config, input_host_tensors, output_host_tensors);

        for (size_t i = 0; i < input_mem_ptrs.size(); ++i) {
            input_mem_ptrs[i]->unlock(stream);
        }

        ev->set();

        return ev;
    }

    void init_kernels(const kernels_cache& , const kernel_impl_params&) override {}

    void update_dispatch_data(const kernel_impl_params& impl_param) override {}

public:
    static std::unique_ptr<primitive_impl> create(const rope_node& arg, const kernel_impl_params& impl_param) {
        return make_unique<rope_impl>();
    }
};

namespace detail {

attach_rope_impl::attach_rope_impl() {
    auto formats = {
        format::bfyx,
    };

    auto types = {
        data_types::f32,
        data_types::f16,
    };

    implementation_map<rope>::add(impl_types::cpu, shape_types::static_shape, rope_impl::create, types, formats);
    implementation_map<rope>::add(impl_types::cpu, shape_types::dynamic_shape, rope_impl::create, types, formats);
}

}  // namespace detail
}  // namespace cpu
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::cpu::rope_impl)
