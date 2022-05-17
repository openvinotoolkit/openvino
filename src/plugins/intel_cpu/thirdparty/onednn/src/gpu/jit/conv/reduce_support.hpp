/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#ifndef GPU_JIT_CONV_REDUCE_SUPPORT_HPP
#define GPU_JIT_CONV_REDUCE_SUPPORT_HPP

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

// Implements reduction of GRF buffer for given layout.
class reduce_t : public func_impl_t {
public:
    IR_DECL_DERIVED_TYPE_ID(reduce_t, func_impl_t)

    static func_t make(const layout_t &src_layout, const layout_t &dst_layout) {
        return func_t(new reduce_t(src_layout, dst_layout));
    }

    bool is_equal(const object_impl_t *obj) const override {
        if (!obj->is<self_type>()) return false;
        auto &other = obj->as<self_type>();

        return (src_layout == other.src_layout)
                && (dst_layout == other.dst_layout);
    }

    size_t get_hash() const override {
        return ir_utils::get_hash(src_layout, dst_layout);
    }

    std::string str() const override {
        std::ostringstream oss;
        oss << "reduce[" << src_layout << ", " << dst_layout << "]";
        return oss.str();
    }

    IR_DEFINE_ARG_GET(dst_buf, 0)
    IR_DEFINE_ARG_GET(src_buf, 1)

    layout_t src_layout;
    layout_t dst_layout;

private:
    reduce_t(const layout_t &src_layout, const layout_t &dst_layout)
        : src_layout(src_layout), dst_layout(dst_layout) {}
};
} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
