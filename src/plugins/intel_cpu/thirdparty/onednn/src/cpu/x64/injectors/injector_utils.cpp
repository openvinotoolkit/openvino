/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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
#include <numeric>
#include "common/broadcast_strategy.hpp"
#include "cpu/x64/injectors/injector_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace injector_utils {

static std::size_t get_vmm_size_bytes(const Xbyak::Xmm &vmm) {
    static constexpr int byte_size_bits = 8;
    return vmm.getBit() / byte_size_bits;
}

static std::size_t calc_vmm_to_preserve_size_bytes(
        const std::initializer_list<Xbyak::Xmm> &vmm_to_preserve) {

    return std::accumulate(vmm_to_preserve.begin(), vmm_to_preserve.end(),
            std::size_t(0u), [](std::size_t accum, const Xbyak::Xmm &vmm) {
                return accum + get_vmm_size_bytes(vmm);
            });
}

register_preserve_guard_t::register_preserve_guard_t(jit_generator *host,
        std::initializer_list<Xbyak::Reg64> reg64_to_preserve,
        std::initializer_list<Xbyak::Xmm> vmm_to_preserve)
    : host_(host)
    , reg64_stack_(reg64_to_preserve)
    , vmm_stack_(vmm_to_preserve)
    , vmm_to_preserve_size_bytes_(
              calc_vmm_to_preserve_size_bytes(vmm_to_preserve)) {

    for (const auto &reg : reg64_to_preserve)
        host_->push(reg);

    if (!vmm_stack_.empty()) {
        host_->sub(host_->rsp, vmm_to_preserve_size_bytes_);

        auto stack_offset = vmm_to_preserve_size_bytes_;
        for (const auto &vmm : vmm_to_preserve) {
            stack_offset -= get_vmm_size_bytes(vmm);
            const auto idx = vmm.getIdx();
            if (vmm.isXMM())
                host_->uni_vmovups(
                        host_->ptr[host_->rsp + stack_offset], Xbyak::Xmm(idx));
            else if (vmm.isYMM())
                host_->uni_vmovups(
                        host_->ptr[host_->rsp + stack_offset], Xbyak::Ymm(idx));
            else
                host_->uni_vmovups(
                        host_->ptr[host_->rsp + stack_offset], Xbyak::Zmm(idx));
        }
    }
}

register_preserve_guard_t::~register_preserve_guard_t() {

    auto tmp_stack_offset = 0;

    while (!vmm_stack_.empty()) {
        const Xbyak::Xmm &vmm = vmm_stack_.top();
        const auto idx = vmm.getIdx();
        if (vmm.isXMM())
            host_->uni_vmovups(
                    Xbyak::Xmm(idx), host_->ptr[host_->rsp + tmp_stack_offset]);
        else if (vmm.isYMM())
            host_->uni_vmovups(
                    Xbyak::Ymm(idx), host_->ptr[host_->rsp + tmp_stack_offset]);
        else
            host_->uni_vmovups(
                    Xbyak::Zmm(idx), host_->ptr[host_->rsp + tmp_stack_offset]);

        tmp_stack_offset += get_vmm_size_bytes(vmm);
        vmm_stack_.pop();
    }

    if (vmm_to_preserve_size_bytes_)
        host_->add(host_->rsp, vmm_to_preserve_size_bytes_);

    while (!reg64_stack_.empty()) {
        host_->pop(reg64_stack_.top());
        reg64_stack_.pop();
    }
}

size_t register_preserve_guard_t::stack_space_occupied() const {
    constexpr static size_t reg64_size = 8;
    const size_t stack_space_occupied
            = vmm_to_preserve_size_bytes_ + reg64_stack_.size() * reg64_size;

    return stack_space_occupied;
};

conditional_register_preserve_guard_t::conditional_register_preserve_guard_t(
        bool condition_to_be_met, jit_generator *host,
        std::initializer_list<Xbyak::Reg64> reg64_to_preserve,
        std::initializer_list<Xbyak::Xmm> vmm_to_preserve)
    : register_preserve_guard_t {condition_to_be_met
                    ? register_preserve_guard_t {host, reg64_to_preserve,
                            vmm_to_preserve}
                    : register_preserve_guard_t {nullptr, {}, {}}} {};

} // namespace injector_utils
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
