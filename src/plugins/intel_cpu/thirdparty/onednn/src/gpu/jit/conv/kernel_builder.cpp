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

#include "gpu/jit/conv/kernel_builder.hpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>
#include <unordered_map>

#include "gpu/jit/conv/config.hpp"
#include "gpu/jit/conv/fma_support.hpp"
#include "gpu/jit/conv/gemm_schedule.hpp"
#include "gpu/jit/conv/ir.hpp"
#include "gpu/jit/conv/message_support.hpp"
#include "gpu/jit/conv/post_op_support.hpp"
#include "gpu/jit/conv/reduce_support.hpp"
#include "gpu/jit/conv/reorder_support.hpp"
#include "gpu/jit/conv/tensor.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

class permutation_injector_t : public ir_mutator_t {
public:
    permutation_injector_t(const grf_permutator_t &grf_perm)
        : grf_perm_(new grf_permutator_t(grf_perm)) {}

    object_t _mutate(const func_call_t *obj) override {
        if (!is_func_call<reorder_t>(obj)) return ir_mutator_t::_mutate(obj);

        auto &func = obj->func.as<reorder_t>();
        auto new_func
                = reorder_t::make(func.src_layout, func.dst_layout, grf_perm_);

        return new_func.call(obj->args);
    }

private:
    std::shared_ptr<grf_permutator_t> grf_perm_;
};

class dpasw_injector_t {
public:
    dpasw_injector_t(ngen::HW hw, const stmt_t &load_mul_stmt,
            const expr_t &c_buf, const stmt_t &c_store_stmt,
            alloc_updater_t &alloc_updater, const expr_t &tg_idx0)
        : hw_(hw)
        , load_mul_stmt_(load_mul_stmt)
        , c_buf_(c_buf)
        , c_store_stmt_(c_store_stmt)
        , alloc_updater_(alloc_updater)
        , tg_idx0_(tg_idx0) {}

    const stmt_t &load_mul_stmt() const { return load_mul_stmt_; }

    const stmt_t &c_store_stmt() const { return c_store_stmt_; }

    void inject() {
        expr_t src2_base;
        extract_dpas_calls(src2_base);

        grf_permutator_t grf_perm(hw_, c_buf_);

        bool was_injected = false;
        int dpas_count = int(dpas_infos_.size());
        for (int i = 0; i < dpas_count;) {
            if (i + 1 < dpas_count) {
                auto &a = dpas_infos_[i];
                auto &b = dpas_infos_[i + 1];
                if (try_convert_to_dpasw(a, b, grf_perm)) {
                    was_injected = true;
                    i += 2;
                    continue;
                }
            }
            if (try_convert_to_dpasw(dpas_infos_[i], grf_perm)) {
                was_injected = true;
            }
            ++i;
        }
        // Nothing to update, no dpas -> dpasw transformation.
        if (!was_injected) return;

        int src2_size = 0;
        object_map_t<stmt_t, int> send2off;
        std::function<int(const stmt_t &)> get_src2_off;
        get_src2_off = [&](const stmt_t &s) {
            auto &si = find_send_info(s);
            if (!si.base_call.is_empty()) return get_src2_off(si.base_call);
            if (!si.prev_send.is_empty()) return get_src2_off(si.prev_send);

            auto it = send2off.find(s);
            if (it != send2off.end()) return it->second;

            auto ret = send2off.insert({s, src2_size});
            if (!ret.second) return ret.first->second;

            int new_size = si.new_reg_buf_size();
            src2_size += new_size;
            return ret.first->second;
        };
        for (auto &si : send_infos_) {
            if (!si.reg_buf_base().is_equal(src2_base)) continue;

            int src2_off = get_src2_off(si.call);
            auto src2_sub = src2_base[src2_off];
            auto new_call = si.new_call;
            if (!new_call.is_empty()) {
                new_call = substitute(
                        new_call, send_t::arg_reg_buf(new_call), src2_sub, 1);
            }

            load_mul_stmt_ = substitute(load_mul_stmt_, si.call, new_call, 1);
            for (auto &d : si.dpas_consumers) {
                auto &di = find_dpas_info(d);
                ir_assert(si.promote_to_dpasw == di.promote_to_dpasw)
                        << "Both send and dpas must be updated.";
                if (di.update_applied) {
                    ir_error_not_expected() << "Can it happen?";
                    continue;
                }
                auto new_call = di.new_call;
                new_call = substitute(new_call, dpas_t::arg_src2(new_call),
                        src2_sub[di.src2_relative_off], 1);
                load_mul_stmt_
                        = substitute(load_mul_stmt_, di.call, new_call, 1);
                di.update_applied = true;
            }
        }

        // Apply permutation to C store.
        c_store_stmt_ = apply_permutation_to_reorder(c_store_stmt_, grf_perm);

        // Update src2 size after applying send updates.
        alloc_updater_.resize(src2_base, src2_size);
    }

private:
    struct send_info_t {
        send_info_t() = default;

        send_info_t(const stmt_t &call) : call(call), new_call(call) {}

        const send_t &send() const {
            return call.as<func_call_t>().func.as<send_t>();
        }

        const send_t &new_send() const {
            ir_assert(!new_call.is_same(call));
            return new_call.as<func_call_t>().func.as<send_t>();
        }

        const std::vector<expr_t> &args() const {
            return call.as<func_call_t>().args;
        }

        const expr_t &reg_buf() const { return send_t::arg_reg_buf(call); }

        const expr_t &reg_buf_base() const {
            return reg_buf().as<ptr_t>().base;
        }

        int reg_buf_size() const { return send().register_size(); }

        int new_reg_buf_size() const {
            if (new_call.is_same(call)) return 0;
            return new_send().register_size();
        }

        void set_new_call(const stmt_t &s, const stmt_t &base = stmt_t()) {
            if (!promote_to_dpasw) {
                promote_to_dpasw = true;
                new_call = s;
                base_call = base;
                return;
            }
            ir_assert(new_call.is_equal(s));
            ir_assert(base_call.is_equal(base));
        }

        void set_prev_send(const stmt_t &s) {
            int prev_size
                    = s.as<func_call_t>().func.as<send_t>().register_size();
            if (reg_buf_size() != prev_size) return;
            prev_send = s;
        }

        stmt_t call;
        std::vector<stmt_t> dpas_consumers;

        bool promote_to_dpasw = false;
        stmt_t new_call;
        stmt_t base_call;
        stmt_t prev_send;
    };

    struct dpas_info_t {
        dpas_info_t() = default;

        dpas_info_t(const stmt_t &call) : call(call), new_call(call) {}

        const dpas_t &dpas() const {
            return call.as<func_call_t>().func.as<dpas_t>();
        }

        const std::vector<expr_t> &args() const {
            return call.as<func_call_t>().args;
        }

        const expr_t &src1_buf() const { return dpas_t::arg_src1(call); }

        const expr_t &src2_buf() const { return dpas_t::arg_src2(call); }

        int src2_size() const { return dpas().src2_size(); }

        void set_new_call(const stmt_t &s, int src2_relative_off) {
            if (!promote_to_dpasw) {
                promote_to_dpasw = true;
                this->src2_relative_off = src2_relative_off;
                new_call = s;
                return;
            }
            ir_assert(this->src2_relative_off == src2_relative_off);
            ir_assert(new_call.is_equal(s));
        }

        stmt_t call;
        stmt_t send_producer;

        bool promote_to_dpasw = false;
        bool update_applied = false;
        int src2_relative_off = 0;
        stmt_t new_call;
    };

    send_info_t &find_send_info(const stmt_t &s) {
        for (auto &si : send_infos_)
            if (si.call.is_same(s)) return si;
        ir_error_not_expected();
        return send_infos_.front();
    }

    dpas_info_t &find_dpas_info(const stmt_t &s) {
        for (auto &si : dpas_infos_)
            if (si.call.is_same(s)) return si;
        ir_error_not_expected();
        return dpas_infos_.front();
    }
    static bool is_send(const stmt_t &s, send_info_t &info) {
        if (!is_func_call<send_t>(s)) return false;
        info = send_info_t(s);
        return true;
    }

    static bool is_dpas(const stmt_t &s, dpas_info_t &info) {
        if (!is_func_call<dpas_t>(s)) return false;
        info = dpas_info_t(s);
        return true;
    }

    void extract_dpas_calls(expr_t &src2_base) {
        object_eq_map_t<expr_t, stmt_t> buf2send;

        auto set_src2_base = [&](const expr_t &ptr) {
            auto &ptr_base = ptr.as<ptr_t>().base;
            if (src2_base.is_empty()) {
                src2_base = ptr_base;
                return;
            }
            // This may need a fix in the future.
            ir_assert(src2_base.is_same(ptr_base));
        };

        // Iterate through dpas and send calls.
        auto stmt_vec = flatten_statements(load_mul_stmt_);
        for (auto &s : stmt_vec) {
            send_info_t send_info;
            if (is_send(s, send_info)) {
                auto &buf = send_info.reg_buf();
                stmt_t prev_send;
                auto it = buf2send.find(buf);
                if (it != buf2send.end()) prev_send = it->second;
                buf2send[buf] = s;
                send_infos_.push_back(send_info);
                if (!prev_send.is_empty()) {
                    send_infos_.back().set_prev_send(prev_send);
                }
                continue;
            }
            dpas_info_t dpas_info;
            if (is_dpas(s, dpas_info)) {
                set_src2_base(dpas_info.src2_buf());
                auto &buf = dpas_info.src2_buf();
                auto it = buf2send.find(buf);
                if (it == buf2send.end()) continue;
                auto &send_info = find_send_info(it->second);
                // Ensure read size matches DPAS src2 size.
                // FIXME: This may not be always the case.
                ir_assert(send_info.reg_buf_size() == dpas_info.src2_size());
                dpas_info.send_producer = send_info.call;
                send_info.dpas_consumers.push_back(s);
                dpas_infos_.push_back(dpas_info);
            }
        }
    }

    // Checks for the following pattern:
    //    dpas.sxr(a_dst, a_src0, src1, src2)
    //    dpas.sxr(b_dst, b_src0, src1, src2 + s * r * 4)
    static bool can_convert_to_dpasw(
            const dpas_info_t &a, const dpas_info_t &b) {
        if (!a.dpas().is_equal(&b.dpas())) return false;
        if (!a.src1_buf().is_equal(b.src1_buf())) return false;

        auto src2_off0 = to_cpp<int>(a.src2_buf().as<ptr_t>().off);
        auto src2_off1 = to_cpp<int>(b.src2_buf().as<ptr_t>().off);

        if (src2_off1 - src2_off0 != a.src2_size()) return false;

        return true;
    }

    bool try_convert_to_dpasw(
            dpas_info_t &a, dpas_info_t &b, grf_permutator_t &grf_perm) {

        // Check if DPAS -> DPASW transformation is possible.
        if (!can_convert_to_dpasw(a, b)) return false;

        // Perform the transformation:
        // Before:
        //   send(slm, a_off, src2[0])
        //   send(slm, b_off, src2[s * r * 4])
        //   dpas.sxr(a_dst, a_src0, src1, src2[0])
        //   dpas.sxr(b_dst, b_src0, src1, src2[s * r * 4])
        // After:
        //   send(slm, a_off + (tg_idx0 % 2) * (b_off - a_off), src2)
        //   dpasw.sxr(p_a_dst, p_a_src0, src1, src2[0])
        //   dpasw.sxr(p_b_dst, p_b_src0, src1, src2[s * r * 4 / 2])
        // Where:
        //   p_a_dst[:] = a_dst[0:rcount / 2] + b_dst[0:rcount / 2]
        //   p_b_dst[:] = a_dst[rcount / 2:rcount] + b_dst[rcount / 2:rcount]
        ir_assert(a.dpas().is_equal(&b.dpas()));
        auto _dpasw = dpas_t::make_dpasw(a.dpas());
        auto &dpasw = _dpasw.as<dpas_t>();

        auto a_args = a.args();
        auto b_args = b.args();
        dpas_t::arg_src2(b_args) -= dpasw.src2_size();

        a.set_new_call(dpasw.call(a.args()), 0);
        b.set_new_call(dpasw.call(b_args), dpasw.src2_size());

        // Record permutation for registers to apply it for the destination
        // store later.
        const auto grf_size = ngen::GRF::bytes(hw_);
        const auto rcount = a.dpas().rcount;
        for (int j = 0; j < rcount; j++) {
            int k = j % (rcount / 2);
            auto a_old = dpas_t::arg_dst(a_args) + grf_size * j;
            auto b_old = dpas_t::arg_dst(b_args) + grf_size * j;
            expr_t grf_new;
            if (j < rcount / 2) {
                grf_new = dpas_t::arg_dst(a_args)[grf_size * k];
            } else {
                grf_new = dpas_t::arg_dst(b_args)[grf_size * k];
            }
            grf_perm.set_permute(a_old, grf_new);
            grf_perm.set_permute(b_old, grf_new + grf_size * rcount / 2);
        }

        auto &a_send = find_send_info(a.send_producer);
        auto &b_send = find_send_info(b.send_producer);

        auto &a_mem_off = send_t::arg_mem_off(a_send.call);
        auto &b_mem_off = send_t::arg_mem_off(b_send.call);
        auto ab_addr_diff = simplify(b_mem_off - a_mem_off);
        ir_assert(is_const(ab_addr_diff));

        auto new_send_args = a_send.args();
        send_t::arg_mem_off(new_send_args)
                += (tg_idx0_ % 2) * to_cpp<int64_t>(ab_addr_diff);

        a_send.set_new_call(a_send.send().call(new_send_args));
        b_send.set_new_call(stmt_t(), a_send.call);

        return true;
    }

    static bool can_convert_to_dpasw(const dpas_info_t &a_dpas,
            const send_info_t &a_send, const expr_t &tg_idx0) {
        if (contains_object(a_send.call, tg_idx0)) return false;
        return a_dpas.dpas().rcount % 2 == 0;
    }

    static func_t create_half_send(const send_t &send) {
        ir_assert(send.data_elems % 2 == 0) << "Can't create half-send.";
        auto _s = send.with_data_elems(send.data_elems / 2);
        auto &s = _s.as<send_t>();
        ir_assert(s.is_supported())
                << "Can't find send reading half of the original send.";
        MAYBE_UNUSED(s);
        return _s;
    }

    bool try_convert_to_dpasw(dpas_info_t &a, grf_permutator_t &grf_perm) {
        if (!can_convert_to_dpasw(a, find_send_info(a.send_producer), tg_idx0_))
            return false;

        // Perform the transformation:
        // Before:
        //   send(slm, a_off, src2[0])
        //   dpas.sxr(a_dst, a_src0, src1, src2[0])
        // After:
        //   send(slm, a_off + (tg_idx0 % 2) * (s * r * 4 / 2), src2)
        //   dpasw.sxr(a_dst, a_src0, src1, src2[0])

        auto _dpasw = dpas_t::make_dpasw(a.dpas());
        auto &dpasw = _dpasw.as<dpas_t>();

        a.set_new_call(dpasw.call(a.args()), 0);

        // Real permutation is not required but it needs to be set anyway.
        const auto grf_size = ngen::GRF::bytes(hw_);
        const auto rcount = a.dpas().rcount;
        for (int j = 0; j < rcount; j++) {
            auto grf = dpas_t::arg_dst(a.args()) + grf_size * j;
            grf_perm.set_permute(grf, grf);
        }

        auto &a_send = find_send_info(a.send_producer);
        auto new_send_args = a_send.args();
        send_t::arg_mem_off(new_send_args)
                += (tg_idx0_ % 2) * to_cpp<int64_t>(a.src2_size() / 2);
        a_send.set_new_call(
                create_half_send(a_send.send()).call(new_send_args));

        return true;
    }

    static stmt_t apply_permutation_to_reorder(
            const stmt_t &stmt, const grf_permutator_t &grf_perm) {
        return permutation_injector_t(grf_perm).mutate(stmt);
    }

    ngen::HW hw_;
    stmt_t load_mul_stmt_;
    expr_t c_buf_;
    stmt_t c_store_stmt_;
    alloc_updater_t &alloc_updater_;
    expr_t tg_idx0_;

    std::vector<dpas_info_t> dpas_infos_;
    std::vector<send_info_t> send_infos_;
};

// Transforms DPAS to DPASW.
void inject_dpasw(ngen::HW hw, stmt_t &load_mul_stmt, const expr_t &c_buf,
        stmt_t &c_store_stmt, alloc_updater_t &alloc_updater,
        const expr_t &tg_idx0) {
    dpasw_injector_t injector(
            hw, load_mul_stmt, c_buf, c_store_stmt, alloc_updater, tg_idx0);
    injector.inject();

    load_mul_stmt = injector.load_mul_stmt();
    c_store_stmt = injector.c_store_stmt();
}

// Adds {Atomic} modifier to DPAS/DPASW instructions when applicable.
stmt_t inject_atomic(const stmt_t &stmt) {
    stmt_t ret = stmt;
    auto stmt_vec = flatten_statements(stmt);
    for (size_t i = 0; i < stmt_vec.size(); i++) {
        bool ok = true;
        ok &= is_func_call<dpas_t>(stmt_vec[i]);
        ok &= (i + 1 < stmt_vec.size()
                && is_func_call<dpas_t>(stmt_vec[i + 1]));
        if (ok) {
            auto &cur_src1 = dpas_t::arg_src1(stmt_vec[i]);
            auto &next_src1 = dpas_t::arg_src1(stmt_vec[i + 1]);
            // Compare src1, apply {Atomic} if they are equal.
            if (cur_src1.is_equal(next_src1)) {
                auto &s = stmt_vec[i];
                auto atomic_attr = instruction_modifier_attr_t::make(
                        ngen_proxy::InstructionModifier().with_atomic());
                ret = substitute(ret, s, atomic_attr.apply_to(s));
            }
        }
    }
    return ret;
}

// Trace for debugging purposes.
void trace_pass(const char *pass_name, const stmt_t &stmt) {
    ir_trace() << "=== After " << pass_name << std::endl;
    ir_trace() << stmt << std::endl;
}

class external_var_visitor_t : public scope_visitor_t {
public:
    void _visit(const var_t *obj) {
        if (!is_expr_defined(obj)) external_vars.insert(obj);
    }

    object_set_t<expr_t> external_vars;
};

stmt_t inject_external_var_let(const stmt_t &_stmt) {
    auto stmt = _stmt;
    external_var_visitor_t v;
    v.visit(stmt);

    for (auto &var : v.external_vars)
        stmt = let_t::make(var, {}, stmt);

    trace_pass("inject_external_var_let", stmt);
    return stmt;
}

class slm_buffer_merger_t : public ir_mutator_t {
public:
    slm_buffer_merger_t() {
        slm_base_ = make_buffer("slm");
        slm_off_.push_back(0);
    }

    const expr_t &slm_base() const { return slm_base_; }

    int slm_size() const { return slm_size_; }

    object_t _mutate(const alloc_t *obj) override {
        if (obj->kind != alloc_kind_t::slm) return ir_mutator_t::_mutate(obj);

        auto new_buf = push(obj);
        auto new_obj = ir_mutator_t::_mutate(obj);
        pop();

        auto &alloc = new_obj.as<alloc_t>();
        new_obj = substitute(alloc.body, alloc.buf, new_buf);

        return new_obj;
    }

private:
    expr_t push(const alloc_t *obj) {
        int cur_off = slm_off_.back();
        expr_t new_buf = slm_base_ + cur_off;
        slm_off_.push_back(cur_off + obj->size);
        slm_size_ = std::max(slm_size_, cur_off + obj->size);
        return new_buf;
    }

    void pop() { slm_off_.pop_back(); }

    expr_t slm_base_;
    std::vector<int> slm_off_;
    int slm_size_ = 0;
};

// Merges all SLM buffers into a single one.
stmt_t merge_slm_buffers(const stmt_t &_stmt) {
    stmt_t stmt = _stmt;
    slm_buffer_merger_t merger;
    stmt = merger.mutate(stmt);
    stmt = alloc_t::make(
            merger.slm_base(), merger.slm_size(), alloc_kind_t::slm, {}, stmt);
    trace_pass("merge_slm_buffers", stmt);
    return stmt;
}

class buffer_offset_lifter_t : public ir_mutator_t {
public:
    object_t _mutate(const func_call_t *obj) {
        if (!obj->func.is<send_t>()) return ir_mutator_t::_mutate(obj);

        auto &mem_buf = send_t::arg_mem_buf(obj);
        if (!mem_buf.is<ptr_t>()) return ir_mutator_t::_mutate(obj);

        auto &base = mem_buf.as<ptr_t>().base;
        auto &off = mem_buf.as<ptr_t>().off;

        std::vector<expr_t> new_args = obj->args;
        send_t::arg_mem_buf(new_args) = base;
        send_t::arg_mem_off(new_args) += off;
        return obj->func.call(new_args, obj->attr);
    }
};

stmt_t lift_buffer_offsets_in_send(const stmt_t &s) {
    buffer_offset_lifter_t lifter;
    auto ret = lifter.mutate(s);
    trace_pass("lift_buffer_offsets_in_send", ret);
    return ret;
}

stmt_t simplify_pass(const stmt_t &s, const constraint_set_t &cset) {
    auto ret = simplify(s, cset);
    trace_pass("simplify_pass", ret);
    return ret;
}

class send_injector_t : public ir_mutator_t {
public:
    send_injector_t(ir_context_t &ir_ctx, const constraint_set_t &cset)
        : ir_ctx_(ir_ctx), cset_(cset) {}

    object_t _mutate(const func_call_t *obj) {
        auto *send = obj->func.as_ptr<send_t>();
        if (!send) return ir_mutator_t::_mutate(obj);

        auto &mem_buf = send_t::arg_mem_buf(obj);
        auto &mem_off = send_t::arg_mem_off(obj);
        auto &reg_buf = send_t::arg_reg_buf(obj);
        auto &mask = send_t::arg_mask(obj);

        ir_assert(is_var(mem_buf)) << mem_buf;

        auto header_buf = ir_ctx_.create_tmp_var(type_t::byte_ptr(), "h");
        auto off_store = simplify_store(
                send->create_offset_store(header_buf, mem_buf, mem_off));

        auto new_call = func_call_t::make(
                obj->func, {mem_buf, header_buf, reg_buf, mask}, obj->attr);
        auto body = stmt_seq_t::make(off_store, new_call);

        // Allocate header.
        return alloc_t::make(
                header_buf, send->header_size(), alloc_kind_t::grf, {}, body);
    }

private:
    stmt_t simplify_store(const stmt_t &_store) const {
        auto &store = _store.as<store_t>();

        auto value = store.value;
        value = simplify(value, cset_);

        // Convert to N-ary form and back to expand multiplications. This
        // helps to find more common subexpressions during the pass.
        value = nary_op_canonicalize(value);
        value = nary_op_back_transform(value);

        return store_t::make(store.buf, store.off, value, store.stride);
    }

    ir_context_t &ir_ctx_;
    const constraint_set_t &cset_;
};

stmt_t inject_send(
        const stmt_t &s, ir_context_t &ir_ctx, const constraint_set_t &cset) {
    auto ret = send_injector_t(ir_ctx, cset).mutate(s);
    trace_pass("inject_send", ret);
    return ret;
}

class alloc_lifter_t : public ir_mutator_t {
public:
    alloc_lifter_t(const stmt_t &root, bool reuse_headers)
        : reuse_headers_(reuse_headers) {
        if (!reuse_headers_) return;
        auto calls = find_objects<func_call_t>(root);
        for (auto &c : calls) {
            if (!is_func_call<send_t>(c)) continue;
            auto header_buf = send_t::arg_mem_off(c);
            ir_assert(is_var(header_buf)) << header_buf;
            header_bufs_.insert(header_buf);
        }
    }

    object_t _mutate(const alloc_t *obj) override {
        if (!do_lift(obj)) return ir_mutator_t::_mutate(obj);
        // Remove alloc and insert it before the compute loop.
        allocs_.push_back(obj);
        return obj->body;
    }

    object_t _mutate(const stmt_group_t *obj) override {
        bool is_compute_loop = (obj->label == stmt_label_t::compute_loop());
        if (is_compute_loop) in_compute_loop_ = true;
        auto new_obj = ir_mutator_t::_mutate(obj);
        if (is_compute_loop) {
            in_compute_loop_ = false;
            // Outermost loop.
            for (auto it = allocs_.rbegin(); it != allocs_.rend(); ++it) {
                auto &a = it->as<alloc_t>();
                new_obj = alloc_t::make(a.buf, a.size, a.kind, a.attr, new_obj);
            }
            allocs_.resize(0);
        }
        return new_obj;
    }

private:
    bool do_lift(const alloc_t *obj) const {
        if (!in_compute_loop_) return false;
        if (reuse_headers_) {
            bool is_header_alloc = (header_bufs_.count(obj->buf) != 0);
            return !is_header_alloc;
        }
        return true;
    }

    bool reuse_headers_;
    object_set_t<expr_t> header_bufs_;

    bool in_compute_loop_ = false;
    std::vector<stmt_t> allocs_;
};

// Lifts alloc statements out of loops.
stmt_t lift_alloc(const stmt_t &s, const conv_config_t &cfg) {
    auto ret = alloc_lifter_t(s, cfg.reuse_headers).mutate(s);
    trace_pass("lift_alloc", ret);
    return ret;
}

// Common subexpression elimination support.

// Represents an expression-candidate to eliminate.
class cse_expr_t {
public:
    cse_expr_t(const expr_t &expr, const ir_path_t &path, int refs = 1,
            const expr_t &cse_var = {})
        : expr(expr), path(path), refs(refs), cse_var(cse_var) {
        ir_trace() << "cse_pass: add expression: " << expr << std::endl;
    }

    void add_usage(const ir_path_t &other_path, bool do_increment = true) {
        if (do_increment) refs++;
        path.merge(other_path);
        ir_trace() << "cse_pass: add usage: " << expr
                   << ", total refs: " << refs << std::endl;
    }

    // Expression to eliminate via let.
    expr_t expr;
    // Path to the innermost IR node where the expression can be defined.
    ir_path_t path;
    // Number of references to the expression.
    int refs;
    // Variable assigned to the expression (if decided to eliminate).
    expr_t cse_var;
};

// Stores information about all expressions subject to CSEing.
class cse_context_t {
public:
    cse_context_t(ir_context_t &ir_ctx) : ir_ctx_(ir_ctx) {}

    ir_context_t &ir_ctx() { return ir_ctx_; }

    bool has(const expr_t &e) const { return cse_exprs_.count(e) != 0; }

    cse_expr_t &find_cse_expr(const expr_t &e) {
        ir_assert(has(e)) << e;
        return cse_exprs_.at(e);
    }

    const cse_expr_t &find_cse_expr(const expr_t &e) const {
        ir_assert(has(e)) << e;
        return cse_exprs_.at(e);
    }

    bool has_var(const expr_t &e) const {
        return !find_cse_expr(e).cse_var.is_empty();
    }

    int get_refs(const expr_t &e) const {
        if (!has(e)) return 0;
        return find_cse_expr(e).refs;
    }

    void register_expr(const expr_t &e, const ir_path_t &path) {
        if (e.type().is_bool()) return; // Ignore booleans.
        auto ret = cse_exprs_.insert({e, cse_expr_t(e, path)});
        ir_assert(ret.second) << e;
        MAYBE_UNUSED(ret);
    }

    void register_expr(const cse_expr_t &cse_expr) {
        auto ret = cse_exprs_.insert({cse_expr.expr, cse_expr});
        ir_assert(ret.second);
        MAYBE_UNUSED(ret);
    }

    expr_t get_or_assign_var(const expr_t &e) {
        auto &cse_expr = find_cse_expr(e);
        if (cse_expr.cse_var.is_empty()) {
            cse_expr.cse_var = ir_ctx_.create_tmp_var(e.type());
            ir_trace() << "cse_pass: assigning var: " << e << " -> "
                       << cse_expr.cse_var << std::endl;
        }
        return cse_expr.cse_var;
    }

    const expr_t &get_var(const expr_t &e) const {
        return find_cse_expr(e).cse_var;
    }

    const ir_path_t &get_path(const expr_t &e) const {
        return find_cse_expr(e).path;
    }

    void add_usage(
            const expr_t &e, const ir_path_t &path, bool do_increment = true) {
        if (e.type().is_bool()) return; // Ignore booleans.
        return find_cse_expr(e).add_usage(path, do_increment);
    }

    void update_expr(const expr_t &old_expr, const expr_t &new_expr) {
        auto it = cse_exprs_.find(old_expr);
        ir_assert(it != cse_exprs_.end()) << old_expr;
        auto &old_cse_expr = it->second;
        auto new_cse_expr = cse_expr_t(new_expr, old_cse_expr.path,
                old_cse_expr.refs, old_cse_expr.cse_var);
        cse_exprs_.erase(it);
        auto ret = cse_exprs_.insert({new_expr, new_cse_expr});
        ir_assert(ret.second);
        MAYBE_UNUSED(ret);
    }

    template <typename F>
    void for_each(const F &f) const {
        for (auto &kv : cse_exprs_)
            f(kv.first);
    }

private:
    ir_context_t &ir_ctx_;
    object_eq_map_t<expr_t, cse_expr_t> cse_exprs_;
};

// Collects statistics about expressions for common subexpression elimination.
class cse_visitor_t : public ir_visitor_t {
public:
    cse_visitor_t(cse_context_t &ctx) : ctx_(ctx) {}

    void _visit(const binary_op_t *obj) override { visit_expr(obj); }
    void _visit(const shuffle_t *obj) override {
        if (is_const_broadcast(obj)) return;
        visit_expr(obj);
    }
    void _visit(const unary_op_t *obj) override { visit_expr(obj); }

#define HANDLE_IR_OBJECT(type) \
    void _visit(const type *obj) override { visit_stmt(obj); }

    HANDLE_STMT_IR_OBJECTS()

#undef HANDLE_IR_OBJECT

private:
    template <typename T>
    void visit_expr(const T *obj) {
        // Exclude loads as they may have side effects.
        if (count_objects<load_t>(obj) > 0) {
            ir_visitor_t::_visit(obj);
            return;
        }

        if (propagate_path_) {
            if (ctx_.has(obj))
                ctx_.add_usage(obj, root_path_, /*do_increment=*/false);
            ir_visitor_t::_visit(obj);
            return;
        }
        if (ctx_.has(obj)) {
            ctx_.add_usage(obj, root_path_);
            propagate_path_ = true;
            ir_visitor_t::_visit(obj);
            propagate_path_ = false;
            return;
        }
        ir_visitor_t::_visit(obj);
        ctx_.register_expr(obj, root_path_);
    }

    template <typename T>
    void visit_stmt(const T *obj) {
        if (std::is_same<T, for_t>::value) {
            visit_for((const object_impl_t *)obj);
            return;
        }
        if (std::is_same<T, let_t>::value) {
            visit_let((const object_impl_t *)obj);
            return;
        }
        root_path_.push(obj);
        ir_visitor_t::_visit(obj);
        root_path_.pop();
    }

    void visit_for(const object_impl_t *_obj) {
        auto *obj = (const for_t *)_obj;

        visit(obj->var);
        visit(obj->init);
        visit(obj->bound);
        root_path_.push(obj);
        visit(obj->body);
        root_path_.pop();
    }

    void visit_let(const object_impl_t *_obj) {
        auto *obj = (const let_t *)_obj;

        visit(obj->var);
        visit(obj->value);
        root_path_.push(obj);
        visit(obj->body);
        root_path_.pop();
    }

    cse_context_t &ctx_;
    ir_path_t root_path_;

    bool propagate_path_ = false;
};

// Verifies all IR paths are correct (for debugging purposes).
class cse_verifier_t : public scope_visitor_t {
public:
    cse_verifier_t(cse_context_t &ctx) : ctx_(ctx) {}

    ~cse_verifier_t() override { ir_assert(to_check_.empty()); }

    void _visit(const binary_op_t *obj) override { visit_expr(obj); }
    void _visit(const shuffle_t *obj) override { return visit_expr(obj); }
    void _visit(const unary_op_t *obj) override { visit_expr(obj); }

#define HANDLE_IR_OBJECT(type) \
    void _visit(const type *obj) override { visit_stmt(obj); }

    HANDLE_STMT_IR_OBJECTS()

#undef HANDLE_IR_OBJECT

    void verify(const stmt_t &s) {
        // Phase 0: collect IR paths for expressions.
        phase_ = 0;
        visit(s);

        // Phase 1: verify all expressions are defined at their path.
        phase_ = 1;
        visit(s);
    }

private:
    template <typename T>
    void visit_expr(const T *obj) {
        // Expressions are not used during phase 1.
        if (phase_ == 1) return;
        if (ctx_.has(obj)) {
            auto &path = ctx_.get_path(obj);
            to_check_[path.back()].push_back(obj);
        }
        scope_visitor_t::_visit(obj);
    }

    template <typename T>
    void visit_stmt(const T *obj) {
        scope_visitor_t::_visit(obj);

        // Statements are not used during phase 0.
        if (phase_ == 0) return;

        // Phase 1: check that all attached expressions are defined at this
        // statement.
        auto it = to_check_.find(obj);
        if (it != to_check_.end()) {
            for (auto &e : it->second) {
                ir_assert(is_expr_defined(e))
                        << "Expression contains undefined variables: " << e;
                MAYBE_UNUSED(e);
            }
            to_check_.erase(it);
        }
    }

    cse_context_t &ctx_;

    int phase_ = 0;
    object_map_t<stmt_t, std::vector<expr_t>> to_check_;
};

// Generates let statements for expressions being eliminated.
class cse_let_generator_t : public ir_visitor_t {
public:
    cse_let_generator_t(const cse_context_t &ctx, const stmt_t &stmt)
        : ctx_(ctx), stmt_(stmt) {}

    void _visit(const binary_op_t *obj) override { visit_expr(obj); }
    void _visit(const shuffle_t *obj) override { visit_expr(obj); }
    void _visit(const unary_op_t *obj) override { visit_expr(obj); }
    void _visit(const var_t *obj) override {
        auto it = all_vars_.find(obj);
        if (it == all_vars_.end()) return;
        if (seen_vars_.count(obj) == 0) generate_for_expr(it->second);
    }

    stmt_t generate() {
        ctx_.for_each([&](const expr_t &e) {
            auto &cse_var = ctx_.get_var(e);
            auto ret = all_vars_.insert({cse_var, e});
            ir_assert(ret.second);
            MAYBE_UNUSED(ret);
        });
        ctx_.for_each([&](const expr_t &e) { generate_for_expr(e); });
        for (auto it = lets_.rbegin(); it != lets_.rend(); ++it) {
            auto &let = it->as<let_t>();
            stmt_ = let_t::make(let.var, let.value, stmt_);
        }
        return stmt_;
    }

private:
    void generate_for_expr(const expr_t &e) {
        auto &cse_var = ctx_.get_var(e);
        if (seen_vars_.count(cse_var) == 1) return;
        visit(e);
    }

    template <typename T>
    void visit_expr(const T *obj) {
        ir_visitor_t::_visit(obj);
        if (ctx_.has(obj) && ctx_.has_var(obj)) {
            auto &var = ctx_.get_var(obj);
            auto ret = seen_vars_.insert(var);
            if (ret.second) lets_.push_back(let_t::make(var, obj));
        }
    }

    const cse_context_t &ctx_;
    stmt_t stmt_;

    object_map_t<expr_t, expr_t> all_vars_; // Var -> expression.
    object_set_t<expr_t> seen_vars_;

    std::vector<stmt_t> lets_;
};

// Eliminiates expressions from the statement.
class cse_mutator_t : public ir_mutator_t {
public:
    cse_mutator_t(cse_context_t &ctx) : ctx_(ctx) {}

    object_t _mutate(const binary_op_t *obj) override {
        return mutate_expr(obj);
    }
    object_t _mutate(const shuffle_t *obj) override { return mutate_expr(obj); }
    object_t _mutate(const unary_op_t *obj) override {
        return mutate_expr(obj);
    }

#define HANDLE_IR_OBJECT(type) \
    object_t _mutate(const type *obj) override { return mutate_stmt(obj); }

    HANDLE_STMT_IR_OBJECTS()

#undef HANDLE_IR_OBJECT

private:
    template <typename T>
    object_t mutate_expr(const T *obj) {
        auto new_obj = ir_mutator_t::_mutate(obj);
        if (ctx_.has(obj) && !new_obj.is_equal(obj)) {
            ctx_.update_expr(obj, new_obj);
        }
        if (ctx_.get_refs(new_obj) > 1) {
            bool has_var = ctx_.has_var(new_obj);
            auto var = ctx_.get_or_assign_var(new_obj);
            auto &path = ctx_.get_path(new_obj);
            if (!has_var) to_update_[path.back()].push_back(new_obj);
            return std::move(var);
        }
        return new_obj;
    }

    template <typename T>
    object_t mutate_stmt(const T *obj) {
        auto new_obj = ir_mutator_t::_mutate(obj);
        auto it = to_update_.find(obj);
        if (it == to_update_.end()) return new_obj;

        cse_context_t local_ctx(ctx_.ir_ctx());
        for (auto &e : it->second) {
            local_ctx.register_expr(ctx_.find_cse_expr(e));
        }
        to_update_.erase(it);

        auto body = get_stmt_body(new_obj);
        cse_let_generator_t g(local_ctx, body);
        body = g.generate();
        new_obj = replace_stmt_body(new_obj, body);
        return new_obj;
    }

    cse_context_t &ctx_;
    object_map_t<stmt_t, std::vector<expr_t>> to_update_;
};

stmt_t eliminate_common_subexprs(const stmt_t &_stmt, ir_context_t &ir_ctx) {
    auto stmt = _stmt;

    cse_context_t ctx(ir_ctx);

    // Collect statistics.
    cse_visitor_t visitor(ctx);
    visitor.visit(stmt);

#ifndef NDEBUG
    // Verify that collected IR paths are correct (cse_expr_t objects are
    // defined at those paths).
    cse_verifier_t verifier(ctx);
    verifier.verify(stmt);
#endif

    // Eliminate subexpressions.
    cse_mutator_t mutator(ctx);
    stmt = mutator.mutate(stmt);

    trace_pass("eliminate_common_subexprs", stmt);
    return stmt;
}

class hoist_exprs_mutator_t : public ir_mutator_t {
public:
    hoist_exprs_mutator_t(ir_context_t &ir_ctx) : ir_ctx_(ir_ctx) {}

    ~hoist_exprs_mutator_t() override { ir_assert(let_vars_.empty()); }

    object_t _mutate(const func_call_t *obj) override {
        if (!obj->func.is<send_t>()) return ir_mutator_t::_mutate(obj);

        std::vector<expr_t> new_args;
        for (auto &e : obj->args) {
            new_args.push_back(hoist_expr(e));
        }

        if (ir_utils::is_equal(new_args, obj->args)) return obj;

        return func_call_t::make(obj->func, new_args, obj->attr);
    }

    object_t _mutate(const stmt_group_t *obj) override {
        if (obj->body.is<for_t>()) {
            loops_.emplace_back(obj->body.as<for_t>().var);
            auto body = ir_mutator_t::_mutate(obj->body.as_ptr<for_t>());
            if (body.is_same(obj->body)) return obj;
            auto new_obj = stmt_group_t::make(obj->label, body);
            return injects_lets_and_pop_loop(new_obj);
        }
        return ir_mutator_t::_mutate(obj);
    }

    object_t _mutate(const store_t *obj) override {
        auto value = hoist_expr(obj->value);
        if (value.is_equal(obj->value)) return obj;
        return store_t::make(obj->buf, obj->off, value, obj->stride);
    }

    object_t _mutate(const for_t *obj) override {
        loops_.emplace_back(obj->var);
        auto new_obj = ir_mutator_t::_mutate(obj);
        return injects_lets_and_pop_loop(new_obj);
    }

    object_t _mutate(const let_t *obj) override {
        bool fully_hoisted = false;
        auto new_value = hoist_expr(obj->value, obj->var, &fully_hoisted);
        if (fully_hoisted) return mutate(obj->body);
        register_let(obj->var, new_value);
        auto new_obj = let_t::make(
                obj->var, new_value, ir_mutator_t::mutate(obj->body));
        unregister_let(obj->var);
        return std::move(new_obj);
    }

private:
    struct loop_info_t {
        loop_info_t(const expr_t &var) : var(var) {}

        expr_t var;
        int var_count = 0;
        std::vector<stmt_t> lets;
    };

    expr_t hoist_expr(const expr_t &expr, const expr_t &expr_var = {},
            bool *fully_hoisted = nullptr) {
        if (expr.is_empty()) return expr;
        if (expr.type().is_ptr()) return expr;
        if (expr.type().is_bool()) return expr;
        if (is_const(expr) || is_shuffle_const(expr) || is_var(expr))
            return expr;

        auto hoisted_expr = hoist_expr_with_add(expr, expr_var, fully_hoisted);
        if (!hoisted_expr.is_equal(expr)) return hoisted_expr;

        // hoist_expr_with_add() doesn't handle cast so try to hoist it manually.
        auto *cast = expr.as_ptr<cast_t>();
        if (!cast) return hoisted_expr;

        auto hoisted_cast_expr = hoist_expr(cast->expr);
        if (!hoisted_cast_expr.is_equal(cast->expr)) {
            hoisted_expr = cast_t::make(
                    cast->type, hoisted_cast_expr, cast->saturate);
        }
        return hoisted_expr;
    }

    expr_t hoist_expr_with_add(const expr_t &expr, const expr_t &expr_var = {},
            bool *fully_hoisted = nullptr) {
        auto cur_expr = nary_op_canonicalize(expr);

        auto is_nary_add = [](const expr_t &e) {
            auto *nary = e.as_ptr<nary_op_t>();
            return nary && (nary->op_kind == op_kind_t::_add);
        };

        for (size_t i = 0; i < loops_.size(); i++) {
            std::vector<expr_t> invariant_args;
            std::vector<expr_t> other_args;
            std::vector<expr_t> nary_args;
            if (is_nary_add(cur_expr)) {
                nary_args = cvt_expr_to_nary_op_args(cur_expr);
            } else {
                nary_args.push_back(cur_expr);
            }
            for (auto &_a : nary_args) {
                auto a = nary_op_back_transform(_a);
                bool is_inv_arg = true;
                for (size_t j = i; j < loops_.size(); j++) {
                    if (!is_invariant(a, loops_[j].var)) is_inv_arg = false;
                }
                if (is_inv_arg) {
                    invariant_args.push_back(_a);
                } else {
                    other_args.push_back(_a);
                }
            }
            // Nothing to hoist for this loop, continue.
            if (invariant_args.empty()) continue;
            if (invariant_args.size() == 1 && is_var(invariant_args[0]))
                continue;

            // Introduce new variable for the invariant sub-expression.
            auto inv_expr = nary_op_back_transform(
                    make_nary_op(op_kind_t::_add, invariant_args));
            expr_t inv_var;
            if (!expr_var.is_empty() && other_args.empty()) {
                // If nothing to hoist further, reuse the old variable and
                // return.
                inv_var = expr_var;
            } else {
                inv_var = ir_ctx_.create_tmp_var(inv_expr.type());
            }
            auto let = let_t::make(inv_var, inv_expr);
            register_let(inv_var, inv_expr);
            loops_[i].lets.push_back(let);

            if (other_args.empty()) {
                if (fully_hoisted) *fully_hoisted = true;
                return inv_var;
            }

            other_args.push_back(inv_var);
            cur_expr = make_nary_op(op_kind_t::_add, other_args);
        }
        return nary_op_back_transform(cur_expr);
    }

    stmt_t injects_lets_and_pop_loop(const stmt_t &_s) {
        stmt_t s = _s;
        // Inject let statements if any.
        auto &lets = loops_.back().lets;
        for (auto it = lets.rbegin(); it != lets.rend(); ++it) {
            auto &let = it->as<let_t>();
            s = let_t::make(let.var, let.value, s);
            unregister_let(let.var);
        }
        loops_.pop_back();
        return s;
    }

    void register_let(const expr_t &var, const expr_t &value) {
        let_vars_.insert({var, value});
    }

    void unregister_let(const expr_t &var) { let_vars_.erase(var); }

    bool is_invariant(const expr_t &e, const expr_t &var) const {
        if (contains_object(e, var)) return false;
        if (!find_objects<load_t>(e).empty()) return false;

        // Check value if this is a let variable.
        auto it = let_vars_.find(e);
        if (it != let_vars_.end()) return is_invariant(it->second, var);

        if (is_var(e)) return true;

        // Check transitive dependencies.
        auto vars = find_unique_objects<var_t>(e);
        for (auto &v : vars) {
            if (!is_invariant(v, var)) return false;
        }
        return true;
    }

    ir_context_t &ir_ctx_;
    std::vector<loop_info_t> loops_;

    object_map_t<expr_t, expr_t> let_vars_;
};

// Moves invariant expressions out of loops.
stmt_t hoist_exprs(const stmt_t &s, ir_context_t &ir_ctx) {
    auto ret = hoist_exprs_mutator_t(ir_ctx).mutate(s);
    trace_pass("hoist_exprs", ret);
    return ret;
}

class loop_strength_reducer_t : public ir_mutator_t {
public:
    loop_strength_reducer_t() {
        // Create top-level dummy loop.
        loops_.emplace_back();
    }

    ~loop_strength_reducer_t() override {
        // Sanity check, all stores must be applied.
        ir_assert(post_inc_stores.empty());
    }

    object_t _mutate(const for_t *obj) override {
        loops_.emplace_back(obj);
        auto new_obj = ir_mutator_t::_mutate(obj);
        return inject_stores_and_pop_loop(new_obj);
    }

    object_t _mutate(const let_t *obj) override {
        int loop_level = int(loops_.size()) - 1;
        auto ret = lets_.insert(
                {obj->var, let_info_t(obj->var, obj->value, loop_level)});
        ir_assert(ret.second);
        MAYBE_UNUSED(ret);
        auto new_obj = ir_mutator_t::_mutate(obj);
        lets_.erase(obj->var);
        return new_obj;
    }

    object_t _mutate(const stmt_group_t *obj) override {
        if (obj->body.is<for_t>()) {
            loops_.emplace_back(obj->body);
            auto body = ir_mutator_t::_mutate(obj->body.as_ptr<for_t>());
            if (body.is_same(obj->body)) return obj;
            auto new_obj = stmt_group_t::make(obj->label, body);
            return inject_stores_and_pop_loop(new_obj);
        }
        return ir_mutator_t::_mutate(obj);
    }

    // Pattern to handle:
    //     for (...) {
    //         store(buf_ptr, ...) <- Write (producer).
    //         // ...
    //         stmt_t(..., buf_ptr, ...) <- Read (consumer).
    //     }
    object_t _mutate(const store_t *obj) override {
        if (loops_.size() == 1) return ir_mutator_t::_mutate(obj);

        // Try to reduce strength, moving the store up.
        int init_store_level = -1;
        stmt_t init_store_stmt = obj;
        post_inc_store_info_t post_inc_store(obj);
        for (int level = int(loops_.size()) - 1; level >= 1; level--) {
            auto &loop_info = loops_[level];
            int refs = count_object(loop_info.loop, obj->buf);
            // Producer and consumer - must be 2 references.
            if (refs != 2) break;

            // Try to insert the store before level-th loop.
            auto &store = init_store_stmt.as<store_t>();
            auto &store_value = store.value;
            auto &loop_var = loop_info.loop_var();

            auto cur_value = substitute_let(store_value, level);
            auto next_value = substitute(cur_value, loop_var, loop_var + 1);
            auto inc = simplify(next_value - cur_value);

            // Cannot eliminate loop variable, break.
            if (contains_object(inc, loop_var)) break;

            // Success, replace store by post-increment store.
            init_store_level = level;

            auto new_store_value
                    = substitute(cur_value, loop_var, loop_info.loop_init());
            init_store_stmt = store_t::make(store.buf, store.off,
                    simplify(new_store_value), store.stride);

            post_inc_store.update(loop_info, inc);
        }

        // Can't do anything, return as is.
        if (init_store_level == -1) return ir_mutator_t::_mutate(obj);

        // Move this store up, remove from here.
        loops_[init_store_level].init_stores.push_back(init_store_stmt);
        if (!post_inc_store.is_empty()) {
            auto ret = post_inc_stores.insert({obj->buf, post_inc_store});
            ir_assert(ret.second);
            MAYBE_UNUSED(ret);
        }
        return stmt_t();
    }

    object_t _mutate(const func_call_t *obj) override {
        for (auto &kv : post_inc_stores) {
            int refs = count_object(obj, kv.first);
            if (refs == 1) {
                auto ret = stmt_seq_t::make(obj, kv.second.stmt());
                post_inc_stores.erase(kv.first);
                return std::move(ret);
            }
        }
        return ir_mutator_t::_mutate(obj);
    }

private:
    struct loop_info_t {
        loop_info_t(const stmt_t &loop = {}) : loop(loop) {}

        const expr_t &loop_var() const { return loop.as<for_t>().var; }

        const expr_t &loop_init() const { return loop.as<for_t>().init; }

        const expr_t &loop_bound() const { return loop.as<for_t>().bound; }

        expr_t loop_extent() const { return loop_bound() - loop_init(); }

        // Loop being analyzed.
        stmt_t loop;
        // Stores to insert before the loop.
        std::vector<stmt_t> init_stores;

        std::vector<stmt_t> lets;
    };

    struct let_info_t {
        let_info_t(const expr_t &var, const expr_t &value, int loop_level)
            : var(var), value(value), loop_level(loop_level) {}

        expr_t var;
        expr_t value;
        int loop_level;
    };

    struct post_inc_store_info_t {
        post_inc_store_info_t(const store_t *obj)
            : store(obj), inc(0), last_iter_cond(true), compensation(0) {}

        stmt_t stmt() const {
            auto load
                    = load_t::make(store->value.type(), store->buf, store->off);
            return store_t::make(store->buf, store->off, load + inc);
        }

        bool is_empty() const { return is_zero(inc); }

        void update(const loop_info_t &loop, const expr_t &loop_inc) {
            inc = simplify(iif_t::make(
                    last_iter_cond, inc - compensation + loop_inc, inc));
            if (last_iter_cond.is_equal(expr_t(true))) {
                last_iter_cond = (loop.loop_var() == loop.loop_bound() - 1);
            } else {
                last_iter_cond = last_iter_cond
                        & (loop.loop_var() == loop.loop_bound() - 1);
            }
            compensation = simplify(loop.loop_extent() * loop_inc);
        }

        const store_t *store;
        expr_t inc;

        expr_t last_iter_cond;
        expr_t compensation;
    };

    // Recursively substitutes all variable from let statements located under
    // the given loop level.
    expr_t substitute_let(const expr_t &_e, int loop_level) const {
        auto e = _e;
        for (;;) {
            bool found = false;
            auto vars = find_unique_objects<var_t>(e);
            for (auto &v : vars) {
                auto it = lets_.find(v);
                if (it == lets_.end()) continue;
                auto &let_info = it->second;
                // Do not substitute top-level let variables.
                if (let_info.loop_level < loop_level) continue;
                found = true;
                e = substitute(e, v, let_info.value);
            }
            if (!found) break;
        }
        return e;
    }

    // Injects initial store statements if any.
    object_t inject_stores_and_pop_loop(const stmt_t &_s) {
        stmt_t s = _s;
        auto &stores = loops_.back().init_stores;
        for (auto it = stores.rbegin(); it != stores.rend(); ++it) {
            s = stmt_seq_t::make(*it, s);
        }
        loops_.pop_back();
        // The top-level dummy loop shouldn't be removed.
        ir_assert(loops_.size() >= 1);
        return std::move(s);
    }

    // Loops, ordered from outermost to innermost. The first loop is dummy, to
    // represent let statements in the top-level scope.
    std::vector<loop_info_t> loops_;

    // Buffers whose references are to be updated.
    object_map_t<expr_t, post_inc_store_info_t> post_inc_stores;

    // Let statements available at the current IR node.
    object_map_t<expr_t, let_info_t> lets_;
};

// Detects and converts expensive expression operations inside a loop to less
// expensive operations. Example:
// Before:
//     for (int j = 0; j < N; j++) {
//         int off = off_i + j * K;
//         a[off] = j;
//     }
// After:
//     int off = off_i;
//     for (int j = 0; j < N; j++) {
//         a[off] = j;
//         off += K;
//     }
stmt_t loop_strength_reduce(const stmt_t &s) {
    auto ret = loop_strength_reducer_t().mutate(s);
    trace_pass("loop_strength_reduce", ret);
    return ret;
}

class let_optimizer_t : public ir_mutator_t {
public:
    // Also track alloc_t and for_t to validate all variable usages.
    object_t _mutate(const alloc_t *obj) override {
        return mutate_scope(obj, obj->buf);
    }
    object_t _mutate(const for_t *obj) override {
        level_++;
        auto new_obj = mutate_scope(obj, obj->var);
        level_--;
        return new_obj;
    }
    object_t _mutate(const let_t *obj) override {
        return mutate_scope(obj, obj->var);
    }

    object_t _mutate(const var_t *obj) override {
        ir_assert(refs_.count(obj) == 1)
                << "Variable is not defined: " << expr_t(obj);
        refs_[obj].update(increment_, level_);
        return ir_mutator_t::_mutate(obj);
    }

private:
    struct ref_info_t {
        ref_info_t(int level = 0)
            : refs(0), min_level(level), max_level(level) {}

        void update(int increment, int level) {
            refs += increment;
            max_level = std::max(max_level, level);
        }

        bool is_same_level() const { return min_level == max_level; }

        int refs;
        int min_level;
        int max_level;
    };

    template <typename T>
    object_t mutate_scope(const T *obj, const expr_t &var) {
        auto ret = refs_.insert({var, ref_info_t(level_)});
        ir_assert(ret.second) << stmt_t(obj);
        MAYBE_UNUSED(ret);

        auto new_obj = ir_mutator_t::_mutate(obj);
        auto &ref_info = refs_[var];

        if (std::is_same<T, let_t>())
            new_obj = mutate_let(new_obj.template as<let_t>(), ref_info);

        refs_.erase(var);
        return new_obj;
    }

    object_t mutate_let(const let_t &obj, const ref_info_t &ref_info) {
        ir_assert(ref_info.refs >= 1);
        if (ref_info.refs == 1) {
            // Variable is not used.
            remove_refs(obj);
            return obj.body;
        }
        // Check following conditions to substitute let value:
        // - 2 references: one from producer, one from consumer - means single usage
        // - Consumer and producer are on the same level (same loop)
        // - Variable is not external
        if (ref_info.refs == 2 && ref_info.is_same_level()
                && !obj.value.is_empty()) {
            return substitute(obj.body, obj.var, obj.value);
        }
        return &obj;
    }

    void remove_refs(const let_t &obj) {
        increment_ = -1;
        mutate(obj.value);
        increment_ = 1;
    }

    int increment_ = 1;
    int level_ = 0;
    object_map_t<expr_t, ref_info_t> refs_;
};

stmt_t optimize_let(const stmt_t &s) {
    auto ret = let_optimizer_t().mutate(s);
    trace_pass("optimize_let", ret);
    return ret;
}

class unrolling_updater_t : public ir_mutator_t {
public:
    object_t _mutate(const let_t *obj) override {
        if (level_ == 0) {
            // Skip top-level let statements.
            return ir_mutator_t::_mutate(obj);
        }
        lets_.push_back(obj);
        auto new_body = mutate(obj->body);
        if (!lets_.back()) {
            // Let was moved to the innermost loop.
            lets_.pop_back();
            return new_body;
        }
        lets_.pop_back();
        if (new_body.is_same(obj->body)) return obj;
        return let_t::make(obj->var, obj->value, new_body);
    }

    object_t _mutate(const for_t *obj) override {
        level_++;
        found_loop_ = false;
        auto new_obj = ir_mutator_t::_mutate(obj);
        level_--;
        if (!found_loop_) {
            // Innermost loop, inject let statements.
            auto body = get_stmt_body(new_obj);
            for (auto it = lets_.rbegin(); it != lets_.rend(); ++it) {
                body = let_t::make((*it)->var, (*it)->value, body);
                *it = nullptr;
            }
            new_obj = replace_stmt_body(new_obj, body);
        }
        found_loop_ = true;
        return new_obj;
    }

private:
    bool found_loop_ = false;
    int level_ = 0;
    std::vector<const let_t *> lets_;
};

// Eliminates let statements from the outer loops to be able to unroll loop
// nest for SLM buffering or prefetch injection. Example:
// Before:
//     for (int i = 0; i < I; i++) {
//         int tmp = TMP;
//         for (int j = 0; j < J; j++) {
//            ...
//         }
//     }
// After:
//     for (int i = 0; i < I; i++) {
//         for (int j = 0; j < J; j++) {
//             int tmp = TMP;
//             ...
//         }
//     }
stmt_t update_loops_for_unrolling(const stmt_t &s, const conv_config_t &cfg) {
    auto ret = s;
    if (cfg.do_loop_unroll) ret = unrolling_updater_t().mutate(s);
    trace_pass("update_loops_for_unrolling", ret);
    return ret;
}

// Helper structure for for_t.
struct loop_info_t {
    loop_info_t(const stmt_t &s) {
        ir_assert(s.is<for_t>()) << s;
        auto &loop = s.as<for_t>();
        stmt = s;
        var = loop.var;
        init_ = loop.init;
        bound_ = loop.bound;

        auto e_size = simplify(bound_ - init_);
        ir_assert(is_const(e_size));
        size_ = to_cpp<int>(e_size);
    }

    int init() const {
        ir_assert(is_const(init_));
        return to_cpp<int>(init_);
    }

    int bound() const {
        ir_assert(is_const(bound_));
        return to_cpp<int>(bound_);
    }

    int size() const { return size_; }

    stmt_t stmt;
    expr_t var;

private:
    expr_t init_;
    expr_t bound_;
    int size_;
};

// Iterates through multiple nested loops with fixed bounds. Used to unroll
// such nested loops.
class multi_loop_iterator_t {
public:
    // Ordered from innermost to outermost.
    multi_loop_iterator_t(const std::vector<loop_info_t> &loops)
        : loops_(loops) {
        for (auto &l : loops)
            var_values_.push_back(l.init());
    }

    int var_value(const expr_t &var) const {
        for (size_t i = 0; i < loops_.size(); i++) {
            if (loops_[i].var.is_same(var)) return var_values_[i];
        }
        ir_error_not_expected();
        return 0;
    }

    void advance() {
        if (loops_.empty()) return;
        for (size_t i = 0; i < loops_.size(); i++) {
            auto &l = loops_[i];
            if (++var_values_[i] < l.bound()) break;
            var_values_[i] = l.init();
        }
        ir_assert(var_values_.back() < loops_.back().bound());
    }

    std::string str() const {
        std::ostringstream oss;
        oss << "multi_loop_iterator_t(";
        for (size_t i = 0; i < loops_.size(); i++) {
            oss << (i != 0 ? ", " : "");
            oss << loops_[i].var << " = " << var_values_[i];
        }
        oss << ")";
        return oss.str();
    }

private:
    std::vector<loop_info_t> loops_;
    std::vector<int> var_values_;
};

// Extracts different parts of the compute iteration and verifies the loop nest
// is properly formed and can be further injected with SLM buffering.
class compute_step_visitor_t : public ir_visitor_t {
public:
    stmt_t find_stmt_group(const stmt_label_t &label) const {
        auto groups = find_stmt_groups(label);
        if (groups.empty()) return stmt_t();
        ir_assert(groups.size() == 1);
        return groups[0];
    }

    std::vector<stmt_t> find_stmt_groups(const stmt_label_t &label) const {
        std::vector<stmt_t> ret;
        for (auto &_g : stmt_groups_) {
            auto &g = _g.as<stmt_group_t>();
            if (g.label == label) ret.push_back(_g);
        }
        return ret;
    }

    const std::vector<stmt_t> &inner_let_stmts() const {
        return inner_let_stmts_;
    }

#define HANDLE_IR_OBJECT(type) \
    void _visit(const type *obj) override { visit_stmt(obj); }

    HANDLE_STMT_IR_OBJECTS()

#undef HANDLE_IR_OBJECT

    template <typename T>
    void visit_stmt(const T *obj) {
        auto obj_type_id = T::_type_id();
        bool is_for = (obj_type_id == for_t::_type_id());
        bool is_stmt_group = (obj_type_id == stmt_group_t::_type_id());
        bool is_let = (obj_type_id == let_t::_type_id());
        bool is_stmt_seq = (obj_type_id == stmt_seq_t::_type_id());

        // Loop may contain:
        // - Another loop
        // - Container statement (stmt_seq_t or stmt_group_t)
        // - Let statement (in the innermost loop only)
        // - Barrier
        if (loop_level_ > 0) {
            bool ok = false;
            if (is_for || is_let || is_stmt_group || is_stmt_seq) {
                ok = true;
            } else if (obj_type_id == func_call_t::_type_id()) {
                auto &call = obj->template as<func_call_t>();
                ok = call.func.is_equal(funcs::barrier_func());
            }

            if (!ok) {
                ir_error_not_expected()
                        << "Found unexpected statement inside loop.\n"
                        << stmt_t(obj);
            }
        }

        bool is_compute_loop = false;
        if (is_stmt_group) {
            auto label = obj->template as<stmt_group_t>().label;
            stmt_groups_.push_back(obj);
            if (utils::one_of(label, stmt_label_t::g2s_load(),
                        stmt_label_t::g2s_store(), stmt_label_t::g2r_load(),
                        stmt_label_t::s2r_load(), stmt_label_t::prefetch(),
                        stmt_label_t::mul())) {
                // Leaf labels, do not visit them.
                return;
            }
            if (label == stmt_label_t::compute_loop()) {
                is_compute_loop = true;
                in_compute_loop_ = true;
            }
        }

        if (is_for) loop_level_++;
        found_loop_ = false;
        ir_visitor_t::_visit(obj);
        if (in_compute_loop_ && is_let) {
            if (found_loop_)
                ir_error_not_expected()
                        << "Let is allowed in the innermost loop only.";

            inner_let_stmts_.push_back(replace_stmt_body(obj, stmt_t()));
        }
        if (is_for) {
            loop_level_--;
            found_loop_ = true;
        }

        if (is_compute_loop) in_compute_loop_ = false;
    }

private:
    bool found_loop_ = false;
    bool in_compute_loop_ = false;
    int loop_level_ = 0;

    std::vector<stmt_t> stmt_groups_;
    std::vector<stmt_t> inner_let_stmts_;
};

// Provides access to different parts of the inner compute iteration.
class compute_step_t {
public:
    compute_step_t(const stmt_t &parent) {
        compute_step_visitor_t v;
        v.visit(parent);

        compute_loop_ = v.find_stmt_group(stmt_label_t::compute_loop());
        g2s_load_ = v.find_stmt_group(stmt_label_t::g2s_load());
        g2s_store_ = v.find_stmt_group(stmt_label_t::g2s_store());
        prefetch_ = v.find_stmt_group(stmt_label_t::prefetch());
        g2r_load_ = v.find_stmt_groups(stmt_label_t::g2r_load());
        s2r_load_ = v.find_stmt_groups(stmt_label_t::s2r_load());
        mul_ = v.find_stmt_groups(stmt_label_t::mul());
        c_zero_out_ = v.find_stmt_group(stmt_label_t::c_zero_out());
        inner_let_stmts_ = v.inner_let_stmts();

        ir_assert(g2r_load_.size() == mul_.size());
        ir_assert(s2r_load_.size() == mul_.size());

        // Assign preload/mul tags to let statements.
        for (auto &_let : inner_let_stmts_) {
            auto &var = _let.as<let_t>().var;
            bool is_preload = (count_object(g2s_load_, var) > 0)
                    || (count_object(prefetch_, var) > 0);
            bool is_mul = count_object(g2r_load_, var) > 0;
            if (is_preload) preload_lets_.insert(_let);
            if (is_mul) mul_lets_.insert(_let);
        }

        // Propagate preload/mul tags up based on dependencies between let
        // statements.
        std::vector<let_info_t> let_infos;
        object_set_t<stmt_t> seen;
        std::function<void(const stmt_t &)> propagate;
        propagate = [&](const stmt_t &_let) {
            if (seen.count(_let) > 0) return;
            auto &let = _let.as<let_t>();
            for (auto &_child : inner_let_stmts_) {
                auto &child = _child.as<let_t>();
                if (_child.is_same(_let)) continue;
                if (contains_object(child.value, let.var)) {
                    // Visit child let statements first.
                    propagate(_child);
                    // Propagate child preload/mul values to this let statement.
                    if (is_preload_let(_child)) preload_lets_.insert(_let);
                    if (is_mul_let(_child)) mul_lets_.insert(_let);
                }
            }
            auto let_info = create_let_info(
                    let, is_preload_let(_let), is_mul_let(_let));
            let_infos.push_back(let_info);
            seen.insert(_let);
        };
        for (auto &_let : inner_let_stmts_)
            propagate(_let);

        // Duplicate lets that are used in both preload and mul contexts.
        duplicate_lets(let_infos);
    }

    // See ir_core.hpp for the description.
    const stmt_t &compute_loop() const { return compute_loop_; }
    const stmt_t &g2s_load() const { return g2s_load_; }
    const stmt_t &g2s_store() const { return g2s_store_; }
    const stmt_t &prefetch() const { return prefetch_; }
    const std::vector<stmt_t> &g2r_load() const { return g2r_load_; }
    const std::vector<stmt_t> &s2r_load() const { return s2r_load_; }
    const std::vector<stmt_t> &mul() const { return mul_; }
    const stmt_t &c_zero_out() const { return c_zero_out_; }
    const std::vector<stmt_t> &inner_let_stmts() const {
        return inner_let_stmts_;
    }

    bool is_preload_let(const stmt_t &s) const {
        return preload_lets_.count(s) > 0;
    }
    bool is_mul_let(const stmt_t &s) const { return mul_lets_.count(s) > 0; }

private:
    struct let_info_t {
        let_info_t(const expr_t &var) : var(var) {}

        expr_t var;
        expr_t preload_var;
        expr_t mul_var;

        bool is_preload() const { return !preload_var.is_empty(); }
        bool is_mul() const { return !mul_var.is_empty(); }

        bool needs_update() const { return is_preload() && is_mul(); }
    };

    let_info_t create_let_info(const let_t &let, bool is_preload, bool is_mul) {
        let_info_t info(let.var);
        if (is_preload && !is_mul) {
            info.preload_var = let.var;
        } else if (!is_preload && is_mul) {
            info.mul_var = let.var;
        } else if (is_preload && is_mul) {
            info.preload_var = create_var_with_suffix(let.var, "p");
            info.mul_var = create_var_with_suffix(let.var, "m");
        }
        return info;
    }

    void duplicate_lets(const std::vector<let_info_t> &let_infos) {
        int nlets = int(inner_let_stmts_.size());
        ir_assert(int(let_infos.size()) == nlets);

        std::vector<stmt_t> new_lets;
        for (int i = nlets - 1; i >= 0; i--) {
            auto &info = let_infos[i];
            auto &old_let = inner_let_stmts_[i].as<let_t>();
            if (!info.needs_update()) {
                auto new_value = update_var(old_let.value, let_infos,
                        info.is_preload(), info.is_mul());
                auto new_let = inner_let_stmts_[i];
                if (!new_value.is_same(old_let.value)) {
                    new_let = let_t::make(old_let.var, new_value, old_let.body);
                }
                new_lets.push_back(new_let);
                continue;
            }

            preload_lets_.erase(&old_let);
            mul_lets_.erase(&old_let);

            auto preload_value
                    = update_var(old_let.value, let_infos, true, false);
            auto preload_let = let_t::make(
                    info.preload_var, preload_value, old_let.body);

            auto mul_value = update_var(old_let.value, let_infos, false, true);
            auto mul_let = let_t::make(info.mul_var, mul_value, old_let.body);

            preload_lets_.insert(preload_let);
            new_lets.push_back(preload_let);

            mul_lets_.insert(mul_let);
            new_lets.push_back(mul_let);

            // Update statements.
            g2s_load_ = update_var(g2s_load_, let_infos, true, false);
            g2s_store_ = update_var(g2s_store_, let_infos, true, false);
            prefetch_ = update_var(prefetch_, let_infos, true, false);
            g2r_load_ = update_var(g2r_load_, let_infos, false, true);
            s2r_load_ = update_var(s2r_load_, let_infos, false, true);
            mul_ = update_var(mul_, let_infos, false, true);
        }

        std::reverse(new_lets.begin(), new_lets.end());
        inner_let_stmts_ = new_lets;
    }

    template <typename T>
    static std::vector<T> update_var(const std::vector<T> &vec,
            const std::vector<let_info_t> &let_infos, bool is_preload,
            bool is_mul) {
        std::vector<T> ret;
        for (auto &v : vec)
            ret.push_back(update_var(v, let_infos, is_preload, is_mul));
        return ret;
    }

    static object_t update_var(const object_t &obj,
            const std::vector<let_info_t> &let_infos, bool is_preload,
            bool is_mul) {
        auto ret = obj;
        for (auto &info : let_infos) {
            if (!info.needs_update()) continue;
            if (!contains_object(ret, info.var)) continue;
            if (is_preload) {
                ir_assert(info.is_preload());
                ret = substitute(ret, info.var, info.preload_var);
            } else if (is_mul) {
                ir_assert(info.is_mul());
                ret = substitute(ret, info.var, info.mul_var);
            }
        }
        return ret;
    }

    static expr_t create_var_with_suffix(
            const expr_t &_var, const std::string &suffix) {
        auto &var = _var.as<var_t>();
        auto new_name = var.name + "_" + suffix;
        return var_t::make(var.type, new_name);
    }

    stmt_t compute_loop_;
    stmt_t g2s_load_;
    stmt_t g2s_store_;
    stmt_t prefetch_;
    std::vector<stmt_t> g2r_load_;
    std::vector<stmt_t> s2r_load_;
    std::vector<stmt_t> mul_;
    stmt_t c_zero_out_;

    std::vector<stmt_t> inner_let_stmts_;

    // Due to loop unrolling the inner let statements may depend on different
    // indices of the outer loops. There are two contexts:
    // - "preload" loop iteration, e.g. index I
    // - "multiplication" loop iteration, e.g. index (I + nbuf)
    // Preloads (either via SLM or via prefetches) for the corresponding
    // multiplication are executed several iterations before the real
    // multiplication. That's why we need to know exactly in which context the
    // given let statement is used. It might be that the same variable is used
    // from two different contexts. In this case it is duplicated and
    // initialized with different values for each case.
    object_set_t<stmt_t> preload_lets_;
    object_set_t<stmt_t> mul_lets_;
};

// Helper class to work with loop nest of the compute loop.
class compute_loop_nest_t {
public:
    compute_loop_nest_t(const stmt_t &root) : root_(root) {
        for (auto &l : find_objects<for_t>(root)) {
            loops_.emplace_back(l);
        }

        if (loops_.empty()) {
            outer_loop_size_ = 1;
            return;
        }

        // Outer loop may not be used for unrolling hence loop iterations must
        // not use its index. If this doesn't hold, assume a dummy outer loop
        // with single iteration.
        auto &outer_info = loops_.back();
        auto &outer_loop = outer_info.stmt.as<for_t>();
        if (count_object(outer_loop.body, outer_loop.var) == 0) {
            outer_loop_size_ = outer_info.size();
        } else {
            outer_loop_size_ = 1;
        }
    }

    const std::vector<loop_info_t> &loops() const { return loops_; }

    // Number of iterations of all loops.
    int size() const {
        int ret = 1;
        for (auto &l : loops_)
            ret *= l.size();
        return ret;
    }

    // Number of iterations in the outermost loop (see comments in ctor).
    int outer_loop_size() const { return outer_loop_size_; }

    template <typename F>
    void for_each_loop_var(const F &f) const {
        for (auto &l : loops_)
            f(l.var);
    }

    // Number of iterations of all loops except the outermost.
    int inner_loops_size() const { return size() / outer_loop_size(); }

private:
    stmt_t root_;
    std::vector<loop_info_t> loops_;
    int outer_loop_size_;
};

struct compute_params_t {
    compute_params_t() = default;

    compute_params_t(int slm_bufs, int gmem_bufs, int slm_buf_size,
            int prefetch_bufs, int inner_loops_iters)
        : slm_bufs(slm_bufs)
        , gmem_bufs(gmem_bufs)
        , slm_buf_size(slm_buf_size)
        , prefetch_bufs(prefetch_bufs) {
        use_slm = (slm_buf_size > 0);
        use_prefetch = (prefetch_bufs > 0);
        ir_assert(!use_slm || !use_prefetch)
                << "Can't have both SLM buffering and prefetch enabled.";
        if (use_slm) {
            ir_assert(utils::one_of(slm_bufs, 1, 2, 3));
            ir_assert(utils::one_of(gmem_bufs, 1, 2));
            preload_bufs = slm_bufs;
            unroll = math::lcm(slm_bufs * gmem_bufs, inner_loops_iters);
        } else if (use_prefetch) {
            preload_bufs = prefetch_bufs;
            ir_assert(slm_bufs == 0);
            ir_assert(gmem_bufs == 0);
            unroll = math::lcm(prefetch_bufs, inner_loops_iters);
        } else {
            preload_bufs = 0;
            ir_assert(slm_bufs == 0);
            ir_assert(gmem_bufs == 0);
            unroll = inner_loops_iters;
        }
    }

    int slm_bufs;
    int gmem_bufs;
    int slm_buf_size;
    int prefetch_bufs;
    int preload_bufs;
    int unroll;

    bool use_slm;
    bool use_prefetch;
};

// Helper class to implement SLM buffering.
class compute_iterator_t {
public:
    compute_iterator_t(const compute_params_t &params,
            const compute_loop_nest_t &loop_nest)
        : params(params)
        , preload_loop_it(loop_nest.loops())
        , mul_loop_it(loop_nest.loops()) {

        int compute_iters = loop_nest.size();
        iters = compute_iters;
        ir_assert(iters >= 1) << "Empty loop is not expected.";

        iters += std::max(0, preload_bufs() - 1) + std::max(0, gmem_bufs() - 1);
        ramp_up_iters
                = std::max(1, preload_bufs() + std::max(0, gmem_bufs() - 1));
        ramp_down_iters = std::min(
                std::max(0, preload_bufs() - 1) + std::max(0, gmem_bufs() - 1),
                iters - ramp_up_iters);
        body_iters = iters - ramp_up_iters - ramp_down_iters;
        body_iters = utils::rnd_dn(body_iters, params.unroll);
        ramp_down_iters = iters - ramp_up_iters - body_iters;

        ir_assert(ramp_up_iters + body_iters + ramp_down_iters == iters);

        iter = 0;
        linear_id = 0;
        riter = iters - 1;
    }

    int unroll() const { return params.unroll; }

    int preload_bufs() const { return params.preload_bufs; }

    int slm_bufs() const { return params.slm_bufs; }

    int gmem_bufs() const { return params.gmem_bufs; }

    compute_iterator_t &operator++() {
        if (do_preload()) preload_loop_it.advance();
        if (do_mul()) mul_loop_it.advance();
        ++iter;
        ++linear_id;
        --riter;
        return *this;
    }

    void advance(int n) {
        ir_assert(n % params.unroll == 0);
        ir_assert(iter + n <= iters);

        iter += n;
        riter -= n;
    }

    bool do_mul() const {
        return iter >= std::max(0, preload_bufs() - 1)
                + std::max(0, gmem_bufs() - 1);
    }

    bool is_first_mul() const {
        return iter
                == std::max(0, preload_bufs() - 1)
                + std::max(0, gmem_bufs() - 1);
    }
    bool is_last_mul() const { return riter == 0; }

    bool do_preload() const {
        if (preload_bufs() == 0) return false;
        return riter >= (preload_bufs() - 1) + std::max(0, gmem_bufs() - 1);
    }

    bool do_prefetch() const {
        if (!params.use_prefetch) return false;
        return do_preload();
    }

    bool do_g2s_load() const {
        if (!params.use_slm) return false;
        return do_preload();
    }

    bool do_s2r_load() const {
        if (!params.use_slm) return false;
        ir_assert(gmem_bufs() >= 1);
        return iter >= (gmem_bufs() - 1) && riter >= (slm_bufs() - 1);
    }

    int gmem_write_buf_index() const {
        ir_assert(do_g2s_load());
        return iter % gmem_bufs();
    }

    int gmem_read_buf_index() const {
        ir_assert(do_s2r_load());
        return (iter - (gmem_bufs() - 1)) % gmem_bufs();
    }

    int slm_read_offset_update() const {
        ir_assert(params.use_slm);
        ir_assert(do_mul());

        int slm_iter = iter - (gmem_bufs() - 1) - (slm_bufs() - 1);
        int cur_slm_idx = slm_iter % slm_bufs();
        int next_slm_idx = (slm_iter + 1) % slm_bufs();
        int ret = next_slm_idx * params.slm_buf_size
                - cur_slm_idx * params.slm_buf_size;
        return ret;
    }

    int slm_write_offset_update() const {
        ir_assert(params.use_slm);
        ir_assert(do_s2r_load());

        int slm_iter = iter - (gmem_bufs() - 1);
        int cur_slm_idx = slm_iter % slm_bufs();
        int next_slm_idx = (slm_iter + 1) % slm_bufs();
        int ret = next_slm_idx * params.slm_buf_size
                - cur_slm_idx * params.slm_buf_size;
        return ret;
    }

    compute_params_t params;
    multi_loop_iterator_t preload_loop_it;
    multi_loop_iterator_t mul_loop_it;

    // ramp_up_iters + body_iters + ramp_down_iters == iters
    int iters;
    int ramp_up_iters;
    int body_iters;
    int ramp_down_iters;

    // Invariant: iter + riter = iters - 1
    int iter;
    int riter;

    int linear_id;
};

// Basic LRU SBID allocator, tries to use the same SBIDs for the same GRF
// buffers.
class sbid_manager_t {
public:
    sbid_manager_t() : tuple_func_(builtin_t::make("tuple")) {}

    ngen_proxy::SBID get_sbid(const expr_t &buf, int index = 0) {
        auto key = tuple_func_.call({buf, expr_t(index)});

        int free_idx = -1;
        for (int i = 0; i < sbid_count; i++) {
            auto &e = entries_[i];
            if (key.is_equal(e.key)) {
                e.time = cur_time_++;
                return ngen_proxy::SBID(i);
            }
            if (free_idx == -1 && e.key.is_empty()) free_idx = i;
        }

        // Not found but there is a free SBID.
        if (free_idx != -1) {
            entries_[free_idx] = {key, cur_time_++};
            return ngen_proxy::SBID(free_idx);
        }

        // Find the oldest SBID and use it.
        int old_idx = 0;
        int old_time = entries_[0].time;
        for (int i = 1; i < sbid_count; i++) {
            if (entries_[i].time < old_time) {
                old_idx = i;
                old_time = entries_[i].time;
            }
        }

        entries_[old_idx] = entry_t({key, cur_time_++});
        return ngen_proxy::SBID(old_idx);
    }

private:
    struct entry_t {
        stmt_t key;
        int time;
    };

    static const int sbid_count = 16;
    std::array<entry_t, sbid_count> entries_;

    func_t tuple_func_;
    int cur_time_ = 0;
};

// Helper to assign SBIDs to IR function calls.
class sbid_assigner_t {
public:
    sbid_assigner_t() = default;

    sbid_assigner_t(sbid_manager_t &external_sbid_mgr)
        : external_sbid_mgr_(&external_sbid_mgr) {}

    stmt_t assign(const stmt_t &stmt) {
        auto stmt_vec = flatten_statements(stmt);
        stmt_t ret = stmt;
        int prefetch_idx = 0;
        for (auto &_s : stmt_vec) {
            if (!_s.is<func_call_t>()) continue;
            auto s = _s;
            if (is_func_call<send_t>(s)) {
                auto &send = s.as<func_call_t>().func.as<send_t>();
                int idx = (send.is_prefetch ? prefetch_idx++ : 0);
                auto sbid = get_sbid(send_t::arg_reg_buf(s), idx);
                s = update_call_with_sbid(s, sbid);
            } else if (is_func_call<dpas_t>(s)) {
                auto &attr = s.as<func_call_t>().attr;
                auto *mod_attr = attr.as_ptr<instruction_modifier_attr_t>();
                if (!mod_attr || !mod_attr->mod.is_atomic) {
                    // Last dpas in Atomic chain.
                    auto sbid = get_sbid(dpas_t::arg_src1(s));
                    s = update_call_with_sbid(s, sbid);
                }
            } else if (s.is<func_call_t>()) {
                auto &c = s.as<func_call_t>();
                if (c.func.is_equal(funcs::signal_func())
                        || c.func.is_equal(funcs::slm_fence_func())
                        || c.func.is_equal(funcs::barrier_func())) {
                    // Use 0 as the key for signals and SLM fences.
                    auto sbid = get_sbid(expr_t(0));
                    s = update_call_with_sbid(s, sbid);
                }
            } else {
                ir_error_not_expected() << s;
            }
            ret = substitute(ret, _s, s);
        }
        return ret;
    }

private:
    ngen_proxy::SBID get_sbid(const expr_t &ptr, int index = 0) {
        auto &sbid_mgr
                = (external_sbid_mgr_ ? *external_sbid_mgr_ : local_sbid_mgr_);
        return sbid_mgr.get_sbid(ptr, index);
    }

    static stmt_t update_call_with_sbid(
            const stmt_t &s, const ngen_proxy::SBID &sbid) {
        return instruction_modifier_attr_t::make(
                ngen_proxy::InstructionModifier().with_sbid(sbid))
                .apply_to(s);
    }

    sbid_manager_t local_sbid_mgr_;
    sbid_manager_t *external_sbid_mgr_ = nullptr;
};

class simple_slm_buffering_injector_t {
public:
    simple_slm_buffering_injector_t(ngen::HW hw, const stmt_t &root,
            const conv_config_t &cfg, ir_context_t &ir_ctx, int ab_slm_size)
        : hw_(hw)
        , cfg_(cfg)
        , ir_ctx_(ir_ctx)
        , ab_slm_size_(ab_slm_size)
        , root_(root)
        , alloc_mgr_(root_)
        , step_(root)
        , loop_nest_(root) {}

    stmt_t inject() {
        ir_assert(cfg_.gmem_bufs == 1) << "GRF buffering is not supported.";
        if (utils::one_of(cfg_.slm_bufs, 0, 1)) return root_;

        ir_assert(cfg_.use_a_slm == cfg_.use_b_slm)
                << "Mixed SLM/GMEM loads are not supported.";

        auto loop = step_.compute_loop();

        // SLM indices are allocated as follows:
        // slm_idx[0] -> slm_buf_store
        // slm_idx[1] -> slm_buf_compute
        // slm_idx[2] -> slm_counter
        auto slm_idx_buf
                = ir_ctx_.create_tmp_var(type_t::byte_ptr(), "slm_idx");
        int slm_idx_size = type_t::s32().size();

        auto slm_idx_load = [&](int off, int elems) {
            return load_t::make(
                    type_t::s32(elems), slm_idx_buf, slm_idx_size * off);
        };

        // Initialize slm_idx.
        int off = 0;
        auto store0 = store_t::make(slm_idx_buf, off, 0);
        off += slm_idx_size;

        auto store1 = store_t::make(slm_idx_buf, off, 1);
        off += slm_idx_size;

        auto store2 = store_t::make(
                slm_idx_buf, off, int_imm_t::make(0, type_t::s32()));

        auto slm_idx_init = store0.append(store1).append(store2);

        auto slm_idx_load2 = slm_idx_load(0, 2);
        auto slm_idx_load4 = slm_idx_load(0, 4);
        auto slm_idx_store = store_t::make(slm_idx_buf, 0,
                slm_idx_load4 + shuffle_t::make_broadcast(1, 4));

        // Update slm_idx.
        auto mask = (slm_idx_load2
                == shuffle_t::make_broadcast(cfg_.slm_bufs, 2));
        auto slm_idx_store_fix = store_t::make(slm_idx_buf, 0,
                shuffle_t::make_broadcast(int_imm_t::make(0, type_t::s32()), 2),
                store_t::default_stride, mask);

        auto slm_idx_update = slm_idx_store.append(slm_idx_store_fix);

        loop = slm_idx_init.append(loop);

        auto &g2s_store_orig = step_.g2s_store();
        auto &s2r_load = step_.s2r_load();
        auto &mul = step_.mul();

        auto g2s_store = g2s_store_orig;

        ir_assert(s2r_load.size() == mul.size());

        stmt_t s2r_mul;
        for (int i = 0; i < int(mul.size()); i++) {
            s2r_mul = s2r_mul.append(s2r_load[i]);
            loop = substitute(loop, s2r_load[i], stmt_t(), 1);
            s2r_mul = s2r_mul.append(mul[i]);
            loop = substitute(loop, mul[i], stmt_t(), 1);
        }

        loop = remove_synchronization(loop);

        s2r_mul = sub_slm_bufs(s2r_mul, slm_idx_load(1, 1));
        g2s_store = sub_slm_bufs(g2s_store, slm_idx_load(0, 1));
        g2s_store = g2s_store.append(slm_idx_update);

        auto s2r_mul_body = s2r_mul;
        auto s2r_mul_tail = s2r_mul;
        auto slm_counter = slm_idx_load(2, 1);
        auto cond = (slm_counter >= cfg_.slm_bufs - 1);

        if (cfg_.slm_bufs == 2) {
            s2r_mul_body = if_t::make(cond, s2r_mul_body);
            g2s_store = g2s_store.append(funcs::barrier());
        } else {
            auto fence_signal = funcs::slm_fence().append(funcs::signal());
            s2r_mul_body = s2r_mul_body.append(funcs::signal());
            s2r_mul_body = if_t::make(cond, s2r_mul_body, fence_signal);
            s2r_mul_body = funcs::barrier_wait().append(s2r_mul_body);
        }

        loop = substitute(
                loop, g2s_store_orig, s2r_mul_body.append(g2s_store), 1);

        if (cfg_.slm_bufs == 3) {
            // Emit initial signal, to match wait-signal pairs in the loop.
            loop = funcs::signal().append(loop);
        }

        // Complete the remaining iterations.
        int rem_iters = cfg_.slm_bufs - 1;
        int mul_start = std::max(0, rem_iters - loop_nest_.size());
        for (int i = 0; i < rem_iters; i++) {
            if (cfg_.slm_bufs == 3) loop = loop.append(funcs::barrier_wait());
            if (i >= mul_start) loop = loop.append(s2r_mul_tail);
            loop = loop.append(slm_idx_update);
            if (cfg_.slm_bufs == 3 && i + 1 < rem_iters)
                loop = loop.append(funcs::signal());
        }

        if (cfg_.assign_sbids) loop = sbid_assigner_t().assign(loop);

        const auto grf_size = ngen::GRF::bytes(hw_);
        loop = alloc_t::make(
                slm_idx_buf, grf_size, alloc_kind_t::grf, {}, loop);

        alloc_updater_t alloc_updater;

        auto slm_buffers = alloc_mgr_.find_buffers(alloc_kind_t::slm);
        ir_assert(slm_buffers.size() == 1);
        auto &slm_buf = slm_buffers[0];
        int non_ab_slm_size = alloc_mgr_.alloc_size(slm_buf) - ab_slm_size_;
        alloc_updater.resize(
                slm_buf, non_ab_slm_size + ab_slm_size_ * cfg_.slm_bufs);

        auto ret = substitute(root_, step_.compute_loop(), loop, 1);
        ret = alloc_updater.update(ret);
        return ret;
    }

    static stmt_t remove_synchronization(const stmt_t &s) {
        auto ret = s;
        for (auto &_c : find_objects<func_call_t>(s)) {
            auto &c = _c.as<func_call_t>();
            if (c.func.is_equal(funcs::signal_func())
                    || c.func.is_equal(funcs::slm_fence_func())
                    || c.func.is_equal(funcs::barrier_func())) {
                ret = substitute(ret, _c, stmt_t(), 1);
            }
        }
        return ret;
    }

    stmt_t sub_slm_bufs(const stmt_t &stmt, const expr_t &slm_idx) const {
        auto stmt_vec = flatten_statements(stmt);

        stmt_t ret = stmt;
        for (auto &s : stmt_vec) {
            if (!is_func_call<send_t>(s)) continue;

            auto &send = s.as<func_call_t>().func.as<send_t>();

            // This is not send to SLM, skip.
            if (send.address_model != ngen_proxy::AddressModel::ModelSLM)
                continue;

            auto new_args = s.as<func_call_t>().args;
            send_t::arg_mem_off(new_args) += ab_slm_size_ * slm_idx;
            auto new_send = send.call(new_args);
            ret = substitute(ret, s, new_send, 1);
        }

        return ret;
    }

    ngen::HW hw_;
    const conv_config_t &cfg_;
    ir_context_t &ir_ctx_;
    int ab_slm_size_;

    stmt_t root_;
    alloc_manager_t alloc_mgr_;
    compute_step_t step_;
    compute_loop_nest_t loop_nest_;
};

// Injects SLM buffering without unrolling based on the config.
stmt_t inject_simple_slm_buffering(ngen::HW hw, const stmt_t &s,
        const conv_config_t &cfg, ir_context_t &ir_ctx, int ab_slm_size) {
    auto ret = simple_slm_buffering_injector_t(hw, s, cfg, ir_ctx, ab_slm_size)
                       .inject();
    trace_pass("inject_simple_slm_buffering", ret);
    return ret;
}

class unrolling_injector_t {
public:
    unrolling_injector_t(const stmt_t &root, const conv_config_t &cfg,
            ir_context_t &ir_ctx, int ab_slm_size)
        : cfg_(cfg)
        , ir_ctx_(ir_ctx)
        , ab_slm_size_(ab_slm_size)
        , root_(root)
        , alloc_mgr_(root_)
        , step_(root)
        , loop_nest_(root) {
        int inner_iters = loop_nest_.inner_loops_size();
        params_ = compute_params_t(cfg_.slm_bufs, cfg_.gmem_bufs, ab_slm_size,
                cfg_.prefetch_bufs, inner_iters);
        if (params_.use_slm) {
            for (auto &b :
                    find_send_buffers(step_.g2s_load(), /*is_mem=*/false)) {
                g2s_reg_bufs_.emplace_back(b, alloc_mgr_.alloc_size(b));
            }
        }
    }

    stmt_t inject() {
        compute_iterator_t it(params_, loop_nest_);
        stmt_t body;

        sbid_manager_t sbid_mgr;

        // Ramp-up.
        for (int i = 0; i < it.ramp_up_iters; i++) {
            body = stmt_seq_t::make(body, create_iteration(it, sbid_mgr));
            ++it;
        }

        // Body.
        if (it.body_iters > 0) {
            stmt_t loop_body;
            for (int i = 0; i < it.unroll(); i++) {
                loop_body = loop_body.append(create_iteration(it, sbid_mgr));
                ++it;
            }
            int extent = it.body_iters / it.unroll();
            if (extent == 1) {
                body = body.append(loop_body);
            } else {
                ir_assert(extent > 0);
                auto for_var = ir_ctx_.create_tmp_var(type_t::s32(), "i");
                body = body.append(for_t::make(for_var, 0, extent, loop_body));
            }
            it.advance(it.body_iters - it.unroll());
        }

        // Ramp-down.
        for (int i = 0; i < it.ramp_down_iters; i++) {
            body = body.append(create_iteration(it, sbid_mgr));
            ++it;
        }

        auto ret = substitute(root_, step_.compute_loop(), body, 1);
        if (params_.use_slm) {
            alloc_updater_t alloc_updater;

            // Update buffer sizes.
            for (auto &b : g2s_reg_bufs_) {
                alloc_updater.resize(
                        b.buf, alloc_mgr_.alloc_size(b.buf) * cfg_.gmem_bufs);
            }

            auto slm_buffers = alloc_mgr_.find_buffers(alloc_kind_t::slm);
            if (!slm_buffers.empty()) {
                ir_assert(slm_buffers.size() == 1);

                auto &slm_buf = slm_buffers[0];
                int non_ab_slm_size
                        = alloc_mgr_.alloc_size(slm_buf) - ab_slm_size_;
                alloc_updater.resize(slm_buf,
                        non_ab_slm_size + ab_slm_size_ * cfg_.slm_bufs);
            }

            ret = alloc_updater.update(ret);
        }

        // Remove zero-out statement for C (handled by sub_fma_acc_with_zero).
        ret = substitute(ret, step_.c_zero_out(), stmt_t(), 1);

        return ret;
    }

private:
    struct buffer_info_t {
        buffer_info_t(const expr_t &buf, int size) : buf(buf), size(size) {}

        expr_t buf;
        int size;
    };

    stmt_t create_iteration(
            const compute_iterator_t &it, sbid_manager_t &sbid_mgr) const {
        auto g2s_load = step_.g2s_load();
        auto g2s_store = step_.g2s_store();
        auto prefetch = step_.prefetch();
        auto g2r_load = step_.g2r_load();
        auto s2r_load = step_.s2r_load();
        auto mul = step_.mul();
        auto lets = step_.inner_let_stmts();

        loop_nest_.for_each_loop_var([&](const expr_t &v) {
            g2s_load = const_fold(substitute(
                    g2s_load, v, expr_t(it.preload_loop_it.var_value(v))));
            g2s_store = const_fold(substitute(
                    g2s_store, v, expr_t(it.preload_loop_it.var_value(v))));
            prefetch = const_fold(substitute(
                    prefetch, v, expr_t(it.preload_loop_it.var_value(v))));
            for (auto &s : g2r_load) {
                s = const_fold(
                        substitute(s, v, expr_t(it.mul_loop_it.var_value(v))));
            }
            for (auto &s : s2r_load) {
                s = const_fold(substitute(
                        s, v, expr_t(it.preload_loop_it.var_value(v))));
            }
            for (int i = 0; i < int(lets.size()); i++) {
                auto &let = lets[i];
                auto &orig_let = step_.inner_let_stmts()[i];
                expr_t var_value;
                bool is_preload_let = step_.is_preload_let(orig_let);
                bool is_mul_let = step_.is_mul_let(orig_let);
                if (is_preload_let && !is_mul_let) {
                    var_value = it.preload_loop_it.var_value(v);
                } else if (is_mul_let && !is_preload_let) {
                    var_value = it.mul_loop_it.var_value(v);
                } else {
                    ir_assert(count_object(let.as<let_t>().value, v) == 0)
                            << "Unexpected reference to variable " << v
                            << " from " << let;
                    continue;
                }
                let = const_fold(substitute(let, v, var_value));
            }
        });

        if (params_.use_slm) {
            g2s_load = sub_gmem_bufs(g2s_load, it, /*is_read=*/false);
            g2s_store = sub_gmem_bufs(g2s_store, it, /*is_read=*/true);

            g2s_store = sub_slm_bufs(g2s_store, it, /*is_read=*/false);
            for (auto &s : s2r_load) {
                s = sub_slm_bufs(s, it, /*is_read=*/true);
            }
        }

        if (it.is_first_mul()) {
            for (auto &m : mul) {
                m = sub_fma_acc_with_zero(m);
            }
        }

        stmt_t iter_stmt;
        if (it.slm_bufs() == 3 && it.do_mul()) {
            iter_stmt = iter_stmt.append(funcs::barrier_wait());
        }

        if (it.do_g2s_load()) iter_stmt = iter_stmt.append(g2s_load);

        if (it.slm_bufs() == 3 && it.iter == it.gmem_bufs()) {
            iter_stmt = iter_stmt.append(funcs::slm_fence());
            iter_stmt = iter_stmt.append(funcs::signal());
        }

        if (it.do_s2r_load() && it.slm_bufs() == 1) {
            iter_stmt = iter_stmt.append(funcs::barrier());
            iter_stmt = iter_stmt.append(g2s_store);
            iter_stmt = iter_stmt.append(funcs::barrier());
        }

        if (it.do_prefetch()) iter_stmt = iter_stmt.append(prefetch);

        if (it.do_mul()) {
            for (size_t i = 0; i < mul.size(); i++) {
                iter_stmt = iter_stmt.append(g2r_load[i]);
                iter_stmt = iter_stmt.append(s2r_load[i]);
                iter_stmt = iter_stmt.append(mul[i]);
            }
            if (it.slm_bufs() == 3 && !it.is_last_mul()) {
                iter_stmt = iter_stmt.append(funcs::signal());
            }
        }
        if (it.do_s2r_load() && it.slm_bufs() >= 2) {
            iter_stmt = iter_stmt.append(g2s_store);
            if (it.slm_bufs() == 2) {
                iter_stmt = iter_stmt.append(funcs::barrier());
            }
        }

        if (cfg_.assign_sbids)
            iter_stmt = sbid_assigner_t(sbid_mgr).assign(iter_stmt);

        iter_stmt = inject_local_let(iter_stmt, lets, it.linear_id);

        return iter_stmt;
    }

    stmt_t sub_gmem_bufs(const stmt_t &stmt, const compute_iterator_t &it,
            bool is_read) const {
        if (it.slm_bufs() == 0) return stmt;
        if (is_read && !it.do_s2r_load()) return stmt;
        if (!is_read && !it.do_g2s_load()) return stmt;

        int buf_idx = (is_read ? it.gmem_read_buf_index()
                               : it.gmem_write_buf_index());
        if (buf_idx == 0) return stmt;

        auto ret = stmt;
        for (auto &b : g2s_reg_bufs_) {
            ret = substitute(ret, b.buf, b.buf[buf_idx * b.size]);
        }
        return ret;
    }

    stmt_t sub_slm_bufs(const stmt_t &stmt, const compute_iterator_t &it,
            bool is_read) const {
        if (it.slm_bufs() <= 1) return stmt;
        if (is_read && !it.do_mul()) return stmt;
        if (!is_read && !it.do_s2r_load()) return stmt;

        int upd = (is_read ? it.slm_read_offset_update()
                           : it.slm_write_offset_update());

        auto stmt_vec = flatten_statements(stmt);

        stmt_t ret = stmt;
        for (auto &s : stmt_vec) {
            auto *call = s.as_ptr<func_call_t>();
            if (!call) continue;
            auto *func = call->func.as_ptr<send_t>();
            if (!func) continue;

            auto &send = call->func.as<send_t>();
            auto &args = call->args;
            auto &mem_buf = send_t::arg_mem_buf(args);
            auto &header_buf = send_t::arg_mem_off(args);

            // This is not send to SLM, skip.
            if (send.address_model != ngen_proxy::AddressModel::ModelSLM)
                continue;

            // May have signed offset.
            auto store_obj = send.create_offset_store(
                    header_buf, mem_buf, upd, /*is_signed_offset=*/true);
            auto &store = store_obj.as<store_t>();
            expr_t old_value
                    = load_t::make(send.address_type(), store.buf, store.off);
            auto post_inc_store = store_t::make(
                    store.buf, store.off, old_value + store.value);
            ret = substitute(ret, s, stmt_seq_t::make(s, post_inc_store), 1);
        }

        return ret;
    }

    static stmt_t sub_fma_acc_with_zero(const stmt_t &stmt) {
        auto stmt_vec = flatten_statements(stmt);

        object_eq_set_t<expr_t> seen_dst;
        stmt_t ret = stmt;
        for (auto &s : stmt_vec) {
            if (is_func_call<dpas_t>(s)) {
                auto &call = s.as<func_call_t>();

                auto &dst = dpas_t::arg_dst(s);
                auto src0 = expr_t(0); // Will be translated to null register.
                auto &src1 = dpas_t::arg_src1(s);
                auto &src2 = dpas_t::arg_src2(s);

                auto new_call = func_call_t::make(
                        call.func, {dst, src0, src1, src2}, call.attr);
                ret = substitute(ret, s, new_call, 1);
            } else if (is_func_call<mad_t>(s)) {
                auto &call = s.as<func_call_t>();

                auto &dst = mad_t::arg_dst(s);
                auto src0 = expr_t(0); // Will be translated to null register.
                auto &src1 = mad_t::arg_src1(s);
                auto &src2 = mad_t::arg_src2(s);

                if (!seen_dst.insert(dst).second) continue;

                auto new_call = func_call_t::make(
                        call.func, {dst, src0, src1, src2}, call.attr);
                ret = substitute(ret, s, new_call, 1);
            }
        }
        return ret;
    }

    // Returns memory buffers if is_mem is true and register buffers otherwise.
    static object_set_t<expr_t> find_send_buffers(
            const stmt_t &s, bool is_mem) {
        object_set_t<expr_t> ret;
        auto calls = find_objects<func_call_t>(s);
        for (auto &_c : calls) {
            auto &c = _c.as<func_call_t>();
            if (!c.func.is<send_t>()) continue;
            auto &buf = (is_mem ? send_t::arg_mem_buf(_c)
                                : send_t::arg_reg_buf(_c));
            ret.insert(buf.as<ptr_t>().base);
        }
        return ret;
    }

    static stmt_t inject_local_let(const stmt_t &_s,
            const std::vector<stmt_t> &enclosed_lets, int id) {
        auto s = _s;

        // Inject let statements from the innermost loop.
        for (auto &_let : enclosed_lets) {
            auto &let = _let.as<let_t>();
            s = let_t::make(let.var, let.value, s);
        }

        // Substitute variables to avoid clashing.
        auto lets = find_objects<let_t>(s);
        for (auto &_let : lets) {
            auto &let = _let.as<let_t>();
            auto &var = let.var.as<var_t>();
            auto local_var = var_t::make(
                    var.type, var.name + "_" + std::to_string(id));
            s = substitute(s, let.var, local_var);
        }
        return s;
    }

    const conv_config_t &cfg_;
    ir_context_t &ir_ctx_;
    int ab_slm_size_;

    stmt_t root_;
    alloc_manager_t alloc_mgr_;
    compute_step_t step_;
    compute_loop_nest_t loop_nest_;
    compute_params_t params_;

    std::vector<buffer_info_t> g2s_reg_bufs_; // For SLM buffering.
};

// Injects loop unrolling based on the config. Possible options:
// - Without preload (no SLM buffering, no prefetch)
// - With SLM buffering
// - With prefetch
stmt_t inject_unrolling(const stmt_t &s, const conv_config_t &cfg,
        ir_context_t &ir_ctx, int ab_slm_size) {
    auto ret = unrolling_injector_t(s, cfg, ir_ctx, ab_slm_size).inject();
    trace_pass("inject_unrolling", ret);
    return ret;
}

class store_splitter_t : public ir_mutator_t {
public:
    store_splitter_t(ngen::HW hw) : hw_(hw) {}

    object_t _mutate(const store_t *obj) override {
        int elems = obj->value.type().elems();
        int elem_size = obj->value.type().scalar().size();
        int stride = (obj->has_default_stride() ? 1 : obj->stride / elem_size);
        int store_size = elem_size * stride * elems;
        const auto grf_size = ngen::GRF::bytes(hw_);
        if (store_size <= 2 * grf_size) return ir_mutator_t::_mutate(obj);

        int step = 2 * grf_size / (stride * elem_size);
        stmt_t new_stmt;
        for (int i = 0; i < elems; i += step) {
            int cur_elems = std::min(step, elems - i);
            ir_assert(math::is_pow2(cur_elems));
            int off = i * stride * elem_size;
            auto store = store_t::make(obj->buf, obj->off + off,
                    split_expr(obj->value, i, i + cur_elems), obj->stride);
            new_stmt = new_stmt.append(store);
        }
        return std::move(new_stmt);
    }

private:
    static expr_t split_expr(const expr_t &e, int beg, int end) {
        auto *shuffle = e.as_ptr<shuffle_t>();
        if (shuffle) return shuffle_t::make(shuffle, beg, end);

        auto *binary = e.as_ptr<binary_op_t>();
        if (binary) {
            auto a = split_expr(binary->a, beg, end);
            auto b = split_expr(binary->b, beg, end);
            return binary_op_t::make(binary->op_kind, a, b);
        }
        ir_error_not_expected();
        return expr_t();
    }

    ngen::HW hw_;
};

// Splits wide GRF stores otherwise unsupported in HW.
stmt_t split_wide_stores(ngen::HW hw, const stmt_t &s) {
    auto ret = store_splitter_t(hw).mutate(s);
    trace_pass("split_wide_stores", ret);
    return ret;
}

class peephole_optimizer_t : public ir_mutator_t {
public:
    object_t _mutate(const binary_op_t *obj) override {
        auto old_obj = ir_mutator_t::_mutate(obj);
        auto new_obj
                = simplify_rewrite_with_ternary(old_obj, /*recursive=*/false);
        auto *ternary = new_obj.as_ptr<ternary_op_t>();
        if (!ternary) return std::move(new_obj);

        switch (ternary->op_kind) {
            case op_kind_t::_add3: {
                bool ok = true;
                // Allowed form: add3(dword/word, dword/word, dword/word).
                ok &= add3_type_ok(ternary->a);
                ok &= add3_type_ok(ternary->b);
                ok &= add3_type_ok(ternary->c);
                ok &= !is_const(ternary->a);
                ok &= !is_const(ternary->b);
                if (!ok) new_obj = old_obj;
                break;
            }
            case op_kind_t::_mad: {
                auto a_type = real_type(ternary->a);
                auto b_type = real_type(ternary->b);
                auto c_type = real_type(ternary->c);
                bool ok = true;
                // Allowed form: mad(dword, dword, word).
                ok &= utils::one_of(a_type, type_t::s32(), type_t::u32());
                ok &= utils::one_of(b_type, type_t::s32(), type_t::u32());
                ok &= utils::one_of(c_type, type_t::s16(), type_t::u16());
                if (!ok) new_obj = old_obj;
                break;
            }
            default: ir_error_not_expected();
        }
        return std::move(new_obj);
    }

private:
    static type_t real_type(const expr_t &e) {
        auto *imm = e.as_ptr<int_imm_t>();
        if (!imm) return e.type();
        if (int_imm_t::try_shrink_type<int16_t>(imm->value))
            return type_t::s16();
        if (int_imm_t::try_shrink_type<int32_t>(imm->value))
            return type_t::s32();
        return type_t::s64();
    }

    static bool add3_type_ok(const expr_t &e) {
        auto t = real_type(e);
        if (!t.is_scalar()) return false;
        switch (t.kind()) {
            case type_kind_t::s32:
            case type_kind_t::u32: return !is_const(e);
            case type_kind_t::s16:
            case type_kind_t::u16: return true;
            default: return false;
        }
    }
};

stmt_t optimize_peephole(const stmt_t &s) {
    auto ret = peephole_optimizer_t().mutate(s);
    trace_pass("optimize_peephole", ret);
    return ret;
}

class if_condition_fixer_t : public ir_mutator_t {
public:
    if_condition_fixer_t(int simd_size) : simd_size_(simd_size) {}

    object_t _mutate(const if_t *obj) override {
        auto _new_obj = ir_mutator_t::_mutate(obj);
        auto &new_obj = _new_obj.as<if_t>();
        auto cond = shuffle_t::make_broadcast(new_obj.cond, simd_size_);
        return if_t::make(cond, new_obj.body, new_obj.else_body);
    }

private:
    int simd_size_;
};

// Injects broadcasts for scalar if conditions. Example:
// Before:
//     if (cond) { ... }
// After (for SIMD8):
//     if (bcast8(cond)) { ... }
stmt_t fixup_if_conditions(const stmt_t &s, const conv_config_t &cfg) {
    auto ret = if_condition_fixer_t(cfg.simd_size).mutate(s);
    trace_pass("fixup_if_conditions", ret);
    return ret;
}

class loop_unroller_t : public ir_mutator_t {
public:
    loop_unroller_t(ir_context_t &ir_ctx) : ir_ctx_(ir_ctx) {}

    object_t _mutate(const for_t *obj) override {
        auto new_obj = ir_mutator_t::_mutate(obj);
        auto &_for = new_obj.as<for_t>();
        // No unrolling.
        if (_for.unroll == 1) return new_obj;

        ir_assert(is_const(obj->init))
                << "Can't unroll loop with non-const bound: " << obj->init;
        ir_assert(is_const(obj->bound))
                << "Can't unroll loop with non-const bound: " << obj->bound;

        auto init = to_cpp<int>(obj->init);
        auto bound = to_cpp<int>(obj->bound);

        ir_assert(_for.unroll == (bound - init))
                << "Only full loop unroll is supported.";

        stmt_t ret;
        for (int i = init; i < bound; i++) {
            auto iter_stmt = substitute(
                    obj->body, obj->var, to_expr(i, obj->var.type()));
            iter_stmt = rename_let_alloc(iter_stmt, i - init);
            ret = ret.append(iter_stmt);
        }
        return std::move(ret);
    }

private:
    stmt_t rename_let_alloc(const stmt_t &s, int idx) {
        auto lets = find_objects<let_t>(s);
        auto ret = s;
        for (auto &_let : lets) {
            auto &let = _let.as<let_t>();
            auto &var = let.var.as<var_t>();
            auto new_var = ir_ctx_.create_tmp_var(var.type, var.name);
            ret = substitute(ret, let.var, new_var);
        }
        auto allocs = find_objects<alloc_t>(s);
        for (auto &_alloc : allocs) {
            auto &alloc = _alloc.as<alloc_t>();
            auto &buf = alloc.buf.as<var_t>();
            auto new_buf = ir_ctx_.create_tmp_var(buf.type, buf.name);
            ret = substitute(ret, alloc.buf, new_buf);
        }
        return ret;
    }

    ir_context_t &ir_ctx_;
};

// Unrolls loops according to their unroll attribute.
// Before:
//     for (int i = 0; i < 2; i++) [unroll: 2] {
//         body(i);
//     }
// After:
//     body(0);
//     body(1);
stmt_t unroll_loops(const stmt_t &s, ir_context_t &ir_ctx) {
    auto ret = loop_unroller_t(ir_ctx).mutate(s);
    trace_pass("unroll_loops", ret);
    return ret;
}

class alloc_injector_t : public ir_mutator_t {
public:
    alloc_injector_t(const stmt_t &root, const std::vector<stmt_t> &allocs,
            bool put_innermost)
        : root_(root), put_innermost_(put_innermost) {
        for (auto &_a : allocs) {
            auto &a = _a.as<alloc_t>();
            if (a.kind != alloc_kind_t::global) ir_assert(a.size > 0) << _a;
            allocs_.insert({a.buf, _a});
        }
        mutate(root_);
        buf_total_refs_ = buf_cur_refs_;
        for (auto &kv : buf_cur_refs_)
            kv.second = 0;
        in_ctor_ = false;
    }

#define HANDLE_IR_OBJECT(type) \
    object_t _mutate(const type *obj) override { return mutate_stmt(obj); }

    HANDLE_STMT_IR_OBJECTS()

#undef HANDLE_IR_OBJECT
private:
    object_t _mutate(const var_t *obj) override {
        if (allocs_.find(obj) != allocs_.end()) buf_cur_refs_[obj]++;
        return obj;
    }

    template <typename T>
    object_t mutate_stmt(const T *obj) {
        if (in_ctor_) return ir_mutator_t::_mutate(obj);
        object_t new_obj = obj;
        object_set_t<expr_t> undef_bufs;
        if (put_innermost_) {
            for (auto &kv : buf_cur_refs_)
                if (kv.second == 0) undef_bufs.insert(kv.first);
            new_obj = ir_mutator_t::_mutate(obj);
        }
        for (auto &kv : allocs_) {
            auto &buf = kv.first;
            if (kv.second.is_empty()) continue; // Already injected.
            bool do_inject = false;
            if (put_innermost_) {
                int cur_refs = buf_cur_refs_[buf];
                int total_refs = buf_total_refs_[buf];
                bool was_undef = (undef_bufs.count(buf) != 0);
                do_inject = was_undef && (cur_refs == total_refs);
            } else {
                do_inject = root_.is_same(obj);
            }
            if (do_inject) {
                auto &a = kv.second.as<alloc_t>();
                new_obj = alloc_t::make(a.buf, a.size, a.kind, a.attr, new_obj);
                kv.second = stmt_t();
            }
        }
        return new_obj;
    }

    bool in_ctor_ = true;
    const stmt_t &root_;
    bool put_innermost_;
    object_map_t<expr_t, stmt_t> allocs_;
    object_map_t<expr_t, int> buf_total_refs_;
    object_map_t<expr_t, int> buf_cur_refs_;
};

stmt_t inject_alloc_stmts(const stmt_t &stmt, const std::vector<stmt_t> &allocs,
        bool put_innermost = false) {
    alloc_injector_t injector(stmt, allocs, put_innermost);
    return injector.mutate(stmt);
}

stmt_t inject_let_stmts(const stmt_t &stmt, const std::vector<stmt_t> &lets) {
    stmt_t ret = stmt;
    for (auto it = lets.rbegin(); it != lets.rend(); ++it) {
        auto &let = it->as<let_t>();
        ret = let_t::make(let.var, let.value, ret);
    }
    return ret;
}

stmt_t create_reorder_stmt(const layout_t &src, const layout_t &dst,
        const expr_t &src_buf, const expr_t &dst_buf) {
    ir_assert(src.ndims() == dst.ndims()) << "Layouts are incompatible.";
    ir_assert(src.elems() == dst.elems()) << "Layouts are incompatible.";
    auto func = reorder_t::make(src, dst);
    return func.call({dst_buf, src_buf});
}

stmt_t create_reduce_stmt(const layout_t &src, const layout_t &dst,
        const expr_t &src_buf, const expr_t &dst_buf, const tensor_t &_sub_tile,
        uint32_t reduction_mask) {
    auto sub_tile = _sub_tile;
    if (sub_tile.is_empty()) sub_tile = tensor_t(src.dims());
    ir_assert(src.ndims() == sub_tile.ndims());
    int ndims = src.ndims();

    // Align dst layout with src layout according to the mask.
    std::vector<int> dst2src(dst.ndims());
    int dst_dim_idx = 0;
    for (int i = 0; i < ndims; i++) {
        if ((reduction_mask & (1 << i)) != 0) {
            dst2src[dst_dim_idx] = i;
            dst_dim_idx++;
        }
    }
    ir_assert(dst_dim_idx == dst.ndims()) << "Incompatible reduction mask.";

    auto dst_blocks = dst.blocks();
    for (auto &b : dst_blocks)
        b.dim_idx = dst2src[b.dim_idx];

    // Create final layouts.
    auto dst_aligned = layout_t(dst.type(), ndims, dst.offset(), dst_blocks);

    std::vector<dim_t> dst_tile_dims = sub_tile.dims();
    std::vector<expr_t> dst_tile_start = sub_tile.start();
    for (int i = 0; i < ndims; i++) {
        if ((reduction_mask & (1 << i)) == 0) {
            dst_tile_dims[i] = 1;
            dst_tile_start[i] = expr_t(0);
            continue;
        }
    }
    dst_aligned = dst_aligned.map(tensor_t(dst_tile_dims, dst_tile_start));

    auto func = reduce_t::make(src, dst_aligned);
    return func.call({dst_buf, src_buf});
}

stmt_t create_zero_out_stmt(ngen::HW hw, const expr_t &buf, int size) {
    stmt_t ret;
    int step_bytes = 2 * ngen::GRF::bytes(hw);
    for (int i = 0; i < size; i += step_bytes) {
        int cur_step_bytes = std::min(step_bytes, size - i);
        ret = ret.append(store_t::make(buf, i,
                shuffle_t::make_broadcast(
                        expr_t(0.0f), cur_step_bytes / sizeof(float))));
    }
    return ret;
}

// Generates loads or stores to move data between memory (global or SLM) and
// GRF. Memory layout is a parameter. GRF layout is deduced automatically,
// according to the decomposition into messages.
class access_builder_t {
public:
    access_builder_t() = default;

    access_builder_t(ngen::HW hw, ir_context_t &ir_ctx,
            const constraint_set_t &cset, const view_t &mem_view,
            const expr_t &mem_buf, const expr_t &reg_buf, bool is_slm,
            bool is_prefetch, bool is_load, ngen_proxy::AtomicOp atomic_op)
        : hw_(hw)
        , ir_ctx_(&ir_ctx)
        , cset_(&cset)
        , mem_view_(mem_view)
        , mem_buf_(mem_buf)
        , reg_buf_(reg_buf)
        , is_slm_(is_slm)
        , is_prefetch_(is_prefetch)
        , is_load_(is_load)
        , atomic_op_(atomic_op) {
        build();
    }

    bool is_slm() const { return is_slm_; }

    bool is_prefetch() const { return is_prefetch_; }

    const layout_t &reg_layout() const { return reg_layout_; }

    int reg_buf_size() const { return reg_buf_size_; }

    const stmt_t &stmt() const { return stmt_; }

    std::string str() const {
        const auto grf_size = ngen::GRF::bytes(hw_);
        std::ostringstream oss;
        oss << "Memory view:          " << mem_view_ << std::endl;
        oss << "Register layout:      " << reg_layout_ << std::endl;
        oss << "Register buffer:      " << reg_buf_ << std::endl;
        oss << "Register buffer size: " << reg_buf_size_ << " ("
            << reg_buf_size_ / grf_size << " regs)" << std::endl;
        oss << "Statement:            " << std::endl << stmt_;
        return oss.str();
    }

private:
    void build() {
        auto send_list = get_send_list(mem_view_.type());

        auto mask_tensor = mem_view_.create_mask_tensor(*cset_);

        // Find the first send candidate matching the layout.
        func_t _send;
        tensor_t send_tensor;
        for (auto &_s_base : send_list) {
            auto &s_base = _s_base.as<send_t>();
            int type_size = mem_view_.type().size();
            int block_bytes_base = s_base.block_size();
            if (block_bytes_base % type_size != 0) continue;
            int elems_per_block_base = block_bytes_base / type_size;

            dim_t elems_per_block = elems_per_block_base;
            dim_t slots = s_base.slots;

            // Check if the view can be decomposed for this send.
            auto tensor
                    = mem_view_.split_into_dense_tile(elems_per_block, slots);
            if (tensor.is_empty()) continue;

            auto _s = s_base.adjust(
                    int(elems_per_block * type_size), int(slots));
            if (_s.is_empty()) continue;
            auto &s = _s.as<send_t>();

            // Check if this send supports the required mask.
            if (!has_compatible_mask(s, mem_view_, tensor, mask_tensor))
                continue;

            // TODO: Check alignment requirements.

            // Success, send is found, stop iterating.
            _send = _s;
            send_tensor = tensor;
            break;
        }
        // Support for prefetch messages is limited. If message is not found,
        // skip prefetch generation.
        if (_send.is_empty() && is_prefetch()) return;
        ir_assert(!_send.is_empty()) << "Can't decompose view into messages.";

        auto &send = _send.as<send_t>();
        reg_layout_ = create_register_layout_for_message(
                send, mem_view_, reg_buf_size_);

        mem_view_.for_each_tile(
                send_tensor, [&](const std::vector<dim_t> &start) {
                    auto tile = tensor_t(send_tensor.dims(), start);
                    auto sub_view = mem_view_.create_sub_view(tile);
                    auto sub_mask_tensor = mask_tensor.map(tile);
                    auto reg_sub_buf = (is_prefetch()
                                    ? expr_t()
                                    : reg_buf_[reg_layout_(start)
                                            * reg_layout_.type().size()]);
                    stmt_ = stmt_seq_t::make(stmt_,
                            create_send_stmt(*ir_ctx_, send, mem_buf_,
                                    reg_sub_buf, sub_view, sub_mask_tensor));
                });
    }

    // Returns a list of send functions that can be used for the access.
    std::vector<func_t> get_send_list(const type_t &data_type) const {
        using namespace ngen_proxy;
        bool is_atomic = (atomic_op_ != AtomicOp::undef);
        Access access_type = (is_load_ ? Access::Read : Access::Write);
        bool use_stateful_msgs = is_atomic;
        AddressModel address_model
                = (is_slm() ? AddressModel::ModelSLM
                            : use_stateful_msgs ? AddressModel::ModelBTS
                                                : AddressModel::ModelA64);
        auto send_list = send_t::get_all(hw_, data_type, access_type,
                address_model, atomic_op_, is_prefetch_);
        return send_list;
    }

    ngen::HW hw_;
    ir_context_t *ir_ctx_;
    const constraint_set_t *cset_;

    view_t mem_view_;
    expr_t mem_buf_;
    layout_t reg_layout_;
    expr_t reg_buf_;
    int reg_buf_size_;
    bool is_slm_;
    bool is_prefetch_;
    bool is_load_;
    stmt_t stmt_;
    ngen_proxy::AtomicOp atomic_op_;
};

class read_builder_t : public access_builder_t {
public:
    read_builder_t() = default;

    read_builder_t(ngen::HW hw, ir_context_t &ir_ctx,
            const constraint_set_t &cset, const view_t &view,
            const expr_t &mem_buf, const expr_t &reg_buf, bool is_slm,
            bool is_prefetch = false)
        : access_builder_t(hw, ir_ctx, cset, view, mem_buf, reg_buf, is_slm,
                is_prefetch, /*is_load=*/true, ngen_proxy::AtomicOp::undef) {}
};

class write_builder_t : public access_builder_t {
public:
    write_builder_t() = default;

    write_builder_t(ngen::HW hw, ir_context_t &ir_ctx,
            const constraint_set_t &cset, const view_t &view,
            const expr_t &mem_buf, const expr_t &reg_buf, bool is_slm,
            ngen_proxy::AtomicOp atomic_op = ngen_proxy::AtomicOp::undef)
        : access_builder_t(hw, ir_ctx, cset, view, mem_buf, reg_buf, is_slm,
                /*is_prefetch=*/false, /*is_load=*/false, atomic_op) {}
};

// Generates loads to the post-op buffer and applies a single post-op.
// There are two types of post-ops:
// - Eltwise: lhs = F(lhs)
// - Binary:  lhs = F(lhs, rhs)
// Binary requires rhs load which may be either:
// - Pre-loaded and used for all updates (preferred approach)
// - Loaded for every tile
// Right-hand side tensor supports implicit broadcasting: value is broadcasted
// across a size one dimension.
class post_op_builder_t {
public:
    post_op_builder_t(ngen::HW hw, ir_context_t &ir_ctx,
            const constraint_set_t &cset, const post_op_t &post_op,
            int &available_pre_load_size)
        : hw_(hw), ir_ctx_(ir_ctx), cset_(cset), post_op_(post_op) {
        if (!post_op_.needs_load()) return;

        // Estimate buffer size required to load full rhs, do not do pre-load
        // if it requires too much GRF memory.
        int estimated_rhs_bytes = 0;

        estimated_rhs_bytes
                += int(post_op.rhs_view().create_dense_vlayout().size());

        if (needs_rhs_convert()) {
            estimated_rhs_bytes += int(post_op.rhs_view()
                                               .create_dense_vlayout()
                                               .retype(type_t::f32())
                                               .size());
        }

        if (estimated_rhs_bytes <= available_pre_load_size) {
            available_pre_load_size -= estimated_rhs_bytes;
            do_preload_ = true;
        }
    }

    // Pre-loads rhs data for the whole update.
    stmt_t build_pre_load() {
        if (!do_preload_) return stmt_t();

        auto rhs_load_reg_buf = make_tmp_rhs_buffer();
        read_builder_t read(hw_, ir_ctx_, cset_, post_op_.rhs_view(),
                post_op_.rhs_buf(), rhs_load_reg_buf, /*is_slm=*/false);
        pre_load_rhs_reg_buf_ = rhs_load_reg_buf;
        pre_load_rhs_reg_layout_ = read.reg_layout();
        if (!needs_rhs_convert()) rhs_reg_buf_ = rhs_load_reg_buf;
        update_rhs_buf_size(rhs_load_reg_buf, read.reg_buf_size());
        return read.stmt();
    }

    // Converts the pre-loaded rhs data to f32.
    stmt_t build_pre_convert() {
        if (!do_preload_ || !needs_rhs_convert()) return stmt_t();

        auto rhs_f32_reg_buf = make_tmp_rhs_buffer();
        auto f32_layout
                = pre_load_rhs_reg_layout_.make_dense().retype(type_t::f32());
        update_rhs_buf_size(rhs_f32_reg_buf, int(f32_layout.size()));

        // Reorder to f32.
        auto ret = create_reorder_stmt(pre_load_rhs_reg_layout_, f32_layout,
                pre_load_rhs_reg_buf_, rhs_f32_reg_buf);

        // Now rhs is converted to f32.
        pre_load_rhs_reg_layout_ = f32_layout;
        rhs_reg_buf_ = rhs_f32_reg_buf;

        return ret;
    }

    // Loads rhs data for one tile.
    stmt_t build_tile_load(const tensor_t &tile) {
        if (!post_op_.needs_load()) return stmt_t();

        stmt_t stmt;
        auto rhs_tile = post_op_.apply_mask(tile);
        if (post_op_.needs_load() && !do_preload_) {
            // Load and convert now.
            auto po = post_op_.create_sub_post_op(rhs_tile);
            auto rhs_load_reg_buf = make_tmp_rhs_buffer();
            read_builder_t read(hw_, ir_ctx_, cset_, po.rhs_view(),
                    po.rhs_buf(), rhs_load_reg_buf,
                    /*is_slm=*/false);
            stmt = stmt.append(read.stmt());

            update_rhs_buf_size(rhs_load_reg_buf, read.reg_buf_size());

            if (needs_rhs_convert()) {
                auto rhs_f32_reg_buf = make_tmp_rhs_buffer();
                auto f32_layout
                        = read.reg_layout().make_dense().retype(type_t::f32());
                update_rhs_buf_size(rhs_f32_reg_buf, int(f32_layout.size()));
                // Reorder to f32.
                stmt = stmt.append(create_reorder_stmt(read.reg_layout(),
                        f32_layout, rhs_load_reg_buf, rhs_f32_reg_buf));

                // Now rhs is converted to f32.
                rhs_reg_layout_ = f32_layout;
                rhs_reg_buf_ = rhs_f32_reg_buf;
            } else {
                rhs_reg_layout_ = read.reg_layout();
                rhs_reg_buf_ = rhs_load_reg_buf;
            }
        } else {
            // Already pre-loaded and pre-converted.
            rhs_reg_layout_ = pre_load_rhs_reg_layout_.map(rhs_tile);
        }
        return stmt;
    }

    // Applies post-op for a single tile.
    stmt_t build_tile_stmt(const tensor_t &tile, const layout_t &lhs_reg_layout,
            const expr_t &lhs_buf) {
        auto po = post_op_.create_sub_post_op(tile);
        if (!po.has_rhs()) {
            // Apply eltwise post-op.
            int lhs_size = lhs_reg_layout.size();
            int lhs_elems = lhs_size / int(sizeof(float));
            return po.eltwise().call({expr_t(lhs_elems), lhs_buf});
        }

        auto lhs_layout = lhs_reg_layout;
        auto rhs_layout = (po.needs_load()
                        ? rhs_reg_layout_
                        : lhs_layout.map(
                                tensor_t(std::vector<dim_t>(tile.ndims(), 1))));

        int inner_dim_idx = lhs_layout.blocks().front().dim_idx;
        bool do_broadcast = po.is_broadcast_dim(inner_dim_idx);
        if (!do_broadcast) layout_t::align_layouts(lhs_layout, rhs_layout);

        auto lhs_blocks = lhs_layout.blocks();
        auto rhs_blocks = rhs_layout.blocks();

        auto &lhs0 = lhs_blocks[0];

        ir_assert(lhs0.dim_idx == inner_dim_idx);
        ir_assert(dim_t(lhs0.stride) == 1);

        if (!do_broadcast) {
            auto &rhs0 = rhs_blocks[0];
            ir_assert(lhs0.dim_idx == rhs0.dim_idx);
            ir_assert(lhs0.block == rhs0.block);
            MAYBE_UNUSED(rhs0);
        }

        std::vector<dim_t> inner_tile_dims(tile.ndims(), 1);
        inner_tile_dims[inner_dim_idx] = lhs0.block;

        auto &lhs_type = lhs_layout.type();
        auto &rhs_type = rhs_layout.type();
        ir_assert(lhs_type == type_t::f32());
        ir_assert(rhs_type == type_t::f32());

        // Handle one inner tile at a time. Inner tile covers a single block
        // with a single dimension.
        stmt_t stmt;
        lhs_layout.for_each_tile(tensor_t(inner_tile_dims),
                [&](const std::vector<dim_t> &lhs_start) {
                    auto rhs_start = po.apply_mask(lhs_start, 0);
                    int lhs_off0 = lhs_layout(lhs_start) * lhs_type.size();
                    int rhs_off0 = rhs_layout(rhs_start) * rhs_type.size();

                    int elems = lhs0.block;
                    int step = (elems < 16 ? 8 : 16);
                    for (int i = 0; i < elems; i += step) {
                        int cur_elems = std::min(step, elems - i);
                        ir_assert(math::is_pow2(cur_elems));
                        auto lhs_vec_type = lhs_type.with_elems(cur_elems);
                        auto rhs_vec_type = rhs_type.with_elems(
                                do_broadcast ? 1 : cur_elems);

                        int lhs_off = lhs_off0 + i * lhs_type.size();
                        int rhs_off = rhs_off0;
                        if (!do_broadcast) rhs_off += i * rhs_type.size();

                        auto lhs = load_t::make(lhs_vec_type, lhs_buf, lhs_off);
                        expr_t rhs;
                        if (po.needs_load()) {
                            int stride
                                    = (do_broadcast ? load_t::default_stride
                                                    : int(rhs_blocks[0].stride)
                                                            * rhs_type.size());
                            rhs = load_t::make(rhs_vec_type, rhs_reg_buf_,
                                    rhs_off, stride);
                        } else {
                            // rhs is scalar and passed in the kernel arguments.
                            rhs = po.rhs_buf();
                            ir_assert(rhs.type().is_scalar());
                        }

                        if (rhs.type().elems() != cur_elems) {
                            rhs = shuffle_t::make_broadcast(rhs, cur_elems);
                        }

                        if (po.rhs_scale() != 1) {
                            // Scale rhs first.
                            rhs = binary_op_t::make(op_kind_t::_mul, rhs,
                                    shuffle_t::make_broadcast(
                                            po.rhs_scale(), cur_elems));
                        }

                        auto new_lhs
                                = binary_op_t::make(po.op_kind(), lhs, rhs);
                        if (new_lhs.type().is_bool()) {
                            // Apply bool -> f32 cast when binary is a comparison op.
                            new_lhs = cast(new_lhs, type_t::f32(cur_elems));
                        }
                        auto store = store_t::make(lhs_buf, lhs_off, new_lhs);
                        stmt = stmt.append(store);
                    }
                });

        // Reset rhs layout.
        rhs_reg_layout_ = layout_t();
        return stmt;
    }

    std::vector<stmt_t> allocs() const {
        std::vector<stmt_t> allocs;
        for (auto &kv : rhs_bufs_)
            allocs.push_back(
                    alloc_t::make(kv.first, kv.second, alloc_kind_t::grf));
        return allocs;
    }

private:
    expr_t make_tmp_rhs_buffer() const {
        auto &rhs_name = post_op_.rhs_buf().as<var_t>().name;
        return ir_ctx_.create_tmp_var(type_t::byte_ptr(), "tmp_" + rhs_name);
    }

    void update_rhs_buf_size(const expr_t &buf, int size) {
        rhs_bufs_[buf] = std::max(rhs_bufs_[buf], size);
    }

    bool needs_rhs_convert() const {
        if (!post_op_.has_rhs()) return false;
        return post_op_.rhs_view().type() != type_t::f32();
    }

    ngen::HW hw_;
    ir_context_t &ir_ctx_;
    const constraint_set_t &cset_;
    post_op_t post_op_;

    bool do_preload_ = false;

    expr_t pre_load_rhs_reg_buf_;
    layout_t pre_load_rhs_reg_layout_;

    expr_t rhs_reg_buf_;
    layout_t rhs_reg_layout_;

    object_map_t<expr_t, int> rhs_bufs_;
};

// Zero pads a register buffer of f32 type.
class zero_pad_builder_t {
public:
    zero_pad_builder_t(const constraint_set_t &cset,
            const post_op_context_t &post_op_ctx, const view_t &mem_view,
            const layout_t &reg_layout, const expr_t &reg_buf)
        : cset_(cset)
        , post_op_ctx_(post_op_ctx)
        , mem_view_(mem_view)
        , reg_layout_(reg_layout)
        , reg_buf_(reg_buf) {
        ir_assert(mem_view_.nvdims() == reg_layout_.ndims())
                << "Incompatible view/layout.";
        build();
    }

    const stmt_t &stmt() const { return stmt_; }

private:
    void build() {
        int max_step = 16; // Handle 16 elements at most in one step.
        auto base_tile = reg_layout_.split_into_max_tile(
                max_step, /*is_dense_tile=*/true);
        reg_layout_.for_each_tile(
                base_tile, [&](const std::vector<dim_t> &start) {
                    tensor_t tile(base_tile.dims(), start);
                    auto sub_layout = reg_layout_.map(tile);
                    auto sub_view = mem_view_.create_sub_view(tile);
                    int elems = tile.elems();
                    int off = reg_layout_(start) * reg_layout_.type().size();
                    auto mask_tensor = create_mask(sub_view, sub_layout);
                    auto mask = mask_tensor.to_expr(elems);
                    auto store = store_t::make(reg_buf_, off,
                            shuffle_t::make_broadcast(expr_t(0.0f), elems),
                            store_t::default_stride, -mask);
                    stmt_ = stmt_.append(store);
                });
    }

    mask_tensor_t create_mask(
            const view_t &view, const layout_t &layout) const {
        mask_tensor_t mask_tensor(layout);
        std::vector<dim_t> args(layout.ndims());
        fill_mask_impl(mask_tensor, 0, args, view, layout);
        mask_tensor.simplify(cset_);
        return mask_tensor;
    }

    void fill_mask_impl(mask_tensor_t &mask_tensor, int idx,
            std::vector<dim_t> &args, const view_t &view,
            const layout_t &layout) const {
        if (idx == layout.ndims()) {
            expr_t mask = bool_imm_t::make(true);
            for (int i = 0; i < layout.ndims(); i++) {
                if (!post_op_ctx_.is_lhs_dim_zero_padded(i)) continue;
                mask &= (view.vstart(i) + args[i] < post_op_ctx_.lhs_dim(i));
            }
            auto off = layout.offset(args, /*ignore_offset=*/true);
            mask_tensor.set_mask(off, mask);
            return;
        }

        for (int i = 0; i < int(layout.dims()[idx]); i++) {
            args[idx] = i;
            fill_mask_impl(mask_tensor, idx + 1, args, view, layout);
        }
    }

    const constraint_set_t &cset_;
    const post_op_context_t &post_op_ctx_;

    view_t mem_view_;
    layout_t reg_layout_;
    expr_t reg_buf_;

    stmt_t stmt_;
};

// Performs the following steps after the computation:
// - Conversion
// - Applying post-ops
// - GRF reorder to match the memory layout
// - Store to the destination
class epilogue_builder_t {
public:
    epilogue_builder_t(const conv_config_t &cfg, ir_context_t &ir_ctx,
            const constraint_set_t &cset, const post_op_context_t &post_op_ctx,
            const tensor_t &tile, const view_t &mem_view,
            const layout_t &reg_layout, const expr_t &mem_buf,
            const expr_t &reg_buf)
        : cfg_(cfg)
        , ir_ctx_(ir_ctx)
        , cset_(cset)
        , post_op_ctx_(post_op_ctx)
        , mem_view_(mem_view)
        , reg_layout_(reg_layout)
        , mem_buf_(mem_buf)
        , reg_buf_(reg_buf) {

        int pre_load_size = pre_load_max_size_;
        for (auto &po : post_op_ctx_.post_ops()) {
            auto sub_po = po.create_sub_post_op(tile);
            post_op_builders_.emplace_back(
                    cfg.hw, ir_ctx, cset_, sub_po, pre_load_size);
        }
        build();
    }

    const stmt_t &stmt() const { return stmt_; }

private:
    // Represents one stage in the flow between multiplication and storing the
    // updated result to memory.
    //
    // Flow with post-ops:
    //   Multiplication ->
    //     M_x -> [R_f32] -> P0_f32 -> ... -> Pn_f32 -> [Z_f32] -> S_y ->
    //   GMEM
    // Flow without post-ops:
    //   Multiplication ->
    //     M_x -> S_y ->
    //   GMEM
    // Where:
    // - x      is data type after multiplication
    // - y      is destination data type
    // - M_x    is a stage after multiplication
    // - R_f32  is a stage after reordering from M_x to f32 (optional)
    // - Pi_f32 is a stage after applying Pi post-op
    // - Z_f32  is a stage after restoring zero padding (optional)
    // - S_y    is a stage before storing data to destination
    struct stage_t {
        stage_t(const layout_t &layout, const expr_t &buf,
                const stmt_t &stmt = stmt_t())
            : layout(layout), buf(buf), stmt(stmt) {}

        void set_next(ngen::HW hw, ir_context_t &ir_ctx, stage_t *next,
                bool force_reorder) {
            if (!next) return;
            bool do_reorder
                    = !layout.is_equal(next->layout, /*compare_offset=*/false);
            if (force_reorder) do_reorder = true;
            if (do_reorder) {
                ir_assert(stmt.is_empty());
                // Generate reorder between stages.
                stmt = create_reorder_stmt(
                        layout, next->layout, buf, next->buf);
            } else {
                // Reuse the same GRF buffer for the next stage.
                int this_off = to_cpp<int>(layout.offset_in_bytes());
                int next_off = to_cpp<int>(next->layout.offset_in_bytes());
                ir_assert(next_off == 0);
                MAYBE_UNUSED(next_off);
                next->set_buf(buf[this_off]);
            }
        }

        void set_buf(const expr_t &buf) {
            // Replace old buffer if there is an assigned statement.
            if (!stmt.is_empty()) { stmt = substitute(stmt, this->buf, buf); }
            this->buf = buf;
        }

        const expr_t &buf_base() const {
            if (buf.is<var_t>()) return buf;
            return buf.as<ptr_t>().base;
        }

        int buf_size() const {
            ir_assert(buf.is_same(buf_base()))
                    << "Size must be queried from another stage.";
            return int(layout.size());
        }

        void prepend_stmt(const stmt_t &stmt) {
            this->stmt = stmt.append(this->stmt);
        }

        layout_t layout;
        expr_t buf;
        stmt_t stmt;
    };

    void build() {
        for (auto &po_builder : post_op_builders_) {
            stmt_ = stmt_.append(po_builder.build_pre_load());
        }

        for (auto &po_builder : post_op_builders_) {
            stmt_ = stmt_.append(po_builder.build_pre_convert());
        }

        auto tmp_type = (post_op_builders_.empty() ? mem_view_.type()
                                                   : type_t::f32());
        int tmp_buf_elems = tmp_buf_size_ / tmp_type.size();
        auto base_tile = mem_view_.split_into_max_tile(
                tmp_buf_elems, /*is_dense=*/false);
        mem_view_.for_each_tile(
                base_tile, [&](const std::vector<dim_t> &start) {
                    build_tile(tensor_t(base_tile.dims(), start));
                });

        // Generate alloc statements for rhs post-op buffers.
        std::vector<stmt_t> allocs;
        for (auto &po_builder : post_op_builders_) {
            auto po_allocs = po_builder.allocs();
            allocs.insert(allocs.end(), po_allocs.begin(), po_allocs.end());
        }
        stmt_ = jit::inject_alloc_stmts(stmt_, allocs, /*put_innermost=*/true);
    }

    // Builds statements for a tile iterating through all stages (see stage_t
    // description).
    void build_tile(const tensor_t &tile) {
        auto mem_sub_view = mem_view_.create_sub_view(tile);
        auto reg_sub_layout = reg_layout_.map(tile);

        auto tmp_reg_buf = ir_ctx_.create_tmp_var(type_t::byte_ptr(), "c_tmp");
        bool restore_zero_padding = post_op_ctx_.need_to_restore_zero_padding();

        // S_y -> GMEM.
        ngen_proxy::AtomicOp atomic_op
                = (cfg_.do_atomic_update ? ngen_proxy::AtomicOp::fadd
                                         : ngen_proxy::AtomicOp::undef);
        write_builder_t r2g(cfg_.hw, ir_ctx_, cset_, mem_sub_view, mem_buf_,
                tmp_reg_buf,
                /*is_slm=*/false, /*atomic_op=*/atomic_op);

        // Initialize stages.
        std::vector<stage_t> stages;
        stages.emplace_back(reg_sub_layout, reg_buf_); // M_x
        if (!post_op_builders_.empty()) {
            auto po_layout = r2g.reg_layout().retype(type_t::f32());
            for (int i = 0; i < int(post_op_builders_.size()); i++) {
                auto buf = ir_ctx_.create_tmp_var(type_t::byte_ptr(), "c_tmp");
                stages.emplace_back(po_layout, buf); // Pi_f32
            }
            if (restore_zero_padding) {
                auto &last = stages.back();
                stages.emplace_back(last.layout, last.buf); // Z_f32.
            }
        }
        stages.emplace_back(r2g.reg_layout(), tmp_reg_buf, r2g.stmt()); // S_y

        int nstages = int(stages.size());
        int npost_ops = int(post_op_builders_.size());

        bool is_dpasw = (cfg_.fma_kind == fma_kind_t::dpasw);

        // Generate reorders between stages and create buffers.
        for (int i = 0; i < nstages; i++) {
            auto *next_stage = (i + 1 < nstages ? &stages[i + 1] : nullptr);
            // Always perform reorder when dpasw is used. This is to ensure
            // that C is properly restored and permuted after dpasw.
            stages[i].set_next(cfg_.hw, ir_ctx_, next_stage,
                    /*force_reorder=*/i == 0 && is_dpasw);
        }

        std::vector<stmt_t> tile_load_stmts;
        for (int i = 0; i < npost_ops; i++) {
            auto &po_builder = post_op_builders_[i];
            // Generate load for post-op.
            tile_load_stmts.push_back(po_builder.build_tile_load(tile));

            // Generate post-op statement.
            auto &s = stages[i + 1];
            s.prepend_stmt(po_builder.build_tile_stmt(tile, s.layout, s.buf));
        }

        // Restore zero padding if needed.
        if (restore_zero_padding) {
            auto &s = stages[nstages - 2];
            zero_pad_builder_t builder(
                    cset_, post_op_ctx_, mem_sub_view, s.layout, s.buf);
            s.prepend_stmt(builder.stmt());
        }

        stmt_t tile_stmt;

        // Add stage statements. Emit stages in blocks to reduce GRF
        // consumption.
        int stage_blk = 8;
        for (int i = 0; i < nstages; i += stage_blk) {
            int stage_beg = i;
            int stage_end = std::min(nstages, i + stage_blk);
            int po_beg = std::max(0, i - 1);
            int po_end = std::min(npost_ops, i + stage_blk - 1);
            stmt_t blk_stmt;
            for (int j = po_beg; j < po_end; j++) {
                blk_stmt = blk_stmt.append(tile_load_stmts[j]);
            }
            for (int j = stage_beg; j < stage_end; j++) {
                blk_stmt = blk_stmt.append(stages[j].stmt);
            }
            tile_stmt = tile_stmt.append(blk_stmt);
        }

        // Generate alloc statements for stage buffers.
        object_set_t<expr_t> seen;
        for (int i = 0; i < nstages; i++) {
            auto &s = stages[i];
            auto &buf = s.buf_base();
            auto ret = seen.insert(buf);
            if (i == 0 || !ret.second) continue;
            tile_stmt = alloc_t::make(
                    buf, s.buf_size(), alloc_kind_t::grf, {}, tile_stmt);
        }

        stmt_ = stmt_.append(tile_stmt);
    }

    const conv_config_t &cfg_;
    ir_context_t &ir_ctx_;
    const constraint_set_t &cset_;
    const post_op_context_t &post_op_ctx_;

    view_t mem_view_;
    layout_t reg_layout_;

    expr_t mem_buf_;
    expr_t reg_buf_;

    // Tile size in bytes. The tile data type is:
    // - the destination data type without post-ops
    // - f32 with post-ops
    static const int tmp_buf_size_ = 128;
    static const int pre_load_max_size_ = 256;

    std::vector<post_op_builder_t> post_op_builders_;

    stmt_t stmt_;
};

class multiply_builder_t {
public:
    multiply_builder_t() = default;

    multiply_builder_t(const conv_config_t &cfg,
            const bmnk_mapper_t &bmnk_mapper, const view_t &a_view,
            const view_t &b_view, const expr_t &a_buf, const expr_t &b_buf,
            const expr_t &c_buf)
        : hw_(cfg.hw)
        , simd_size_(cfg.simd_size)
        , bmnk_mapper_(bmnk_mapper)
        , a_view_(a_view)
        , b_view_(b_view)
        , a_buf_(a_buf)
        , b_buf_(b_buf)
        , c_buf_(c_buf) {
        switch (cfg.fma_kind) {
            case fma_kind_t::dpasw:
            case fma_kind_t::dpas:
                if (try_build_dpas()) return;
                break;
            case fma_kind_t::mad:
                if (try_build_mad()) return;
                break;
            default: ir_error_not_expected() << "Unknown FMA kind.";
        }

        ir_error_not_expected()
                << "Can't decompose into multiplication instructions.";
    }

    const stmt_t &stmt() const { return stmt_; }

    const layout_t &c_layout() const { return c_layout_; }

    ngen_proxy::Bundle a_grf_bundle() {
        if (!do_transpose_) return ngen_proxy::Bundle();
        return ngen_proxy::Bundle(1, ngen_proxy::Bundle::any);
    }

    ngen_proxy::Bundle b_grf_bundle() {
        if (do_transpose_) return ngen_proxy::Bundle();
        return ngen_proxy::Bundle(1, ngen_proxy::Bundle::any);
    }

    ngen_proxy::Bundle c_grf_bundle() {
        return ngen_proxy::Bundle(0, ngen_proxy::Bundle::any);
    }

    std::string str() const {
        std::ostringstream oss;
        oss << "A view:    " << a_view_ << std::endl;
        oss << "B view:    " << b_view_ << std::endl;
        oss << "C layout:  " << c_layout_ << std::endl;
        oss << "Statement: " << std::endl << stmt_;
        return oss.str();
    }

private:
    struct loop_info_t {
        loop_info_t() = default;

        loop_info_t(const expr_t &var, bmnk_kind_t bmnk_kind, int dim)
            : var(var), bmnk_kind(bmnk_kind), dim(dim) {}

        expr_t var;
        bmnk_kind_t bmnk_kind;

        int dim;
        int a_idx = -1;
        int b_idx = -1;
        int c_idx = -1;
        int block = 1;
    };

    bool try_build_dpas() {
        ir_assert(a_view_.can_convert_to_vlayout())
                << "Views are not supported with dpas/dpasw.";
        ir_assert(b_view_.can_convert_to_vlayout())
                << "Views are not supported with dpas/dpasw.";

        auto a_layout = a_view_.create_vlayout();
        auto b_layout = b_view_.create_vlayout();

        bmnk_block_mapper_t from_bmnk_mapper(bmnk_mapper_);
        from_bmnk_mapper.push_blocks(abc_kind_t::a, a_layout.blocks());
        from_bmnk_mapper.push_blocks(abc_kind_t::b, b_layout.blocks());

        // Convert to MNK layouts.
        a_layout = bmnk_mapper_.map_to_bmnk(
                abc_kind_t::a, {bmnk_kind_t::m, bmnk_kind_t::k}, a_layout);
        b_layout = bmnk_mapper_.map_to_bmnk(
                abc_kind_t::b, {bmnk_kind_t::k, bmnk_kind_t::n}, b_layout);

        multiply_desc_t desc(a_layout, b_layout, /*force_c_upconvert=*/true);
        if (!dpas_t::matches_types(
                    hw_, desc.a_type(), desc.b_type(), desc.c_type()))
            return false;

        int sdepth = 8;
        int rcount = std::min(utils::rnd_up_pow2(desc.n()), 8);
        auto _dpas = dpas_t::make(/*is_dpasw=*/false, simd_size_, sdepth,
                rcount, desc.c_type(), desc.a_type(), desc.b_type());
        if (_dpas.as<dpas_t>().matches(desc)) {
            build_dpas(from_bmnk_mapper, _dpas.as<dpas_t>(), desc);
            return true;
        }

        // Try to transpose and flip: C += A * B -> C^T = B^T * A^T.
        rcount = std::min(utils::rnd_up_pow2(desc.m()), 8);
        desc = multiply_desc_t(
                b_layout.transpose(), a_layout.transpose(), true);
        _dpas = dpas_t::make(/*is_dpasw=*/false, /*exec_size=*/simd_size_,
                sdepth, rcount, desc.c_type(), desc.a_type(), desc.b_type());

        if (_dpas.as<dpas_t>().matches(desc)) {
            do_transpose_ = true;
            build_dpas(from_bmnk_mapper, _dpas.as<dpas_t>(), desc);
            return true;
        }
        return false;
    }

    void build_dpas(const bmnk_block_mapper_t &from_bmnk_mapper,
            const dpas_t &dpas, const multiply_desc_t &desc) {
        int m_blk = dpas.simd_size;
        int n_blk = dpas.rcount;
        int k_blk = dpas.sdepth * 4 / dpas.src1_type.size();

        c_layout_ = compute_dpas_c_layout(m_blk, n_blk, dpas.c_layout(), desc);

        expr_t a_buf = a_buf_;
        expr_t b_buf = b_buf_;
        if (do_transpose_) std::swap(a_buf, b_buf);

        for (int i_k = 0; i_k < desc.k(); i_k += k_blk) {
            for (int i_m = 0; i_m < desc.m(); i_m += m_blk) {
                for (int i_n = 0; i_n < desc.n(); i_n += n_blk) {
                    std::vector<int> a_args = {i_m, 0};
                    std::vector<int> b_args = {0, i_n};
                    std::vector<int> c_args = {i_m, i_n};
                    auto a = a_buf[desc.a_layout()(a_args)
                            * desc.a_type().size()];
                    auto b = b_buf[desc.b_layout()(b_args)
                            * desc.b_type().size()];
                    auto c = c_buf_[c_layout_(c_args) * desc.c_type().size()];
                    stmt_ = stmt_.append(dpas(c, c, a, b));
                }
            }
        }

        // Transpose C layout back if needed.
        if (do_transpose_) c_layout_ = c_layout_.transpose();

        // Convert C layout back to problem notation.
        c_layout_ = from_bmnk_mapper.map_from_bmnk(
                abc_kind_t::c, {bmnk_kind_t::m, bmnk_kind_t::n}, c_layout_);
    }

    static layout_t compute_dpas_c_layout(int m_blk, int n_blk,
            const layout_t &blk_layout, const multiply_desc_t &desc) {
        auto c_layout = blk_layout;
        c_layout = c_layout.add_outer_block(1, desc.n() / n_blk);
        c_layout = c_layout.add_outer_block(0, desc.m() / m_blk);
        return c_layout;
    }

    bool try_build_mad() {
        auto loops = create_loop_nest();

        if (try_build_mad_kmn_block_by_n(loops)) return true;
        if (try_build_mad_kmn_block_by_b(loops)) return true;

        return false;
    }

    std::vector<loop_info_t> create_loop_nest() const {
        object_map_t<expr_t, loop_info_t> loops;
        for (auto *view : {&a_view_, &b_view_}) {
            abc_kind_t abc_kind
                    = (view == &a_view_ ? abc_kind_t::a : abc_kind_t::b);
            for (int i = 0; i < view->nvdims(); i++) {
                auto &var = bmnk_mapper_.var(abc_kind, i);
                int dim = int(view->vdims()[i]);
                if (dim == 1) continue;

                if (loops.count(var) > 0) continue;
                loops[var] = loop_info_t(var, bmnk_mapper_.bmnk_kind(var), dim);
            }
        }

        std::vector<loop_info_t> ret;
        for (auto &kv : loops) {
            auto &loop = kv.second;
            loop.a_idx = bmnk_mapper_.dim_idx(abc_kind_t::a, loop.var);
            loop.b_idx = bmnk_mapper_.dim_idx(abc_kind_t::b, loop.var);
            loop.c_idx = bmnk_mapper_.dim_idx(abc_kind_t::c, loop.var);
            ret.push_back(kv.second);
        }
        return ret;
    }

    // Order of loops: BKMN, block by N.
    bool try_build_mad_kmn_block_by_n(std::vector<loop_info_t> &_loops) {
        return try_build_mad_impl(_loops,
                {bmnk_kind_t::b, bmnk_kind_t::k, bmnk_kind_t::m,
                        bmnk_kind_t::n},
                bmnk_kind_t::n);
    }

    // Order of loops: BKMN, block by B.
    bool try_build_mad_kmn_block_by_b(std::vector<loop_info_t> &_loops) {
        return try_build_mad_impl(_loops,
                {bmnk_kind_t::b, bmnk_kind_t::k, bmnk_kind_t::m,
                        bmnk_kind_t::n},
                bmnk_kind_t::b);
    }

    bool try_build_mad_impl(std::vector<loop_info_t> &_loops,
            const std::vector<bmnk_kind_t> &loop_order,
            bmnk_kind_t block_bmnk_kind) {
        auto loops = _loops;
        int nloops = int(loops.size());
        std::sort(loops.begin(), loops.end(),
                [&](const loop_info_t &a, const loop_info_t &b) {
                    int a_key = ir_utils::find_index(loop_order, a.bmnk_kind);
                    int b_key = ir_utils::find_index(loop_order, b.bmnk_kind);
                    ir_assert(a_key != -1);
                    ir_assert(b_key != -1);
                    return a_key < b_key;
                });

        int block_idx = -1;
        for (int i = 0; i < nloops; i++) {
            auto &l = loops[i];
            if (l.bmnk_kind == block_bmnk_kind) {
                ir_assert(block_idx == -1) << "Can't block 2+ dimensions.";
                block_idx = i;
            }
        }

        // Couldn't find N dimension, try different blocking scheme.
        if (block_idx == -1) return false;

        auto &block_loop = loops[block_idx];

        int block = simd_size_;
        while (block >= 1) {
            if (block_loop.dim % block == 0) break;
            block /= 2;
        }

        ir_assert(block >= 1) << "Invalid block size.";
        block_loop.block = block;

        int a_stride = 0;
        int b_stride = 0;

        // Ensure that A tile is dense.
        if (block_loop.a_idx != -1) {
            std::vector<dim_t> tile_dims(a_view_.nvdims(), 1);
            tile_dims[block_loop.a_idx] = block;
            auto layout = a_view_.create_pseudo_vlayout();
            auto tile = layout.map(tensor_t(tile_dims));
            if (!is_1d_strided(tile)) return false;
            a_stride = tile.blocks()[0].stride;
        }

        // Ensure that B tile is dense.
        if (block_loop.b_idx != -1) {
            std::vector<dim_t> tile_dims(b_view_.nvdims(), 1);
            tile_dims[block_loop.b_idx] = block;
            auto layout = b_view_.create_pseudo_vlayout();
            auto tile = layout.map(tensor_t(tile_dims));
            if (!is_1d_strided(tile)) return false;
            b_stride = tile.blocks()[0].stride;
        }

        build_mad(loops, block_loop, a_stride, b_stride);
        return true;
    }

    static bool is_1d_strided(const layout_t &layout) {
        auto &blocks = layout.blocks();
        if (blocks.size() > 1) return false;
        return true;
    }

    void build_mad(const std::vector<loop_info_t> &loops,
            const loop_info_t &block_loop, int a_stride, int b_stride) {
        ir_assert(utils::one_of(
                block_loop.bmnk_kind, bmnk_kind_t::b, bmnk_kind_t::n))
                << "Unsupported blocking (expected blocking by B or N).";

        auto &a_type = a_view_.type();
        auto &b_type = b_view_.type();
        auto c_type = multiply_desc_t::get_c_type(a_type, b_type,
                /*force_c_upconvert=*/false);

        int block = block_loop.block;
        auto _mad = mad_t::make(
                c_type, block, a_type, a_stride, b_type, b_stride);
        auto &mad = _mad.as<mad_t>();

        c_layout_ = compute_mad_c_layout(c_type, loops, block_loop);

        int nloops = int(loops.size());
        std::vector<int> bounds(loops.size());
        for (int i = 0; i < nloops; i++) {
            bounds[i] = loops[i].dim / loops[i].block;
        }
        std::vector<int> a_idx(a_view_.nvdims());
        std::vector<int> b_idx(b_view_.nvdims());
        std::vector<int> c_idx(c_layout_.ndims());
        ir_utils::for_each(bounds, [&](const std::vector<int> &idx) {
            for (int i = 0; i < nloops; i++) {
                int full_idx = idx[i] * loops[i].block;
                auto &loop = loops[i];
                if (loop.a_idx != -1) a_idx[loop.a_idx] = full_idx;
                if (loop.b_idx != -1) b_idx[loop.b_idx] = full_idx;
                if (loop.c_idx != -1) c_idx[loop.c_idx] = full_idx;
            }
            int a_off = a_view_(a_idx) * a_type.size();
            int b_off = b_view_(b_idx) * b_type.size();
            int c_off = c_layout_(c_idx) * c_type.size();
            stmt_ = stmt_.append(mad(c_buf_[c_off], c_buf_[c_off],
                    a_buf_[a_off], b_buf_[b_off]));
        });
    }

    layout_t compute_mad_c_layout(const type_t &c_type,
            const std::vector<loop_info_t> &loops,
            const loop_info_t &block_loop) const {
        layout_t c_layout(c_type, bmnk_mapper_.ndims(abc_kind_t::c), 0, {});

        int c_dim_idx = bmnk_mapper_.dim_idx(abc_kind_t::c, block_loop.var);
        c_layout = c_layout.add_outer_block(c_dim_idx, block_loop.block);

        for (size_t i = 0; i < loops.size(); i++) {
            if (loops[i].bmnk_kind == bmnk_kind_t::k) continue;
            int dim_idx = bmnk_mapper_.dim_idx(abc_kind_t::c, loops[i].var);
            int bound = loops[i].dim / loops[i].block;
            c_layout = c_layout.add_outer_block(dim_idx, bound);
        }
        return c_layout;
    }

    ngen::HW hw_;
    int simd_size_;
    bmnk_mapper_t bmnk_mapper_;

    bool do_transpose_ = false;

    view_t a_view_;
    view_t b_view_;
    layout_t c_layout_;

    expr_t a_buf_;
    expr_t b_buf_;
    expr_t c_buf_;

    stmt_t stmt_;
};

layout_t get_fma_friendly_layout(abc_kind_t abc_kind, int simd_size,
        const layout_t &bmnk_layout, const type_t &a_type,
        const type_t &b_type) {
    bool is_a = (abc_kind == abc_kind_t::a);
    int mn_idx = (is_a ? 0 : 1);
    int k_idx = (is_a ? 1 : 0);

    dim_t mn_blk = bmnk_layout.dim(mn_idx);
    dim_t k_blk = bmnk_layout.dim(k_idx);

    // Cannot calculate correct r_count when !is_a, but rcount is effectively
    // ignored in that case as rcount mainly effects b_layout.
    int rcount = is_a && mn_blk < 8 ? utils::rnd_up_pow2(mn_blk) : 8;
    auto _dpas = dpas_t::make(/*is_dpasw=*/false, simd_size, /*sdepth=*/8,
            rcount, type_t::undef(), b_type, a_type);
    auto &dpas = _dpas.as<dpas_t>();

    auto dpas_layout = (is_a ? dpas.b_layout() : dpas.a_layout());
    dpas_layout = dpas_layout.transpose();

    ir_assert(dpas_layout.dim(k_idx) == k_blk);
    MAYBE_UNUSED(k_blk);

    dim_t dpas_mn_blk = dpas_layout.dim(mn_idx);
    dpas_layout = dpas_layout.add_outer_block(mn_idx, mn_blk / dpas_mn_blk);

    return dpas_layout;
}

layout_t convert_to_fma_friendly_type(const conv_config_t &cfg,
        abc_kind_t abc_kind, const layout_t &layout, const type_t &a_type,
        const type_t &b_type, bool *changed = nullptr) {
    if (changed) *changed = false;
    if (cfg.fma_kind != fma_kind_t::mad) return layout;

    if (a_type.is_x8() && b_type.is_x8()) {
        if (changed) *changed = true;
        return layout.retype(type_t::s16()).make_strided(2);
    }
    // f16/bf16 mixed mode mad requires src2 to be f32
    if (a_type.is_bf16() || b_type.is_bf16()
            || (a_type.is_f32() && b_type.is_f16())
            || (a_type.is_f16() && b_type.is_f32())) {
        if (changed) *changed = true;
        return layout.retype(type_t::f32());
    }
    return layout;
}

layout_t convert_to_fma_friendly_layout(const conv_config_t &cfg,
        abc_kind_t abc_kind, const bmnk_mapper_t &bmnk_mapper,
        const layout_t &layout, const type_t &a_type, const type_t &b_type,
        bool *changed = nullptr) {
    if (changed) *changed = false;
    if (!cfg.allow_grf_reorder) return layout;

    // GRF reorder is only supported with dpas/dpasw.
    if (!utils::one_of(cfg.fma_kind, fma_kind_t::dpas, fma_kind_t::dpasw)) {
        // mad may require type conversion.
        return convert_to_fma_friendly_type(
                cfg, abc_kind, layout, a_type, b_type, changed);
    }

    std::vector<bmnk_kind_t> bmnk_kinds;
    if (abc_kind == abc_kind_t::a) {
        bmnk_kinds.push_back(bmnk_kind_t::m);
        bmnk_kinds.push_back(bmnk_kind_t::k);
    } else {
        bmnk_kinds.push_back(bmnk_kind_t::k);
        bmnk_kinds.push_back(bmnk_kind_t::n);
    }

    auto bmnk_layout = bmnk_mapper.map_to_bmnk(abc_kind, bmnk_kinds, layout);

    auto dpas_layout = get_fma_friendly_layout(
            abc_kind, cfg.simd_size, bmnk_layout, a_type, b_type);
    if (dpas_layout == bmnk_layout) return layout;

    if (changed) *changed = true;

    bmnk_block_mapper_t from_bmnk_mapper(bmnk_mapper);
    from_bmnk_mapper.push_blocks(abc_kind, layout.blocks());

    auto fma_layout
            = from_bmnk_mapper.map_from_bmnk(abc_kind, bmnk_kinds, dpas_layout);
    fma_layout = fma_layout.make_dense();
    return fma_layout;
}

class b_reduce_context_t {
public:
    b_reduce_context_t(const conv_config_t &cfg)
        : cfg_(cfg), reduce_condition_(true) {
        if (cfg.do_b_reduction) b_reduced_reg_buf_ = make_buffer("b_reduced");
    }

    // Setters for B reduced memory buffer/view.
    void set_b_reduced_mem_buf(const expr_t &buf) { b_reduced_mem_buf_ = buf; }
    void set_b_reduced_view(const view_t &v) { b_reduced_view_ = v; }

    // Sets the condition to update B reduced output. Reduction is done across
    // K for B (KxN tensor) so M dimension should be checked before the update.
    void set_reduce_condition(const expr_t &cond) { reduce_condition_ = cond; }

    // Global memory buffer.
    const expr_t &b_reduced_mem_buf() const { return b_reduced_mem_buf_; }

    // Register buffer.
    const expr_t &b_reduced_reg_buf() const { return b_reduced_reg_buf_; }
    int b_reduced_size() const { return b_reduced_size_; }

    // Memory view.
    const view_t &b_reduced_thr_view() const { return b_reduced_thr_view_; }

    // Register layout.
    const layout_t &b_reduced_reg_layout() const {
        return b_reduced_reg_layout_;
    }

    void init_reduced_thr_view(
            const tensor_t &b_thr_tile, const expr_t &cond = expr_t()) {
        ir_assert(b_reduced_thr_view_.is_empty()) << "Can't initialize twice.";

        auto b_reduced_thr_tile = b_to_b_reduced_tile(b_thr_tile);
        b_reduced_thr_view_
                = b_reduced_view_.create_sub_view(b_reduced_thr_tile);
        b_reduced_reg_layout_ = b_reduced_thr_view_.create_dense_vlayout();
        b_reduced_size_ = b_reduced_reg_layout_.size();
        b_reduced_size_ = utils::rnd_up(b_reduced_size_, cfg_.grf_size());

        if (!cond.is_empty()) reduce_condition_ &= cond;
    }

    stmt_t create_reduce_stmt(const layout_t &b_layout, const expr_t &b_buf,
            const tensor_t &sub_tile = tensor_t()) {
        auto reduction_stmt
                = jit::create_reduce_stmt(b_layout, b_reduced_reg_layout_,
                        b_buf, b_reduced_reg_buf_, sub_tile, (1 << 1));
        return reduction_stmt;
    }

    stmt_t create_store_stmt(
            ir_context_t &ir_ctx, const constraint_set_t &cset) const {
        write_builder_t r2g(cfg_.hw, ir_ctx, cset, b_reduced_thr_view_,
                b_reduced_mem_buf_, b_reduced_reg_buf_, /*is_slm=*/false,
                ngen_proxy::AtomicOp::fadd);
        // TODO: Check that layouts match.
        auto ret = r2g.stmt();
        if (!reduce_condition_.is_empty()) {
            ret = if_t::make(reduce_condition_, ret);
        }
        return ret;
    }

private:
    tensor_t b_to_b_reduced_tile(const tensor_t &b_tile) const {
        std::vector<dim_t> dims;
        std::vector<expr_t> start;
        for (int i = 0; i < b_tile.ndims(); i++) {
            if ((reduction_mask_ & (1 << i)) != 0) {
                dims.push_back(b_tile(i));
                start.push_back(b_tile.start(i));
            }
        }
        return tensor_t(dims, start);
    }

    const conv_config_t &cfg_;

    expr_t reduce_condition_;

    expr_t b_reduced_mem_buf_;
    expr_t b_reduced_reg_buf_;

    view_t b_reduced_view_;
    view_t b_reduced_thr_view_;

    layout_t b_reduced_reg_layout_;
    int b_reduced_size_ = 0;

    uint32_t reduction_mask_ = (1 << 1);
};

class load_multiply_builder_t {
public:
    load_multiply_builder_t(const conv_config_t &cfg, ir_context_t &ir_ctx,
            const constraint_set_t &cset, const gemm_schedule_t &gemm_schedule,
            b_reduce_context_t &b_reduce_ctx, const expr_t &ap_buf,
            const expr_t &a_slm_buf, const expr_t &bp_buf,
            const expr_t &b_slm_buf, const view_t &ap_x_view,
            const view_t &bp_x_view)
        : cfg_(cfg)
        , ir_ctx_(ir_ctx)
        , cset_(cset)
        , gemm_schedule_(gemm_schedule)
        , b_reduce_ctx_(b_reduce_ctx)
        , ap_buf_(ap_buf)
        , a_slm_buf_(a_slm_buf)
        , bp_buf_(bp_buf)
        , b_slm_buf_(b_slm_buf) {
        ir_assert(cfg_.a_sub_tiles == 1 || cfg_.b_sub_tiles == 1)
                << "At most one tensor can be tiled.";

        ab_tmp_buf_ = make_buffer("ab_tmp");
        a_buf_ = make_buffer("a");
        b_buf_ = make_buffer("b");
        c_buf_ = make_buffer("c");

        // Views to multiply by a thread.
        a_thr_view_ = ap_x_view.create_sub_view(gemm_schedule_.a_thr_tile());
        b_thr_view_ = bp_x_view.create_sub_view(gemm_schedule_.b_thr_tile());

        // Initialize view for reduced B.
        if (cfg_.do_b_reduction && !cfg_.use_b_slm) {
            b_reduce_ctx_.init_reduced_thr_view(
                    gemm_schedule_.b_thr_tile(/*is_relative=*/false));
        }

        // TODO: Specify loops over sub-tiles in the schedule, use unrolling.
        // Sub-tile indices.
        a_idx_ = ir_ctx_.create_tmp_var(type_t::s32(), "a_idx");
        b_idx_ = ir_ctx_.create_tmp_var(type_t::s32(), "b_idx");

        // Sub-tile views.
        a_i_view_ = create_sub_tile_view(abc_kind_t::a, a_thr_view_,
                cfg_.a_sub_tiles, a_idx_, bmnk_kind_t::m, &a_i_outer_blocks_,
                a_i_tile_);
        b_j_view_ = create_sub_tile_view(abc_kind_t::b, b_thr_view_,
                cfg_.b_sub_tiles, b_idx_, bmnk_kind_t::n, &b_j_outer_blocks_,
                b_j_tile_);

        build();
    }

    const std::vector<stmt_t> &allocs() const { return allocs_; }

    const stmt_t &load_mul_stmt() const { return load_mul_stmt_; }

    const expr_t &c_buf() const { return c_buf_; }

    const layout_t &c_reg_layout() const { return c_reg_layout_; }

    const alloc_attr_t &c_attr() const { return c_attr_; }

private:
    struct sub_tile_info_t {
        bool is_loaded = false;
        view_t reg_view;
        int reg_buf_size;
    };

    view_t create_sub_tile_view(abc_kind_t abc_kind, const view_t &thr_view,
            int sub_tiles, const expr_t &idx, bmnk_kind_t bmnk_kind,
            std::vector<block_t> *outer_blocks, tensor_t &sub_tile) const {
        auto &bmnk_mapper = gemm_schedule_.bmnk_mapper();
        auto layout = thr_view.create_pseudo_vlayout();
        dim_t mn_dim = 1;
        for (auto &b : layout.blocks()) {
            auto b_bmnk_kind = bmnk_mapper.bmnk_kind(abc_kind, b.dim_idx);
            if (b_bmnk_kind == bmnk_kind) mn_dim *= b.block;
        }

        std::vector<dim_t> sub_tile_dims(thr_view.nvdims(), 1);
        dim_t mn_sub_tile_dim = ir_utils::safe_divide(mn_dim, dim_t(sub_tiles));
        for (auto &b : layout.blocks()) {
            auto b_bmnk_kind = bmnk_mapper.bmnk_kind(abc_kind, b.dim_idx);
            if (b_bmnk_kind == bmnk_kind) {
                if (mn_sub_tile_dim == 1) continue;
                dim_t next_block;
                if (mn_sub_tile_dim % b.block == 0) {
                    next_block = b.block;
                } else {
                    next_block
                            = ir_utils::safe_divide(b.block, mn_sub_tile_dim);
                }
                sub_tile_dims[b.dim_idx] *= next_block;
                mn_sub_tile_dim /= next_block;
            } else {
                sub_tile_dims[b.dim_idx] *= b.block;
            }
        }
        grid_info_t grid({sub_tiles}, {idx});
        sub_tile = layout.split(tensor_t(sub_tile_dims), grid, outer_blocks);
        return thr_view.create_sub_view(sub_tile);
    }

    const type_t &a_type() const { return a_i_view_.type(); }
    const type_t &b_type() const { return b_j_view_.type(); }

    void build() {
        a_sub_tiles_.resize(cfg_.a_sub_tiles);
        b_sub_tiles_.resize(cfg_.b_sub_tiles);
        for (int i = 0; i < cfg_.a_sub_tiles; i++) {
            for (int j = 0; j < cfg_.b_sub_tiles; j++) {
                build_sub_tile(i, j);
            }
        }

        if (tmp_buf_size_ > 0) {
            register_buffer(ab_tmp_buf_, tmp_buf_size_, alloc_kind_t::grf);
        }

        // C layout in problem notation.
        auto c_layout = c_sub_tile_layout_;

        // Add outer blocks coming from A/B sub-tiles.
        auto &bmnk_mapper = gemm_schedule_.bmnk_mapper();
        for (auto &b : a_i_outer_blocks_) {
            auto &var = bmnk_mapper.var(abc_kind_t::a, b.dim_idx);
            int c_dim_idx = bmnk_mapper.dim_idx(abc_kind_t::c, var);
            c_layout = c_layout.add_outer_block(c_dim_idx, b.block);
        }
        for (auto &b : b_j_outer_blocks_) {
            auto &var = bmnk_mapper.var(abc_kind_t::b, b.dim_idx);
            int c_dim_idx = bmnk_mapper.dim_idx(abc_kind_t::c, var);
            c_layout = c_layout.add_outer_block(c_dim_idx, b.block);
        }

        c_reg_layout_ = c_layout;
    }

    void build_sub_tile(int i, int j) {
        bool is_first = (i == 0 && j == 0);

        stmt_t ab_s2r_load;
        stmt_t ab_g2r_load;
        load_sub_tile(abc_kind_t::a, i, ab_s2r_load, ab_g2r_load);
        load_sub_tile(abc_kind_t::b, j, ab_s2r_load, ab_g2r_load);

        load_mul_stmt_ = load_mul_stmt_.append(
                stmt_group_t::make(stmt_label_t::g2r_load(i + j), ab_g2r_load));
        load_mul_stmt_ = load_mul_stmt_.append(
                stmt_group_t::make(stmt_label_t::s2r_load(i + j), ab_s2r_load));

        auto &a_i_view = a_sub_tiles_[i].reg_view;
        auto &b_j_view = b_sub_tiles_[j].reg_view;

        // Multiply C_i_j += A_i x B_j in GEMM notation.
        multiply_builder_t mul_builder(cfg_, gemm_schedule_.bmnk_mapper(),
                a_i_view, b_j_view, a_buf_, b_buf_, c_buf_[c_buf_off_]);
        c_sub_tile_layout_ = mul_builder.c_layout();
        c_buf_off_ += c_sub_tile_layout_.size();
        ir_trace() << "Multiply (" << i << ", " << j << "):\n"
                   << mul_builder.str() << std::endl;

        load_mul_stmt_ = load_mul_stmt_.append(stmt_group_t::make(
                stmt_label_t::mul(i + j), mul_builder.stmt()));

        if (!is_first) {
            ir_assert(mul_builder.c_layout() == c_sub_tile_layout_)
                    << "Sub-tile layouts must be equal.";
            return;
        }

        c_attr_ = grf_alloc_attr_t::make(mul_builder.c_grf_bundle());

        auto a_attr = grf_alloc_attr_t::make(mul_builder.a_grf_bundle());
        register_buffer(a_buf_, a_sub_tiles_[i].reg_buf_size, alloc_kind_t::grf,
                a_attr);

        auto b_attr = grf_alloc_attr_t::make(mul_builder.b_grf_bundle());
        register_buffer(b_buf_, b_sub_tiles_[j].reg_buf_size, alloc_kind_t::grf,
                b_attr);
    }

    // Loads A_i or B_j sub-tile.
    void load_sub_tile(abc_kind_t abc_kind, int i, stmt_t &ab_s2r_load,
            stmt_t &ab_g2r_load) {
        bool is_a = (abc_kind == abc_kind_t::a);
        auto &info = (is_a ? a_sub_tiles_[i] : b_sub_tiles_[i]);
        if (info.is_loaded) return;

        auto &bmnk_mapper = gemm_schedule_.bmnk_mapper();

        auto &x_view = (is_a ? a_i_view_ : b_j_view_);
        auto &x_tile = (is_a ? a_i_tile_ : b_j_tile_);
        auto &x_idx = (is_a ? a_idx_ : b_idx_);

        auto view = x_view.substitute(x_idx, i);
        auto tile = x_tile.substitute(x_idx, i);

        bool use_x_slm = (is_a ? cfg_.use_a_slm : cfg_.use_b_slm);
        auto &x_slm_buf = (is_a ? a_slm_buf_ : b_slm_buf_);
        auto &x_gmem_buf = (is_a ? ap_buf_ : bp_buf_);
        auto &x_buf = (use_x_slm ? x_slm_buf : x_gmem_buf);
        auto &x_reg_buf = (is_a ? a_buf_ : b_buf_);

        layout_t load_layout;
        view_t reg_view;
        stmt_t stmt;
        load_sub_tile_impl(abc_kind, i, view, x_buf, x_reg_buf, use_x_slm,
                load_layout, reg_view, stmt);

        auto reg_layout = load_layout;

        if (!is_a && cfg_.do_b_reduction && !cfg_.use_b_slm) {
            auto reduce_stmt = b_reduce_ctx_.create_reduce_stmt(
                    reg_layout, b_buf_, tile);
            stmt = stmt.append(reduce_stmt);
        }

        bool changed;
        auto fma_layout = convert_to_fma_friendly_layout(cfg_, abc_kind,
                bmnk_mapper, reg_layout, a_type(), b_type(), &changed);

        if (changed) {
            if (fma_layout.type() != reg_layout.type()) {
                reg_view = reg_view.retype(fma_layout.type());
            }
            reg_layout = fma_layout;
            reg_view.set_tlayout(reg_layout);
            stmt = substitute(stmt, x_reg_buf, ab_tmp_buf_);
            stmt = stmt.append(create_reorder_stmt(
                    load_layout, reg_layout, ab_tmp_buf_, x_reg_buf));
            tmp_buf_size_ = std::max(tmp_buf_size_, int(load_layout.size()));
        }

        if (use_x_slm) {
            ab_s2r_load = ab_s2r_load.append(stmt);
        } else {
            ab_g2r_load = ab_g2r_load.append(stmt);
        }
        info.is_loaded = true;
        info.reg_view = reg_view;
        info.reg_buf_size = reg_layout.size();
    }

    void load_sub_tile_impl(abc_kind_t abc_kind, int sub_tile_idx,
            const view_t &_mem_view, const expr_t &buf, const expr_t &reg_buf,
            bool is_slm, layout_t &reg_layout, view_t &reg_view, stmt_t &stmt) {
        bool is_a = (abc_kind == abc_kind_t::a);

        view_t mem_view;
        bool load_buffered = false;

        // Using buffered view is enabled only when:
        // - Loading directly from global memory
        // - FMA kind is mad (dpas implementation is more strict and requires
        //   layouts, not views)
        // - Loading A tensor (A - activations for FWD/BWD_D where we may have
        //   overlapping when applying KW blocking )
        if (!is_slm && is_a && cfg_.fma_kind == fma_kind_t::mad) {
            load_buffered
                    = _mem_view.try_create_buffer_view(mem_view, reg_view);
        }

        if (!load_buffered) mem_view = _mem_view;

        read_builder_t read(
                cfg_.hw, ir_ctx_, cset_, mem_view, buf, reg_buf, is_slm);
        ir_trace() << (is_a ? "A" : "B") << " GMEM/SLM to GRF load #"
                   << sub_tile_idx << ":\n"
                   << read.str() << std::endl;

        if (load_buffered) {
            reg_view.set_tlayout(read.reg_layout());
        } else {
            reg_view = view_t(read.reg_layout());
        }

        reg_layout = read.reg_layout();
        stmt = read.stmt();
    }

    void register_buffer(const stmt_t &alloc) {
        ir_assert(alloc.is<alloc_t>());
        allocs_.push_back(alloc);
    }

    void register_buffer(const expr_t &buf, int size, alloc_kind_t kind,
            const alloc_attr_t &attr = {}) {
        register_buffer(alloc_t::make(buf, size, kind, attr));
    }

    const conv_config_t &cfg_;
    ir_context_t ir_ctx_;
    const constraint_set_t &cset_;
    const gemm_schedule_t &gemm_schedule_;
    b_reduce_context_t &b_reduce_ctx_;

    expr_t ap_buf_;
    expr_t a_slm_buf_;

    expr_t bp_buf_;
    expr_t b_slm_buf_;

    layout_t c_reg_layout_;

    expr_t ab_tmp_buf_;
    expr_t a_buf_;
    expr_t b_buf_;
    expr_t c_buf_;

    int tmp_buf_size_ = 0;

    // Per-thread views to multiply.
    view_t a_thr_view_;
    view_t b_thr_view_;

    // Sub-tile indices.
    expr_t a_idx_;
    expr_t b_idx_;

    // Sub-tile views.
    view_t a_i_view_;
    view_t b_j_view_;

    tensor_t a_i_tile_;
    tensor_t b_j_tile_;

    std::vector<sub_tile_info_t> a_sub_tiles_;
    std::vector<sub_tile_info_t> b_sub_tiles_;

    std::vector<block_t> a_i_outer_blocks_;
    std::vector<block_t> b_j_outer_blocks_;

    std::vector<stmt_t> allocs_;

    stmt_t load_mul_stmt_;

    int c_buf_off_ = 0;
    layout_t c_sub_tile_layout_;
    alloc_attr_t c_attr_;
};

class slm_reduce_builder_t {
public:
    slm_reduce_builder_t(const conv_config_t &cfg, ir_context_t &ir_ctx,
            const constraint_set_t &cset, const grid_info_t &tg_grid,
            const expr_t &reg_buf, const layout_t &reg_layout,
            const tensor_t &thr_tile)
        : cfg_(cfg)
        , ir_ctx_(ir_ctx)
        , cset_(cset)
        , tg_grid_(tg_grid)
        , reg_buf_(reg_buf)
        , reg_layout_(reg_layout)
        , thr_tile_(thr_tile) {
        ir_assert(tg_grid_.dim(2) > 1);

        tmp_reg_buf_ = ir_ctx_.create_tmp_var(type_t::byte_ptr());
        slm_buf_ = make_buffer("reduce_slm");

        build();
    }

    layout_t reg_layout() const { return reg_layout_; }

    tensor_t thr_tile() const { return thr_tile_; }

    stmt_t stmt() const { return stmt_; }

private:
    void build() {
        int ndims = reg_layout_.ndims();
        int tg_ndims = tg_grid_.ndims();

        // Create SLM layout to store all intermediate buffers from the thread
        // group.
        layout_t slm_layout(reg_layout_.type(), ndims + tg_ndims,
                reg_layout_.offset(), reg_layout_.blocks());
        for (int i = tg_ndims - 1; i >= 0; i--) {
            slm_layout = slm_layout.add_outer_block(ndims + i, tg_grid_.dim(i));
        }

        slm_buf_size_ = slm_layout.size();

        // Write thread tile to SLM.
        std::vector<dim_t> write_dims = reg_layout_.dims();
        std::vector<expr_t> write_start(ndims + tg_ndims, 0);
        write_dims.resize(ndims + tg_ndims, 1);
        for (int i = tg_ndims - 1; i >= 0; i--) {
            write_start[ndims + i] = tg_grid_.idx(i);
        }
        auto write_tile = tensor_t(write_dims, write_start);
        write_builder_t write(cfg_.hw, ir_ctx_, cset_,
                view_t(slm_layout.map(write_tile)), slm_buf_, reg_buf_,
                /*is_slm=*/true);
        stmt_ = stmt_.append(funcs::barrier());
        stmt_ = stmt_.append(write.stmt());
        stmt_ = stmt_.append(funcs::barrier());

        auto &write_layout = write.reg_layout();
        ir_assert(write_layout == reg_layout_) << "Incompatible layouts.";

        // Redistribute the layout to read/reduce all k-axis tiles from every
        // thread.
        auto local_thr_tile = reg_layout_.split(tg_grid_.sub_grid({2}));
        reg_layout_ = reg_layout_.map(tensor_t(local_thr_tile.dims()));

        stmt_t reduce_stmt;
        std::vector<dim_t> read_dims(ndims + tg_ndims, 1);
        std::vector<expr_t> read_start(ndims + tg_ndims);
        for (int i = 0; i < ndims; i++) {
            read_dims[i] = local_thr_tile(i);
            read_start[i] = local_thr_tile.start(i);
        }
        read_start[ndims + 0] = tg_grid_.idx(0);
        read_start[ndims + 1] = tg_grid_.idx(1);
        for (int i = 0; i < tg_grid_.dim(2); i++) {
            read_start[ndims + 2] = i;
            tensor_t read_tile(read_dims, read_start);
            read_builder_t read(cfg_.hw, ir_ctx_, cset_,
                    view_t(slm_layout.map(read_tile)), slm_buf_, tmp_reg_buf_,
                    /*is_slm=*/true);
            reduce_stmt = reduce_stmt.append(read.stmt());

            tmp_reg_buf_size_
                    = std::max(tmp_reg_buf_size_, read.reg_buf_size());
            auto read_layout = read.reg_layout();
            for (int j = 0; j < 2; j++)
                ir_assert(read_layout.dim(ndims + j) == 1);
            read_layout = layout_t(read_layout.type(), ndims + 1,
                    read_layout.offset(), read_layout.blocks());
            reduce_stmt = reduce_stmt.append(
                    create_reduce_stmt(read_layout, reg_layout_, tmp_reg_buf_,
                            reg_buf_, tensor_t(), reduction_mask()));
        }

        stmt_ = stmt_.append(
                create_zero_out_stmt(cfg_.hw, reg_buf_, reg_layout_.size()));
        stmt_ = stmt_.append(reduce_stmt);

        stmt_ = alloc_t::make(
                slm_buf_, slm_buf_size_, alloc_kind_t::slm, {}, stmt_);
        stmt_ = alloc_t::make(
                tmp_reg_buf_, tmp_reg_buf_size_, alloc_kind_t::grf, {}, stmt_);

        thr_tile_ = thr_tile_.create_sub_tensor(local_thr_tile);
    }

    uint32_t reduction_mask() const {
        int k_dim_idx = reg_layout_.ndims();
        uint32_t mask = 0xFFFFFFFF;
        mask &= ~(1 << k_dim_idx);
        return mask;
    }

    const conv_config_t &cfg_;
    ir_context_t &ir_ctx_;
    const constraint_set_t &cset_;
    grid_info_t tg_grid_;

    expr_t reg_buf_;
    layout_t reg_layout_;
    tensor_t thr_tile_;

    expr_t tmp_reg_buf_;
    int tmp_reg_buf_size_ = 0;

    expr_t slm_buf_;
    int slm_buf_size_ = 0;

    stmt_t stmt_;
};

class compute_builder_t {
public:
    compute_builder_t(const conv_config_t &cfg, ir_context_t &ir_ctx,
            constraint_set_t &cset)
        : cfg_(cfg)
        , ir_ctx_(ir_ctx)
        , cset_(cset)
        , b_reduce_ctx_(cfg)
        , g2s_ctx_(ir_ctx) {}

    int ab_slm_size() const { return ab_slm_size_; }

    const stmt_t &c_zero_out_stmt() const { return c_zero_out_stmt_; }
    const stmt_t &b_reduced_zero_out_stmt() const {
        return b_reduced_zero_out_stmt_;
    }

    stmt_t zero_out_stmt() const {
        stmt_t ret;
        ret = ret.append(c_zero_out_stmt());
        ret = ret.append(b_reduced_zero_out_stmt());
        return ret;
    }

    stmt_t iter_stmt() const {
        stmt_t stmt;
        bool use_prefetch = !prefetch_stmt_.is_empty();
        bool use_slm = !g2s_load_stmt_.is_empty();
        if (use_prefetch) {
            stmt = stmt.append(stmt_group_t::make(
                    stmt_label_t::prefetch(), prefetch_stmt_));
        } else if (use_slm) {
            stmt = stmt.append(stmt_group_t::make(
                    stmt_label_t::g2s_load(), g2s_load_stmt_));
            stmt = stmt.append(funcs::barrier());
            stmt = stmt.append(stmt_group_t::make(
                    stmt_label_t::g2s_store(), g2s_store_stmt_));
            stmt = stmt.append(funcs::barrier());
        }
        stmt = stmt.append(load_mul_stmt_);
        return stmt;
    }

    const stmt_t &c_store_stmt() const { return c_store_stmt_; }
    const stmt_t &b_reduced_store_stmt() const { return b_reduced_store_stmt_; }

    stmt_t inject_compute_alloc_stmts(const stmt_t &stmt) const {
        return jit::inject_alloc_stmts(stmt, compute_allocs_);
    }

    stmt_t inject_out_alloc_stmts(const stmt_t &stmt) const {
        return jit::inject_alloc_stmts(stmt, out_allocs_);
    }

    stmt_t inject_let_stmts(const stmt_t &stmt) const {
        return jit::inject_let_stmts(stmt, g2s_ctx_.grid_idx_lets);
    }

    void set_gemm_schedule(const gemm_schedule_t &gemm_schedule) {
        gemm_schedule_ = gemm_schedule;
    }

    // Setters for original AP/BP/CP buffers (P - problem notation).
    void set_ap_buf(const expr_t &buf) { ap_buf_ = buf; }
    void set_bp_buf(const expr_t &buf) { bp_buf_ = buf; }
    void set_cp_buf(const expr_t &buf) { cp_buf_ = buf; }
    void set_b_reduced_mem_buf(const expr_t &buf) {
        b_reduce_ctx_.set_b_reduced_mem_buf(buf);
    }

    void set_b_reduced_view(const view_t &v) {
        b_reduce_ctx_.set_b_reduced_view(v);
    }

    void set_post_op_context(const post_op_context_t &post_op_ctx) {
        post_op_ctx_ = post_op_ctx;
    }

    void set_reduce_condition(const expr_t &cond) {
        b_reduce_ctx_.set_reduce_condition(cond);
    }

    void build() {
        // Initialize SLM buffers.
        expr_t a_slm_buf = make_buffer("a_slm");
        expr_t b_slm_buf = make_buffer("b_slm");

        view_t ap_gmem_view = gemm_schedule_.a_tg_view();
        view_t bp_gmem_view = gemm_schedule_.b_tg_view();

        // Views to multiply by a thread group (either GMEM or SLM).
        view_t ap_x_view;
        view_t bp_x_view;
        prepare_gmem_to_slm("A", cfg_.use_a_slm, gemm_schedule_.a_tg_tile(),
                ap_gmem_view, ap_buf_, a_slm_buf, ap_x_view, g2s_ctx_);
        prepare_gmem_to_slm("B", cfg_.use_b_slm, gemm_schedule_.b_tg_tile(),
                bp_gmem_view, bp_buf_, b_slm_buf, bp_x_view, g2s_ctx_);
        prepare_prefetch("A", cfg_.use_prefetch, ap_gmem_view, ap_buf_);
        prepare_prefetch("B", cfg_.use_prefetch, bp_gmem_view, bp_buf_);

        if (ap_x_view.is_empty()) ap_x_view = ap_gmem_view;
        if (bp_x_view.is_empty()) bp_x_view = bp_gmem_view;

        for (auto &bi : g2s_ctx_.bufs) {
            register_compute_buffer(bi.buf, bi.size, alloc_kind_t::grf);
        }

        load_multiply_builder_t load_mul_builder(cfg_, ir_ctx_, cset_,
                gemm_schedule_, b_reduce_ctx_, ap_buf_, a_slm_buf, bp_buf_,
                b_slm_buf, ap_x_view, bp_x_view);

        load_mul_stmt_ = load_mul_builder.load_mul_stmt();
        compute_allocs_.insert(compute_allocs_.end(),
                load_mul_builder.allocs().begin(),
                load_mul_builder.allocs().end());

        auto c_buf = load_mul_builder.c_buf();
        auto c_attr = load_mul_builder.c_attr();
        int c_size = load_mul_builder.c_reg_layout().size();
        register_out_buffer(c_buf, c_size, alloc_kind_t::grf, c_attr);

        auto c_thr_reg_layout = load_mul_builder.c_reg_layout();
        auto thr_tile = gemm_schedule_.c_thr_tile(/*is_relative=*/false);

        if (gemm_schedule_.with_thread_group_k_slicing()) {
            slm_reduce_builder_t slm_reduce_builder(cfg_, ir_ctx_, cset_,
                    gemm_schedule_.tg_grid(), c_buf, c_thr_reg_layout,
                    thr_tile);
            c_store_stmt_ = c_store_stmt_.append(slm_reduce_builder.stmt());
            c_thr_reg_layout = slm_reduce_builder.reg_layout();
            thr_tile = slm_reduce_builder.thr_tile();
        }

        auto c_thr_mem_view = gemm_schedule_.c_view().create_sub_view(thr_tile);
        epilogue_builder_t c_m2g(cfg_, ir_ctx_, cset_, post_op_ctx_, thr_tile,
                c_thr_mem_view, c_thr_reg_layout, cp_buf_, c_buf);
        ir_trace() << "C GRF to GMEM store:\n" << c_m2g.stmt() << std::endl;

        c_zero_out_stmt_ = stmt_group_t::make(stmt_label_t::c_zero_out(),
                create_zero_out_stmt(cfg_.hw, c_buf, c_size));
        c_store_stmt_ = c_store_stmt_.append(c_m2g.stmt());

        if (cfg_.do_b_reduction) {
            auto &ctx = b_reduce_ctx_;
            b_reduced_zero_out_stmt_ = create_zero_out_stmt(
                    cfg_.hw, ctx.b_reduced_reg_buf(), ctx.b_reduced_size());
            b_reduced_store_stmt_ = ctx.create_store_stmt(ir_ctx_, cset_);
            register_out_buffer(ctx.b_reduced_reg_buf(), ctx.b_reduced_size(),
                    alloc_kind_t::grf);
        }

        // Replace DPAS by DPASW when applicable.
        if (cfg_.fma_kind == fma_kind_t::dpasw) {
            alloc_updater_t alloc_updater;
            inject_dpasw(cfg_.hw, load_mul_stmt_, c_buf, c_store_stmt_,
                    alloc_updater, gemm_schedule_.tg_grid().idx(0));
            for (auto &a : compute_allocs_) {
                a = alloc_updater.update(a);
            }
            for (auto &a : out_allocs_) {
                a = alloc_updater.update(a);
            }
        }

        // Assign {Atomic} for DPAS(W) when applicable.
        load_mul_stmt_ = inject_atomic(load_mul_stmt_);
    }

private:
    struct buf_info_t {
        buf_info_t(const std::string &tag, const expr_t &buf)
            : tag(tag), buf(buf) {}

        std::string tag;
        expr_t buf;
        int size = 0;
    };

    struct g2s_context_t {
        g2s_context_t(ir_context_t &ir_ctx) : ir_ctx(ir_ctx) {}

        expr_t create_buf(const char *tag, bool force_reuse = false) {
            if (reuse_buffers || force_reuse) {
                for (auto &bi : bufs) {
                    if (bi.tag == tag) return bi.buf;
                }
            }
            auto buf = ir_ctx.create_tmp_var(type_t::byte_ptr(), tag);
            bufs.emplace_back(tag, buf);
            return buf;
        }

        void set_buf_size(const expr_t &buf, int size) {
            for (auto &bi : bufs) {
                if (bi.buf.is_same(buf)) bi.size = std::max(bi.size, size);
            }
        }

        expr_t create_tmp_grid_idx() {
            auto var = ir_ctx.create_tmp_var(type_t::s32(), "idx");
            tmp_grid_idxs.insert({var, expr_t()});
            return var;
        }

        void set_grid_idx_value(const expr_t &idx, const expr_t &value) {
            auto &old = tmp_grid_idxs[idx];
            ir_assert(old.is_empty());
            old = substitute_grid_idx_value(value);
        }

        expr_t substitute_grid_idx_value(const expr_t &_e) {
            auto e = _e;
            auto vars = find_unique_objects<var_t>(e);
            for (auto &v : vars) {
                auto it = tmp_grid_idxs.find(v);
                if (it == tmp_grid_idxs.end()) continue;
                e = substitute(e, v, it->second);
            }
            return e;
        }

        void register_grid(const grid_info_t &grid) {
            for (int i = 0; i < grid.ndims(); i++) {
                auto &idx = grid.idx(i);
                auto it = tmp_grid_idxs.find(idx);
                if (it == tmp_grid_idxs.end()) continue;
                grid_idx_lets.emplace_back(let_t::make(idx, it->second));
            }
        }

        ir_context_t &ir_ctx;
        grid_info_t prev_load_grid;
        bool reuse_buffers = false;
        std::vector<buf_info_t> bufs;

        object_map_t<expr_t, expr_t> tmp_grid_idxs;
        std::vector<stmt_t> grid_idx_lets;
    };

    void register_compute_buffer(const expr_t &buf, int size, alloc_kind_t kind,
            const alloc_attr_t &attr = {}) {
        compute_allocs_.push_back(alloc_t::make(buf, size, kind, attr));
    }

    void register_out_buffer(const expr_t &buf, int size, alloc_kind_t kind,
            const alloc_attr_t &attr = {}) {
        out_allocs_.push_back(alloc_t::make(buf, size, kind, attr));
    }

    // Handles GMEM to SLM load for A and B. Done in two steps:
    // 1. Load: GMEM -> GRF (temporary)
    // 2. Store: GRF (temporary) -> SLM
    void prepare_gmem_to_slm(const char *tag, bool use_x_slm,
            const tensor_t &tg_tile, const view_t &x_gmem_view,
            const expr_t &xp_buf, const expr_t &x_slm_buf, view_t &x_slm_view,
            g2s_context_t &g2s_ctx) {
        if (!use_x_slm) return;

        grid_info_t load_grid = gemm_schedule_.tg_grid();
        for (;;) {
            bool ok = prepare_gmem_to_slm_impl(tag, use_x_slm, tg_tile,
                    x_gmem_view, xp_buf, x_slm_buf, x_slm_view, load_grid,
                    g2s_ctx);
            if (ok) {
                g2s_ctx.prev_load_grid = load_grid;
                g2s_ctx.register_grid(load_grid);
                return;
            }

            // Reduce grid and try again.
            auto grid_idx = g2s_ctx.create_tmp_grid_idx();
            int dim_idx;
            expr_t grid_idx_value;
            auto new_load_grid
                    = load_grid.halven(grid_idx, dim_idx, grid_idx_value);
            if (new_load_grid.is_empty()) break;

            if (new_load_grid == g2s_ctx.prev_load_grid) {
                new_load_grid = load_grid.halven(
                        grid_idx, dim_idx, grid_idx_value, /*first=*/false);
                g2s_ctx.reuse_buffers = true;
            }
            g2s_ctx.set_grid_idx_value(grid_idx, grid_idx_value);

            cset_.add_constraint(grid_idx >= 0);
            cset_.add_constraint(grid_idx < new_load_grid.dim(dim_idx));

            load_grid = new_load_grid;
        }
        ir_error_not_expected() << "Can't create GMEM -> SLM loads/stores.";
    }

    bool prepare_gmem_to_slm_impl(const char *tag, bool use_x_slm,
            const tensor_t &tg_tile, const view_t &x_gmem_view,
            const expr_t &xp_buf, const expr_t &x_slm_buf, view_t &x_slm_view,
            const grid_info_t &load_grid, g2s_context_t &g2s_ctx) {
        bool is_a = (tag[0] == 'A');

        auto xp_slm_layout = create_slm_layout(
                x_gmem_view, is_a ? abc_kind_t::a : abc_kind_t::b, load_grid);

        auto grid_cond = load_grid.slice_condition();

        tensor_t thr_tile;
        // Per-thread view to load from GMEM to SLM.
        auto x_g2s_view = x_gmem_view.split(load_grid, thr_tile);
        auto slm_thr_layout = xp_slm_layout.map(thr_tile);

        // Ensure that each thread writes a dense region to SLM. If the layout
        // is not dense, return and try with smaller grid.
        if (!slm_thr_layout.is_dense()) return false;

        register_compute_buffer(
                x_slm_buf, xp_slm_layout.size(), alloc_kind_t::slm);
        ab_slm_size_ += xp_slm_layout.size();

        // Temporary GRF buffer.
        expr_t x_g2s_reg_buf = g2s_ctx.create_buf("g2s");

        // GMEM -> GRF load.
        read_builder_t x_read(cfg_.hw, ir_ctx_, cset_, x_g2s_view, xp_buf,
                x_g2s_reg_buf, /*is_slm=*/false);
        ir_trace() << tag << " GMEM to GRF load:\n"
                   << x_read.str() << std::endl;

        g2s_ctx.set_buf_size(x_g2s_reg_buf, x_read.reg_buf_size());

        auto load_stmt = x_read.stmt();
        if (!grid_cond.is_empty()) load_stmt = if_t::make(grid_cond, load_stmt);
        g2s_load_stmt_ = g2s_load_stmt_.append(load_stmt);

        // GRF -> SLM store.
        write_builder_t x_write(cfg_.hw, ir_ctx_, cset_, view_t(slm_thr_layout),
                x_slm_buf, x_g2s_reg_buf, /*is_slm=*/true);
        ir_trace() << tag << " GRF to SLM store:\n"
                   << x_write.str() << std::endl;
        auto store_stmt = x_write.stmt();

        auto &read_layout = x_read.reg_layout();
        auto &write_layout = x_write.reg_layout();
        if (read_layout != write_layout) {
            if (cfg_.allow_grf_reorder) {
                // Temporary GRF buffer.
                expr_t tmp_buf
                        = g2s_ctx.create_buf("g2s_tmp", /*force_reuse=*/true);
                auto reorder_stmt = create_reorder_stmt(
                        read_layout, write_layout, x_g2s_reg_buf, tmp_buf);
                g2s_ctx.set_buf_size(tmp_buf, x_write.reg_buf_size());
                store_stmt = substitute(store_stmt, x_g2s_reg_buf, tmp_buf);
                store_stmt = reorder_stmt.append(store_stmt);
            } else {
                ir_error_not_expected() << "Requested register layouts for "
                                        << tag << " do not match: "
                                        << "read: " << read_layout
                                        << ", write: " << write_layout;
            }
        }
        // Generate reduction statement for B.
        if (!is_a && cfg_.do_b_reduction) {
            auto absolute_thr_tile = tg_tile.create_sub_tensor(thr_tile);
            b_reduce_ctx_.init_reduced_thr_view(absolute_thr_tile, grid_cond);
            auto reduce_stmt = b_reduce_ctx_.create_reduce_stmt(
                    read_layout, x_g2s_reg_buf);
            store_stmt = reduce_stmt.append(store_stmt);
        }
        if (!grid_cond.is_empty())
            store_stmt = if_t::make(grid_cond, store_stmt);
        g2s_store_stmt_ = g2s_store_stmt_.append(store_stmt);

        x_slm_view = view_t(xp_slm_layout);

        return true;
    }

    void prepare_prefetch(const char *tag, bool use_prefetch,
            const view_t &x_gmem_view, const expr_t &xp_buf) {
        if (!use_prefetch) return;

        // Per-thread view to prefetch from GMEM.
        auto thr_view = x_gmem_view.split(gemm_schedule_.tg_grid());

        // GMEM prefetch.
        read_builder_t x_prefetch(cfg_.hw, ir_ctx_, cset_, thr_view, xp_buf,
                expr_t(), /*is_slm=*/false, /*is_prefetch=*/true);
        ir_trace() << tag << " GMEM prefetch:\n"
                   << x_prefetch.str() << std::endl;

        prefetch_stmt_ = prefetch_stmt_.append(x_prefetch.stmt());
    }

    layout_t create_slm_layout(const view_t &tg_view, abc_kind_t abc_kind,
            const grid_info_t &load_grid) const {
        auto layout = tg_view.create_dense_vlayout();
        auto &a_type = gemm_schedule_.a_view().type();
        auto &b_type = gemm_schedule_.b_view().type();
        auto ret = convert_to_fma_friendly_layout(cfg_, abc_kind,
                gemm_schedule_.bmnk_mapper(), layout, a_type, b_type);
        if (cfg_.pad_slm) ret = pad_slm_layout(ret, load_grid);
        return ret;
    }

    // SLM has 65 dword-granularity banks (Xe_HP):
    //      banks:   [bank 0] [bank 1] [bank 2] ... [bank 0]
    // byte offsets: | 0      | 4      | 8      ... | 4 * 65
    // SLM reads don't have conflicts. During SLM writes each fused EU writes
    // 64 bytes (in total 128 bytes per clock). If there are repeating banks
    // between 128 bytes the write takes 2 clocks to complete.
    // Assume that every X-axis thread (across tg_dim[0]) writes the
    // corresponding outer block of the layout. The goal is to ensure that the
    // stride between outer blocks allows to avoid duplicated banks.
    layout_t pad_slm_layout(
            const layout_t &layout, const grid_info_t &load_grid) const {
        auto tg_dim0 = load_grid.dim(0);
        auto tg_dim1 = load_grid.dim(1);
        int type_size = layout.type().size();

        ir_assert(layout.elems() % tg_dim0 == 0) << layout;
        dim_t inner_block = layout.elems() / tg_dim0;

        ir_assert((inner_block * type_size) % tg_dim1 == 0) << layout;
        dim_t per_thr_bytes = (inner_block * type_size) / tg_dim1;

        std::vector<dim_t> multi_blocks = {inner_block, tg_dim0};
        auto l = layout.split_into_multi_blocks(multi_blocks);

        auto padded_blocks = l.blocks();
        dim_t stride = -1;
        dim_t remaining_elems = inner_block;
        bool past_inner_block = false;
        for (auto &b : padded_blocks) {
            if (past_inner_block) {
                if (stride == -1) {
                    dim_t stride_bytes = find_min_stride_without_conflicts(
                            per_thr_bytes, dim_t(b.stride) * type_size);
                    ir_assert(stride_bytes % type_size == 0);
                    stride = stride_bytes / type_size;
                }
                b.stride = stride;
                stride = b.stride * b.block;
                continue;
            }
            ir_assert(remaining_elems % b.block == 0);
            remaining_elems /= b.block;
            if (remaining_elems == 1) past_inner_block = true;
        }
        return layout_t(
                layout.type(), layout.ndims(), layout.offset(), padded_blocks);
    }

    dim_t find_min_stride_without_conflicts(
            dim_t inner_bytes, dim_t dense_stride_bytes) const {
        int write_step = 64;
        int stride_step = 16;
        dim_t stride_beg = dense_stride_bytes;
        dim_t stride_end = 2 * dense_stride_bytes;
        const int slm_banks = 65;
        for (dim_t s = stride_beg; s < stride_end; s += stride_step) {
            bool ok = true;
            for (dim_t off0 = 0; off0 < inner_bytes; off0 += write_step) {
                // Check banks for a single SLM write.
                bool found[slm_banks] = {false};
                for (dim_t off = off0; off < off0 + write_step;
                        off += sizeof(uint32_t)) {
                    int bank0 = (off / sizeof(uint32_t)) % slm_banks;
                    int bank1 = ((off + s) / sizeof(uint32_t)) % slm_banks;
                    if (found[bank0]) {
                        ok = false;
                        break;
                    }
                    found[bank0] = true;
                    if (found[bank1]) {
                        ok = false;
                        break;
                    }
                    found[bank1] = true;
                }
                if (ok) return s;
            }
        }

        ir_warning()
                << "Couldn't find stride without conflicts for SLM padding."
                << std::endl;

        return dense_stride_bytes;
    }

    const conv_config_t &cfg_;
    ir_context_t &ir_ctx_;
    constraint_set_t &cset_;
    post_op_context_t post_op_ctx_;
    b_reduce_context_t b_reduce_ctx_;

    g2s_context_t g2s_ctx_;

    gemm_schedule_t gemm_schedule_;

    expr_t ap_buf_;
    expr_t bp_buf_;
    expr_t cp_buf_;

    std::vector<stmt_t> compute_allocs_;
    std::vector<stmt_t> out_allocs_;
    int ab_slm_size_ = 0;

    stmt_t g2s_load_stmt_;
    stmt_t g2s_store_stmt_;
    stmt_t prefetch_stmt_;
    stmt_t load_mul_stmt_;

    stmt_t c_zero_out_stmt_;
    stmt_t c_store_stmt_;

    stmt_t b_reduced_zero_out_stmt_;
    stmt_t b_reduced_store_stmt_;
};

void kernel_builder_t::build() {
    ir_context_t ir_ctx;
    constraint_set_t init_cset;

    int grid_ndims = 3;
    kernel_grid_ = grid_info_t(grid_ndims);
    tg_grid_ = grid_info_t(grid_ndims);
    for (int i = 0; i < grid_ndims; i++) {
        local_id_[i]
                = var_t::make(type_t::u16(), "local_id" + std::to_string(i));
        kernel_grid_.dim(i) = cfg_.kernel_grid_dim[i];
        kernel_grid_.idx(i)
                = var_t::make(type_t::s32(), "grid_idx" + std::to_string(i));
        tg_grid_.dim(i) = cfg_.tg_grid_dim[i];
        tg_grid_.idx(i)
                = var_t::make(type_t::s32(), "tg_idx" + std::to_string(i));

        init_cset.add_constraint(kernel_grid_.idx(i) >= 0);
        init_cset.add_constraint(kernel_grid_.idx(i) < cfg_.kernel_grid_dim[i]);
        init_cset.add_constraint(tg_grid_.idx(i) >= 0);
        init_cset.add_constraint(tg_grid_.idx(i) < cfg_.tg_grid_dim[i]);
    }

    gemm_schedule_t gemm_schedule(init_cset, kernel_grid_, tg_grid_);

    std::vector<stmt_t> init_stmts;
    for (int i = 0; i < grid_ndims; i++) {
        auto value = local_id_[i];
        if (i == 0) value /= cfg_.simd_size;
        init_stmts.push_back(let_t::make(tg_grid_.idx(i), value));
    }

    // Initialize memory buffers.
    std::vector<stmt_t> inner_lets;

    view_t a_view;
    view_t b_view;
    view_t c_view;
    view_t bp_reduced_view;

    expr_t ap_buf;
    expr_t bp_buf;
    expr_t cp_buf;
    expr_t b_reduced_mem_buf;
    expr_t b_reduction_condition;

    if (cfg_.is_fwd) {
        init_fwd(gemm_schedule, a_view, b_view, c_view, ap_buf, bp_buf, cp_buf);
    } else if (cfg_.is_bwd_d) {
        init_bwd_d(
                gemm_schedule, a_view, b_view, c_view, ap_buf, bp_buf, cp_buf);
    } else if (cfg_.is_bwd_w) {
        init_bwd_w(gemm_schedule, a_view, b_view, c_view, bp_reduced_view,
                ap_buf, bp_buf, cp_buf, b_reduced_mem_buf,
                b_reduction_condition);
    } else {
        ir_error_not_expected();
    }

    gemm_schedule.finalize();

    post_op_context_t post_op_ctx(
            pd_, cfg_, gemm_schedule.c_view(), kernel_arg_info_);
    compute_builder_t cb(cfg_, ir_ctx, init_cset);

    cb.set_gemm_schedule(gemm_schedule);
    cb.set_ap_buf(ap_buf);
    cb.set_bp_buf(bp_buf);
    cb.set_cp_buf(cp_buf);
    cb.set_b_reduced_mem_buf(b_reduced_mem_buf);
    cb.set_b_reduced_view(bp_reduced_view);
    cb.set_post_op_context(post_op_ctx);
    cb.set_reduce_condition(b_reduction_condition);

    cb.build();

    std::vector<stmt_t> allocs;
    for (int i = 0; i < kernel_arg_info_.nargs(); i++) {
        auto &var = kernel_arg_info_.arg_var(i);
        if (!var.type().is_ptr()) continue;
        allocs.push_back(alloc_t::make(var, 0, alloc_kind_t::global));
    }

    // Create IR statements.
    stmt_t loop_stmt = cb.iter_stmt();
    loop_stmt = gemm_schedule.create_loop_nest(loop_stmt);
    loop_stmt = stmt_group_t::make(stmt_label_t::compute_loop(), loop_stmt);
    loop_stmt = cb.inject_compute_alloc_stmts(loop_stmt);

    auto c_store_stmt
            = stmt_group_t::make(stmt_label_t::c_store(), cb.c_store_stmt());
    stmt_ = loop_stmt;
    stmt_ = stmt_seq_t::make(cb.zero_out_stmt(), stmt_);
    stmt_ = stmt_seq_t::make(stmt_, cb.b_reduced_store_stmt());
    stmt_ = stmt_seq_t::make(stmt_, c_store_stmt);

    stmt_ = cb.inject_out_alloc_stmts(stmt_);
    stmt_ = cb.inject_let_stmts(stmt_);

    stmt_ = gemm_schedule.create_bind_stmt(stmt_);
    stmt_ = inject_let_stmts(stmt_, init_stmts);
    stmt_ = inject_alloc_stmts(stmt_, allocs);

    stmt_ = inject_external_var_let(stmt_);
    stmt_ = merge_slm_buffers(stmt_);
    if (!cfg_.do_loop_unroll && (cfg_.use_a_slm || cfg_.use_b_slm)) {
        stmt_ = inject_simple_slm_buffering(
                cfg_.hw, stmt_, cfg_, ir_ctx, cb.ab_slm_size());
    }
    stmt_ = lift_buffer_offsets_in_send(stmt_);
    stmt_ = simplify_pass(stmt_, init_cset);
    stmt_ = inject_send(stmt_, ir_ctx, init_cset);
    stmt_ = split_wide_stores(cfg_.hw, stmt_);
    stmt_ = lift_alloc(stmt_, cfg_);
    stmt_ = eliminate_common_subexprs(stmt_, ir_ctx);
    stmt_ = hoist_exprs(stmt_, ir_ctx);
    if (cfg_.do_loop_unroll) stmt_ = loop_strength_reduce(stmt_);
    stmt_ = optimize_let(stmt_);
    if (cfg_.do_loop_unroll) {
        stmt_ = update_loops_for_unrolling(stmt_, cfg_);
        stmt_ = inject_unrolling(stmt_, cfg_, ir_ctx, cb.ab_slm_size());
    }
    stmt_ = fixup_if_conditions(stmt_, cfg_);
    stmt_ = unroll_loops(stmt_, ir_ctx);
    stmt_ = simplify_pass(stmt_, init_cset);
    stmt_ = optimize_let(stmt_);
    stmt_ = optimize_peephole(stmt_);
    stmt_ = stmt_group_t::make(stmt_label_t::kernel(), stmt_);

    ir_trace() << "Kernel body:\n" << stmt_ << std::endl;
}

namespace {
bool need_src_or_dst_check(
        bool is_fwd, int o, int i, int k, int p, int s, int d) {
    if (is_fwd) {
        int i_min = -p;
        int i_max = (o - 1) * s - p + (k - 1) * (1 + d);
        return (i_min < 0) || (i_max >= i);
    }
    // Backward.
    int os_min = p - (k - 1) * (1 + d);
    int os_max = (o - 1) + p;
    return (os_min < 0) || (os_max >= i * s);
}

} // namespace

void kernel_builder_t::init_fwd(gemm_schedule_t &gemm_schedule,
        view_t &src_view, view_t &wei_view, view_t &dst_view, expr_t &src_buf,
        expr_t &wei_buf, expr_t &dst_buf) {
    // Unify layouts.
    auto src_layout = cfg_.src_layout;
    auto wei_layout = cfg_.wei_layout;
    auto dst_layout = cfg_.dst_layout;
    normalize_conv_layouts(src_layout, wei_layout, dst_layout, cfg_.with_groups,
            cfg_.g, cfg_.is_dw, cfg_.reduced_to_1d, /*add_groups=*/true);

    // Initialize views.
    auto mb = var_t::make(type_t::s32(), "mb");
    auto ic = var_t::make(type_t::s32(), "ic");
    auto oc = var_t::make(type_t::s32(), "oc");
    auto od = var_t::make(type_t::s32(), "od");
    auto oh = var_t::make(type_t::s32(), "oh");
    auto ow = var_t::make(type_t::s32(), "ow");
    auto kd = var_t::make(type_t::s32(), "kd");
    auto kh = var_t::make(type_t::s32(), "kh");
    auto kw = var_t::make(type_t::s32(), "kw");
    auto g = var_t::make(type_t::s32(), "g");

    // Initialize masks.
    expr_t id_mask, ih_mask, iw_mask;
    expr_t od_mask, oh_mask, ow_mask;
    expr_t src_mb_mask, dst_mb_mask;
    expr_t wei_oc_mask, dst_oc_mask;
    expr_t src_g_mask, wei_g_mask, dst_g_mask;
    expr_t kw_mask;

    bool check_ow = (cfg_.ow % cfg_.ow_tg_blk != 0);
    bool check_iw = check_ow
            || need_src_or_dst_check(cfg_.is_fwd, cfg_.ow, cfg_.iw, cfg_.kw,
                    cfg_.pw, cfg_.sw, cfg_.dw);
    bool check_ih = need_src_or_dst_check(
            cfg_.is_fwd, cfg_.oh, cfg_.ih, cfg_.kh, cfg_.ph, cfg_.sh, cfg_.dh);
    bool check_id = need_src_or_dst_check(
            cfg_.is_fwd, cfg_.od, cfg_.id, cfg_.kd, cfg_.pd, cfg_.sd, cfg_.dd);
    bool check_kw = (cfg_.kw % cfg_.kw_blk != 0);

    int src_g = int(src_layout.dim(1));
    int src_g_inner_blk = ir_utils::max_pow2_divisor(src_g);
    src_g_inner_blk = std::min(src_g_inner_blk, cfg_.g_thr_blk);

    int wei_g = int(wei_layout.dim(0));
    int wei_g_inner_blk = ir_utils::max_pow2_divisor(wei_g);
    wei_g_inner_blk = std::min(wei_g_inner_blk, cfg_.g_thr_blk);

    int wei_oc = int(wei_layout.dim(1));
    int wei_oc_inner_blk = ir_utils::max_pow2_divisor(wei_oc);
    wei_oc_inner_blk = std::min(wei_oc_inner_blk, cfg_.oc_thr_blk);

    int dst_g = int(dst_layout.dim(1));
    int dst_g_inner_blk = ir_utils::max_pow2_divisor(dst_g);
    dst_g_inner_blk = std::min(dst_g_inner_blk, cfg_.g_thr_blk);

    int dst_oc = int(dst_layout.dim(2));
    int dst_oc_inner_blk = ir_utils::max_pow2_divisor(dst_oc);
    dst_oc_inner_blk = std::min(dst_oc_inner_blk, cfg_.oc_thr_blk);

    bool check_src_g = (src_g % cfg_.g_tg_blk != 0);
    bool check_wei_g = (wei_g % cfg_.g_tg_blk != 0);
    bool check_wei_oc = (wei_oc % cfg_.oc_tg_blk != 0);
    bool check_dst_g = (dst_g % cfg_.g_tg_blk != 0);
    bool check_dst_oc = (dst_oc % cfg_.oc_tg_blk != 0);

    int src_mb = int(src_layout.dim(0));
    int dst_mb = int(dst_layout.dim(0));

    bool check_src_mb = (src_mb % cfg_.mb_tg_blk != 0);
    bool check_dst_mb = (dst_mb % cfg_.mb_tg_blk != 0);

    auto &x = view_t::placeholder_var();
    if (check_id) id_mask = (x >= 0) & (x < cfg_.id);
    if (check_ih) ih_mask = (x >= 0) & (x < cfg_.ih);
    if (check_iw) iw_mask = (x >= 0) & (x < cfg_.iw);
    if (check_ow) ow_mask = (x >= 0) & (x < cfg_.ow);
    if (check_src_g)
        src_g_mask = (x / src_g_inner_blk < src_g / src_g_inner_blk);
    if (check_wei_g)
        wei_g_mask = (x / wei_g_inner_blk < wei_g / wei_g_inner_blk);
    if (check_wei_oc)
        wei_oc_mask = (x / wei_oc_inner_blk < wei_oc / wei_oc_inner_blk);
    if (check_dst_g)
        dst_g_mask = (x / dst_g_inner_blk < dst_g / dst_g_inner_blk);
    if (check_dst_oc)
        dst_oc_mask = (x / dst_oc_inner_blk < dst_oc / dst_oc_inner_blk);
    if (check_kw) kw_mask = (x < cfg_.kw);
    if (check_src_mb) src_mb_mask = (x < src_mb);
    if (check_dst_mb) dst_mb_mask = (x < dst_mb);

    // Source.
    src_view = view_t({mb, g, ic, od, oh, ow, kd, kh, kw}, 6);
    src_view.set_vdim(mb, cfg_.mb);
    src_view.set_vdim(g, cfg_.g);
    src_view.set_vdim(ic, cfg_.ic);
    src_view.set_vdim(od, cfg_.od);
    src_view.set_vdim(oh, cfg_.oh);
    src_view.set_vdim(ow, cfg_.ow);
    src_view.set_vdim(kd, cfg_.kd);
    src_view.set_vdim(kh, cfg_.kh);
    src_view.set_vdim(kw, cfg_.kw);
    src_view.set_tdim(0, mb, src_mb_mask);
    src_view.set_tdim(1, g, src_g_mask);
    src_view.set_tdim(2, ic);
    src_view.set_tdim(3, od * cfg_.sd - cfg_.pd + kd * (1 + cfg_.dd), id_mask);
    src_view.set_tdim(4, oh * cfg_.sh - cfg_.ph + kh * (1 + cfg_.dh), ih_mask);
    src_view.set_tdim(5, ow * cfg_.sw - cfg_.pw + kw * (1 + cfg_.dw), iw_mask);
    src_view.set_tlayout(src_layout);

    // Weights.
    wei_view = view_t({g, oc, ic, kd, kh, kw}, 6);
    wei_view.set_vdim(g, cfg_.g);
    wei_view.set_vdim(oc, cfg_.oc);
    wei_view.set_vdim(ic, cfg_.ic);
    wei_view.set_vdim(kd, cfg_.kd);
    wei_view.set_vdim(kh, cfg_.kh);
    wei_view.set_vdim(kw, cfg_.kw);
    wei_view.set_tdim(0, g, wei_g_mask);
    wei_view.set_tdim(1, oc, wei_oc_mask);
    wei_view.set_tdim(2, ic);
    wei_view.set_tdim(3, kd);
    wei_view.set_tdim(4, kh);
    wei_view.set_tdim(5, kw, kw_mask);
    wei_view.set_tlayout(wei_layout);

    // Destination.
    dst_view = view_t({mb, g, oc, od, oh, ow}, 6);
    dst_view.set_vdim(mb, cfg_.mb);
    dst_view.set_vdim(g, cfg_.g);
    dst_view.set_vdim(oc, cfg_.oc);
    dst_view.set_vdim(od, cfg_.od);
    dst_view.set_vdim(oh, cfg_.oh);
    dst_view.set_vdim(ow, cfg_.ow);
    dst_view.set_tdim(0, mb, dst_mb_mask);
    dst_view.set_tdim(1, g, dst_g_mask);
    dst_view.set_tdim(2, oc, dst_oc_mask);
    dst_view.set_tdim(3, od, od_mask);
    dst_view.set_tdim(4, oh, oh_mask);
    dst_view.set_tdim(5, ow, ow_mask);
    dst_view.set_tlayout(dst_layout);

    // Initialize GEMM schedule.
    gemm_schedule.set_a_view(src_view);
    gemm_schedule.set_b_view(wei_view);
    gemm_schedule.set_c_view(dst_view);
    gemm_schedule.set_b_vars({g});
    gemm_schedule.set_m_vars({mb, od, oh, ow});
    gemm_schedule.set_n_vars({oc});
    gemm_schedule.set_k_vars({ic, kd, kh, kw});

    expr_t g_tg_blk_idx, g_inner;
    expr_t oc_tg_blk_idx, oc_thr_blk_idx, oc_inner;
    expr_t mb_tg_blk_idx, mb_thr_blk_idx, mb_inner;
    expr_t ow_tg_blk_idx, ow_thr_blk_idx, ow_inner;
    expr_t kw_outer, kw_inner;
    expr_t ic_thr_blk_idx, ic_outer, ic_inner;

    gemm_schedule.split(g, cfg_.g_tg_blk, g_tg_blk_idx, g_inner);
    gemm_schedule.split(oc, cfg_.oc_tg_blk, cfg_.oc_thr_blk, oc_tg_blk_idx,
            oc_thr_blk_idx, oc_inner);
    gemm_schedule.split(mb, cfg_.mb_tg_blk, cfg_.mb_thr_blk, mb_tg_blk_idx,
            mb_thr_blk_idx, mb_inner);
    gemm_schedule.split(ow, cfg_.ow_tg_blk, cfg_.ow_thr_blk, ow_tg_blk_idx,
            ow_thr_blk_idx, ow_inner);
    gemm_schedule.split(ic, cfg_.ic_blk * cfg_.ic_thr_dim, cfg_.ic_blk,
            ic_outer, ic_thr_blk_idx, ic_inner);
    gemm_schedule.split(kw, cfg_.kw_blk, kw_outer, kw_inner);

    auto g_odhw_idx = gemm_schedule.fuse({g_tg_blk_idx, od, oh, ow_tg_blk_idx});
    auto mb_ow_thr_blk_idx = gemm_schedule.fuse(mb_thr_blk_idx, ow_thr_blk_idx);

    gemm_schedule.bind(oc_tg_blk_idx, kernel_grid_.idx(0));
    gemm_schedule.bind(g_odhw_idx, kernel_grid_.idx(1));
    gemm_schedule.bind(mb_tg_blk_idx, kernel_grid_.idx(2));
    gemm_schedule.bind(oc_thr_blk_idx, tg_grid_.idx(0));
    gemm_schedule.bind(mb_ow_thr_blk_idx, tg_grid_.idx(1));
    gemm_schedule.bind(ic_thr_blk_idx, tg_grid_.idx(2));

    gemm_schedule.tensorize(g_inner);
    gemm_schedule.tensorize(oc_inner);
    gemm_schedule.tensorize(mb_inner);
    gemm_schedule.tensorize(ow_inner);
    gemm_schedule.tensorize(kw_inner);
    gemm_schedule.tensorize(ic_inner);

    gemm_schedule.reorder({ic_outer, kd, kh, kw_outer, oc_thr_blk_idx,
            mb_ow_thr_blk_idx, ic_thr_blk_idx});

    src_buf = kernel_arg_info_.find_arg("src");
    wei_buf = kernel_arg_info_.find_arg("wei");
    dst_buf = kernel_arg_info_.find_arg("dst");
}

void kernel_builder_t::init_bwd_d(gemm_schedule_t &gemm_schedule,
        view_t &dst_view, view_t &wei_view, view_t &src_view, expr_t &dst_buf,
        expr_t &wei_buf, expr_t &src_buf) {
    // Unify layouts.
    auto src_layout = cfg_.src_layout;
    auto wei_layout = cfg_.wei_layout;
    auto dst_layout = cfg_.dst_layout;
    normalize_conv_layouts(src_layout, wei_layout, dst_layout, cfg_.with_groups,
            cfg_.g, cfg_.is_dw, cfg_.reduced_to_1d, /*add_groups=*/false);

    // Initialize views.
    auto mb = var_t::make(type_t::s32(), "mb");
    auto ic = var_t::make(type_t::s32(), "ic");
    auto oc = var_t::make(type_t::s32(), "oc");
    auto id = var_t::make(type_t::s32(), "id");
    auto ih = var_t::make(type_t::s32(), "ih");
    auto iw = var_t::make(type_t::s32(), "iw");
    auto kd = var_t::make(type_t::s32(), "kd");
    auto kh = var_t::make(type_t::s32(), "kh");
    auto kw = var_t::make(type_t::s32(), "kw");

    // Initialize masks.
    expr_t id_mask, ih_mask, iw_mask;
    expr_t od_mask(true), oh_mask(true), ow_mask(true);
    expr_t src_mb_mask, dst_mb_mask;
    expr_t wei_oc_mask, dst_oc_mask;
    expr_t wei_ic_mask, src_ic_mask;

    bool check_iw = (cfg_.iw % cfg_.iw_tg_blk != 0);
    bool check_ow = check_iw
            || need_src_or_dst_check(cfg_.is_fwd, cfg_.ow, cfg_.iw, cfg_.kw,
                    cfg_.pw, cfg_.sw, cfg_.dw);
    bool check_oh = need_src_or_dst_check(
            cfg_.is_fwd, cfg_.oh, cfg_.ih, cfg_.kh, cfg_.ph, cfg_.sh, cfg_.dh);
    bool check_od = need_src_or_dst_check(
            cfg_.is_fwd, cfg_.od, cfg_.id, cfg_.kd, cfg_.pd, cfg_.sd, cfg_.dd);

    int wei_ic = int(cfg_.wei_layout.dim(cfg_.with_groups ? 2 : 1));
    int src_ic = int(cfg_.src_layout.dim(1));

    int wei_ic_inner_blk = ir_utils::max_pow2_divisor(wei_ic);
    int src_ic_inner_blk = ir_utils::max_pow2_divisor(src_ic);
    wei_ic_inner_blk = std::min(wei_ic_inner_blk, cfg_.ic_thr_blk);
    src_ic_inner_blk = std::min(src_ic_inner_blk, cfg_.ic_thr_blk);

    bool check_wei_ic = (wei_ic % cfg_.ic_tg_blk != 0);
    bool check_src_ic = (src_ic % cfg_.ic_tg_blk != 0);

    int src_mb = int(cfg_.src_layout.dim(0));
    int dst_mb = int(cfg_.src_layout.dim(0));

    bool check_src_mb = (src_mb % cfg_.mb_tg_blk != 0);
    bool check_dst_mb = (dst_mb % cfg_.mb_tg_blk != 0);

    auto &x = view_t::placeholder_var();
    if (check_od) od_mask = (x >= 0) & (x < cfg_.od);
    if (check_oh) oh_mask = (x >= 0) & (x < cfg_.oh);
    if (check_ow) ow_mask = (x >= 0) & (x < cfg_.ow);
    if (check_iw) iw_mask = (x >= 0) & (x < cfg_.iw);
    if (check_wei_ic)
        wei_ic_mask = (x / wei_ic_inner_blk < wei_ic / wei_ic_inner_blk);
    if (check_src_ic)
        src_ic_mask = (x / src_ic_inner_blk < src_ic / src_ic_inner_blk);
    if (check_src_mb) src_mb_mask = (x < src_mb);
    if (check_dst_mb) dst_mb_mask = (x < dst_mb);

    // Destination.
    dst_view = view_t({mb, oc, id, ih, iw, kd, kh, kw}, 5);
    dst_view.set_vdim(mb, cfg_.mb);
    dst_view.set_vdim(oc, cfg_.oc);
    dst_view.set_vdim(id, cfg_.id);
    dst_view.set_vdim(ih, cfg_.ih);
    dst_view.set_vdim(iw, cfg_.iw);
    dst_view.set_vdim(kd, cfg_.kd);
    dst_view.set_vdim(kh, cfg_.kh);
    dst_view.set_vdim(kw, cfg_.kw);
    dst_view.set_tdim(0, mb, src_mb_mask);
    dst_view.set_tdim(1, oc);

    auto od = id - kd * (1 + cfg_.dd) + cfg_.pd;
    auto oh = ih - kh * (1 + cfg_.dh) + cfg_.ph;
    auto ow = iw - kw * (1 + cfg_.dw) + cfg_.pw;
    dst_view.set_tdim(2, od / cfg_.sd, od_mask & (od % cfg_.sd == 0));
    dst_view.set_tdim(3, oh / cfg_.sh, oh_mask & (oh % cfg_.sh == 0));
    dst_view.set_tdim(4, ow / cfg_.sw, ow_mask & (ow % cfg_.sw == 0));

    dst_view.set_tlayout(dst_layout);

    // Weights.
    wei_view = view_t({oc, ic, kd, kh, kw}, 5);
    wei_view.set_vdim(ic, cfg_.ic);
    wei_view.set_vdim(oc, cfg_.oc);
    wei_view.set_vdim(kd, cfg_.kd);
    wei_view.set_vdim(kh, cfg_.kh);
    wei_view.set_vdim(kw, cfg_.kw);
    wei_view.set_tdim(0, oc);
    wei_view.set_tdim(1, ic, wei_ic_mask);
    wei_view.set_tdim(2, kd);
    wei_view.set_tdim(3, kh);
    wei_view.set_tdim(4, kw);
    wei_view.set_tlayout(wei_layout);

    // Source.
    src_view = view_t({mb, ic, id, ih, iw}, 5);
    src_view.set_vdim(mb, cfg_.mb);
    src_view.set_vdim(ic, cfg_.ic);
    src_view.set_vdim(id, cfg_.id);
    src_view.set_vdim(ih, cfg_.ih);
    src_view.set_vdim(iw, cfg_.iw);
    src_view.set_tdim(0, mb, dst_mb_mask);
    src_view.set_tdim(1, ic, src_ic_mask);
    src_view.set_tdim(2, id, id_mask);
    src_view.set_tdim(3, ih, ih_mask);
    src_view.set_tdim(4, iw, iw_mask);
    src_view.set_tlayout(src_layout);

    // Initialize GEMM schedule.
    gemm_schedule.set_a_view(dst_view);
    gemm_schedule.set_b_view(wei_view);
    gemm_schedule.set_c_view(src_view);
    gemm_schedule.set_m_vars({mb, id, ih, iw});
    gemm_schedule.set_n_vars({ic});
    gemm_schedule.set_k_vars({oc, kd, kh, kw});

    expr_t ic_tg_blk_idx, ic_thr_blk_idx, ic_inner;
    expr_t mb_tg_blk_idx, mb_inner;
    expr_t iw_tg_blk_idx, iw_thr_blk_idx, iw_inner;
    expr_t oc_blk_idx, oc_inner;

    gemm_schedule.split(ic, cfg_.ic_tg_blk, cfg_.ic_thr_blk, ic_tg_blk_idx,
            ic_thr_blk_idx, ic_inner);
    gemm_schedule.split(mb, cfg_.mb_tg_blk, mb_tg_blk_idx, mb_inner);
    gemm_schedule.split(iw, cfg_.iw_tg_blk, cfg_.iw_thr_blk, iw_tg_blk_idx,
            iw_thr_blk_idx, iw_inner);
    gemm_schedule.split(oc, cfg_.oc_blk, oc_blk_idx, oc_inner);

    auto idhw_idx = gemm_schedule.fuse(id, ih, iw_tg_blk_idx);
    gemm_schedule.bind(ic_tg_blk_idx, kernel_grid_.idx(0));
    gemm_schedule.bind(idhw_idx, kernel_grid_.idx(1));
    gemm_schedule.bind(mb_tg_blk_idx, kernel_grid_.idx(2));
    gemm_schedule.bind(ic_thr_blk_idx, tg_grid_.idx(0));
    gemm_schedule.bind(iw_thr_blk_idx, tg_grid_.idx(1));

    gemm_schedule.tensorize(ic_inner);
    gemm_schedule.tensorize(mb_inner);
    gemm_schedule.tensorize(iw_inner);
    gemm_schedule.tensorize(oc_inner);

    gemm_schedule.reorder({oc_blk_idx, kd, kh, kw});

    src_buf = kernel_arg_info_.find_arg("src");
    wei_buf = kernel_arg_info_.find_arg("wei");
    dst_buf = kernel_arg_info_.find_arg("dst");
}

void kernel_builder_t::init_bwd_w(gemm_schedule_t &gemm_schedule,
        view_t &src_view, view_t &dst_view, view_t &wei_view, view_t &bia_view,
        expr_t &src_buf, expr_t &dst_buf, expr_t &wei_buf, expr_t &bia_buf,
        expr_t &bia_reduction_condition) {
    // Unify layouts.
    auto src_layout = cfg_.src_layout;
    auto wei_layout = cfg_.wei_layout;
    auto dst_layout = cfg_.dst_layout;
    normalize_conv_layouts(src_layout, wei_layout, dst_layout, cfg_.with_groups,
            cfg_.g, cfg_.is_dw, cfg_.reduced_to_1d, /*add_groups=*/false);

    // Initialize thread group views.
    auto mb = var_t::make(type_t::s32(), "mb");
    auto ic = var_t::make(type_t::s32(), "ic");
    auto oc = var_t::make(type_t::s32(), "oc");
    auto od = var_t::make(type_t::s32(), "od");
    auto oh = var_t::make(type_t::s32(), "oh");
    auto ow = var_t::make(type_t::s32(), "ow");
    auto kd = var_t::make(type_t::s32(), "kd");
    auto kh = var_t::make(type_t::s32(), "kh");
    auto kw = var_t::make(type_t::s32(), "kw");

    // Initialize masks.
    expr_t id_mask(true), ih_mask(true), iw_mask(true);
    expr_t od_mask, oh_mask, ow_mask;
    expr_t src_mb_mask, src_ic_mask;
    expr_t dst_mb_mask, dst_oc_mask;
    expr_t wei_oc_mask, wei_ic_mask;
    expr_t kw_mask;

    bool check_ow = (cfg_.ow % cfg_.ow_tg_blk != 0);
    bool check_oh = (cfg_.oh % cfg_.oh_tg_blk != 0);
    bool check_od = (cfg_.od % cfg_.od_tg_blk != 0);
    bool check_kw = (cfg_.kw % cfg_.kw_blk != 0);
    bool check_iw = check_kw
            || need_src_or_dst_check(/*is_fwd=*/true, cfg_.ow, cfg_.iw, cfg_.kw,
                    cfg_.pw, cfg_.sw, cfg_.dw);
    bool check_ih = need_src_or_dst_check(/*is_fwd=*/true, cfg_.oh, cfg_.ih,
            cfg_.kh, cfg_.ph, cfg_.sh, cfg_.dh);
    bool check_id = need_src_or_dst_check(/*is_fwd=*/true, cfg_.od, cfg_.id,
            cfg_.kd, cfg_.pd, cfg_.sd, cfg_.dd);
    bool check_iw_min = check_iw;
    bool check_ih_min = check_ih;
    bool check_id_min = check_id;
    bool check_iw_max = (check_iw || check_ow);
    bool check_ih_max = (check_ih || check_oh);
    bool check_id_max = (check_id || check_od);

    int src_ic = int(cfg_.src_layout.dim(1));
    int dst_oc = int(cfg_.dst_layout.dim(1));
    int wei_oc = int(cfg_.wei_layout.dim(cfg_.with_groups ? 1 : 0));
    int wei_ic = int(cfg_.wei_layout.dim(cfg_.with_groups ? 2 : 1));

    int src_ic_inner_blk = ir_utils::max_pow2_divisor(src_ic);
    int dst_oc_inner_blk = ir_utils::max_pow2_divisor(dst_oc);
    int wei_oc_inner_blk = ir_utils::max_pow2_divisor(wei_oc);
    int wei_ic_inner_blk = ir_utils::max_pow2_divisor(wei_ic);
    src_ic_inner_blk = std::min(src_ic_inner_blk, cfg_.ic_thr_blk);
    dst_oc_inner_blk = std::min(dst_oc_inner_blk, cfg_.oc_thr_blk);
    wei_oc_inner_blk = std::min(wei_oc_inner_blk, cfg_.oc_thr_blk);
    wei_ic_inner_blk = std::min(wei_ic_inner_blk, cfg_.ic_thr_blk);

    bool check_src_ic = (src_ic % cfg_.ic_tg_blk != 0);
    bool check_dst_oc = (dst_oc % cfg_.oc_tg_blk != 0);
    bool check_wei_oc = (wei_oc % cfg_.oc_tg_blk != 0);
    bool check_wei_ic = (wei_ic % cfg_.ic_tg_blk != 0);

    auto &x = view_t::placeholder_var();
    if (check_id_min) id_mask &= (x >= 0);
    if (check_ih_min) ih_mask &= (x >= 0);
    if (check_iw_min) iw_mask &= (x >= 0);
    if (check_id_max) id_mask &= (x < cfg_.id);
    if (check_ih_max) ih_mask &= (x < cfg_.ih);
    if (check_iw_max) iw_mask &= (x < cfg_.iw);
    if (check_od) od_mask = (x < cfg_.od);
    if (check_oh) oh_mask = (x < cfg_.oh);
    if (check_ow) ow_mask = (x < cfg_.ow);
    if (check_src_ic)
        src_ic_mask = (x / src_ic_inner_blk < src_ic / src_ic_inner_blk);
    if (check_dst_oc)
        dst_oc_mask = (x / dst_oc_inner_blk < dst_oc / dst_oc_inner_blk);
    if (check_wei_oc)
        wei_oc_mask = (x / wei_oc_inner_blk < wei_oc / wei_oc_inner_blk);
    if (check_wei_ic)
        wei_ic_mask = (x / wei_ic_inner_blk < wei_ic / wei_ic_inner_blk);
    if (check_kw) kw_mask = (x < cfg_.kw);

    // Source.
    src_view = view_t({mb, ic, od, oh, ow, kw}, 5);
    src_view.set_vdim(mb, cfg_.mb);
    src_view.set_vdim(ic, cfg_.ic);
    src_view.set_vdim(od, cfg_.od);
    src_view.set_vdim(oh, cfg_.oh);
    src_view.set_vdim(ow, cfg_.ow);
    src_view.set_vdim(kw, cfg_.kw);
    src_view.set_tdim(0, mb, src_mb_mask);
    src_view.set_tdim(1, ic, src_ic_mask);
    src_view.set_tdim(2, od * cfg_.sd - cfg_.pd + kd * (1 + cfg_.dd), id_mask);
    src_view.set_tdim(3, oh * cfg_.sh - cfg_.ph + kh * (1 + cfg_.dh), ih_mask);
    src_view.set_tdim(4, ow * cfg_.sw - cfg_.pw + kw * (1 + cfg_.dw), iw_mask);
    src_view.set_tlayout(src_layout);

    // Weights.
    wei_view = view_t({oc, ic, kd, kh, kw}, 5);
    wei_view.set_vdim(oc, cfg_.oc);
    wei_view.set_vdim(ic, cfg_.ic);
    wei_view.set_vdim(kd, cfg_.kd);
    wei_view.set_vdim(kh, cfg_.kh);
    wei_view.set_vdim(kw, cfg_.kw);
    wei_view.set_tdim(0, oc, wei_oc_mask);
    wei_view.set_tdim(1, ic, wei_ic_mask);
    wei_view.set_tdim(2, kd);
    wei_view.set_tdim(3, kh);
    wei_view.set_tdim(4, kw, kw_mask);
    wei_view.set_tlayout(wei_layout);

    // Destination.
    dst_view = view_t({mb, oc, od, oh, ow}, 5);
    dst_view.set_vdim(mb, cfg_.mb);
    dst_view.set_vdim(oc, cfg_.oc);
    dst_view.set_vdim(od, cfg_.od);
    dst_view.set_vdim(oh, cfg_.oh);
    dst_view.set_vdim(ow, cfg_.ow);
    dst_view.set_tdim(0, mb, dst_mb_mask);
    dst_view.set_tdim(1, oc, dst_oc_mask);
    dst_view.set_tdim(2, od, od_mask);
    dst_view.set_tdim(3, oh, oh_mask);
    dst_view.set_tdim(4, ow, ow_mask);
    dst_view.set_tlayout(dst_layout);

    // Bias.
    if (cfg_.with_bias) {
        expr_t bia_oc_mask;
        if (cfg_.oc % cfg_.oc_tg_blk != 0) bia_oc_mask = (x < cfg_.oc);
        bia_view = view_t({oc}, 1);
        bia_view.set_vdim(oc, cfg_.oc, 0);
        bia_view.set_tdim(0, oc, bia_oc_mask);
        bia_view.set_tlayout(cfg_.bia_layout);
    }

    // Initialize GEMM schedule.
    gemm_schedule.set_a_view(src_view);
    gemm_schedule.set_b_view(dst_view);
    gemm_schedule.set_c_view(wei_view);
    gemm_schedule.set_m_vars({ic, kw});
    gemm_schedule.set_n_vars({oc});
    gemm_schedule.set_k_vars({mb, od, oh, ow});

    expr_t mb_tg_blk_idx, mb_thr_blk_idx, mb_inner;
    expr_t oc_tg_blk_idx, oc_thr_blk_idx, oc_inner;
    expr_t ic_tg_blk_idx, ic_thr_blk_idx, ic_inner;
    expr_t od_tg_blk_idx, od_inner;
    expr_t oh_tg_blk_idx, oh_inner;
    expr_t ow_tg_blk_idx, ow_thr_blk_idx, ow_inner;
    expr_t kw_tg_blk_idx, kw_inner;

    gemm_schedule.split(mb, cfg_.mb_tg_blk, cfg_.mb_blk, mb_tg_blk_idx,
            mb_thr_blk_idx, mb_inner);
    gemm_schedule.split(ic, cfg_.ic_tg_blk, cfg_.ic_thr_blk, ic_tg_blk_idx,
            ic_thr_blk_idx, ic_inner);
    gemm_schedule.split(oc, cfg_.oc_tg_blk, cfg_.oc_thr_blk, oc_tg_blk_idx,
            oc_thr_blk_idx, oc_inner);
    gemm_schedule.split(od, cfg_.od_tg_blk, od_tg_blk_idx, od_inner);
    gemm_schedule.split(oh, cfg_.oh_tg_blk, oh_tg_blk_idx, oh_inner);
    gemm_schedule.split(ow, cfg_.ow_tg_blk, cfg_.ow_thr_blk, ow_tg_blk_idx,
            ow_thr_blk_idx, ow_inner);
    gemm_schedule.split(kw, cfg_.kw_tg_blk, kw_tg_blk_idx, kw_inner);

    auto odhw_tg_blk_kdhw_ic_tg_blk_idx
            = gemm_schedule.fuse({od_tg_blk_idx, oh_tg_blk_idx, ow_tg_blk_idx,
                    kd, kh, kw_tg_blk_idx, ic_tg_blk_idx});

    gemm_schedule.bind(oc_tg_blk_idx, kernel_grid_.idx(0));
    gemm_schedule.bind(odhw_tg_blk_kdhw_ic_tg_blk_idx, kernel_grid_.idx(1));
    gemm_schedule.bind(mb_tg_blk_idx, kernel_grid_.idx(2));

    gemm_schedule.bind(oc_thr_blk_idx, tg_grid_.idx(0));
    gemm_schedule.bind(ic_thr_blk_idx, tg_grid_.idx(1));

    gemm_schedule.reorder({od_inner, oh_inner, ow_inner, mb_thr_blk_idx});

    int ow_unroll = cfg_.is_dpas_fma() ? cfg_.ow_tg_blk / cfg_.ow_thr_blk : 1;
    gemm_schedule.unroll(mb_thr_blk_idx, cfg_.mb_tg_blk / cfg_.mb_blk);
    gemm_schedule.unroll(ow_thr_blk_idx, ow_unroll);
    gemm_schedule.tensorize(oc_inner);
    gemm_schedule.tensorize(ic_inner);
    gemm_schedule.tensorize(mb_inner);
    gemm_schedule.tensorize(ow_inner);
    gemm_schedule.tensorize(kw_inner);

    src_buf = kernel_arg_info_.find_arg("src");
    wei_buf = kernel_arg_info_.find_arg("wei");
    dst_buf = kernel_arg_info_.find_arg("dst");

    if (cfg_.with_bias) {
        bia_buf = kernel_arg_info_.find_arg("bia");
        bia_reduction_condition = expr_t(true);
        if (cfg_.kd > 1) bia_reduction_condition &= (kd == 0);
        if (cfg_.kh > 1) bia_reduction_condition &= (kh == 0);
        if (cfg_.kw > 1) bia_reduction_condition &= (kw_tg_blk_idx == 0);
        if (cfg_.ic_tg_dim > 1) bia_reduction_condition &= (ic_tg_blk_idx == 0);
        if (!cfg_.use_b_slm && tg_grid_.dim(1) > 1) {
            bia_reduction_condition &= (tg_grid_.idx(1) == 0);
        }
    }
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
