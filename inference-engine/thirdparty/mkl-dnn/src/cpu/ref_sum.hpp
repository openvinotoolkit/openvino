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

#ifndef REF_SUM_HPP
#define REF_SUM_HPP

#include "cpu_sum.hpp"
#include "reorder_pd.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

struct ref_sum_t: public cpu_primitive_t {
    using cpu_memory_pd_t = cpu_memory_t::pd_t;

    struct pd_t: public cpu_sum_pd_t {
        pd_t(const memory_desc_t *output_d, int n, const float *scales,
                const cpu_memory_pd_t **input_pds, const primitive_attr_t *attr)
            : cpu_sum_pd_t(output_d, n, scales, input_pds, attr) {}
        pd_t(const pd_t &rhs): cpu_sum_pd_t(rhs) {
            for (size_t i = 0; i < rhs.scales_.size(); ++i) {
                scales_.push_back(rhs.scales_[i]);
            }
            for (size_t i = 0; i < rhs.reorder_pds_.size(); ++i) {
                reorder_pds_.push_back(
                       (const reorder_pd_t *)rhs.reorder_pds_[i]->clone());
            }
        }

        ~pd_t() {
            for (size_t i = 0; i < reorder_pds_.size(); ++i) {
                delete reorder_pds_[i];
            }
        }

        static status_t create(sum_pd_t **sum_pd,
                const memory_desc_t *output_d, int n, const float *scales,
                const memory_pd_t **input_pds, const primitive_attr_t *attr) {
            auto _pd = new pd_t(output_d, n, scales,
                    (const cpu_memory_pd_t **)input_pds, attr);
            if (_pd == nullptr) return out_of_memory;
            if (_pd->init() != success) { delete _pd; return unimplemented; }
            return safe_ptr_assign<sum_pd_t>(*sum_pd, _pd);
        }

        virtual status_t create_primitive(primitive_t **primitive,
                const primitive_at_t *inputs, const primitive_t **outputs)
                const override {
            double ms = get_msec();
            nstl::vector<primitive_t *> reorders;
            reorders.resize(n_);
            for (int i = 0; i < n_; ++i)
                CHECK(reorder_pds_[i]->create_primitive(&reorders[i],
                            &inputs[i], outputs));

            primitive_t::input_vector ins(inputs, inputs + n_);
            primitive_t::output_vector outs(outputs, outputs + 1);
            auto ret = safe_ptr_assign<primitive_t>(*primitive,
                     new ref_sum_t(this, ins, outs, reorders));
            ms = get_msec() - ms;
            if (mkldnn_verbose()->level >= 2) {
                printf("mkldnn_verbose,create,%s,%g\n", this->info(), ms);
                fflush(0);
            }
            return ret;
        }
        virtual pd_t *clone() const override { return new pd_t(*this); }
        virtual const char *name() const override { return "ref:any"; }

        virtual status_t init() override {
            bool ok = cpu_sum_pd_t::init() == success;
            if (!ok) return unimplemented;

            for (int i = 0; i < n_; ++i) {
                auto r_impls = engine_->get_reorder_implementation_list();
                for (auto r = r_impls; *r; ++r) {
                    primitive_attr_t dummy_attr;
                    dummy_attr.output_scales_.set(scales_[i]);
                    reorder_pd_t *r_pd;
                    if (i != 0) {
                        dummy_attr.post_ops_.append_sum(1.0);
                    }
                    if ((*r)(&r_pd, &src_pds_[i], &dst_pd_, &dummy_attr)
                            == status::success) {
                        r_pd->init_info();
                        reorder_pds_.push_back(r_pd);
                        break;
                    }
                }
            }
            ok = utils::everyone_is(reorder_pds_.size(), scales_.size());
            return ok ? success : unimplemented;
        }

        nstl::vector<const reorder_pd_t *> reorder_pds_;
    };

    ref_sum_t(const pd_t *apd, const input_vector &inputs,
            const output_vector &outputs, nstl::vector<primitive_t *> reorders)
        : cpu_primitive_t(apd, inputs, outputs),
        reorders_(reorders) {}

    ~ref_sum_t() {
        const auto n = reorders_.size();
        for (size_t i = 0; i < n; ++i)
            delete reorders_[i];
    }

    virtual void execute(event_t *e) const {
        const auto n = reorders_.size();
        for (size_t i = 0; i < n; ++i) {
            event_t ei;
            reorders_[i]->execute(&ei);
        }
        e->set_state(event_t::ready);
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
    nstl::vector<primitive_t *> reorders_;
};

}
}
}

#endif
