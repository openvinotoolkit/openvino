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

#ifndef REF_CONCAT_HPP
#define REF_CONCAT_HPP

#include "cpu_concat.hpp"
#include "reorder_pd.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

struct ref_concat_t: public cpu_primitive_t {
    using cpu_memory_pd_t = cpu_memory_t::pd_t;

    struct pd_t: public cpu_concat_pd_t {
        pd_t(const memory_desc_t *output_d, int n, int concat_dim,
                const cpu_memory_pd_t **input_pds, const primitive_attr_t *attr)
            : cpu_concat_pd_t(output_d, n, concat_dim, input_pds, attr) {}
        pd_t(const pd_t &rhs)
            : cpu_concat_pd_t(rhs)
        {
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

        static status_t create(concat_pd_t **concat_pd,
                const memory_desc_t *output_d, int n, int concat_dim,
                const memory_pd_t **input_pds, const primitive_attr_t *attr) {
            auto _pd = new pd_t(output_d, n, concat_dim,
                    (const cpu_memory_pd_t **)input_pds, attr);
            if (_pd == nullptr) return out_of_memory;
            if (_pd->init() != success) { delete _pd; return unimplemented; }
            return safe_ptr_assign<concat_pd_t>(*concat_pd, _pd);
        }
        virtual status_t create_primitive(primitive_t **primitive,
                const primitive_at_t *inputs,
                const primitive_t **outputs) const override {
            double ms = get_msec();
            auto n = n_inputs();
            nstl::vector<primitive_t *> reorders;
            reorders.resize(n);
            for (int i = 0; i < n; ++i) {
                CHECK(reorder_pds_[i]->create_primitive(&reorders[i],
                        &inputs[i], outputs));
            }
            primitive_t::input_vector ins(inputs, inputs + n_);
            primitive_t::output_vector outs(outputs, outputs + 1);
            auto ret = safe_ptr_assign<primitive_t>(*primitive,
                    new ref_concat_t(this, ins, outs, reorders));
            ms = get_msec() - ms;
            if (mkldnn_verbose()->level >= 2) {
                printf("mkldnn_verbose,create,%s,%g\n", this->info(), ms);
                fflush(0);
            }
            return ret;
        }
        virtual pd_t *clone() const override { return  new pd_t(*this); }
        virtual const char *name() const override { return "ref:any"; }

        virtual status_t init() override {
            assert(engine()->kind() == engine_kind::cpu);

            bool ok = cpu_concat_pd_t::init() == success;
            if (!ok) return unimplemented;

            for (int i = 0; i < n_; ++i) {
                auto r_impls = engine_->get_reorder_implementation_list();
                for (auto r = r_impls; *r; ++r) {
                    const primitive_attr_t dummy_attr; /* alpha == 1. */
                    reorder_pd_t *r_pd;
                    if ((*r)(&r_pd, &src_pds_[i], &src_image_pds_[i],
                                &dummy_attr) == status::success) {
                        r_pd->init_info();
                        reorder_pds_.push_back(r_pd);
                        break;
                    }
                }
            }
            return (size_t)n_ == reorder_pds_.size() ? success : unimplemented;
        }

        nstl::vector<const reorder_pd_t *> reorder_pds_;
    };

    ref_concat_t(const pd_t *apd, const input_vector &inputs,
            const output_vector &outputs, nstl::vector<primitive_t *> reorders)
        : cpu_primitive_t(apd, inputs, outputs),
        reorders_(reorders) {}

    ~ref_concat_t() {
        const auto n = reorders_.size();
        for (size_t i = 0; i < n; ++i)
            delete reorders_[i];
    }

    virtual void execute(event_t *e) const {
        for (size_t i = 0; i < reorders_.size(); ++i) {
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
