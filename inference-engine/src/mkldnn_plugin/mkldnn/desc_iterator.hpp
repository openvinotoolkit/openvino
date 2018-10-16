// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "mkldnn.hpp"

#include <string>
#include <mkldnn_types.h>
#include <mkldnn.h>

namespace mkldnn {

template <> struct handle_traits<mkldnn_primitive_desc_iterator_t> {
    static constexpr auto destructor = &mkldnn_primitive_desc_iterator_destroy;
};

struct primitive_desc_iterator : public handle<mkldnn_primitive_desc_iterator_t> {
    template <typename T>
    primitive_desc_iterator(const T &adesc, const engine &aengine) {
        mkldnn_primitive_desc_iterator_t result;
        error::wrap_c_api(mkldnn_primitive_desc_iterator_create(
                &result, &adesc.data, aengine.get(), nullptr),
                "could not create a primitive descriptor iterator");
        reset(result);
    }

    template <typename T>
    primitive_desc_iterator(const T &adesc, const mkldnn::primitive_attr &aattr, const engine &aengine) {
        mkldnn_primitive_desc_iterator_t result;
        error::wrap_c_api(mkldnn_primitive_desc_iterator_create_v2(
                &result, &adesc.data, aattr.get(), aengine.get(), nullptr),
                          "could not create a primitive descriptor iterator");
        reset(result);
    }

    template <typename T, typename TF>
    primitive_desc_iterator(const T &adesc, const engine &aengine, const TF &hint_fwd_primitive_desc) {
        mkldnn_primitive_desc_iterator_t result;
        error::wrap_c_api(mkldnn_primitive_desc_iterator_create(&result,
                        &adesc.data,
                        aengine.get(),
                        hint_fwd_primitive_desc.get()),
                "could not create a primitive descriptor iterator");
        reset(result);
    }

    template <typename T, typename TF>
    primitive_desc_iterator(const T &adesc, const mkldnn::primitive_attr &aattr,
            const engine &aengine, const TF &hint_fwd_primitive_desc) {
        mkldnn_primitive_desc_iterator_t result;
        error::wrap_c_api(mkldnn_primitive_desc_iterator_create_v2(&result,
                        &adesc.data,
                        aattr.get(),
                        aengine.get(),
                        hint_fwd_primitive_desc.get()),
                "could not create a primitive descriptor iterator");
        reset(result);
    }


    memory::primitive_desc fetch() const {
        memory::primitive_desc adesc;
        mkldnn_primitive_desc_t cdesc;

        cdesc = mkldnn_primitive_desc_iterator_fetch(get());

        adesc.reset(cdesc);
        return adesc;
    }

    bool next() {
        mkldnn_status_t status = mkldnn_primitive_desc_iterator_next(get());
        return status == mkldnn_status_t::mkldnn_success;
    }

    memory::primitive_desc src_primitive_desc(size_t index = 0) const {
        memory::primitive_desc adesc;
        memory::primitive_desc cdesc_elem;
        mkldnn_primitive_desc_t cdesc;
        cdesc_elem.reset(mkldnn_primitive_desc_iterator_fetch(get()));
        const_mkldnn_primitive_desc_t const_cdesc =
                mkldnn_primitive_desc_query_pd(cdesc_elem.get(),
                                               mkldnn::convert_to_c(src_pd), index);
        error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                          "could not clone a src primititve descriptor");
        adesc.reset(cdesc);
        return adesc;
    }

    memory::primitive_desc dst_primitive_desc(size_t index = 0) const {
        memory::primitive_desc adesc;
        memory::primitive_desc cdesc_elem;
        mkldnn_primitive_desc_t cdesc;
        cdesc_elem.reset(mkldnn_primitive_desc_iterator_fetch(get()));
        const_mkldnn_primitive_desc_t const_cdesc =
                mkldnn_primitive_desc_query_pd(cdesc_elem.get(),
                                               mkldnn::convert_to_c(dst_pd), index);
        error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                          "could not clone a dst primitive descriptor");
        adesc.reset(cdesc);
        return adesc;
    }


    memory::primitive_desc diff_src_primitive_desc(size_t index = 0) const {
        memory::primitive_desc adesc;
        memory::primitive_desc cdesc_elem;
        mkldnn_primitive_desc_t cdesc;
        cdesc_elem.reset(mkldnn_primitive_desc_iterator_fetch(get()));
        const_mkldnn_primitive_desc_t const_cdesc =
                mkldnn_primitive_desc_query_pd(cdesc_elem.get(),
                                               mkldnn::convert_to_c(diff_src_pd), index);
        error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                          "could not clone a diff_src primititve descriptor");
        adesc.reset(cdesc);
        return adesc;
    }

    memory::primitive_desc weights_primitive_desc(size_t index = 0) const {
        memory::primitive_desc adesc;
        memory::primitive_desc cdesc_elem;
        mkldnn_primitive_desc_t cdesc;
        cdesc_elem.reset(mkldnn_primitive_desc_iterator_fetch(get()));
        const_mkldnn_primitive_desc_t const_cdesc =
                mkldnn_primitive_desc_query_pd(cdesc_elem.get(),
                                               mkldnn::convert_to_c(weights_pd), index);
        error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                          "could not clone a weights primitive descriptor");
        adesc.reset(cdesc);
        return adesc;
    }

    memory::primitive_desc diff_dst_primitive_desc(size_t index = 0) const {
        memory::primitive_desc adesc;
        memory::primitive_desc cdesc_elem;
        mkldnn_primitive_desc_t cdesc;
        cdesc_elem.reset(mkldnn_primitive_desc_iterator_fetch(get()));
        const_mkldnn_primitive_desc_t const_cdesc =
                mkldnn_primitive_desc_query_pd(cdesc_elem.get(),
                                               mkldnn::convert_to_c(diff_dst_pd), index);
        error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                          "could not clone a diff_dst primitive descriptor");
        adesc.reset(cdesc);
        return adesc;
    }

    std::string get_impl_info_str() const {
        memory::primitive_desc cdesc_elem;
        cdesc_elem.reset(mkldnn_primitive_desc_iterator_fetch(get()));
        const char *info;
        error::wrap_c_api(mkldnn_primitive_desc_query(cdesc_elem.get(),
                        mkldnn::convert_to_c(impl_info_str), 0, &info),
                "could not query info string of primitive descriptor");
        return std::string(info);
    }

    template <typename T>
    void getPrimitiveDescriptor(T& pdesc) const {
        mkldnn_primitive_desc_t cdesc;

        memory::primitive_desc cdescpd;

        cdescpd.reset(mkldnn_primitive_desc_iterator_fetch(get()));
        error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, cdescpd.get()),
                          "could not clone a src primititve descriptor");
        pdesc.reset(cdesc);
    }
};

}  // namespace mkldnn
