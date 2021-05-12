// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include <mutex>
#include <unordered_map>

#include <itt.hpp>
#include <ngraph/node.hpp>

namespace ngraph
{
    namespace pass
    {
        class PerfCounters
        {
            PerfCounters(PerfCounters const&) = delete;
            PerfCounters& operator=(PerfCounters const&) = delete;

        public:
            PerfCounters() = default;

            openvino::itt::handle_t operator[](::ngraph::Node::type_info_t const& type_inf);

        private:
            using key = ::ngraph::Node::type_info_t const*;
            using value = openvino::itt::handle_t;
            using counters_map = std::unordered_map<key, value>;

            std::mutex m_mutex;
            counters_map m_counters;
        };
    } // namespace pass
} // namespace ngraph
