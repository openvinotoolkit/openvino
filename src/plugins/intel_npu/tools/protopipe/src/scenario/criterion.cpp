//
// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "criterion.hpp"

#include <chrono>
#include <cstdint>
#include <ostream>
#include "utils/logger.hpp"

#include "utils/utils.hpp"

void ITermCriterion::setWorkloadTrigger(std::shared_ptr<WorkloadTypeInfo> workload_ptr) {
    workload_type = workload_ptr;
}
void ITermCriterion::checkWorkloadTrigger() {
    if (workload_type) {
        if (customCheck(workload_type->workload_config.change_interval)) {
            // update based on changes
            uint64_t next_index = workload_index % workload_type->workload_config.change_to.size();
            if (!workload_type->workload_config.repeat &&
                workload_index >= workload_type->workload_config.change_to.size()) {
                return;
            }
            std::string next_value = workload_type->workload_config.change_to[next_index];
            workload_index++;
            LOG_INFO() << "Update workload type to " << next_value << " after " << workload_index * workload_type->workload_config.change_interval << " seconds/iterations" << std::endl;

            workload_type->wl_onnx->set(next_value);
            workload_type->wl_ov->set(next_value);
        }
    }
}


Iterations::Iterations(uint64_t num_iters): m_num_iters(num_iters), m_counter(0) {
}

bool Iterations::check() const {
    return m_counter != m_num_iters;
}

bool Iterations::customCheck(uint64_t value) {
    return m_counter != 0 && m_counter % value == 0;
}

void Iterations::update() {
    ++m_counter;
}

void Iterations::init() {
    m_counter = 0;
}

ITermCriterion::Ptr Iterations::clone() const {
    return std::make_shared<Iterations>(*this);
}

TimeOut::TimeOut(uint64_t time_in_us): m_time_in_us(time_in_us), m_start_ts(-1) {
}

bool TimeOut::check() const {
    return utils::timestamp<std::chrono::microseconds>() - m_start_ts < m_time_in_us;
}

bool TimeOut::customCheck(uint64_t value) {
    auto now = utils::timestamp<std::chrono::microseconds>() - m_start_ts;
    if (now - last_update >= (value * 1'000'000)) {
        last_update = now;
        return true;
    }
    return false;
}

void TimeOut::update(){/* do nothing */};

void TimeOut::init() {
    m_start_ts = utils::timestamp<std::chrono::microseconds>();
}

ITermCriterion::Ptr TimeOut::clone() const {
    return std::make_shared<TimeOut>(*this);
}

CombinedCriterion::CombinedCriterion(ITermCriterion::Ptr lhs, ITermCriterion::Ptr rhs): m_lhs(lhs), m_rhs(rhs) {
}

CombinedCriterion::CombinedCriterion(const CombinedCriterion& other) {
    m_lhs = other.m_lhs->clone();
    m_rhs = other.m_rhs->clone();
}

bool CombinedCriterion::check() const {
    return m_lhs->check() && m_rhs->check();
}

bool CombinedCriterion::customCheck(uint64_t value) {
    return m_lhs->customCheck(value) && m_rhs->customCheck(value);
}

void CombinedCriterion::update() {
    m_lhs->update();
    m_rhs->update();
};

void CombinedCriterion::init() {
    m_lhs->init();
    m_rhs->init();
}

ITermCriterion::Ptr CombinedCriterion::clone() const {
    return std::make_shared<CombinedCriterion>(*this);
}
