//
// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "criterion.hpp"

#include <chrono>

#include "utils/utils.hpp"

Iterations::Iterations(uint64_t num_iters): m_num_iters(num_iters), m_counter(0) {
}

bool Iterations::check() const {
    return m_counter != m_num_iters;
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
