//
// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>

#include "simulation/workload_type.hpp"
struct ITermCriterion {
    using Ptr = std::shared_ptr<ITermCriterion>;
    virtual void init() = 0;
    virtual void update() = 0;
    virtual bool check() const = 0;
    virtual bool customCheck(uint64_t value) = 0;
    virtual ITermCriterion::Ptr clone() const = 0;
    void setWorkloadTrigger(std::shared_ptr<WorkloadTypeInfo>);
    void checkWorkloadTrigger();
    std::shared_ptr<WorkloadTypeInfo> workload_type;
    uint64_t workload_index = 0;
};

class Iterations : public ITermCriterion {
public:
    Iterations(uint64_t num_iters);

    void init() override;
    void update() override;
    bool check() const override;
    bool customCheck(uint64_t value) override;
    ITermCriterion::Ptr clone() const override;

private:
    uint64_t m_num_iters;
    uint64_t m_counter;
};

class TimeOut : public ITermCriterion {
public:
    TimeOut(uint64_t time_in_us);

    void init() override;
    void update() override;
    bool check() const override;
    bool customCheck(uint64_t value) override;

    ITermCriterion::Ptr clone() const override;

private:
    uint64_t m_time_in_us;
    uint64_t m_start_ts;
    uint64_t last_update = 0;
};

class CombinedCriterion : public ITermCriterion {
public:
    CombinedCriterion(ITermCriterion::Ptr lhs, ITermCriterion::Ptr rhs);
    CombinedCriterion(const CombinedCriterion& other);

    void init() override;
    void update() override;
    bool check() const override;
    bool customCheck(uint64_t value) override;
    ITermCriterion::Ptr clone() const override;

private:
    ITermCriterion::Ptr m_lhs, m_rhs;
};
