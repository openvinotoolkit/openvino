// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gmock/gmock.h>

#include <memory>

#include "intel_npu/common/isection_type_evaluator.hpp"
#include "mock_sections.hpp"

constexpr double VALID_THRESHOLD = 10.0;

class MockCapability_1 : public ISectionTypeEvaluator {
public:
    explicit MockCapability_1(std::shared_ptr<MockSection_1> section)
        : ISectionTypeEvaluator(MockTypes::MOCK_1),
          m_section(std::move(section)) {}

private:
    bool evaluate() const override;

    std::shared_ptr<MockSection_1> m_section;
};

class MockCapability_2 : public ISectionTypeEvaluator {
public:
    explicit MockCapability_2(std::shared_ptr<MockSection_2> section)
        : ISectionTypeEvaluator(MockTypes::MOCK_2),
          m_section(std::move(section)) {}

private:
    bool evaluate() const override;

    std::shared_ptr<MockSection_2> m_section;
};

class MockCapability_3 : public ISectionTypeEvaluator {
public:
    explicit MockCapability_3(std::shared_ptr<MockSection_3> section)
        : ISectionTypeEvaluator(MockTypes::MOCK_3),
          m_section(std::move(section)) {}

private:
    bool evaluate() const override;

    std::shared_ptr<MockSection_3> m_section;
};

// TODO rename these as well
class MockCapability : public ISectionTypeEvaluator {
public:
    explicit MockCapability(SectionType type) : ISectionTypeEvaluator(static_cast<CREToken>(type)) {}
    MOCK_METHOD(bool, evaluate, (), (const, override));
};

// mocking the potential driver query to verify if a capability is supported
class IDriver {
public:
    virtual ~IDriver() = default;
    virtual bool supports_section(SectionType type) const = 0;
};

class MockDriver : public IDriver {
public:
    MOCK_METHOD(bool, supports_section, (SectionType type), (const, override));
};

class DriverCapability : public ISectionTypeEvaluator {
public:
    DriverCapability(SectionType type, const IDriver& driver)
        : ISectionTypeEvaluator(static_cast<CREToken>(type)),
          m_driver(driver),
          m_type(type) {}

private:
    bool evaluate() const override;

    const IDriver& m_driver;
    SectionType m_type;
};
