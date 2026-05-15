// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gmock/gmock.h>

#include <memory>

#include "intel_npu/common/icapability.hpp"
#include "mock_sections.hpp"

constexpr double VALID_THRESHOLD = 10.0;

class MockCapability_1 : public ICapability {
public:
    explicit MockCapability_1(std::shared_ptr<MockSection_1> section)
        : ICapability(MockTypes::MOCK_1),
          m_section(std::move(section)) {}

    bool lazy_check_support() const override;

private:
    std::shared_ptr<MockSection_1> m_section;
};

class MockCapability_2 : public ICapability {
public:
    explicit MockCapability_2(std::shared_ptr<MockSection_2> section)
        : ICapability(MockTypes::MOCK_2),
          m_section(std::move(section)) {}

    bool lazy_check_support() const override;

private:
    std::shared_ptr<MockSection_2> m_section;
};

class MockCapability_3 : public ICapability {
public:
    explicit MockCapability_3(std::shared_ptr<MockSection_3> section)
        : ICapability(MockTypes::MOCK_3),
          m_section(std::move(section)) {}

    bool lazy_check_support() const override;

private:
    std::shared_ptr<MockSection_3> m_section;
};

class MockCapability : public ICapability {
public:
    explicit MockCapability(SectionType type) : ICapability(static_cast<CRE::Token>(type)) {}
    MOCK_METHOD(bool, lazy_check_support, (), (const, override));
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

class DriverCapability : public ICapability {
public:
    DriverCapability(SectionType type, const IDriver& driver)
        : ICapability(static_cast<CRE::Token>(type)),
          m_driver(driver),
          m_type(type) {}

    bool lazy_check_support() const override;

private:
    const IDriver& m_driver;
    SectionType m_type;
};
