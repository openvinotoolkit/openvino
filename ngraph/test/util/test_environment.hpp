// Copyright (C) 2019-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <iostream>
#include <map>
#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"

namespace ngraph
{
    namespace test
    {
        class TestEnvironment;

        class Summary;

        class SummaryDestroyer
        {
        private:
            Summary* p_instance;

        public:
            ~SummaryDestroyer();

            void initialize(Summary* p);
        };

        struct PassRate
        {
            enum Statuses
            {
                PASSED,
                FAILED,
                SKIPPED
            };
            unsigned long passed = 0;
            unsigned long failed = 0;
            unsigned long skipped = 0;

            PassRate() = default;

            PassRate(unsigned long p, unsigned long f, unsigned long s)
            {
                passed = p;
                failed = f;
                skipped = s;
            }

            float getPassrate() const
            {
                if (passed == 0 && failed == 0)
                {
                    return 0.;
                }
                else if (passed != 0 && failed == 0)
                {
                    return 100.;
                }
                else
                {
                    return (passed / (passed + failed)) * 100.;
                }
            }
        };

        class Summary
        {
        private:
            static Summary* p_instance;
            static SummaryDestroyer destroyer;
            std::map<ngraph::NodeTypeInfo, PassRate> opsStats = {};
            std::string deviceName;

        protected:
            Summary() = default;

            Summary(const Summary&);

            Summary& operator=(Summary&);

            ~Summary() = default;

            std::map<ngraph::NodeTypeInfo, PassRate> getOPsStats() { return opsStats; }
            std::string getDeviceName() const { return deviceName; }
            friend class SummaryDestroyer;

            friend class TestEnvironment;

        public:
            void updateOPsStats(ngraph::NodeTypeInfo op, PassRate::Statuses status);

            void setDeviceName(std::string device) { deviceName = device; }
            static Summary& getInstance();
        };

        class TestEnvironment : public ::testing::Environment
        {
        public:
            void TearDown() override;

        private:
            std::string reportFileName = "report.xml";
        };
    }
}