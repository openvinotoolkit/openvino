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
                    return 100.f * passed / (passed + failed);
                }
            }
        };

        class Summary
        {
        private:
            static Summary* p_instance;
            static SummaryDestroyer destroyer;
            std::map<ngraph::NodeTypeInfo, PassRate> opsStats;
            std::string deviceName;

            Summary() = default;

            Summary(const Summary&) = delete;
            Summary& operator=(const Summary&) = delete;
            Summary(Summary&&) = delete;
            Summary& operator=(Summary&&) = delete;

            ~Summary() = default;

            friend class SummaryDestroyer;

        public:
            const std::map<ngraph::NodeTypeInfo, PassRate>& getOPsStats() const { return opsStats; }

            const std::string& getDeviceName() const { return deviceName; }

            void updateOPsStats(const std::shared_ptr<ngraph::Function>& function,
                                PassRate::Statuses status);
            void updateOPsStats(const std::shared_ptr<ngraph::Node>& node,
                                PassRate::Statuses status);
            void updateOPsStats(ngraph::NodeTypeInfo op, PassRate::Statuses status);

            void setDeviceName(std::string device) { deviceName = device; }
            static Summary& getInstance();
        };

        class TestEnvironment : public ::testing::Environment
        {
        public:
            void TearDown() override;

        private:
            void writeReport(const std::string& reportFileName = "report.xml") const;
        };

        inline const ::testing::TestResult* current_test_result()
        {
            return ::testing::UnitTest::GetInstance()->current_test_info()->result();
        }

        inline PassRate::Statuses
            summary_status(const ::testing::TestResult* result = current_test_result())
        {
            if (!result)
            {
                return PassRate::SKIPPED;
            }
            if (result->Failed())
            {
                return PassRate::FAILED;
            }
            return PassRate::PASSED;
        }

        class FunctionReporter
        {
        public:
            FunctionReporter(std::shared_ptr<ngraph::Function> function)
                : function(function)
            {
            }
            void operator()() const
            {
                Summary::getInstance().updateOPsStats(function, summary_status());
            }

        private:
            std::shared_ptr<ngraph::Function> function;
        };

        class NodeReporter
        {
        public:
            NodeReporter(std::shared_ptr<ngraph::Node> node)
                : node(std::move(node))
            {
            }
            void operator()() const
            {
                Summary::getInstance().updateOPsStats(node, summary_status());
            }

        private:
            std::shared_ptr<ngraph::Node> node;
        };

        inline FunctionReporter reporter(std::shared_ptr<ngraph::Function> f)
        {
            return {std::move(f)};
        }

        inline NodeReporter reporter(std::shared_ptr<ngraph::op::Op> op) { return {std::move(op)}; }

        //Idea copy from https://github.com/microsoft/GSL
        template <typename F>
        class Finally
        {
        public:
            Finally(F f)
                : f{std::move(f)}
            {
            }
            Finally(const Finally&) = delete;
            Finally& operator=(const Finally&) = delete;
            Finally(Finally&& other)
                : f(std::move(other.f))
            {
                other.empty = true;
            }
            Finally& operator=(Finally&& other) = delete;
            ~Finally()
            {
                if (!empty)
                {
                    f();
                }
            }

        private:
            bool empty{false};
            F f;
        };

        template <typename F>
        inline Finally<F> finally(F f)
        {
            return Finally<F>(std::move(f));
        }

    } // namespace test
} // namespace ngraph

#define CONCATENATE2(S, N) S##N
#define CONCATENATE(S, N) CONCATENATE2(S, N)
#define REPORT(NODE)                                                                               \
    const auto CONCATENATE(CALL_FINALLY___LINE_, __LINE__) =                                       \
        ::ngraph::test::finally(::ngraph::test::reporter(NODE))
