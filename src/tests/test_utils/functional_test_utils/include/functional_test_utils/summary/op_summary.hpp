// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/openvino.hpp"
#include "openvino/opsets/opset.hpp"
#include "openvino/opsets/opset10.hpp"
#include "summary.hpp"

namespace ov {
namespace test {
namespace utils {

class OpSummary;

class OpSummaryDestroyer {
private:
    OpSummary* p_instance;

public:
    ~OpSummaryDestroyer();

    void initialize(OpSummary* p);
};

class OpSummary : public virtual Summary {
private:
    static OpSummary* p_instance;
    static bool extractBody;
    std::map<ov::NodeTypeInfo, PassRate> opsStats = {};
    unsigned short int downgrade_coefficient;

    std::string get_opset_number(const std::string& opset_full_name);

protected:
    OpSummary(unsigned short int downgrade_coefficient = 1);
    static OpSummary& createInstance(unsigned short int downgrade_coefficient = 1);
    static OpSummaryDestroyer destroyer;
    friend class OpSummaryDestroyer;

public:
    static OpSummary& getInstance();
    static void setDowngradeCoefficient(unsigned short int downgrade_coefficient = 1);

    std::map<ov::NodeTypeInfo, PassRate> getOPsStats() {
        return opsStats;
    }

    static void setExtractBody(bool val) {
        extractBody = val;
    }
    static bool getExtractBody() {
        return extractBody;
    }

    std::map<std::string, PassRate> getStatisticFromReport();
    void saveReport() override;

    void updateOPsStats(const std::shared_ptr<ov::Model>& model,
                        const PassRate::Statuses& status,
                        double rel_influence_coef = 1);
    void updateOPsImplStatus(const std::shared_ptr<ov::Model>& model, const bool implStatus);

    void updateOPsStats(const ov::NodeTypeInfo& op, const PassRate::Statuses& status, double rel_influence_coef = 1);
    void updateOPsImplStatus(const ov::NodeTypeInfo& op, const bool implStatus);
};

}  // namespace utils
}  // namespace test
}  // namespace ov
