// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "summary.hpp"

namespace ov {
namespace test {
namespace utils {

enum class ov_entity { ov_plugin, ov_compiled_model, ov_infer_request, undefined };

class ApiSummary;

class ApiSummaryDestroyer {
private:
    ApiSummary* p_instance;

public:
    ~ApiSummaryDestroyer();

    void initialize(ApiSummary* p);
};

class ApiSummary : public virtual Summary {
private:
    static ApiSummary* p_instance;
    std::map<ov_entity, std::map<std::string, PassRate>> apiStats;
    static const std::map<ov_entity, std::string> apiInfo;
    ov_entity getOvEntityByName(const std::string& name);

protected:
    ApiSummary();
    static ApiSummaryDestroyer destroyer;
    friend class ApiSummaryDestroyer;

public:
    static ApiSummary& getInstance();
    inline void getStatisticFromReport(const std::string& filePath);
    std::map<ov_entity, std::map<std::string, PassRate>> getApiStats() {
        return apiStats;
    }
    void updateStat(ov_entity, const std::string& device, PassRate::Statuses, double rel_influence_coef = 1);
    void saveReport() override;
};

}  // namespace utils
}  // namespace test
}  // namespace ov
