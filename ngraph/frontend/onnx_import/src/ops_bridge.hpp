// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>

#include "ngraph/except.hpp"
#include "onnx_import/core/operator_set.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace error
        {
            struct UnknownOperator : ngraph_error
            {
                UnknownOperator(const std::string& name, const std::string& domain)
                    : ngraph_error{(domain.empty() ? "" : domain + ".") + name}
                {
                }
            };

            struct UnknownDomain : ngraph_error
            {
                explicit UnknownDomain(const std::string& domain)
                    : ngraph_error{domain}
                {
                }
            };

            struct UnsupportedVersion : ngraph_error
            {
                UnsupportedVersion(const std::string& name,
                                   std::int64_t version,
                                   const std::string& domain)
                    : ngraph_error{
                          "Unsupported operator version: " + (domain.empty() ? "" : domain + ".") +
                          name + ":" + std::to_string(version)}
                {
                }
            };

        } // namespace error

        class OperatorsBridge
        {
        public:
            static constexpr const int LATEST_SUPPORTED_ONNX_OPSET_VERSION = ONNX_OPSET_VERSION;

            OperatorsBridge(const OperatorsBridge&) = delete;
            OperatorsBridge& operator=(const OperatorsBridge&) = delete;
            OperatorsBridge(OperatorsBridge&&) = delete;
            OperatorsBridge& operator=(OperatorsBridge&&) = delete;

            static OperatorSet get_operator_set(const std::string& domain,
                                                std::int64_t version = -1)
            {
                return instance()._get_operator_set(domain, version);
            }

            static void register_operator(const std::string& name,
                                          std::int64_t version,
                                          const std::string& domain,
                                          Operator fn)
            {
                instance()._register_operator(name, version, domain, std::move(fn));
            }

            static void unregister_operator(const std::string& name,
                                            std::int64_t version,
                                            const std::string& domain)
            {
                instance()._unregister_operator(name, version, domain);
            }

            static bool is_operator_registered(const std::string& name,
                                               std::int64_t version,
                                               const std::string& domain)
            {
                return instance()._is_operator_registered(name, version, domain);
            }

        private:
            // Registered operators structure
            // {
            //    domain_1: {
            //      op_type_1: {
            //          version_1: {func_handle},
            //          version_2: {func_handle},
            //          ...
            //      },
            //      op_type_2: { ... }
            //      ...
            //    },
            //    domain_2: { ... },
            //    ...
            // }
            std::unordered_map<std::string,
                               std::unordered_map<std::string, std::map<std::int64_t, Operator>>>
                m_map;

            OperatorsBridge();

            static OperatorsBridge& instance()
            {
                static OperatorsBridge instance;
                return instance;
            }

            void _register_operator(const std::string& name,
                                    std::int64_t version,
                                    const std::string& domain,
                                    Operator fn);
            void _unregister_operator(const std::string& name,
                                      std::int64_t version,
                                      const std::string& domain);
            OperatorSet _get_operator_set(const std::string& domain, std::int64_t version);

            bool _is_operator_registered(const std::string& name,
                                         std::int64_t version,
                                         const std::string& domain);

            std::mutex lock;
        };

        const std::string OPENVINO_ONNX_DOMAIN = "org.openvinotoolkit";

    } // namespace onnx_import

} // namespace ngraph
