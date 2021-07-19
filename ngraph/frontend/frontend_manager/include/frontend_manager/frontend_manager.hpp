// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include "frontend.hpp"
#include "frontend_manager_defs.hpp"
#include "ngraph/variant.hpp"

namespace ngraph
{
    namespace frontend
    {
        // -------------- FrontEndManager -----------------
        using FrontEndFactory = std::function<FrontEnd::Ptr()>;

        /// \brief Frontend management class, loads available frontend plugins on construction
        /// Allows load of frontends for particular framework, register new and list available
        /// frontends This is a main frontend entry point for client applications
        class FRONTEND_API FrontEndManager final
        {
        public:
            /// \brief Default constructor. Searches and loads of available frontends
            FrontEndManager();

            /// \brief Default move constructor
            FrontEndManager(FrontEndManager&&);

            /// \brief Default move assignment operator
            FrontEndManager& operator=(FrontEndManager&&);

            /// \brief Default destructor
            ~FrontEndManager();

            /// \brief Loads frontend by name of framework and capabilities
            ///
            /// \param framework Framework name. Throws exception if name is not in list of
            /// available frontends
            ///
            /// \return Frontend interface for further loading of models
            FrontEnd::Ptr load_by_framework(const std::string& framework);

            /// \brief Loads frontend by model fragments described by each FrontEnd documentation.
            /// Selects and loads appropriate frontend depending on model file extension and other
            /// file info (header)
            ///
            /// \param framework
            /// Framework name. Throws exception if name is not in list of available frontends
            ///
            /// \return Frontend interface for further loading of model
            template <typename... Types>
            FrontEnd::Ptr load_by_model(const Types&... vars)
            {
                return load_by_model_impl({make_variant(vars)...});
            }

            /// \brief Gets list of registered frontends
            std::vector<std::string> get_available_front_ends() const;

            /// \brief Register frontend with name and factory creation method
            ///
            /// \param name Name of front end
            ///
            /// \param creator Creation factory callback. Will be called when frontend is about to
            /// be created
            void register_front_end(const std::string& name, FrontEndFactory creator);

        private:
            class Impl;

            FrontEnd::Ptr load_by_model_impl(const std::vector<std::shared_ptr<Variant>>& variants);

            std::unique_ptr<Impl> m_impl;
        };

        // --------- Plugin exporting information --------------

        /// \brief Each frontend plugin is responsible to export GetAPIVersion function returning
        /// version of frontend API used for this plugin
        /// If version is not matched with OV_FRONTEND_API_VERSION - plugin will not be loaded by
        /// FrontEndManager
        using FrontEndVersion = uint64_t;

        /// \brief Each frontend plugin is responsible to export GetFrontEndData function returning
        /// heap-allocated pointer to this structure. Will be used by FrontEndManager during loading
        /// of plugins
        struct FrontEndPluginInfo
        {
            std::string m_name;
            FrontEndFactory m_creator;
        };

    } // namespace frontend

    template <>
    class FRONTEND_API VariantWrapper<std::shared_ptr<std::istream>>
        : public VariantImpl<std::shared_ptr<std::istream>>
    {
    public:
        static constexpr VariantTypeInfo type_info{"Variant::std::shared_ptr<std::istream>", 0};
        const VariantTypeInfo& get_type_info() const override { return type_info; }
        VariantWrapper(const value_type& value)
            : VariantImpl<value_type>(value)
        {
        }
    };

#if defined(ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    template <>
    class FRONTEND_API VariantWrapper<std::wstring> : public VariantImpl<std::wstring>
    {
    public:
        static constexpr VariantTypeInfo type_info{"Variant::std::wstring", 0};
        const VariantTypeInfo& get_type_info() const override { return type_info; }
        VariantWrapper(const value_type& value)
            : VariantImpl<value_type>(value)
        {
        }
    };
#endif

} // namespace ngraph
