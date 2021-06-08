// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include "frontend.hpp"
#include "frontend_manager_defs.hpp"

namespace ngraph
{
    namespace frontend
    {
        /// Capabilities for requested FrontEnd
        /// In general, frontend implementation may be divided into several libraries by capability
        /// level It will allow faster load of frontend when only limited usage is expected by
        /// client application as well as binary size can be minimized by removing not needed parts
        /// from application's package
        namespace FrontEndCapabilities
        {
            /// \brief Just reading and conversion, w/o any modifications; intended to be used in
            /// Reader
            static const int FEC_DEFAULT = 0;

            /// \brief Topology cutting capability
            static const int FEC_CUT = 1;

            /// \brief Query entities by names, renaming and adding new names for operations and
            /// tensors
            static const int FEC_NAMES = 2;

            /// \brief Partial model conversion and decoding capability
            static const int FEC_WILDCARDS = 4;
        }; // namespace FrontEndCapabilities

        // -------------- FrontEndManager -----------------
        using FrontEndCapFlags = int;
        using FrontEndFactory = std::function<FrontEnd::Ptr(FrontEndCapFlags fec)>;

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
            /// \param fec Frontend capabilities. It is recommended to use only
            /// those capabilities which are needed to minimize load time
            ///
            /// \return Frontend interface for further loading of models
            FrontEnd::Ptr
                load_by_framework(const std::string& framework,
                                  FrontEndCapFlags fec = FrontEndCapabilities::FEC_DEFAULT);

            /// \brief Loads frontend by model file path. Selects and loads appropriate frontend
            /// depending on model file extension and other file info (header)
            ///
            /// \param framework
            /// Framework name. Throws exception if name is not in list of available frontends
            ///
            /// \param fec Frontend capabilities. It is recommended to use only those capabilities
            /// which are needed to minimize load time
            ///
            /// \return Frontend interface for further loading of model
            FrontEnd::Ptr load_by_model(const std::string& path,
                                        FrontEndCapFlags fec = FrontEndCapabilities::FEC_DEFAULT);

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

} // namespace ngraph
