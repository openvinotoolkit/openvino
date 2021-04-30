// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/except.hpp>
#include <ngraph/env_util.hpp>

#include "frontend_manager/frontend_manager.hpp"
#include "plugin_loader.hpp"

namespace ngraph
{
    namespace frontend
    {

#define FRONT_END_NOT_IMPLEMENTED(NAME) throw std::runtime_error(#NAME " is not implemented for this FrontEnd class")
#define FRONT_END_ASSERT(EXPRESSION) \
        { if (!(EXPRESSION)) throw "AssertionFailed"; }

        //----------- FrontEndManager ---------------------------
        class FrontEndManager::Impl
        {
            std::vector<PluginHandle> m_loadedLibs; // must be a first class member (destroyed last)
            std::map<std::string, FrontEndFactory> m_factories;
        public:
            Impl()
            {
                registerPlugins();
            }

            ~Impl() = default;

            FrontEnd::Ptr loadByFramework(const std::string& framework, FrontEndCapabilities fec)
            {
                FRONT_END_ASSERT(m_factories.count(framework))
                return m_factories[framework](fec);
            }

            std::vector<std::string> availableFrontEnds() const
            {
                std::vector<std::string> keys;

                std::transform(m_factories.begin(), m_factories.end(),
                               std::back_inserter(keys),
                               [](const std::pair<std::string, FrontEndFactory>& item)
                               {
                                   return item.first;
                               });
                return keys;
            }

            FrontEnd::Ptr loadByModel(const std::string& path, FrontEndCapabilities fec)
            {
                FRONT_END_NOT_IMPLEMENTED(loadByModel);
            }

            void registerFrontEnd(const std::string& name, FrontEndFactory creator)
            {
                m_factories.insert({name, creator});
            }

        private:
            void registerPlugins()
            {
                auto registerFromDir = [&](const std::string& dir)
                {
                    if (!dir.empty())
                    {
                        auto plugins = loadPlugins(dir);
                        for (auto& plugin : plugins)
                        {
                            registerFrontEnd(plugin.m_pluginInfo.m_name, plugin.m_pluginInfo.m_creator);
                            m_loadedLibs.push_back(std::move(plugin.m_libHandle));
                        }
                    }
                };
                std::string envPath = ngraph::getenv_string("OV_FRONTEND_PATH");
                if (!envPath.empty())
                {
                    auto start = 0u;
                    auto sepPos = envPath.find(PathSeparator, start);
                    while (sepPos != std::string::npos)
                    {
                        registerFromDir(envPath.substr(start, sepPos - start));
                        start = sepPos + 1;
                        sepPos = envPath.find(PathSeparator, start);
                    }
                    registerFromDir(envPath.substr(start, sepPos));
                }
            }
        };

        FrontEndManager::FrontEndManager() : m_impl(new Impl())
        {
        }

        FrontEndManager::~FrontEndManager() = default;

        FrontEnd::Ptr FrontEndManager::loadByFramework(const std::string& framework, FrontEndCapabilities fec)
        {
            return m_impl->loadByFramework(framework, fec);
        }

        FrontEnd::Ptr FrontEndManager::loadByModel(const std::string& path, FrontEndCapabilities fec)
        {
            return m_impl->loadByModel(path, fec);
        }

        std::vector<std::string> FrontEndManager::availableFrontEnds() const
        {
            return m_impl->availableFrontEnds();
        }

        void FrontEndManager::registerFrontEnd(const std::string& name, FrontEndFactory creator)
        {
            m_impl->registerFrontEnd(name, creator);
        }

        //----------- FrontEnd ---------------------------

        FrontEnd::FrontEnd() = default;

        FrontEnd::~FrontEnd() = default;

        InputModel::Ptr FrontEnd::loadFromFile(const std::string& paths) const
        {
            FRONT_END_NOT_IMPLEMENTED(loadFromFile);
        }

        InputModel::Ptr FrontEnd::loadFromFiles(const std::vector<std::string>& paths) const
        {
            FRONT_END_NOT_IMPLEMENTED(loadFromFiles);
        }

        InputModel::Ptr FrontEnd::loadFromMemory(const void *model) const
        {
            FRONT_END_NOT_IMPLEMENTED(loadFromMemory);
        }

        InputModel::Ptr FrontEnd::loadFromMemoryFragments(const std::vector<const void *>& modelParts) const
        {
            FRONT_END_NOT_IMPLEMENTED(loadFromMemoryFragments);
        }

        InputModel::Ptr FrontEnd::loadFromStream(std::istream& path) const
        {
            FRONT_END_NOT_IMPLEMENTED(loadFromStream);
        }

        InputModel::Ptr FrontEnd::loadFromStreams(const std::vector<std::istream *>& paths) const
        {
            FRONT_END_NOT_IMPLEMENTED(loadFromStreams);
        }

        std::shared_ptr<ngraph::Function> FrontEnd::convert(InputModel::Ptr model) const
        {
            FRONT_END_NOT_IMPLEMENTED(convert);
        }

        std::shared_ptr<ngraph::Function> FrontEnd::convert(std::shared_ptr<ngraph::Function>) const
        {
            FRONT_END_NOT_IMPLEMENTED(convert);
        }

        std::shared_ptr<ngraph::Function> FrontEnd::convertPartially(InputModel::Ptr model) const
        {
            FRONT_END_NOT_IMPLEMENTED(convertPartially);
        }

        std::shared_ptr<ngraph::Function> FrontEnd::decode(InputModel::Ptr model) const
        {
            FRONT_END_NOT_IMPLEMENTED(convertDecodingOnly);
        }

        void FrontEnd::normalize(std::shared_ptr<ngraph::Function> function) const
        {
            FRONT_END_NOT_IMPLEMENTED(normalize);
        }

        //----------- InputModel ---------------------------
        std::vector<Place::Ptr> InputModel::getInputs() const
        {
            FRONT_END_NOT_IMPLEMENTED(getInputs);
        }

        std::vector<Place::Ptr> InputModel::getOutputs() const
        {
            FRONT_END_NOT_IMPLEMENTED(getOutputs);
        }

        Place::Ptr InputModel::getPlaceByTensorName(const std::string& tensorName) const
        {
            FRONT_END_NOT_IMPLEMENTED(getPlaceByTensorName);
        }

        Place::Ptr InputModel::getPlaceByOperationName(const std::string& operationName)
        {
            FRONT_END_NOT_IMPLEMENTED(getPlaceByOperationName);
        }

        Place::Ptr InputModel::getPlaceByOperationAndInputPort(const std::string& operationName, int inputPortIndex)
        {
            FRONT_END_NOT_IMPLEMENTED(getPlaceByOperationAndInputPort);
        }

        Place::Ptr InputModel::getPlaceByOperationAndOutputPort(const std::string& operationName, int outputPortIndex)
        {
            FRONT_END_NOT_IMPLEMENTED(getPlaceByOperationAndOutputPort);
        }

        void InputModel::setNameForTensor(Place::Ptr tensor, const std::string& newName)
        {
            FRONT_END_NOT_IMPLEMENTED(setNameForTensor);
        }

        void InputModel::addNameForTensor(Place::Ptr tensor, const std::string& newName)
        {
            FRONT_END_NOT_IMPLEMENTED(addNameForTensor);
        }

        void InputModel::setNameForOperation(Place::Ptr operation, const std::string& newName)
        {
            FRONT_END_NOT_IMPLEMENTED(setNameForOperation);
        }

        void InputModel::freeNameForTensor(const std::string& name)
        {
            FRONT_END_NOT_IMPLEMENTED(freeNameForTensor);
        }

        void InputModel::freeNameForOperation(const std::string& name)
        {
            FRONT_END_NOT_IMPLEMENTED(freeNameForOperation);
        }

        void InputModel::setNameForDimension(Place::Ptr place, size_t shapeDimIndex, const std::string& dimName)
        {
            FRONT_END_NOT_IMPLEMENTED(setNameForDimension);
        }

        void InputModel::cutAndAddNewInput(Place::Ptr place, const std::string& newNameOptional)
        {
            FRONT_END_NOT_IMPLEMENTED(cutAndAddNewInput);
        }

        void InputModel::cutAndAddNewOutput(Place::Ptr place, const std::string& newNameOptional)
        {
            FRONT_END_NOT_IMPLEMENTED(cutAndAddNewOutput);
        }

        Place::Ptr InputModel::addOutput(Place::Ptr place)
        {
            FRONT_END_NOT_IMPLEMENTED(addOutput);
        }

        void InputModel::removeOutput(Place::Ptr place)
        {
            FRONT_END_NOT_IMPLEMENTED(removeOutput);
        }

        void InputModel::removeInput(Place::Ptr place)
        {
            FRONT_END_NOT_IMPLEMENTED(removeInput);
        }

        void InputModel::overrideAllOutputs(const std::vector<Place::Ptr>& outputs)
        {
            FRONT_END_NOT_IMPLEMENTED(overrideAllOutputs);
        }

        void InputModel::overrideAllInputs(const std::vector<Place::Ptr>& inputs)
        {
            FRONT_END_NOT_IMPLEMENTED(overrideAllInputs);
        }

        void
        InputModel::extractSubgraph(const std::vector<Place::Ptr>& inputs, const std::vector<Place::Ptr>& outputs)
        {
            FRONT_END_NOT_IMPLEMENTED(extractSubgraph);
        }

        // Setting tensor properties
        void InputModel::setDefaultShape(Place::Ptr place, const ngraph::Shape&)
        {
            FRONT_END_NOT_IMPLEMENTED(setDefaultShape);
        }

        void InputModel::setPartialShape(Place::Ptr place, const ngraph::PartialShape&)
        {
            FRONT_END_NOT_IMPLEMENTED(setPartialShape);
        }

        ngraph::PartialShape InputModel::getPartialShape(Place::Ptr place) const
        {
            FRONT_END_NOT_IMPLEMENTED(setPartialShape);
        }

        void InputModel::setElementType(Place::Ptr place, const ngraph::element::Type&)
        {
            FRONT_END_NOT_IMPLEMENTED(setElementType);
        }

        void InputModel::setTensorValue(Place::Ptr place, const void *)
        {
            FRONT_END_NOT_IMPLEMENTED(setTensorValue);
        }

        void InputModel::setTensorPartialValue(Place::Ptr place, const void *minValue, const void *maxValue)
        {
            FRONT_END_NOT_IMPLEMENTED(setTensorPartialValue);
        }

        //----------- Place ---------------------------
        std::vector<std::string> Place::getNames() const
        {
            FRONT_END_NOT_IMPLEMENTED(getNames);
        }

        std::vector<Place::Ptr> Place::getConsumingOperations(int outputPortIndex) const
        {
            FRONT_END_NOT_IMPLEMENTED(getConsumingOperations);
        }

        Place::Ptr Place::getTargetTensor(int outputPortIndex) const
        {
            FRONT_END_NOT_IMPLEMENTED(getTargetTensor);
        }

        Place::Ptr Place::getProducingOperation(int inputPortIndex) const
        {
            FRONT_END_NOT_IMPLEMENTED(getProducingOperation);
        }

        Place::Ptr Place::getProducingPort() const
        {
            FRONT_END_NOT_IMPLEMENTED(getProducingPort);
        }

        Place::Ptr Place::getInputPort(int inputPortIndex) const
        {
            FRONT_END_NOT_IMPLEMENTED(getInputPort);
        }

        Place::Ptr Place::getInputPort(const std::string& intputName, int inputPortIndex) const
        {
            FRONT_END_NOT_IMPLEMENTED(getInputPort);
        }

        Place::Ptr Place::getOutputPort(int outputPortIndex) const
        {
            FRONT_END_NOT_IMPLEMENTED(getOutputPort);
        }

        Place::Ptr Place::getOutputPort(const std::string& outputName, int outputPortIndex) const
        {
            FRONT_END_NOT_IMPLEMENTED(getOutputPort);
        }

        std::vector<Place::Ptr> Place::getConsumingPorts() const
        {
            FRONT_END_NOT_IMPLEMENTED(getConsumingPorts);
        }

        bool Place::isInput() const
        {
            FRONT_END_NOT_IMPLEMENTED(isInput);
        }

        bool Place::isOutput() const
        {
            FRONT_END_NOT_IMPLEMENTED(isOutput);
        }

        bool Place::isEqual(Ptr another) const
        {
            FRONT_END_NOT_IMPLEMENTED(isEqual);
        }

        bool Place::isEqualData(Ptr another) const
        {
            FRONT_END_NOT_IMPLEMENTED(isEqualData);
        }

        Place::Ptr Place::getSourceTensor(int inputPortIndex) const
        {
            FRONT_END_NOT_IMPLEMENTED(getSourceTensor);
        }

    } // namespace frontend
} // namespace ngraph
