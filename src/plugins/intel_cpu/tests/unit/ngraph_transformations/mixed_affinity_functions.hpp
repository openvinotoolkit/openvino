// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <map>

#include "ngraph/ngraph.hpp"
#include "openvino/core/model.hpp"
#include "common_test_utils/ngraph_test_utils.hpp"


/* This file contains definitions of relatively simple functions (models) that will be used
 * to test mixed affinity behavior. All the functions are expected to be direct descendants of
 * MixedAffinityFunctionBase, so their constructors take only one (input_shapes) argument.
 */

using BSMarkup = std::unordered_map<std::string, size_t>;
class MixedAffinityFunctionBase {
public:
    explicit MixedAffinityFunctionBase(const std::vector<ov::PartialShape>& input_shapes) : input_shapes(input_shapes) {}
    std::shared_ptr<ov::Model> getOriginal(const BSMarkup& markup = {});
    std::shared_ptr<ov::Model> getReference(const BSMarkup& markup = {});

protected:
    virtual std::shared_ptr<ov::Model> initOriginal();
    virtual std::shared_ptr<ov::Model> initReference();

    const std::vector<ov::PartialShape> input_shapes;
private:
    void markup_model(const std::shared_ptr<ov::Model>& m, const BSMarkup& markup);
};

class ConvWithBiasFunction : public MixedAffinityFunctionBase {
public:
    explicit ConvWithBiasFunction(const std::vector<ov::PartialShape>& input_shapes);
protected:
    std::shared_ptr<ov::Model> initOriginal() override;
};

class ConvWithTransposeFunction : public MixedAffinityFunctionBase {
public:
    explicit ConvWithTransposeFunction(const std::vector<ov::PartialShape>& input_shapes);
protected:
    std::shared_ptr<ov::Model> initOriginal() override;
};

class ConvWithReshapeFunction : public MixedAffinityFunctionBase {
public:
    explicit ConvWithReshapeFunction(const std::vector<ov::PartialShape>& input_shapes);
protected:
    std::shared_ptr<ov::Model> initOriginal() override;
};

class ConvWithSplitAndResultFunction : public MixedAffinityFunctionBase {
public:
    explicit ConvWithSplitAndResultFunction(const std::vector<ov::PartialShape>& input_shapes);
protected:
    std::shared_ptr<ov::Model> initOriginal() override;
};

class TwoConvAndAddFunction : public MixedAffinityFunctionBase {
public:
    explicit TwoConvAndAddFunction(const std::vector<ov::PartialShape>& input_shapes);
protected:
    std::shared_ptr<ov::Model> initOriginal() override;
};

class TwoConvWithS2BFunction : public MixedAffinityFunctionBase {
public:
    explicit TwoConvWithS2BFunction(const std::vector<ov::PartialShape>& input_shapes);
protected:
    std::shared_ptr<ov::Model> initOriginal() override;
};
