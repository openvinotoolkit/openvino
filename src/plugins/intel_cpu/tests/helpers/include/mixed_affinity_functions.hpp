// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <map>

#include <openvino/core/model.hpp>


/* This file contains definitions of relatively simple functions (models) that will be used
 * to test mixed affinity behavior. All the functions are expected to be direct descendants of
 * MixedAffinityFunctionBase, so their constructors take only one (input_shapes) argument.
 */

using MixedAffinityMarkup = std::unordered_map<std::string, std::pair<size_t, size_t>>;

MixedAffinityMarkup transformBSMarkup(const std::unordered_map<std::string, size_t>& markup);

class MixedAffinityFunctionBase {
public:
    explicit MixedAffinityFunctionBase(const std::vector<ov::PartialShape>& input_shapes) : input_shapes(input_shapes) {}
    std::shared_ptr<ov::Model> getOriginal(const MixedAffinityMarkup& markup = {});
    std::shared_ptr<ov::Model> getReference();

protected:
    virtual std::shared_ptr<ov::Model> initOriginal();
    virtual std::shared_ptr<ov::Model> initReference();

    const std::vector<ov::PartialShape> input_shapes;
private:
    void markup_model(const std::shared_ptr<ov::Model>& m, const MixedAffinityMarkup& markup);
};

class ConvWithBiasFunction : public MixedAffinityFunctionBase {
public:
    explicit ConvWithBiasFunction(const std::vector<ov::PartialShape>& input_shapes);
protected:
    std::shared_ptr<ov::Model> initOriginal() override;
    std::shared_ptr<ov::Model> initReference() override;
};

class Int8ConvWithDqSubFunction : public MixedAffinityFunctionBase {
public:
    explicit Int8ConvWithDqSubFunction(const std::vector<ov::PartialShape>& input_shapes);
protected:
    std::shared_ptr<ov::Model> initOriginal() override;
    std::shared_ptr<ov::Model> initReference() override;
};

class GrConvWithParamFunction : public MixedAffinityFunctionBase {
public:
    explicit GrConvWithParamFunction(const std::vector<ov::PartialShape>& input_shapes);
protected:
    std::shared_ptr<ov::Model> initOriginal() override;
    std::shared_ptr<ov::Model> initReference() override;
};

class ConvWithTransposeFunction : public MixedAffinityFunctionBase {
public:
    explicit ConvWithTransposeFunction(const std::vector<ov::PartialShape>& input_shapes);
protected:
    std::shared_ptr<ov::Model> initOriginal() override;
    std::shared_ptr<ov::Model> initReference() override;
};

class ConvWithReshapeFunction : public MixedAffinityFunctionBase {
public:
    explicit ConvWithReshapeFunction(const std::vector<ov::PartialShape>& input_shapes);
protected:
    std::shared_ptr<ov::Model> initOriginal() override;
    std::shared_ptr<ov::Model> initReference() override;
};

class ConvWithSplitAndResultFunction : public MixedAffinityFunctionBase {
public:
    explicit ConvWithSplitAndResultFunction(const std::vector<ov::PartialShape>& input_shapes);
protected:
    std::shared_ptr<ov::Model> initOriginal() override;
    std::shared_ptr<ov::Model> initReference() override;
};

class ConvolutionsAndSplitFunction : public MixedAffinityFunctionBase {
public:
    explicit ConvolutionsAndSplitFunction(const std::vector<ov::PartialShape>& input_shapes);
protected:
    std::shared_ptr<ov::Model> initOriginal() override;
    std::shared_ptr<ov::Model> initReference() override;
};

class TwoConvAndAddFunction : public MixedAffinityFunctionBase {
public:
    explicit TwoConvAndAddFunction(const std::vector<ov::PartialShape>& input_shapes);
protected:
    std::shared_ptr<ov::Model> initOriginal() override;
    std::shared_ptr<ov::Model> initReference() override;
};

class TwoConvWithS2BFunction : public MixedAffinityFunctionBase {
public:
    explicit TwoConvWithS2BFunction(const std::vector<ov::PartialShape>& input_shapes);
protected:
    std::shared_ptr<ov::Model> initOriginal() override;
    std::shared_ptr<ov::Model> initReference() override;
};

class ConvAndAddWithParameterFunction : public MixedAffinityFunctionBase {
public:
    explicit ConvAndAddWithParameterFunction(const std::vector<ov::PartialShape>& input_shapes);
protected:
    std::shared_ptr<ov::Model> initOriginal() override;
    std::shared_ptr<ov::Model> initReference() override;
};

class ConvWithTransposeAndAddFunction : public MixedAffinityFunctionBase {
public:
    explicit ConvWithTransposeAndAddFunction(const std::vector<ov::PartialShape>& input_shapes);
protected:
    std::shared_ptr<ov::Model> initOriginal() override;
    std::shared_ptr<ov::Model> initReference() override;
};

class ConvWithConcatFunction : public MixedAffinityFunctionBase {
public:
    explicit ConvWithConcatFunction(const std::vector<ov::PartialShape>& input_shapes);
protected:
    std::shared_ptr<ov::Model> initOriginal() override;
    std::shared_ptr<ov::Model> initReference() override;
};