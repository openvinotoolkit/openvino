// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <set>
#include <list>
#include <array>

#include <gtest/gtest.h>

#include <vpu/utils/range.hpp>
#include <vpu/utils/containers.hpp>

using namespace testing;

//
// VPU_IterRangeTest
//

class VPU_IterRangeTest: public ::testing::Test {
protected:
    const int count = 10;
    std::list<int> list;

    void SetUp() override {
        for (int i = 0; i < count; ++i) {
            list.push_back(i);
        }
    }
};

TEST_F(VPU_IterRangeTest, PreservesIterationOrder) {
    auto contRange = vpu::contRange(list);

    int gold = 0;
    auto innerIt = list.cbegin();
    for (auto cit = contRange.cbegin(); cit != contRange.end(); cit++) {
        ASSERT_EQ(*cit, *innerIt++) << "Values given by owner and inner containers differ";
        gold++;
    }
    ASSERT_EQ(gold, count) << "Owner and inner ranges differ in length";
}

TEST_F(VPU_IterRangeTest, RespectsInnerPushBacksWhileIteration) {
    auto contRange = vpu::contRange(list);

    int gold = 0;
    auto innerIt = list.begin();

    for (auto val : contRange) {
        if (gold < 5) {
            // duplicate first 5 elements of the head, inserting them after the tail
            list.push_back(*innerIt);
        }
        ASSERT_EQ(val, *innerIt++) << "Values given by owner and inner containers differ";
        gold++;
    }

    ASSERT_EQ(gold, count + 5) << "Initial inner container size was not preserved";
}

TEST_F(VPU_IterRangeTest, RespectsInnerRemovalsWhileIteration) {
    auto contRange = vpu::contRange(list);

    int gold = 0;
    auto innerIt = list.begin();
    auto innerRevIt = list.end();

    for (auto val : contRange) {
        if (gold < 5) {
            // removing elements from the end
            innerRevIt = list.erase(--innerRevIt);
        }
        ASSERT_EQ(val, *innerIt++) << "Values given by owner and inner containers differ";
        gold++;
    }

    ASSERT_EQ(gold, count - 5) << "Removals were ignored";
}

TEST_F(VPU_IterRangeTest, SurvivesInnerInsertionsWhileIteration) {
    auto contRange = vpu::contRange(list);

    int gold = 0;
    auto innerIt = list.begin();

    for (auto it = contRange.begin(); it != contRange.end(); ++it) {
        ASSERT_EQ(*it, *innerIt) << "Values given by owner and inner containers differ";

        if (gold < 10) {
            // duplicate head elements of inner, inserting them just before the current iterator
            list.insert(innerIt, *innerIt);
        }
        gold++;
        innerIt++;
    }

    ASSERT_EQ(gold, count) << "Insertions at the head influenced iteration";
}

//
// VPU_MapRangeTest
//

class VPU_MapRangeTest: public ::testing::Test {
protected:
    struct InnerStruct final : public vpu::EnableHandleFromThis<InnerStruct> {
        int val = 0;
        vpu::IntrusivePtrListNode<InnerStruct> node;
        explicit InnerStruct(int val) : val(val), node(this) {}
    };

    const int count = 10;
    std::list<int> list;
    std::vector<int> vec;

    const static std::function<int(int)> incFunc;
    const static std::function<double(int)> incAndConvertFunc;

    void SetUp() override {
        for (int i = 0; i < count; ++i) {
            list.push_back(i);
            vec.push_back(i);
        }
    }
};

const std::function<int(int)> VPU_MapRangeTest::incFunc = [](int val) { return val + 1; };

const std::function<double(int)> VPU_MapRangeTest::incAndConvertFunc = [](int val)
        { return static_cast<double>(val + 1); };

TEST_F(VPU_MapRangeTest, PreservesIterationOrder) {
    auto mapRange = vpu::mapRange(vpu::contRange(vec), incFunc);

    int gold = 0;
    auto innerIt = vec.cbegin();
    for (auto cit = mapRange.cbegin(); cit != mapRange.end(); ++cit) {
        int mappedExpectation = incFunc(*innerIt);
        ASSERT_EQ(*cit, mappedExpectation) << "Values given by map and inner containers differ";
        gold++;
        innerIt++;
    }
    ASSERT_EQ(gold, count) << "Owner and inner ranges differ in length";
}

TEST_F(VPU_MapRangeTest, MapToAnotherType) {
    auto mapRange = vpu::mapRange(vpu::contRange(vec), incAndConvertFunc);

    int gold = 0;
    auto innerIt = vec.cbegin();
    for (auto cit = mapRange.cbegin(); cit != mapRange.end(); ++cit) {
        const int base = *innerIt;
        const double mappedExpectation = incAndConvertFunc(base);
        ASSERT_EQ(*cit, mappedExpectation) << "Values given by map and inner containers differ";
        gold++;
        innerIt++;
    }
    ASSERT_EQ(gold, count) << "Owner and inner ranges differ in length";
}

TEST_F(VPU_MapRangeTest, CountSharedPointers) {
    std::vector<std::shared_ptr<InnerStruct>> nodesExternalVector;
    vpu::IntrusivePtrList<InnerStruct> list(&InnerStruct::node);

    for (int i = 0; i < count; ++i) {
        auto innerStructSPtr = std::make_shared<InnerStruct>(i);
        ASSERT_EQ(1, innerStructSPtr.use_count()) << "single instance of shared pointer ";
        nodesExternalVector.push_back(innerStructSPtr);
        ASSERT_EQ(2, innerStructSPtr.use_count()) << "stack instance of shared pointer plus copy in vector";
        list.push_back(innerStructSPtr);
        ASSERT_EQ(2, innerStructSPtr.use_count()) << "intrusive list keeps weak pointer only";
    }

    auto mapRange = vpu::mapRange(
            vpu::contRange(list),
            [](const vpu::Handle<InnerStruct>& innerPtr) {
                return incFunc(innerPtr->val);
            });

    for (int i = 0; i < count; ++i) {
        ASSERT_EQ(1, nodesExternalVector[i].use_count()) << "intrusive list keeps weak pointer only";
    }
}

TEST_F(VPU_MapRangeTest, IterationOverIntrusiveListSurvivesElementRemoval) {
    std::vector<std::shared_ptr<InnerStruct>> nodesExternalVector;
    vpu::IntrusivePtrList<InnerStruct> list(&InnerStruct::node);

    for (int i = 0; i < count; ++i) {
        auto innerStructSPtr = std::make_shared<InnerStruct>(i);
        nodesExternalVector.push_back(innerStructSPtr);
        list.push_back(innerStructSPtr);
    }

    auto mapRange = vpu::mapRange(
        vpu::contRange(list),
        [](const vpu::Handle<InnerStruct>& innerPtr) {
            return incFunc(innerPtr->val);
        });

    int i = 0;
    for (auto mprit = mapRange.cbegin(); mprit != mapRange.cend(); ++mprit, ++i) {
        ASSERT_EQ(2, nodesExternalVector[i].use_count()) << "intrusive list's iterator keeps shared pointer too";
        ASSERT_EQ(*mprit, incFunc(i)) << "mapped value must conform to increment function";

        list.pop_front();

        ASSERT_EQ(1, nodesExternalVector[i].use_count()) << "removal of element releases shared pointer of its iterator";
    }
}

//
// VPU_FilterRangeTest
//

class VPU_FilterRangeTest: public ::testing::Test {
protected:
    struct InnerStruct final : public vpu::EnableHandleFromThis<InnerStruct> {
        int val = 0;
        vpu::IntrusivePtrListNode<InnerStruct> node;
        explicit InnerStruct(int val) : val(val), node(this) {}
    };

    const int count = 10;
    std::vector<int> vec;

    const static std::function<bool(int)> evenFunc;

    virtual void SetUp() override {
        for (int i = 0; i < count; ++i) {
            vec.push_back(i);
        }
    }
};

const std::function<bool(int)> VPU_FilterRangeTest::evenFunc = [](int val) { return val % 2 == 0; };

TEST_F(VPU_FilterRangeTest, FilteringOnlyEvenNumbers) {
    auto filterRange = vpu::filterRange(vpu::contRange(vec), evenFunc);

    int i = 0;
    for (auto val : filterRange) {
        ASSERT_EQ(val, i);
        i += 2;
    }
    ASSERT_EQ(i, count);
}

TEST_F(VPU_FilterRangeTest, FilteringOutFirst) {
    auto filterRange = vpu::filterRange(
        vpu::contRange(vec),
        [](int val) {
            return val != 0;
        });

    int gold = 1;
    for (auto val : filterRange) {
        ASSERT_EQ(val, gold);
        gold++;
    }
    ASSERT_EQ(gold, count);
}

TEST_F(VPU_FilterRangeTest, FilteringOutLast) {
    auto filterRange = vpu::filterRange(
        vpu::contRange(vec),
        [&](int val) {
            return val != count - 1;
        });

    int gold = 0;
    for (auto val : filterRange) {
        ASSERT_EQ(val, gold);
        gold++;
    }
    ASSERT_EQ(gold, count - 1);
}

TEST_F(VPU_FilterRangeTest, CountSharedPointers) {
    std::list<std::shared_ptr<InnerStruct>> nodesExternalList;
    vpu::IntrusivePtrList<InnerStruct> list(&InnerStruct::node);

    for (int i = 0; i < count; ++i) {
        auto inner = std::make_shared<InnerStruct>(i);
        ASSERT_EQ(1, inner.use_count()) << "single instance of shared pointer";
        nodesExternalList.push_back(inner);
        ASSERT_EQ(2, inner.use_count()) << "stack instance of shared pointer plus copy in vector";
        list.push_back(inner);
        ASSERT_EQ(2, inner.use_count()) << "intrusive list keeps weak pointer only";
    }

    for (auto cit = nodesExternalList.cbegin(); cit != nodesExternalList.end(); ++cit) {
        ASSERT_EQ(1, cit->use_count()) << "intrusive list keeps weak pointer only";
    }

    auto filterRange = vpu::filterRange<vpu::NonNull>(vpu::contRange(list));

    for (auto cit = nodesExternalList.cbegin(); cit != nodesExternalList.end(); ++cit) {
        ASSERT_EQ(1, cit->use_count()) << "intrusive list keeps weak pointer only";
    }
}

TEST_F(VPU_FilterRangeTest, IterationOverIntrusiveListSurvivesElementRemoval) {
    std::list<std::shared_ptr<InnerStruct>> nodesExternalList;
    vpu::IntrusivePtrList<InnerStruct> list(&InnerStruct::node);

    for (int i = 0; i < count; ++i) {
        auto inner = std::make_shared<InnerStruct>(i);
        nodesExternalList.push_back(inner);
        list.push_back(inner);
    }

    auto filterRange = vpu::filterRange<vpu::NonNull>(vpu::contRange(list));

    int gold = 0;
    for (const auto& ptr : filterRange) {
        ASSERT_EQ(ptr->val, gold);
        list.pop_front();
        gold++;
    }
    ASSERT_EQ(gold, count);

    for (auto cit = nodesExternalList.cbegin(); cit != nodesExternalList.end(); ++cit) {
        ASSERT_EQ(1, cit->use_count()) << "intrusive list keeps weak pointer only";
    }
}

TEST_F(VPU_FilterRangeTest, IterationOverIntrusiveListWhileElementsBeingRemoved) {
    std::list<std::shared_ptr<InnerStruct>> nodesExternalList;
    vpu::IntrusivePtrList<InnerStruct> list(&InnerStruct::node);

    for (int i = 0; i < count; ++i) {
        auto inner = std::make_shared<InnerStruct>(i);
        nodesExternalList.push_back(inner);
        list.push_back(inner);
    }

    auto filterRange = vpu::filterRange(
        vpu::contRange(list),
        [](const vpu::Handle<InnerStruct>& innerPtr) {
            return evenFunc(innerPtr->val);
        });

    int gold = 0;
    for (const auto& ptr : filterRange) {
        ASSERT_EQ(ptr->val, gold);
        // remove even & odd front elems
        list.pop_front();
        list.pop_front();
        gold += 2;
    }
    ASSERT_EQ(gold, count);

    for (auto cit = nodesExternalList.cbegin(); cit != nodesExternalList.end(); ++cit) {
        ASSERT_EQ(1, cit->use_count()) << "intrusive list keeps weak pointer only";
    }
}

TEST_F(VPU_FilterRangeTest, IterationOverEmptyIntrusiveListWhereAllElementsFilteredOut) {
    std::list<std::shared_ptr<InnerStruct>> nodesExternalList;
    vpu::IntrusivePtrList<InnerStruct> list(&InnerStruct::node);

    for (int i = 0; i < count; ++i) {
        auto inner = std::make_shared<InnerStruct>(i);
        nodesExternalList.push_back(inner);
        list.push_back(inner);
    }

    auto filterRange = vpu::filterRange(
        vpu::contRange(list),
        [](const vpu::Handle<InnerStruct>& innerPtr) {
            return (innerPtr->val < 0);
        });

    for (const auto& ptr : filterRange) {
        ASSERT_TRUE(false) << "Must not see any item in filtered list";
    }

    for (auto cit = list.cbegin(); cit != list.cend(); ++cit) {
        if (evenFunc((*cit)->val)) {
            list.erase(cit);
        }
    }

    for (const auto& ptr : filterRange) {
        ASSERT_TRUE(false) << "Must not see any item in filtered list";
    }

    for (auto cit = nodesExternalList.cbegin(); cit != nodesExternalList.end(); ++cit) {
        ASSERT_EQ(1, cit->use_count()) << "intrusive list keeps weak pointer only";
    }
}
