// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <string>

#include "common_test_utils/test_constants.hpp"

#include "conformance.hpp"

#include "functional_test_utils/skip_tests_config.hpp"

std::vector<std::string> disabledTestPatterns() {
    std::vector<std::string> skippedTestsRegExp;
    std::string targetDevice = ConformanceTests::targetDevice;
    if (targetDevice.find(CommonTestUtils::DEVICE_MYRIAD) != std::string::npos) {
        std::vector<std::string> myriadSkips{
                // TODO: Issue: 53061
                R"(.*ReduceMean_334003.*)", R"(.*ReduceMean_334702.*)", R"(.*Select_1172002.*)", R"(.*Select_1178458.*)",
                R"(.*Divide_567867.*)", R"(.*Divide_1377278.*)", R"(.*Divide_1030260.*)", R"(.*Pad_274459.*)", R"(.*Pad_276201.*)",
                R"(.*Pad_276193.*)", R"(.*Add_1070296.*)", R"(.*Add_284191.*)", R"(.*Add_1087636.*)", R"(.*VariadicSplit_614428.*)",
                R"(.*VariadicSplit_642662.*)", R"(.*VariadicSplit_412430.*)", R"(.*VariadicSplit_475191.*)", R"(.*VariadicSplit_475203.*)",
                R"(.*VariadicSplit_279359.*)", R"(.*VariadicSplit_31490.*)", R"(.*VariadicSplit_1444989.*)", R"(.*VariadicSplit_31499.*)",
                R"(.*VariadicSplit_279373.*)", R"(.*VariadicSplit_279345.*)", R"(.*VariadicSplit_31508.*)", R"(.*VariadicSplit_535004.*)",
                R"(.*VariadicSplit_148861.*)", R"(.*VariadicSplit_475199.*)", R"(.*VariadicSplit_412439.*)", R"(.*VariadicSplit_412448.*)",
                R"(.*VariadicSplit_475175.*)", R"(.*VariadicSplit_427269.*)", R"(.*VariadicSplit_1070300.*)", R"(.*VariadicSplit_478455.*)",
                R"(.*VariadicSplit_416024.*)", R"(.*VariadicSplit_478469.*)", R"(.*VariadicSplit_416006.*)", R"(.*VariadicSplit_35108.*)",
                R"(.*VariadicSplit_478473.*)", R"(.*VariadicSplit_416015.*)", R"(.*VariadicSplit_280446.*)", R"(.*VariadicSplit_6686.*)",
                R"(.*VariadicSplit_1087640.*)", R"(.*VariadicSplit_1443292.*)", R"(.*VariadicSplit_35099.*)", R"(.*VariadicSplit_280460.*)",
                R"(.*VariadicSplit_35090.*)", R"(.*VariadicSplit_430827.*)", R"(.*VariadicSplit_478461.*)", R"(.*VariadicSplit_280432.*)",
                R"(.*VariadicSplit_539066.*)", R"(.*VariadicSplit_149040.*)", R"(.*VariadicSplit_614761.*)", R"(.*VariadicSplit_629649.*)",
                R"(.*Unsqueeze_65214.*)", R"(.*Unsqueeze_4838.*)", R"(.*Unsqueeze_6661.*)", R"(.*Multiply_65211.*)", R"(.*Multiply_4833.*)",
                R"(.*Multiply_48331.*)", R"(.*Multiply_6656.*)", R"(.*Subtract_534968.*)", R"(.*Subtract_567907.*)", R"(.*Gather_21153.*)",
                R"(.*Gather_302464.*)", R"(.*Gather_588814.*)", R"(.*Gather_588810.*)", R"(.*Gather_952843.*)", R"(.*Gather_588818.*)",
                R"(.*Gather_588871.*)", R"(.*Gather_48276.*)", R"(.*Gather_1652589.*)", R"(.*Gather_535000.*)", R"(.*Gather_360953.*)",
                R"(.*Gather_588867.*)", R"(.*Gather_48283.*)", R"(.*Gather_1311696.*)", R"(.*Gather_588796.*)", R"(.*Gather_4847.*)",
                R"(.*Gather_4803.*)", R"(.*Gather_284209.*)", R"(.*Gather_4869.*)", R"(.*Gather_97426.*)", R"(.*Gather_360963.*)", R"(.*Gather_1570871.*)",
                R"(.*Gather_513092.*)", R"(.*Gather_629616.*)", R"(.*Gather_48287.*)", R"(.*Gather_284215.*)", R"(.*Gather_48335.*)", R"(.*Gather_1444.*)",
                R"(.*Gather_4810.*)", R"(.*Gather_1395.*)", R"(.*Gather_1653002.*)", R"(.*Gather_53520.*)", R"(.*Gather_567864.*)", R"(.*Gather_362776.*)",
                R"(.*Gather_567856.*)", R"(.*Gather_567919.*)", R"(.*Gather_567830.*)", R"(.*Gather_567923.*)", R"(.*Gather_567860.*)", R"(.*Gather_539062.*)",
                R"(.*Gather_1304430.*)", R"(.*Gather_938915.*)", R"(.*Gather_53527.*)", R"(.*Convert_284211.*)", R"(.*Convert_4830.*)", R"(.*Convert_567844.*)",
                R"(.*Convert_6653.*)", R"(.*FakeQuantize_738613.*)", R"(.*FakeQuantize_738639.*)", R"(.*Split_65164.*)", R"(.*Split_274446.*)",
                R"(.*Split_361038.*)", R"(.*Split_330570.*)", R"(.*Split_276188.*)", R"(.*Split_68625.*)", R"(.*Split_333279.*)", R"(.*Split_362843.*)",
                R"(.*Split_68647.*)", R"(.*Floor_6658.*)", R"(.*TopK_1172093.*)", R"(.*TopK_1172038.*)", R"(.*TopK_1178526.*)", R"(.*TopK_515250.*)",
                R"(.*TopK_1178477.*)", R"(.*Reshape_1105704.*)", R"(.*Reshape_1118646.*)", R"(.*Reshape_588858.*)", R"(.*Reshape_567910.*)",
                // TODO: Crashes: Should be handled
                R"(.*CTCGreedyDecoderSeqLen_271798.*)", R"(.*MVN_276196.*)", R"(.*MVN_274454.*)",
                // hung
                R"(.*Clamp_1178483.*)",
                // lost results
                R"(.*Add_1087636.*)", R"(.*Add_2868.*)", R"(.*Add_2979.*)", R"(.*Add_53543.*)",
        };
        skippedTestsRegExp.insert(skippedTestsRegExp.end(),
                                  std::make_move_iterator(myriadSkips.begin()),
                                  std::make_move_iterator(myriadSkips.end()));
    }
    if (targetDevice.find(CommonTestUtils::DEVICE_GPU) != std::string::npos) {
        std::vector<std::string> gpuSkips{
                // TODO: Issue: 53062
                R"(.*TensorIterator_1195103.*)", R"(.*TensorIterator_1195129.*)", R"(.*TensorIterator_1653010.*)", R"(.*TensorIterator_199835.*)",
                R"(.*TensorIterator_303641.*)", R"(.*TensorIterator_303667.*)", R"(.*TensorIterator_337260.*)", R"(.*TensorIterator_362846.*)",
                R"(.*TensorIterator_362869.*)", R"(.*TensorIterator_362906.*)", R"(.*TensorIterator_365496.*)", R"(.*TensorIterator_365522.*)",
                R"(.*TensorIterator_365556.*)", R"(.*TensorIterator_365579.*)", R"(.*TensorIterator_616544.*)", R"(.*TensorIterator_616570.*)",
                R"(.*TensorIterator_864978.*)", R"(.*TensorIterator_865004.*)", R"(.*TensorIterator_865030.*)", R"(.*TensorIterator_865056.*)",
                R"(.*TensorIterator_865082.*)", R"(.*TensorIterator_865108.*)", R"(.*TensorIterator_972832.*)", R"(.*TensorIterator_972858.*)",
                R"(.*TensorIterator_972884.*)", R"(.*TensorIterator_972910.*)", R"(.*TensorIterator_972936.*)", R"(.*TensorIterator_972962.*)",
                R"(.*TensorIterator_972988.*)", R"(.*TensorIterator_973014.*)", R"(.*TensorIterator_1194007.*)", R"(.*TensorIterator_1194033.*)",
                R"(.*TensorIterator_302500.*)", R"(.*TensorIterator_337024.*)", R"(.*TensorIterator_361044.*)", R"(.*TensorIterator_361067.*)",
                R"(.*TensorIterator_361108.*)", R"(.*TensorIterator_361131.*)", R"(.*TensorIterator_364171.*)", R"(.*TensorIterator_364197.*)",
                R"(.*TensorIterator_364231.*)", R"(.*TensorIterator_364254.*)", R"(.*TensorIterator_615626.*)", R"(.*TensorIterator_615652.*)",
                R"(.*TensorIterator_865760.*)", R"(.*TensorIterator_865812.*)", R"(.*TensorIterator_865838.*)", R"(.*TensorIterator_865864.*)",
                R"(.*TensorIterator_973781.*)", R"(.*TensorIterator_973807.*)",
                // Hung:
                R"(.*AvgPool_1199829.*)", R"(.*AvgPool_1201153.*)", R"(.*GroupConvolution_330567.*)", R"(.*ROIPooling_1199827.*)",
                R"(.*MaxPool_43108.*)",
        };
        skippedTestsRegExp.insert(skippedTestsRegExp.end(),
                                  std::make_move_iterator(gpuSkips.begin()),
                                  std::make_move_iterator(gpuSkips.end()));
    }
    if (targetDevice.find(CommonTestUtils::DEVICE_CPU) != std::string::npos) {
        std::vector<std::string> cpuSkips{
            // Hung:
            R"(.*AvgPool_1199829.*)", R"(.*AvgPool_1201153.*)", R"(.*ROIPooling_1199827.*)",
        };
        skippedTestsRegExp.insert(skippedTestsRegExp.end(),
                                  std::make_move_iterator(cpuSkips.begin()),
                                  std::make_move_iterator(cpuSkips.end()));
    }
    if (targetDevice.find(CommonTestUtils::DEVICE_GNA) != std::string::npos) {
        std::vector<std::string> gnaSkips{
            // Lost results
            R"(.*Add_1087636.*)", R"(.*Add_2868.*)", R"(.*Add_2979.*)", R"(.*Add_53543.*)",
            // hung
            R"(.*Concat_535028.*)", R"(.*Concat_377139.*)", R"(.*Concat_379481.*)", R"(.*Concat_539044.*)", R"(.*Concat_539074.*)",
            R"(.*Concat_534956.*)",
        };
        skippedTestsRegExp.insert(skippedTestsRegExp.end(),
                                  std::make_move_iterator(gnaSkips.begin()),
                                  std::make_move_iterator(gnaSkips.end()));
    }
    return skippedTestsRegExp;
}
