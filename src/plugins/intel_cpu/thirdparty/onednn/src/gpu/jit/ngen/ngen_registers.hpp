/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/*
 * Do not #include this file directly; ngen uses it internally.
 */

#if defined(NGEN_CPP11) && defined(NGEN_GLOBAL_REGS)
#define constexpr_reg const
#define constexpr_reg_const const
#else
#define constexpr_reg constexpr
#define constexpr_reg_const constexpr const
#endif

static constexpr_reg IndirectRegisterFrame indirect{};
#ifdef NGEN_SHORT_NAMES
static constexpr_reg_const IndirectRegisterFrame &r = indirect;
#endif

static constexpr_reg GRF r0{0}, r1{1}, r2{2}, r3{3}, r4{4}, r5{5}, r6{6}, r7{7};
static constexpr_reg GRF r8{8}, r9{9}, r10{10}, r11{11}, r12{12}, r13{13}, r14{14}, r15{15};
static constexpr_reg GRF r16{16}, r17{17}, r18{18}, r19{19}, r20{20}, r21{21}, r22{22}, r23{23};
static constexpr_reg GRF r24{24}, r25{25}, r26{26}, r27{27}, r28{28}, r29{29}, r30{30}, r31{31};
static constexpr_reg GRF r32{32}, r33{33}, r34{34}, r35{35}, r36{36}, r37{37}, r38{38}, r39{39};
static constexpr_reg GRF r40{40}, r41{41}, r42{42}, r43{43}, r44{44}, r45{45}, r46{46}, r47{47};
static constexpr_reg GRF r48{48}, r49{49}, r50{50}, r51{51}, r52{52}, r53{53}, r54{54}, r55{55};
static constexpr_reg GRF r56{56}, r57{57}, r58{58}, r59{59}, r60{60}, r61{61}, r62{62}, r63{63};
static constexpr_reg GRF r64{64}, r65{65}, r66{66}, r67{67}, r68{68}, r69{69}, r70{70}, r71{71};
static constexpr_reg GRF r72{72}, r73{73}, r74{74}, r75{75}, r76{76}, r77{77}, r78{78}, r79{79};
static constexpr_reg GRF r80{80}, r81{81}, r82{82}, r83{83}, r84{84}, r85{85}, r86{86}, r87{87};
static constexpr_reg GRF r88{88}, r89{89}, r90{90}, r91{91}, r92{92}, r93{93}, r94{94}, r95{95};
static constexpr_reg GRF r96{96}, r97{97}, r98{98}, r99{99}, r100{100}, r101{101}, r102{102}, r103{103};
static constexpr_reg GRF r104{104}, r105{105}, r106{106}, r107{107}, r108{108}, r109{109}, r110{110}, r111{111};
static constexpr_reg GRF r112{112}, r113{113}, r114{114}, r115{115}, r116{116}, r117{117}, r118{118}, r119{119};
static constexpr_reg GRF r120{120}, r121{121}, r122{122}, r123{123}, r124{124}, r125{125}, r126{126}, r127{127};
static constexpr_reg GRF r128{128}, r129{129}, r130{130}, r131{131}, r132{132}, r133{133}, r134{134}, r135{135};
static constexpr_reg GRF r136{136}, r137{137}, r138{138}, r139{139}, r140{140}, r141{141}, r142{142}, r143{143};
static constexpr_reg GRF r144{144}, r145{145}, r146{146}, r147{147}, r148{148}, r149{149}, r150{150}, r151{151};
static constexpr_reg GRF r152{152}, r153{153}, r154{154}, r155{155}, r156{156}, r157{157}, r158{158}, r159{159};
static constexpr_reg GRF r160{160}, r161{161}, r162{162}, r163{163}, r164{164}, r165{165}, r166{166}, r167{167};
static constexpr_reg GRF r168{168}, r169{169}, r170{170}, r171{171}, r172{172}, r173{173}, r174{174}, r175{175};
static constexpr_reg GRF r176{176}, r177{177}, r178{178}, r179{179}, r180{180}, r181{181}, r182{182}, r183{183};
static constexpr_reg GRF r184{184}, r185{185}, r186{186}, r187{187}, r188{188}, r189{189}, r190{190}, r191{191};
static constexpr_reg GRF r192{192}, r193{193}, r194{194}, r195{195}, r196{196}, r197{197}, r198{198}, r199{199};
static constexpr_reg GRF r200{200}, r201{201}, r202{202}, r203{203}, r204{204}, r205{205}, r206{206}, r207{207};
static constexpr_reg GRF r208{208}, r209{209}, r210{210}, r211{211}, r212{212}, r213{213}, r214{214}, r215{215};
static constexpr_reg GRF r216{216}, r217{217}, r218{218}, r219{219}, r220{220}, r221{221}, r222{222}, r223{223};
static constexpr_reg GRF r224{224}, r225{225}, r226{226}, r227{227}, r228{228}, r229{229}, r230{230}, r231{231};
static constexpr_reg GRF r232{232}, r233{233}, r234{234}, r235{235}, r236{236}, r237{237}, r238{238}, r239{239};
static constexpr_reg GRF r240{240}, r241{241}, r242{242}, r243{243}, r244{244}, r245{245}, r246{246}, r247{247};
static constexpr_reg GRF r248{248}, r249{249}, r250{250}, r251{251}, r252{252}, r253{253}, r254{254}, r255{255};

static constexpr_reg NullRegister null{};
static constexpr_reg AddressRegister a0{0};
static constexpr_reg AccumulatorRegister acc0{0}, acc1{1};
static constexpr_reg SpecialAccumulatorRegister acc2{2,0}, acc3{3,1}, acc4{4,2}, acc5{5,3}, acc6{6,4}, acc7{7,5}, acc8{8,6}, acc9{9,7};
static constexpr_reg SpecialAccumulatorRegister mme0{4,0}, mme1{5,1}, mme2{6,2}, mme3{7,3}, mme4{8,4}, mme5{9,5}, mme6{10,6}, mme7{11,7};
static constexpr_reg SpecialAccumulatorRegister nomme = SpecialAccumulatorRegister::createNoMME();
static constexpr_reg SpecialAccumulatorRegister noacc = nomme;
static constexpr_reg FlagRegister f0{0}, f1{1};
static constexpr_reg FlagRegister f0_0{0,0}, f0_1{0,1}, f1_0{1,0}, f1_1{1,1};
static constexpr_reg ChannelEnableRegister ce0{0};
static constexpr_reg StackPointerRegister sp{0};
static constexpr_reg StateRegister sr0{0}, sr1{1};
static constexpr_reg ControlRegister cr0{0};
static constexpr_reg NotificationRegister n0{0};
static constexpr_reg InstructionPointerRegister ip{};
static constexpr_reg ThreadDependencyRegister tdr0{0};
static constexpr_reg PerformanceRegister tm0{0};
static constexpr_reg PerformanceRegister tm1{1};
static constexpr_reg PerformanceRegister tm2{2};
static constexpr_reg PerformanceRegister tm3{3};
static constexpr_reg PerformanceRegister tm4{4};
static constexpr_reg PerformanceRegister pm0{0,3}, tp0{0,4};
static constexpr_reg DebugRegister dbg0{0};
static constexpr_reg FlowControlRegister fc0{0}, fc1{1}, fc2{2}, fc3{3};

static constexpr_reg InstructionModifier NoDDClr = InstructionModifier::createNoDDClr();
static constexpr_reg InstructionModifier NoDDChk = InstructionModifier::createNoDDChk();
static constexpr_reg InstructionModifier AccWrEn = InstructionModifier::createAccWrCtrl();
static constexpr_reg InstructionModifier NoSrcDepSet = AccWrEn;
static constexpr_reg InstructionModifier Breakpoint = InstructionModifier::createDebugCtrl();
static constexpr_reg InstructionModifier sat = InstructionModifier::createSaturate();
static constexpr_reg InstructionModifier NoMask = InstructionModifier::createMaskCtrl(true);
static constexpr_reg InstructionModifier AutoSWSB = InstructionModifier::createAutoSWSB();
static constexpr_reg InstructionModifier Serialize = InstructionModifier::createSerialized();
static constexpr_reg InstructionModifier EOT = InstructionModifier::createEOT();
static constexpr_reg InstructionModifier Align1 = InstructionModifier::createAccessMode(0);
static constexpr_reg InstructionModifier Align16 = InstructionModifier::createAccessMode(1);

static constexpr_reg InstructionModifier Switch{ThreadCtrl::Switch};
static constexpr_reg InstructionModifier Atomic{ThreadCtrl::Atomic};
static constexpr_reg InstructionModifier NoPreempt{ThreadCtrl::NoPreempt};

#ifdef NGEN_SHORT_NAMES
static constexpr_reg_const InstructionModifier &W = NoMask;
#endif

static constexpr_reg PredCtrl anyv = PredCtrl::anyv;
static constexpr_reg PredCtrl allv = PredCtrl::allv;
static constexpr_reg PredCtrl any2h = PredCtrl::any2h;
static constexpr_reg PredCtrl all2h = PredCtrl::all2h;
static constexpr_reg PredCtrl any4h = PredCtrl::any4h;
static constexpr_reg PredCtrl all4h = PredCtrl::all4h;
static constexpr_reg PredCtrl any8h = PredCtrl::any8h;
static constexpr_reg PredCtrl all8h = PredCtrl::all8h;
static constexpr_reg PredCtrl any16h = PredCtrl::any16h;
static constexpr_reg PredCtrl all16h = PredCtrl::all16h;
static constexpr_reg PredCtrl any32h = PredCtrl::any32h;
static constexpr_reg PredCtrl all32h = PredCtrl::all32h;

static constexpr_reg InstructionModifier x_repl = InstructionModifier{PredCtrl::x};
static constexpr_reg InstructionModifier y_repl = InstructionModifier{PredCtrl::y};
static constexpr_reg InstructionModifier z_repl = InstructionModifier{PredCtrl::z};
static constexpr_reg InstructionModifier w_repl = InstructionModifier{PredCtrl::w};

static constexpr_reg InstructionModifier ze{ConditionModifier::ze};
static constexpr_reg InstructionModifier eq{ConditionModifier::eq};
static constexpr_reg InstructionModifier nz{ConditionModifier::ne};
static constexpr_reg InstructionModifier ne{ConditionModifier::nz};
static constexpr_reg InstructionModifier gt{ConditionModifier::gt};
static constexpr_reg InstructionModifier ge{ConditionModifier::ge};
static constexpr_reg InstructionModifier lt{ConditionModifier::lt};
static constexpr_reg InstructionModifier le{ConditionModifier::le};
static constexpr_reg InstructionModifier ov{ConditionModifier::ov};
static constexpr_reg InstructionModifier un{ConditionModifier::un};
static constexpr_reg InstructionModifier eo{ConditionModifier::eo};

static constexpr_reg InstructionModifier M0 = InstructionModifier::createChanOff(0);
static constexpr_reg InstructionModifier M4 = InstructionModifier::createChanOff(4);
static constexpr_reg InstructionModifier M8 = InstructionModifier::createChanOff(8);
static constexpr_reg InstructionModifier M12 = InstructionModifier::createChanOff(12);
static constexpr_reg InstructionModifier M16 = InstructionModifier::createChanOff(16);
static constexpr_reg InstructionModifier M20 = InstructionModifier::createChanOff(20);
static constexpr_reg InstructionModifier M24 = InstructionModifier::createChanOff(24);
static constexpr_reg InstructionModifier M28 = InstructionModifier::createChanOff(28);
static inline InstructionModifier ExecutionOffset(int off) { return InstructionModifier::createChanOff(off); }
#ifdef NGEN_SHORT_NAMES
static inline InstructionModifier M(int off) { return ExecutionOffset(off); }
#endif

static constexpr_reg SBID sb0{0}, sb1{1}, sb2{2}, sb3{3}, sb4{4}, sb5{5}, sb6{6}, sb7{7};
static constexpr_reg SBID sb8{8}, sb9{9}, sb10{10}, sb11{11}, sb12{12}, sb13{13}, sb14{14}, sb15{15};

static constexpr_reg AddressBase A32 = AddressBase::createA32(true);
static constexpr_reg AddressBase A32NC = AddressBase::createA32(false);
static constexpr_reg AddressBase A64 = AddressBase::createA64(true);
static constexpr_reg AddressBase A64NC = AddressBase::createA64(false);
static constexpr_reg AddressBase SLM = AddressBase::createSLM();

static inline AddressBase Surface(uint8_t index) { return AddressBase::createBTS(index); }
static inline AddressBase CC(uint8_t index) { return AddressBase::createCC(index); }
static inline AddressBase SC(uint8_t index) { return AddressBase::createSC(index); }

static inline AddressBase BTI(uint8_t index) { return AddressBase::createBTS(index); }
static inline AddressBase SS(uint32_t index) { return AddressBase::createSS(index); }
static inline AddressBase BSS(uint32_t index) { return AddressBase::createBSS(index); }

static constexpr_reg DataSpecLSC D8{DataSizeLSC::D8};
static constexpr_reg DataSpecLSC D16{DataSizeLSC::D16};
static constexpr_reg DataSpecLSC D32{DataSizeLSC::D32};
static constexpr_reg DataSpecLSC D64{DataSizeLSC::D64};
static constexpr_reg DataSpecLSC D8U32{DataSizeLSC::D8U32};
static constexpr_reg DataSpecLSC D16U32{DataSizeLSC::D16U32};

static constexpr_reg DataSpecLSC D8T = DataSpecLSC(DataSizeLSC::D8) | DataSpecLSC::createTranspose();
static constexpr_reg DataSpecLSC D16T = DataSpecLSC(DataSizeLSC::D16) | DataSpecLSC::createTranspose();
static constexpr_reg DataSpecLSC D32T = DataSpecLSC(DataSizeLSC::D32) | DataSpecLSC::createTranspose();
static constexpr_reg DataSpecLSC D64T = DataSpecLSC(DataSizeLSC::D64) | DataSpecLSC::createTranspose();
static constexpr_reg DataSpecLSC D8U32T = DataSpecLSC(DataSizeLSC::D8U32) | DataSpecLSC::createTranspose();
static constexpr_reg DataSpecLSC D16U32T = DataSpecLSC(DataSizeLSC::D16U32) | DataSpecLSC::createTranspose();

static constexpr_reg DataSpecLSC V1 = DataSpecLSC::createV(1,0);
static constexpr_reg DataSpecLSC V2 = DataSpecLSC::createV(2,1);
static constexpr_reg DataSpecLSC V3 = DataSpecLSC::createV(3,2);
static constexpr_reg DataSpecLSC V4 = DataSpecLSC::createV(4,3);
static constexpr_reg DataSpecLSC V8 = DataSpecLSC::createV(8,4);
static constexpr_reg DataSpecLSC V16 = DataSpecLSC::createV(16,5);
static constexpr_reg DataSpecLSC V32 = DataSpecLSC::createV(32,6);
static constexpr_reg DataSpecLSC V64 = DataSpecLSC::createV(64,7);

static constexpr_reg DataSpecLSC V1T = DataSpecLSC::createV(1,0) | DataSpecLSC::createTranspose();
static constexpr_reg DataSpecLSC V2T = DataSpecLSC::createV(2,1) | DataSpecLSC::createTranspose();
static constexpr_reg DataSpecLSC V3T = DataSpecLSC::createV(3,2) | DataSpecLSC::createTranspose();
static constexpr_reg DataSpecLSC V4T = DataSpecLSC::createV(4,3) | DataSpecLSC::createTranspose();
static constexpr_reg DataSpecLSC V8T = DataSpecLSC::createV(8,4) | DataSpecLSC::createTranspose();
static constexpr_reg DataSpecLSC V16T = DataSpecLSC::createV(16,5) | DataSpecLSC::createTranspose();
static constexpr_reg DataSpecLSC V32T = DataSpecLSC::createV(32,6) | DataSpecLSC::createTranspose();
static constexpr_reg DataSpecLSC V64T = DataSpecLSC::createV(64,7) | DataSpecLSC::createTranspose();

static constexpr_reg DataSpecLSC transpose = DataSpecLSC::createTranspose();

static constexpr_reg CacheSettingsLSC L1UC_L3UC = CacheSettingsLSC::L1UC_L3UC;
static constexpr_reg CacheSettingsLSC L1UC_L3C  = CacheSettingsLSC::L1UC_L3C;
static constexpr_reg CacheSettingsLSC L1C_L3UC  = CacheSettingsLSC::L1C_L3UC;
static constexpr_reg CacheSettingsLSC L1C_L3C   = CacheSettingsLSC::L1C_L3C;
static constexpr_reg CacheSettingsLSC L1S_L3UC  = CacheSettingsLSC::L1S_L3UC;
static constexpr_reg CacheSettingsLSC L1S_L3C   = CacheSettingsLSC::L1S_L3C;
static constexpr_reg CacheSettingsLSC L1IAR_L3C = CacheSettingsLSC::L1IAR_L3C;
static constexpr_reg CacheSettingsLSC L1UC_L3WB = CacheSettingsLSC::L1UC_L3WB;
static constexpr_reg CacheSettingsLSC L1WT_L3UC = CacheSettingsLSC::L1WT_L3UC;
static constexpr_reg CacheSettingsLSC L1WT_L3WB = CacheSettingsLSC::L1WT_L3WB;
static constexpr_reg CacheSettingsLSC L1S_L3WB  = CacheSettingsLSC::L1S_L3WB;
static constexpr_reg CacheSettingsLSC L1WB_L3WB = CacheSettingsLSC::L1WB_L3WB;
