// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
//  pwl_design.cpp : simple activation function designer
//

#include <vector>
#include <iostream>
#include <limits>
#include <cstdint>
#include <map>

#ifdef _NO_MKL_
#include <cmath>
#include <backend/make_pwl.hpp>

#define SCOPY(num, in, inci, out, inco) for (int i_ = 0; i_ < *(num); i_++) *(out + i_ * *(inco)) = *(in + i_ * *(inci));
#define SSCAL(num, scale, inout, inco)  for (int i_ = 0; i_ < *(num); i_++) *(inout + i_ * *(inco)) = *(scale) * *(inout + i_ * *(inco));
#define TANH(num, in, out) for (int i_ = 0; i_ < num; i_++) *(out+i_) = tanh(*(in+i_))
#else
#include <mkl.h>
#define SCOPY(num, in, incx, out, incy) scopy(num, in, incx, out, incy)
#define SSCAL(num, scale, inout, incx) sscal(num, scale, inout, incx)
#define TANH(num, in, out) vsTanh(num, in, out)
#endif

#include "pwl.h"
#include "gna_plugin_log.hpp"
#include "backend/dnn_types.h"
#include "gna_slope_scale.h"
#include "round_float_define.hpp"

double first_deriv_tanh(const double x) { return(1.0 - tanh(x) * tanh(x)); }
double first_deriv_exp(const double x) { return(exp(x)); }
double first_deriv_log(const double x) { return(1.0 / x); }


std::map<std::string, std::vector<pwl_t>> pwl_search_map {
    {"log", {{1.0769533473860933e-05 , 8.4918474385631271e-06 , -11.662751279293021 , 92854.532875275778 , -12.451257806448908},
        {1.7021658371797054e-05 , 1.3421901942456181e-05 , -11.204973371284382 , 58748.682305649265 , -11.993492424439317},
        {2.6901160981803783e-05 , 2.121301943569138e-05 , -10.747255484868321 , 37173.116828541606 , -11.5358095346374},
        {4.2508975575310986e-05 , 3.3523097254749539e-05 , -10.289651523932033 , 23524.443637281987 , -11.078263735848511},
        {6.7159638615065504e-05 , 5.2968285045173431e-05 , -9.8322142997398423 , 14889.895488146327 , -10.620906528248819},
        {0.00010607938536412906 , 8.3674876369534097e-05 , -9.3749963641228913 , 9426.9022823556988 , -10.163791247146683},
        {0.00016750484167401942 , 0.0001321476299142904 , -8.9180484530997628 , 5969.97668847147 , -9.7069667231248324},
        {0.00026440839561966089 , 0.00020863498397988411 , -8.4614207323653048 , 3782.0281676624659 , -9.2504841185370346},
        {0.00041721037390891224 , 0.00032927394464933357 , -8.0051607849959225 , 2396.8723275762272 , -8.794388391117776},
        {0.00065803047888043399 , 0.00051945736028058084 , -7.5493154189054579 , 1519.6864462895246 , -8.3387277287491912},
        {0.001037362756270747 , 0.00081911613772069059 , -7.0939280363180339 , 963.98294035052527 , -7.8835420192465913},
        {0.0016345178691889663 , 0.0012909987974520291 , -6.6390412024897918 , 611.8012037985186 , -7.4288758208733832},
        {0.0025739995666866798 , 0.0020336387647946299 , -6.1846931764806961 , 388.50045390148483 , -6.9747627596750643},
        {0.004051067401148134 , 0.0032016468460507675 , -5.7309215067520842 , 246.84852187761314 , -6.5212432982738378},
        {0.0063717878399278975 , 0.0050374410331472524 , -5.2777584251958025 , 156.94182309926316 , -6.0683436046929682},
        {0.010015399744147635 , 0.0079208190427630248 , -4.8252358236823785 , 99.846239345996764 , -5.6160998176424242},
        {0.015731959416888035 , 0.012446344656992419 , -4.3733791100375914 , 63.564872848992557 , -5.1645294255940524},
        {0.024693977573234764 , 0.019544055268788288 , -3.9222140374798418 , 40.495703741299124 , -4.7136643095482684},
        {0.038733765155746253 , 0.030667594605217476 , -3.4717584839571183 , 25.817268111660667 , -4.2635119962197363},
        {0.060710894498842323 , 0.048087203788268418 , -3.0220317632779459 , 16.471508256546421 , -3.8141005375106394},
        {0.095087507492990303 , 0.07534567638130947 , -2.5730436069008258 , 10.516628591549917 , -3.3654261013821722},
        {0.14881589164648179 , 0.1179673976691515 , -2.1248067941840345 , 6.7197124509762807 , -2.9175137851107023},
        {0.23272951343232598 , 0.18456019070518948 , -1.6773223736744836 , 4.2968336299589422 , -2.4703468078481774},
        {0.36368021928596278 , 0.28852643303350278 , -1.2305967272577258 , 2.7496683816440872 , -2.0239487374384986},
        {0.56789454936972605 , 0.45071871177964989 , -0.78462174664264084 , 1.7608902939988476 , -1.5782879515390904},
        {0.88610468810159015 , 0.70355895583540895 , -0.33939781495255494 , 1.1285348259949077 , -1.1333885987534271},
        {1.3816335789411809 , 1.0974241494088073 , 0.10509277274225079 , 0.72378090344789758 , -0.68920186958239626},
        {2.1526786672071778 , 1.7105408556958117 , 0.54885493633765814 , 0.46453751562343976 , -0.24575546313966701},
        {3.3517682459038256 , 2.6643098700159529 , 0.99191642472855346 , 0.29834998324305195 , 0.19701961965499604},
        {5.2150907594192395 , 4.1470315870077039 , 1.4342864241471516 , 0.19175121702222525 , 0.63908807030881409},
        {8.1092990368633888 , 6.4506248753490381 , 1.8760032407108322 , 0.12331522064412515 , 1.0805430109146834},
        {12.601356824411839 , 10.02743923506598 , 2.3170788926824022 , 0.079356533898219603 , 1.5213360711125516},
        {19.571181442374591 , 15.578135417317394 , 2.7575629024279547 , 0.051095535695910903 , 1.9615897281366799},
        {30.377546048984698 , 24.187602070756132 , 3.1974682131414882 , 0.032919051406834175 , 2.4012352971662185},
        {47.129543514739666 , 37.535139255177242 , 3.6368564758700774 , 0.02121811342575923 , 2.840431633702059},
        {73.079549140200285 , 58.219475920898645 , 4.0757390773799447 , 0.013683718793633217 , 3.2790801405656671},
        {113.27901633813272 , 90.261352592829454 , 4.5141911073789212 , 0.0088277602712848978 , 3.7173855249275034},
        {175.50631439356027 , 139.88074429844423 , 4.9522192021630707 , 0.0056978006942677408 , 4.155206600184707},
        {271.85671771564853 , 216.69890827860715 , 5.3899137902216161 , 0.0036784082747808367 , 4.5928067328736137},
        {420.92464268160882 , 335.59785527230628 , 5.8272726607059671 , 0.0023757221568907016 , 5.02998540013055},
        {651.67787232666603 , 519.59734639612975 , 6.2644043286254485 , 0.0015345004678918 , 5.4670819574652496},
        {1008.5639944354145 , 804.30768425774158 , 6.7012924752877234 , 0.00099150872479816517 , 5.9038143889239647},
        {1561.0124734004803 , 1244.8233916434529 , 7.1380676425712917 , 0.00064060987150321626 , 6.3406214896063817},
        {2415.300015438163 , 1926.4004262204676 , 7.5746926191112163 , 0.00041402724034620179 , 6.7771103668414092},
        {0 , 2981 , 8.0113255703134367 , 0 , 0}}},
    {"exp", {{-5.2905549738656035 , -7.6246190071105957 , -0.0029375872840971921 , 0.0050389629907875762 , 0.035482585711588618},
        {-3.2765565204702316 , -3.966387017312524 , 0.015496108324210485 , 0.037758052285013388 , 0.16525915670649505},
        {-2.304345998844584 , -2.7128986917228044 , 0.062825386060480992 , 0.099824064363453618 , 0.33363795967454735},
        {-1.6505759560844804 , -1.9420942111377082 , 0.1397702221420461 , 0.19193932815516293 , 0.512534480241849},
        {-1.1551903296891044 , -1.3825157257724856 , 0.24717534067313054 , 0.31499757780654358 , 0.6826644455709191},
        {-0.75535692384923692 , -0.94198675918566444 , 0.38594089810159893 , 0.46984288666759971 , 0.82852667624004861},
        {-0.41999557300201118 , -0.57832149976517178 , 0.55680643336844471 , 0.65704972856644539 , 0.93679241781329048},
        {-0.13153483933686491 , -0.2688406706913587 , 0.76015072810791207 , 0.87674872926137593 , 0.99585644451033684},
        {0.4235904473306159 , 0 , 0.94308787650971637 , 1.5274359002153208 , 0.94308787650971637},
        {1.0937368353927797 , 0.79581116380659767 , 2.1586384179000495 , 2.9854092373666563 , -0.2171835817276766},
        { 1.5979006351086957 , 1.3669113028909896 , 3.8636060485839758 , 4.9426451092152783 , -2.8925514173812577},
        { 2.0041923346353099 , 1.8147648658250355 , 6.0771872710645756 , 7.420098519377718 , -7.3885468228624731},
        { 2.3452256111159864 , 2.1843822125126793 , 8.8197843979578803 , 10.435626855401082 , -13.975613281399871},
        { 2.6387547172826245 , 2.4991598196640292 , 12.1046860486254 , 13.99576406829804 , -22.872965156362628},
        { 0 , 2.7725581832447883 , 15.931105041960471 , -0 , -0}}},
    {"sigmoid", {{-6.0269768546940687 , -10 , -0.0033685324745532531 , 0.0024011761556240077 , 0.020643229081686823},
        {-3.4572777895083773 , -4.2646607997060624 , 0.010403027257608216 , 0.029619100828046807 , 0.13671844548152082},
        {-2.302945392313446 , -2.7960754970003254 , 0.053901203413037058 , 0.082620267964448268 , 0.28491371022403178},
        {-1.4431692770391085 , -1.8482809500056467 , 0.13220824286098024 , 0.15455301637191463 , 0.41786563878710092},
        {-0.58709153507881506 , -1.0390898867848257 , 0.25727116250295479 , 0.22963741468060328 , 0.49588507772498291},
        {0.58709153507881362 , 0 , 0.50411492227501709 , 0.22963741468060325 , 0.50411492227501709},
        {1.4431692770391091 , 1.0390898867848253 , 0.7427288374970451 , 0.15455301637191451 , 0.58213436121289919},
        {2.3029453923134513 , 1.8482809500056492 , 0.86779175713902001 , 0.082620267964447991 , 0.71508628977596878},
        {3.4572777895083746 , 2.7960754970003263 , 0.94609879658696283 , 0.029619100828046918 , 0.86328155451847877},
        {6.0269768546940705 , 4.2646607997060606 , 0.98959697274239178 , 0.0024011761556240298 , 0.97935677091831308},
        { 0 , 10 , 1.0033685324745534 , 0 , 0}}},
    {"tanh", {{-3.0134884273470361 , -5 , -1.0067370649491065 , 0.0096047046224959371 , -0.95871354183662683},
        {-1.7286388947541886 , -2.1323303998530339 , -0.979193945484784 , 0.11847640331218724 , -0.72656310903695842},
        {-1.1514726961567241 , -1.3980377485001632 , -0.892197593173926 , 0.3304810718577928 , -0.43017257955193672},
        {-0.72158463851955434 , -0.92414047500282348 , -0.73558351427803959 , 0.61821206548765828 , -0.16426872242579849},
        {-0.29354576753940709 , -0.51954494339241275 , -0.48545767499409032 , 0.91854965872241312 , -0.0082298445500341155},
        {0.29354576753940703 , 0 , 0.0082298445500341155 , 0.91854965872241323 , 0.0082298445500341155},
        {0.72158463851955434 , 0.51954494339241275 , 0.48545767499409037 , 0.6182120654876585 , 0.16426872242579826},
        {1.1514726961567245 , 0.92414047500282415 , 0.73558351427804003 , 0.33048107185779213 , 0.43017257955193755},
        {1.7286388947541889 , 1.3980377485001632 , 0.89219759317392588 , 0.11847640331218723 , 0.72656310903695842},
        {3.0134884273470322 , 2.1323303998530312 , 0.97919394548478356 , 0.0096047046224960447 , 0.95871354183662627},
        { 0 , 5 , 1.0067370649491065 , 0 , 0}}},
};

double sigmoid(const double x) { return(0.5 * (1.0 + tanh(x / 2))); }
double first_deriv_sigmoid(const double x) { return(sigmoid(x) * (1.0 - sigmoid(x))); }
double relu(const double x) { if (x < 0) { return(0.0); } else { return(x); } }
double leaky_relu(const double x) { if (x < 0.0) { return(LEAKYRELU_SLOPE*x); } else { return(x); } }
double clipping(const double x, const double lbound, const double ubound) { return((x < lbound)?lbound:((x > ubound)?ubound:x)); }

double pivot_search(std::vector<pwl_t>& result, double(*f)(const double),
                                    double(*first_deriv_f)(const double),
                                    const uint32_t N,
                                    const double alpha_0,
                                    const double alpha_N,
                                    const double threshold,
                                    const bool negative) {
    std::vector<std::vector<double>> t(N + 1);
    std::vector<std::vector<double>> alpha(N + 1);
    std::vector<std::vector<double>> epsilon(N + 1);
    std::vector<std::vector<double>> d(N + 1);
    bool same_epsilon = false;
    double Delta;
    double epsilon_final = 0.0;
    double max_epsilon = 0.0;
    double max_epsilon_prev;
    double min_epsilon;
    double sgn = (negative) ? -1.0 : 1.0;
    int j;

    if ( f == nullptr ||
        first_deriv_f == nullptr ||
        threshold < 0) {
        return epsilon_final;
    }
    // Figure 4:  Box #1
    j = 0;
    Delta = 1.0;

    for (int i = 0; i < N; i++) {
        t[i].push_back(alpha_0 + (static_cast<double>((i + 1)) / static_cast<double>((N + 1))) * (alpha_N - alpha_0));
    }

    while (true) {
        // Figure 4:  Box #2
        alpha[0].resize(j + 1);
        alpha[0][j] = alpha_0;
        for (int i = 1; i < N; i++) {
            alpha[i].resize(j + 1);
            alpha[i][j] = (f(t[i - 1][j]) - f(t[i][j]) + first_deriv_f(t[i][j]) * t[i][j] - first_deriv_f(t[i - 1][j]) * t[i - 1][j])
                / (first_deriv_f(t[i][j]) - first_deriv_f(t[i - 1][j]));
        }
        alpha[N].resize(j + 1);
        alpha[N][j] = alpha_N;

        // Figure 4:  Box #3
        for (int i = 0; i < N; i++) {
            epsilon[i].resize(j + 1);
            epsilon[i][j] = sgn * (first_deriv_f(t[i][j]) * (alpha[i][j] - t[i][j]) + f(t[i][j]) - f(alpha[i][j]));
        }
        epsilon[N].resize(j + 1);
        epsilon[N][j] = sgn * (first_deriv_f(t[N - 1][j]) * (alpha[N][j] - t[N - 1][j]) + f(t[N - 1][j]) - f(alpha[N][j]));

        // Figure 4:  Test for completion
        max_epsilon_prev = max_epsilon;
        max_epsilon = fabs(epsilon[0][j]);
        min_epsilon = fabs(epsilon[0][j]);
        for (int i = 1; i < N + 1; i++) {
            if (fabs(epsilon[i][j]) > max_epsilon) max_epsilon = fabs(epsilon[i][j]);
            if (fabs(epsilon[i][j]) < min_epsilon) min_epsilon = fabs(epsilon[i][j]);
        }
        if ((j == PWL_MAX_ITERATIONS) || (max_epsilon - min_epsilon < threshold * min_epsilon)) {
            pwl_t value;
            result.resize(0);
            epsilon_final = (max_epsilon + min_epsilon) / 4.0;  // Andrzej's modification
            for (int i = 0; i < N; i++) {
                double val, val_next;
                value.t = t[i][j];
                value.alpha = alpha[i][j];
                val = sgn * first_deriv_f(value.t) * (value.alpha - value.t) + sgn * f(value.t) - epsilon_final;
                val_next = sgn * first_deriv_f(value.t) * (alpha[i + 1][j] - value.t) + sgn * f(value.t) - epsilon_final;
                value.beta = val;
                value.m = (val_next - val) / (alpha[i + 1][j] - value.alpha);
                value.b = (val - value.m * value.alpha);
                result.push_back(value);
            }
            value.t = value.m = value.b = 0.0;
            value.alpha = alpha[N][j];
            value.beta = sgn * first_deriv_f(t[N - 1][j]) * (alpha[N][j] - t[N - 1][j]) + sgn * f(t[N - 1][j]) - epsilon_final;
            result.push_back(value);
            if (j == PWL_MAX_ITERATIONS) {
                std::cerr << "Error:  failed to converge in pivot_search!" << std::endl;
            }
            return(epsilon_final);
        }

        if (j > 0) {
            if (max_epsilon > max_epsilon_prev) {
                j = j - 1;
                Delta = Delta / 2;
            } else if (max_epsilon == max_epsilon_prev) {
                if (!same_epsilon) {
                    same_epsilon = true;
                } else {
                    j = j - 1;
                    Delta = Delta / 2;
                    same_epsilon = false;
                }
            }
        }

        // Figure 4:  Box #4
        for (int i = 0; i < N; i++) {
            d[i].resize(j + 1);
            d[i][j] = Delta * (epsilon[i + 1][j] - epsilon[i][j]) /
                ((epsilon[i + 1][j] / (alpha[i + 1][j] - t[i][j])) + (epsilon[i][j] / (t[i][j] - alpha[i][j])));
        }

        // Figure 4:  Box #5
        for (int i = 0; i < N; i++) {
            t[i].resize(j + 2);
            t[i][j + 1] = t[i][j] + d[i][j];
        }
        t[N].resize(j + 2);

        j = j + 1;
    }
}

double calculate_error_pct(const DnnActivationType fun,
                            const double l_bound,
                            const double u_bound,
                            const double offset,
                            const int samples) {
    double delta = (u_bound - l_bound) / (samples + 1);
    double min_val = 0.0;
    double max_val = 0.0;

    if ( delta < 0 ) {
        return 0.0;
    }

    switch (fun) {
        case kActSigmoid:
            min_val = max_val = sigmoid(l_bound); break;
        case kActTanh:
            min_val = max_val = tanh(l_bound); break;\
        case kActExp:
            min_val = max_val = exp(l_bound);
            break;
        case kActLog:
            min_val = max_val = log(l_bound);
            break;
        default:
            break;
    }

    for (int i = 0; i < samples; i++) {
        double arg = l_bound + i * delta;
        double val = 0.0;
        switch (fun) {
            case kActSigmoid:
                val = sigmoid(arg);
                break;
            case kActTanh:
                val = tanh(arg);
                break;
            case kActExp:
                val = exp(arg);
                break;
            case kActLog:
                val = log(arg);
                break;
            default:
                break;
        }
        if (val > max_val) max_val = val;
        if (val < min_val) min_val = val;
    }

    return(100.0 * fabs(offset) / (max_val - min_val));
}

bool split_search(const DnnActivationType fun,
                    const double l_bound,
                    const double u_bound) {
    bool is_split = false;
    if (l_bound > u_bound) {
        return is_split;
    }

    switch (fun) {
        case kActSigmoid:
        case kActTanh:
        case kActExp:
            if ((l_bound < 0.0) && (u_bound > 0.0)) {
                is_split = true;
            }
            break;
        default:
            is_split = false;
    }
    return(is_split);
}

inline std::vector<pwl_t> negative_pwl(const std::vector<pwl_t>& pwl) {
    std::vector<pwl_t> new_pwl;
    new_pwl = pwl;
    for (uint32_t i = 0; i < pwl.size(); i++) {
        new_pwl[i].m = -pwl[i].m;
        new_pwl[i].b = -pwl[i].b;
        new_pwl[i].beta = -pwl[i].beta;
    }

    return(new_pwl);
}

std::vector<pwl_t> pwl_search(const DnnActivationType fun,
                                const double l_bound,
                                const double u_bound,
                                const double threshold,
                                const double allowed_err_pct,
                                const int samples,
                                double& err_pct) {
    std::vector<pwl_t> pwl;
    double err = 0.0;
    int n_segments = 1;

    if (l_bound > u_bound ||
        threshold < 0) {
        return pwl;
    }

    if (split_search(fun, l_bound, u_bound)) {
        std::vector<pwl_t> pwl2;
        double err_pct1 = 0.0, err_pct2 = 0.0;

        pwl = pwl_search(fun, l_bound, 0.0, threshold, allowed_err_pct, samples, err_pct1);
        pwl = negative_pwl(pwl);
        pwl2 = pwl_search(fun, 0.0, u_bound, threshold, allowed_err_pct, samples, err_pct2);
        if (fun == kActExp) {
            pwl2 = negative_pwl(pwl2);  // both regions of exp are concave
        }
        // merge
        pwl.pop_back();  // remove final alpha and beta from first half
        pwl.insert(pwl.end(), pwl2.begin(), pwl2.end());  // concatenate the two halves
        err_pct = (err_pct1 + err_pct2) / 2;  // this is not quite correct but should give an indication

    } else {
        if (fun == kActIdentity) {
            pwl.resize(2);
            pwl[0].alpha = pwl[0].t = pwl[0].beta = -std::numeric_limits<float>::infinity();
            pwl[0].m = 1.0;
            pwl[0].b = 0.0;
            pwl[1].alpha = std::numeric_limits<float>::infinity();
            pwl[1].beta = std::numeric_limits<float>::infinity();

        } else if (fun == kActKaldiLstmClipping) {
            pwl.resize(4);
            pwl[0].alpha = pwl[0].t = pwl[0].beta = -std::numeric_limits<float>::infinity();
            pwl[0].m = 0.0;
            pwl[0].b = pwl[0].beta = KALDI_LSTM_CLIP_LOWER;
            //pwl[1].alpha = pwl[0].t = pwl[1].beta = KALDI_LSTM_CLIP_LOWER;
            pwl[1].alpha = pwl[1].t = pwl[1].beta = KALDI_LSTM_CLIP_LOWER;
            pwl[1].m = 1.0;
            pwl[1].b = 0.0;
            //pwl[2].alpha = pwl[0].t = pwl[1].beta = KALDI_LSTM_CLIP_UPPER;
            pwl[2].alpha = pwl[2].t = pwl[2].beta = KALDI_LSTM_CLIP_UPPER;
            pwl[2].m = 0.0;
            pwl[2].b = KALDI_LSTM_CLIP_UPPER;
            pwl[3].alpha = pwl[3].beta = std::numeric_limits<float>::infinity();

        } else {
            bool negative = false;

            switch (fun) {
                case kActSigmoid:
                    if (u_bound == 0) negative = true;  // make left half convex
                    err = pivot_search(pwl, sigmoid, first_deriv_sigmoid, n_segments, l_bound, u_bound, threshold, negative);
                    break;
                case kActTanh:
                    if (u_bound == 0) negative = true;  // make left half convex
                    err = pivot_search(pwl, tanh, first_deriv_tanh, n_segments, l_bound, u_bound, threshold, negative);
                    break;
                case kActExp:
                    negative = true;  // make function convex
                    err = pivot_search(pwl, exp, first_deriv_exp, n_segments, l_bound, u_bound, threshold, negative);
                    break;
                case kActLog:
                    err = pivot_search(pwl, log, first_deriv_log, n_segments, l_bound, u_bound, threshold, negative);
                    break;
                default:
                    break;
            }
            err_pct = calculate_error_pct(fun, l_bound, u_bound, err, samples);

            while ((n_segments < PWL_MAX_ITERATIONS) && (allowed_err_pct < err_pct)) {
                n_segments += 1;
                switch (fun) {
                    case kActSigmoid:
                        err = pivot_search(pwl, sigmoid, first_deriv_sigmoid, n_segments, l_bound, u_bound, threshold, negative);
                        break;
                    case kActTanh:
                        err = pivot_search(pwl, tanh, first_deriv_tanh, n_segments, l_bound, u_bound, threshold, negative);
                        break;
                    case kActExp:
                        err = pivot_search(pwl, exp, first_deriv_exp, n_segments, l_bound, u_bound, threshold, negative);
                        break;
                    case kActLog:
                        err = pivot_search(pwl, log, first_deriv_log, n_segments, l_bound, u_bound, threshold, negative);
                        break;
                    default:
                        break;
                }
                err_pct = calculate_error_pct(fun, l_bound, u_bound, err, samples);
            }

            if (n_segments >= PWL_MAX_ITERATIONS) {
                std::cerr << "Error:  failed to converge in pwl_search!" << std::endl;
            }
        }
    }
    return(pwl);
}


void PwlDesignOpt16(const DnnActivation activation_type,
                    std::vector<intel_pwl_segment_t> &ptr_segment,
                    const float scale_in,
                    const float scale_out,
                    const uint32_t n) {
    std::vector<pwl_t> pwl;
    double err_pct = 0.0;
    switch (activation_type) {
        case kActSigmoid:
            if ( pwl_search_map.find("sigmoid") == pwl_search_map.end() ) {
                pwl = pwl_search(kActSigmoid, -SIGMOID_DOMAIN, SIGMOID_DOMAIN, PWL_DESIGN_THRESHOLD, PWL_MAX_ERR_PERCENT, PWL_DESIGN_SAMPLES, err_pct);
            } else {
                pwl = pwl_search_map["sigmoid"];
            }

            make_gna_pwl(activation_type, pwl, -SIGMOID_DOMAIN, SIGMOID_DOMAIN, scale_in, scale_out, ptr_segment, n);
            break;
        case kActTanh:
            if ( pwl_search_map.find("tanh") == pwl_search_map.end() ) {
                pwl = pwl_search(kActTanh, -TANH_DOMAIN, TANH_DOMAIN, PWL_DESIGN_THRESHOLD, PWL_MAX_ERR_PERCENT, PWL_DESIGN_SAMPLES, err_pct);
            } else {
                pwl = pwl_search_map["tanh"];
            }
            make_gna_pwl(activation_type, pwl, -TANH_DOMAIN, TANH_DOMAIN, scale_in, scale_out, ptr_segment, n);
            break;
        case kActRelu:
            make_gna_pwl(activation_type, pwl, -1.0, 1.0, scale_in, scale_out, ptr_segment, n);
            break;
        case kActLeakyRelu:
            make_gna_pwl(activation_type, pwl, -1.0, 1.0, scale_in, scale_out, ptr_segment, n);
            break;
        case kActIdentity:
            make_gna_pwl(activation_type, pwl, -1.0, 1.0, scale_in, scale_out, ptr_segment, n);
            break;
        case kActKaldiLstmClipping:
            make_gna_pwl(activation_type, pwl, KALDI_LSTM_CLIP_LOWER, KALDI_LSTM_CLIP_UPPER, scale_in, scale_out, ptr_segment, n);
            break;
        case kActDivByN:
            make_gna_pwl(activation_type, pwl, -1.0, 1.0, scale_in, scale_out, ptr_segment, n);
            break;
        case kActLog: {
            double x_min = (1 + ~XBASEMASK) / scale_in;
            double x_max = ((INT32_MAX / scale_in) < LOG_DOMAIN) ? (INT32_MAX / scale_in) : LOG_DOMAIN;
            if ( pwl_search_map.find("log") == pwl_search_map.end() ) {
                pwl = pwl_search(kActLog, x_min, x_max, PWL_DESIGN_THRESHOLD, 0.066*PWL_MAX_ERR_PERCENT, PWL_DESIGN_SAMPLES, err_pct);
            } else {
                pwl = pwl_search_map["log"];
            }
            make_gna_pwl(activation_type, pwl, x_min, x_max, scale_in, scale_out, ptr_segment, n);
            break;
        }
        case kActExp: {
            double x_min = -log(scale_out);
            double x_max = x_min + log(INT16_MAX);
            if ( pwl_search_map.find("exp") == pwl_search_map.end() ) {
                pwl = pwl_search(kActExp, x_min, x_max, PWL_DESIGN_THRESHOLD, 0.5*PWL_MAX_ERR_PERCENT, PWL_DESIGN_SAMPLES, err_pct);
            } else {
                pwl = pwl_search_map["exp"];
            }
            make_gna_pwl(activation_type, pwl, x_min, x_max, scale_in, scale_out, ptr_segment, n);
            break;
        }
        default:
            break;
    }
}

void PwlDesign16(const DnnActivation activation_type,
                 intel_pwl_segment_t *ptr_segment,
                 const uint32_t num_segments,
                 const float scale_in,
                 const float scale_out,
                 const uint32_t n) {
    switch (activation_type) {
        case kActSigmoid:
           {
                gnalog() <<  "=========================== Sigmoid Segments===========================\n";
                uint32_t num_segment_size = 0;
                int32_t offset = 0;
                ptr_segment[0].xBase = static_cast<int32_t>(INT32_MIN & XBASEMASK);  // zero out the 2 lsb
                num_segment_size = static_cast<int32_t>(SIGMOID_DOMAIN * scale_in / ((num_segments-2) / 2) + 0.5);
                offset = -static_cast<int32_t>(num_segment_size * (num_segments-2) / 2);
                for (uint32_t i = 1; i < num_segments; i++) {
                    ptr_segment[i].xBase = static_cast<int32_t>(offset & XBASEMASK);  // zero out the 2 lsb
                    offset += num_segment_size;
                }
                for (uint32_t i = 0; i < num_segments; i++) {
                    int32_t xbase = static_cast<int32_t>(ptr_segment[i].xBase & XBASEMASK);
                    int32_t xbasenext = (i < num_segments-1) ? static_cast<int32_t>(ptr_segment[i+1].xBase & XBASEMASK) : INT32_MAX;
                    float floatarg = static_cast<float>(xbase / (2 * scale_in));
                    float floatargnext = static_cast<float>(xbasenext / (2 * scale_in));
                    float floatval, floatvalnext, slope;
                    TANH(1, &floatarg, &floatval);
                    floatval = 0.5f * (1.0f + floatval);
                    TANH(1, &floatargnext, &floatvalnext);
                    floatvalnext = 0.5f * (1.0f + floatvalnext);
                    slope = scale_out*(floatvalnext - floatval) / static_cast<float>(xbasenext - xbase);
                    {
                        // find best scale factor
                        uint64_t slope_scale;
                        uint32_t slope_scale_index;
                        for (slope_scale_index = 3; slope_scale_index > 0; slope_scale_index--) {
                            slope_scale = static_cast<uint64_t>(1) << (8 * (1 + slope_scale_index));
                            if (((slope * slope_scale) <= 32767.0) && ((slope * slope_scale) >= -32768.0))
                                break;
                        }
                        slope_scale = static_cast<uint64_t>(1) << (8 * (1 + slope_scale_index));
                        ptr_segment[i].slope = FLOAT_TO_INT16(slope * slope_scale);

                        ptr_segment[i].xBase = ptr_segment[i].xBase | slope_scale_index;
                    }
                    ptr_segment[i].yBase = FLOAT_TO_INT16(floatval * scale_out);
                    gnalog() << (static_cast<int32_t>((ptr_segment[i].xBase & XBASEMASK))/scale_out)
                             << " "
                             << (static_cast<float>((ptr_segment[i].yBase))/scale_out)
                             << " "
                             << (slope/scale_out)
                             << "\n";
                }
            }
            break;
        case kActTanh:
            {
                gnalog() <<  "=========================== Tanh Segments===========================\n";
                uint32_t num_segment_size = 0;
                int32_t offset = 0;
                ptr_segment[0].xBase = static_cast<int32_t>(INT32_MIN & XBASEMASK);  // zero out the 2 lsb
                num_segment_size = static_cast<int32_t>(TANH_DOMAIN * scale_in / ((num_segments-2) / 2) + 0.5);
                offset = -static_cast<int32_t>(num_segment_size * (num_segments-2) / 2);
                for (uint32_t i = 1; i < num_segments; i++) {
                    ptr_segment[i].xBase = static_cast<int32_t>(offset & XBASEMASK);  // zero out the 2 lsb
                    offset += num_segment_size;
                }
                for (uint32_t i = 0; i < num_segments; i++) {
                    int32_t xbase = static_cast<int32_t>(ptr_segment[i].xBase & XBASEMASK);
                    int32_t xbasenext = (i < num_segments-1) ?
                                                    static_cast<int32_t>(ptr_segment[i+1].xBase & XBASEMASK) :
                                                    INT32_MAX;
                    float floatarg = static_cast<float>(xbase / scale_in);
                    float floatargnext = static_cast<float>(xbasenext / scale_in);
                    float floatval, floatvalnext, slope;
                    TANH(1, &floatarg, &floatval);
                    TANH(1, &floatargnext, &floatvalnext);
                    slope = scale_out * (floatvalnext - floatval) /
                                                static_cast<float>(xbasenext - xbase);
                    {
                        // find best scale factor
                        uint64_t slope_scale;
                        uint32_t slope_scale_index;
                        for (slope_scale_index = 3; slope_scale_index > 0; slope_scale_index--) {
                            slope_scale = static_cast<uint64_t>(1) << (8 * (1 + slope_scale_index));
                            if (((slope * slope_scale) <= 32767.0) && ((slope * slope_scale) >= -32768.0))
                                break;
                        }
                        slope_scale = static_cast<uint64_t>(1) << (8 * (1 + slope_scale_index));
                        ptr_segment[i].slope = FLOAT_TO_INT16(slope * slope_scale);
                        ptr_segment[i].xBase = ptr_segment[i].xBase | slope_scale_index;
                    }
                    ptr_segment[i].yBase = FLOAT_TO_INT16(floatval * scale_out);
                    gnalog() << (static_cast<int32_t>((ptr_segment[i].xBase & XBASEMASK))/scale_out)
                             << " "
                             << (static_cast<float>((ptr_segment[i].yBase))/scale_out)
                             << " "
                             << (slope/scale_out)
                             << "\n";
                }
            }
            break;
        case kActRelu:
            std::cerr << "Rectilinear activation function design not yet implemented!" << std::endl;
            throw -1;
        case kActIdentity:
        case kActKaldiLstmClipping:  // clipping of IDENTITY is more aggressive than Kaldi
            {
                float slope = 0.0;
                int64_t x_lower_limit = static_cast<int64_t>((INT16_MIN / scale_out) * scale_in - 0.5);
                int64_t x_upper_limit = static_cast<int64_t>((INT16_MAX / scale_out) * scale_in + 0.5);
                int16_t y_lower_limit = INT16_MIN;
                int16_t y_upper_limit = INT16_MAX;
                if (activation_type == kActKaldiLstmClipping)
                    gnalog() << "=========================== Clipping Segments ===========================\n";
                else
                    gnalog() << "=========================== Identity Segments ===========================\n";
                if (x_lower_limit < INT32_MIN) {
                    std::cerr << "Warning:  saturation in PwlDesign16! " << x_lower_limit  << " < INT32_MIN"<< std::endl;
                    x_lower_limit = INT32_MIN;
                    y_lower_limit = static_cast<int16_t>((scale_out / scale_in)*static_cast<float>(INT32_MIN) - 0.5);
                }
                if (x_upper_limit > INT32_MAX) {
                    std::cerr << "Warning:  saturation in PwlDesign16! " << x_upper_limit  << " > INT32_MAX"<< std::endl;
                    x_upper_limit = INT32_MAX;
                    y_upper_limit = static_cast<int16_t>((scale_out / scale_in)*static_cast<float>(INT32_MAX) + 0.5);
                }
                slope =
                    static_cast<float>(static_cast<uint64_t>(y_upper_limit) - static_cast<uint64_t>(y_lower_limit)) /
                                               static_cast<float>(static_cast<uint64_t>(x_upper_limit) - static_cast<uint64_t>(x_lower_limit));
                ptr_segment[0].xBase = static_cast<int32_t>(INT32_MIN & XBASEMASK);  // zero out the 2 lsb
                ptr_segment[0].yBase = y_lower_limit;
                ptr_segment[0].slope = 0;

                gnalog() << ptr_segment[0].xBase / scale_in
                    << " " << ptr_segment[0].yBase / scale_out
                    << " " << 0
                    << "\n";

                ptr_segment[1].xBase = static_cast<int32_t>(x_lower_limit & XBASEMASK);
                ptr_segment[1].yBase = y_lower_limit;
                {
                    // find best scale factor
                    uint64_t slope_scale = 0;
                    uint32_t slope_scale_index = 0;
                    for (slope_scale_index = 3; slope_scale_index > 0; slope_scale_index--) {
                        slope_scale = static_cast<uint64_t>(1) << (8 * (1 + slope_scale_index));
                        if (((slope * slope_scale) <= std::numeric_limits<int16_t>::max()) &&
                                    ((slope * slope_scale) >= std::numeric_limits<int16_t>::min()))
                            break;
                    }
                    slope_scale = static_cast<uint64_t>(1) << (8 * (1 + slope_scale_index));
                    ptr_segment[1].slope = FLOAT_TO_INT16(slope * slope_scale);
                    ptr_segment[1].xBase = ptr_segment[1].xBase | slope_scale_index;
                }
                ptr_segment[2].xBase = static_cast<int32_t>(x_upper_limit & XBASEMASK);
                ptr_segment[2].yBase = y_upper_limit;
                ptr_segment[2].slope = 0;
            }
            break;
        default:
            fprintf(stderr, "Activation function design for %s not yet implemented!\n", intel_dnn_activation_name[activation_type]);
            throw -1;
    }
}

void PwlApply16(intel_dnn_component_t *component, uint32_t num_subset_size) {
    if (component->orientation_in == kDnnInterleavedOrientation) {  // subsets only supported in interleaved orientation
        PwlApply16(component, 0, num_subset_size - 1, 0, component->num_columns_in - 1);
    } else {
        PwlApply16(component, 0, component->num_rows_in - 1, 0, component->num_columns_in - 1);
    }
}

void PwlApply16(intel_dnn_component_t *component,
                uint32_t num_row_start,
                uint32_t num_row_end,
                uint32_t num_col_start,
                uint32_t num_col_end) {
    uint32_t num_saturate = 0;
    uint32_t num_segments = component->op.pwl.num_segments;
    if (num_segments > 0) {
        intel_pwl_segment_t *ptr_segment = component->op.pwl.ptr_segments;
        for (int i = num_row_start; i <= num_row_end; i++) {
            int32_t *ptr_input = reinterpret_cast<int32_t *>(component->ptr_inputs) + i * component->num_columns_in;
            int16_t *ptr_output = reinterpret_cast<int16_t *>(component->ptr_outputs) + i * component->num_columns_in;
            for (int j = num_col_start; j <= num_col_end; j++) {
                int32_t xbase = (int32_t) (ptr_segment[0].xBase & XBASEMASK);
                int32_t input = ptr_input[j];
                if (input <= xbase) {
                    ptr_output[j] = ptr_segment[0].yBase;
                } else {
                    uint32_t slope_shift;
                    int16_t slope, ybase;
                    int64_t diff, prod, prod_shift, sum;
                    uint32_t k = num_segments / 2;
                    uint32_t k_upper = num_segments;
                    uint32_t k_lower = 0;
                    while (k_upper > k_lower + 1) {
                        xbase = (int32_t) (ptr_segment[k].xBase & XBASEMASK);
                        if (xbase > input) {
                            k_upper = k;
                            k = (k + k_lower) / 2;
                        } else {
                            k_lower = k;
                            k = (k_upper + k) / 2;
                        }
                    }
                    xbase = (int32_t) (ptr_segment[k].xBase & XBASEMASK);
                    slope_shift = ((ptr_segment[k].xBase & ~XBASEMASK) + 1) * 8;
                    slope = ptr_segment[k].slope;
                    ybase = ptr_segment[k].yBase;
                    diff = (int64_t) input - (int64_t) xbase;
                    prod = diff * slope;
                    prod_shift = prod >> slope_shift;
                    sum = prod_shift + (int64_t) ybase;
                    if (sum > 32767LL) {
                        ptr_output[j] = 32767;
                        num_saturate++;
                    } else if (sum < -32768LL) {
                        ptr_output[j] = -32768;
                        num_saturate++;
                    } else {
                        ptr_output[j] = (int16_t) sum;
                    }
                }
            }
        }
    }

    if (num_saturate > 0) {
        fprintf(stderr, "Warning:  %d saturations in PwlApply16!\n", num_saturate);
    }
}

void PwlApply32(intel_dnn_component_t *component, uint32_t num_subset_size) {
    if (component->orientation_in == kDnnInterleavedOrientation) {  // subsets only supported in interleaved orientation
        PwlApply32(component, 0, num_subset_size - 1, 0, component->num_columns_in - 1);
    } else {
        PwlApply32(component, 0, component->num_rows_in - 1, 0, component->num_columns_in - 1);
    }
}

void PwlApply32(intel_dnn_component_t *component,
                uint32_t num_row_start,
                uint32_t num_row_end,
                uint32_t num_col_start,
                uint32_t num_col_end) {
    intel_piecewiselinear_t *transform = reinterpret_cast<intel_piecewiselinear_t *>(&component->op.pwl);
    float *ptr_in = reinterpret_cast<float *>(component->ptr_inputs);
    float *ptr_out = reinterpret_cast<float *>(component->ptr_outputs);
    uint32_t num_columns = component->num_columns_in;
    switch (transform->func_id.type) {
        case kActSigmoid:
            for (uint32_t i = num_row_start; i <= num_row_end; i++) {
                for (uint32_t j = num_col_start; j <= num_col_end; j++) {
                    ptr_out[i * num_columns + j] = 0.5 * (1.0 + tanh(0.5 * ptr_in[i * num_columns + j]));
                }
            }
            break;
        case kActTanh:
            for (uint32_t i = num_row_start; i <= num_row_end; i++) {
                for (uint32_t j = num_col_start; j <= num_col_end; j++) {
                    ptr_out[i * num_columns + j] = tanh(ptr_in[i * num_columns + j]);
                }
            }
            break;
        case kActRelu:
            for (uint32_t i = num_row_start; i <= num_row_end; i++) {
                for (uint32_t j = num_col_start; j <= num_col_end; j++) {
                    ptr_out[i * num_columns + j] =
                            (ptr_in[i * num_columns + j] < 0.0f) ? ptr_in[i * num_columns + j] *
                                                                   transform->func_id.negative_slope : ptr_in[
                                    i * num_columns + j];
                }
            }
            break;
        case kActIdentity:
            for (uint32_t i = num_row_start; i <= num_row_end; i++) {
                for (uint32_t j = num_col_start; j <= num_col_end; j++) {
                    ptr_out[i * num_columns + j] = ptr_in[i * num_columns + j];
                }
            }
            break;
        case kActKaldiLstmClipping:
            for (uint32_t i = num_row_start; i <= num_row_end; i++) {
                for (uint32_t j = num_col_start; j <= num_col_end; j++) {
                    float val = ptr_in[i * num_columns + j];
                    if (val > KALDI_LSTM_CLIP_UPPER) {
                        ptr_out[i * num_columns + j] = KALDI_LSTM_CLIP_UPPER;
                    } else if (val < KALDI_LSTM_CLIP_LOWER) {
                        ptr_out[i * num_columns + j] = KALDI_LSTM_CLIP_LOWER;
                    } else {
                        ptr_out[i * num_columns + j] = val;
                    }
                }
            }
            break;
        case kActDivByN:
            for (uint32_t i = num_row_start; i <= num_row_end; i++) {
                for (uint32_t j = num_col_start; j <= num_col_end; j++) {
                    ptr_out[i * num_columns + j] = ptr_in[i * num_columns + j]/(float)(num_row_end-num_row_start+1);
                }
            }
            break;
        case kActExp:
            for (uint32_t i = num_row_start; i <= num_row_end; i++) {
                for (uint32_t j = num_col_start; j <= num_col_end; j++) {
                    ptr_out[i * num_columns + j] = exp(ptr_in[i * num_columns + j]);
                }
            }
            break;
        case kActLog:
            for (uint32_t i = num_row_start; i <= num_row_end; i++) {
                for (uint32_t j = num_col_start; j <= num_col_end; j++) {
                    ptr_out[i * num_columns + j] = log(ptr_in[i * num_columns + j]);
                }
            }
            break;
        case kActCustom:
            // break;
        default:fprintf(stderr, "Unknown piecewise linear function type!\n");
            throw -1;
    }
}
