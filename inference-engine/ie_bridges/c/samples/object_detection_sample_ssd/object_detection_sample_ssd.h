// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stdlib.h>

/// @brief message for help argument
static const char* help_message = "Print a usage message.";

/// @brief message for model argument
static const char* model_message = "Required. Path to an .xml file with a trained model.";

/// @brief message for images argument
static const char* image_message = "Required. Path to one or more images or folder with images.";

/// @brief message for assigning cnn calculation to device
static const char* target_device_message = "Optional. Specify the target device to infer. "
                                           "Default value is CPU. Use \"-d HETERO:<comma-separated_devices_list>\" format to specify "
                                           "HETERO plugin. "
                                           "Sample will look for a suitable plugin for device specified.";

/// @brief message for plugin custom kernels desc
static const char* custom_plugin_config_message = "Required for GPU, MYRIAD, HDDL custom kernels. "
                                                  "Absolute path to the .xml config file with the kernels descriptions.";

/// @brief message for user extension library argument
static const char* custom_ex_library_message = "Required for CPU plugin custom layers. "
                                               "Absolute path to a shared library with the kernels implementations.";

/// @brief message for config argument
static const char* config_message = "Path to the configuration file. Default value: \"config\".";
/**
 * \brief This function show a help message
 */
static void showUsage() {
    printf("\nobject_detection_sample_ssd_c [OPTION]\n");
    printf("Options:\n\n");
    printf("    -h                      %s\n", help_message);
    printf("    -m \"<path>\"             %s\n", model_message);
    printf("    -i \"<path>\"             %s\n", image_message);
    printf("      -l \"<absolute_path>\"  %s\n", custom_ex_library_message);
    printf("          Or\n");
    printf("      -c \"<absolute_path>\"  %s\n", custom_plugin_config_message);
    printf("    -d \"<device>\"           %s\n", target_device_message);
    printf("    -g                  %s\n", config_message);
}

int opterr = 1;
int optind = 1;
int optopt;
char* optarg;

#define ERR(s, c)               \
    if (opterr) {               \
        fputs(argv[0], stderr); \
        fputs(s, stderr);       \
        fputc('\'', stderr);    \
        fputc(c, stderr);       \
        fputs("\'\n", stderr);  \
    }

/**
 * @brief Check command line arguments with available options
 * @param int argc - count of args
 * @param char *argv[] - array values of args
 * @param char *opts - array of options
 * @return option name or -1(fail)
 */
static int getopt(int argc, char** argv, char* opts) {
    static int sp = 1;
    register int c = 0;
    register char* cp = NULL;

    if (sp == 1) {
        if (optind >= argc || argv[optind][0] != '-' || argv[optind][1] == '\0')
            return -1;
        else if (strcmp(argv[optind], "--") == 0) {
            optind++;
            return -1;
        }
        optopt = c = argv[optind][sp];
        if (c == ':' || (cp = strchr(opts, c)) == 0) {
            ERR(": unrecognized option -- ", c);
            showUsage();
            if (argv[optind][++sp] == '\0') {
                optind++;
                sp = 1;
            }
            return ('?');
        }
        if (*++cp == ':') {
            if (argv[optind][sp + 1] != '\0')
                optarg = &argv[optind++][sp + 1];
            else if (++optind >= argc) {
                ERR(": option requires an argument -- ", c);
                sp = 1;
                return ('?');
            } else
                optarg = argv[optind++];
            sp = 1;
        } else {
            if (argv[optind][++sp] == '\0') {
                sp = 1;
                optind++;
            }
            optarg = NULL;
        }
    }
    return (c);
}
