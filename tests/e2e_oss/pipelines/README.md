# Tests and tests configurations

This folder contains common test utilities and test classes that specify testing
pipelines to execute via pytest runners.

    e2e_oss/pipelines/
    |__ experimental/
    |__ no_label/          Should be decided later
    |__ production/        Test classes for internal models

All models are divided by memory consumption into lightweight and heavyweight. 

Any new E2E test should be placed in a correct folder, and should be run in parallel accurately (i.e. meet "memory consumption" requirements)

* Model is determined as heavy if MO memory consumption greater than 2 Gb
* Model is determined as light if MO memory consumption less than 2 Gb
