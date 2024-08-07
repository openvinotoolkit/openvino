# Smart CI Overview

Smart CI is a feature designed to optimize pre-commit CI workflow by running only the necessary
builds and tests required to validate changes in a given pull request (PR).

For example, if a PR changes only the CPU plugin, GPU plugin tests are skipped in the pre-commit stage
for this PR, as they are unrelated. This approach reduces execution time for isolated changes
in product components and minimizes the load on limited hardware resources.

> **Product component** is a distinct functional unit responsible for a specific product feature.
It is defined by a set of files in the repository (for example, openvino/src/_some_folder_/**)
containing the feature implementation.

This document describes the Smart CI implementation in OpenVINO GitHub Actions pre-commit
workflows and provides instructions on how to add or modify its rules.

>**NOTE**: Basic understanding of [GitHub Actions workflows](https://docs.github.com/en/actions) is required.

## Table of Contents

* [Implementation](#implementation)
* [Configuration of Smart CI Rules](#configuration-of-smart-ci-rules)
  * [Product Components Definition: .github/labeler.yml](#product-components-definition)
  * [Definition of Dependencies between Components: .github/components.yml](#definition-of-dependencies-between-components)
  * [Specifics of Pipeline Behavior](#specifics-of-pipeline-behavior)
* [How to Contribute](#how-to-contribute)
  * [Adding a New Component](#adding-a-new-component)
  * [Adding Validation for a Component](#adding-validation-for-a-component)
  * [Adding Support for Smart CI to a Workflow](#adding-support-for-smart-ci-to-a-workflow)
  * [Skipping an Entire Workflow for Specific Changes](#skipping-an-entire-workflow-for-specific-changes)
  * [Adding Smart CI for Components Outside the OpenVINO Repository](#adding-smart-ci-for-components-outside-the-openvino-repository)


## Implementation

Smart CI is implemented as a [custom GitHub Action](https://docs.github.com/en/actions/creating-actions/about-custom-actions)
stored in the openvino repository at [.github/actions/smart-ci](../../../../.github/actions/smart-ci).
In GitHub Actions workflows, this action is called as the first step in a separate job:
```yaml
jobs:
  Smart_CI:
    outputs:
      affected_components: "${{ steps.smart_ci.outputs.affected_components }}"
      skip_workflow: "${{ steps.smart_ci.outputs.skip_workflow }}"
    steps:
      - name: Get affected components
        id: smart_ci
        uses: ./.github/actions/smart-ci
        ...
```
It takes PR data as input and [outputs](https://docs.github.com/en/actions/using-jobs/defining-outputs-for-jobs)
a list of product components affected by the PR, along with the validation scope for each component.
The validation scope can be either "build" only or both "build" and "test", assuming that a component
must be built before testing.

Example of such output for a PR that changes only the TensorFlow Frontend component including
files inside src/frontends/tensorflow:
```
changed_component_names: {'TF_FE'}  # TF_FE is an alias we chose for TensorFlow Frontend component
affected_components={
    "TF_FE": {"test": true, "build": true},
    "MO": {"test": true, "build": true},
    "CPU": {"build": true},
    "Python_API": {"build": true},
    ...
}
```

Once the Smart CI job is finished, validation jobs start. Based on the output from Smart CI,
some jobs can be skipped entirely, while in other jobs only specific steps are skipped.
This is done via GitHub Actions [conditions](https://docs.github.com/en/actions/using-jobs/using-conditions-to-control-job-execution).
For example, the following job, called TensorFlow_Hub_Models_Tests, will be executed only if the PR
is related to the "TF_FE" component and requires running the "test" scope for it:
```yaml
TensorFlow_Hub_Models_Tests:
  needs: [Build, Smart_CI]
  ...
  if: fromJSON(needs.smart_ci.outputs.affected_components).TF_FE.test
  steps:
    - ...
```

## Configuration of Smart CI Rules

Smart CI operates based on the rules described in two configuration files stored in
the OpenVINO repository.

### Product Components Definition

[.github/labeler.yml](../../../../.github/labeler.yml)

This file contains a mapping of source code paths to corresponding component names. It serves
as a configuration for [actions/labeler](https://github.com/marketplace/actions/labeler?version=v4.3.0)
GitHub Action used to automatically assign labels to PRs based on the PR changeset. It is reused
for Smart CI purposes, so that each label described in this configuration is considered a component name.
The labeler action automatically identifies which components were changed in each PR. For example:
```yaml
'category: CPU':
- 'src/plugins/intel_cpu/**/*'
- 'src/common/snippets/**/*'
- 'thirdparty/xbyak/**/*'
```
If a PR changes at least one file matching any of the [minimatch glob patterns](https://github.com/isaacs/minimatch#readme)
above, the label "category: CPU" is assigned to this PR. GitHub Actions workflows that use
Smart CI feature consider the component named "CPU" as changed (the "category:" prefix is omitted the in component name).

### Definition of Dependencies between Components

[.github/components.yml](../../../../.github/components.yml)

Some components are not entirely independent, and changing them may impact other components as well. In this case,
validation for the changed component itself (including both build and tests) is required,
along with validation for dependent components (either only build or both build and tests).
This file describes the relationships between components, for example:
```yaml
PyTorch_FE:       # Component name
  revalidate:     # Defines the list of components to revalidate (build + test) if the component above was changed
    - MO          # This component depends on PyTorch_FE and requires full revalidation
  build:          # Defines the list of components to build if the PyTorch_FE was changed (test runs for them are skipped)
    - CPU         # This component and the component below must be built if PyTorch_FE was changed
    - Python_API
```
For the example above, the following pipeline will be executed on changes applied only to PyTorch_FE:

* Build for PyTorch_FE
* Tests for PyTorch_FE
* Build for MO
* Tests for MO
* Build for CPU
* Build for Python_API

>**NOTE**: Dependencies are **not** transitive. For example, if a component "A" depends on component "B",
and component "B" depends on component "C", it is not automatically assumed that "A" depends on
"C". Each component must specify all its dependents explicitly.

### Specifics of Pipeline Behavior
* If the changed component **is not defined** in components.yml, it is assumed that it affects all other components,
and the full validation scope is executed.

* If **more than one** component is impacted by a PR, all jobs required to validate all these components are executed.

* If a PR changes files that **are not related to any known component**, the full validation scope is executed.
This is to avoid potential regressions by unlabeled changes. For that, a [patched](https://github.com/akladiev/labeler/releases/tag/v4.3.1) version of [actions/labeler v4.3.0](https://github.com/marketplace/actions/labeler?version=v4.3.0) with the same functionality but with an additional feature is used. This enables developers to detect cases
where a PR modifies files that do not match any of the patterns in the labeler.yml configuration.

## How to Contribute

### Adding a New Component

1. Add a new record to [.github/labeler.yml](../../../../.github/labeler.yml).
The root-level key is the component (label) name, and the value is a set of globs to define which
source code paths are related to this component.
See [labeler usage](https://github.com/marketplace/actions/labeler?version=v4.3.0) to learn more about
globs syntax.
2. Add a new record to [.github/components.yml](../../../../.github/components.yml).
The root-level key is the component name, which is the same as the label name defined in the previous step,
but with the prefix "category:" omitted (if any). If there are spaces in the label name, replace them with underscores.
For example: `'category: LP transformations'` in labeler.yml -> `LP_transformations` in components.yml.
To fill the value, review other components in components.yml and choose those that can be impacted by changes in the new component.
Put components requiring full revalidation (build and test) under the `revalidate` key; and those requiring
only build under the `build` key. Example record:
    ```yaml
    your_component:
      revalidate:
        - component_1
        - component_2
      build:
        - component_3
    ```
    If your component does not impact anything else, specify an empty list under both
`revalidate` and `build`:
    ```yaml
    your_component:
      revalidate: []
      build: []
    ```
    If you want to explicitly show that a component impacts all other components, use the "all" notation as a value under
`revalidate`. This causes the full pipeline to be executed on changes to your component, equivalent to completely
omitting the record about it in components.yml. Alternatively, use the "all" notation as a value under `build`.
This means that changes to your component will cause building (but not testing) all other components:
    ```yaml
    your_component_1:
      build: 'all'

    your_component_2:
      revalidate: 'all'
    ```
4. Review other components listed in components.yml. Check if your component requires to be validated when changes
occur in any of the listed components. If it does, add your component name under the `revalidate` or `build` sections of the
corresponding components.

### Adding Validation for a Component
If you want to add a new validation job to test your new component or choose an existing one, go to the
desired workflow in [.github/workflows](../../../../.github/workflows). The main ones include
[ubuntu_22.yml](../../../../.github/workflows/ubuntu_22.yml), [windows.yml](../../../../.github/workflows/windows.yml) and
[mac.yml](../../../../.github/workflows/mac.yml). If Smart CI is enabled for the pipeline, you will find the Smart_CI job
in the beginning of the workflow:
```yaml
jobs:
  Smart_CI:
    ...
    steps:
      - name: Get affected components
        id: smart_ci
      ...
```

Alternatively, you can create a separate workflow for testing your component.
Refer to the following pages for more information:
- [Adding support for Smart CI to a workflow](#adding-support-for-smart-ci-to-a-workflow)
- [Using workflows](https://docs.github.com/en/actions/using-workflows/about-workflows) -
official GitHub documentation


Once you have a job that validates your component:
* Add Smart_CI to the "[needs](https://docs.github.com/en/actions/using-jobs/using-jobs-in-a-workflow#defining-prerequisite-jobs)"
  block for this job. This ensures that the job has access to the Smart CI outputs:
  ```yaml
  job_that_validates_your_component:
    needs: Smart_CI  # if other job is already specified here, add Smart_CI to the list like that: [Other_Job_ID, Smart_CI]
    ...
  ```
* Add an ["if" condition](https://docs.github.com/en/actions/using-jobs/using-conditions-to-control-job-execution) to
  refer to the Smart CI output. To run the entire job conditionally, add it on the same level as the "needs" key:
  ```yaml
  # The job below will start if YOUR_COMPONENT is affected and "test" scope is required
  job_that_validates_your_component:
    needs: [Build, Smart_CI]
    ...
    if: fromJSON(needs.smart_ci.outputs.affected_components).YOUR_COMPONENT.test # or <...>.build, if needed
    steps:
      - ...
  ```
  If only a specific step within the job must be executed on changes to your component, add "if" to the required steps.
  The syntax is described in the [GitHub Actions documentation](https://docs.github.com/en/actions/creating-actions/metadata-syntax-for-github-actions#runsstepsif):
  ```yaml
  job_that_validates_your_component:
    needs: [Build, Smart_CI]
    ...
    steps:
      # The step below will start if YOUR_COMPONENT is affected and the "build" scope is required
      - name: step_name
        if: fromJSON(needs.smart_ci.outputs.affected_components).YOUR_COMPONENT.build # or <...>.test, if needed
  ```
  >**NOTE**: When adding the Smart CI condition to a step within a job, make sure that the job itself
  is not skipped on changes only to your component. For that, examine the "if" condition on the job level.
  It should either be absent (in this case, the job will always be executed), or return "True" in all cases
  you want your conditional step to be executed.

  You can also use any boolean operators to write complex conditions, for example:
  ```yaml
    # The below condition will force the job/step to run when either COMPONENT_1 or COMPONENT_2 was changed
    if: fromJSON(needs.smart_ci.outputs.affected_components).COMPONENT_1.test ||
        fromJSON(needs.smart_ci.outputs.affected_components).COMPONENT_2.test
  ```
  See the [expressions](https://docs.github.com/en/actions/learn-github-actions/expressions) page
  to learn more about expressions in conditions.

### Adding Support for Smart CI to a Workflow
To use Smart CI in a workflow, add the following code under the `jobs` block before all other jobs that will use it:
```yaml
jobs:
  Smart_CI:
    runs-on: ubuntu-latest
    outputs:
      affected_components: "${{ steps.smart_ci.outputs.affected_components }}"
      skip_workflow: "${{ steps.smart_ci.outputs.skip_workflow }}"
    steps:
      - name: checkout action
        uses: actions/checkout@v4
        with:
          sparse-checkout: .github/actions/smart-ci

      - name: Get affected components
        id: smart_ci
        uses: ./.github/actions/smart-ci
        with:
          repository: ${{ github.repository }}
          pr: ${{ github.event.number }}
          commit_sha: ${{ github.sha }}
          component_pattern: "category: (.*)"
          repo_token: ${{ secrets.GITHUB_TOKEN }}
```
If needed, more parameters can be passed to the "Get affected components" step, the full list is available in
[.github/actions/smart-ci/action.yml](../../../../.github/actions/smart-ci/action.yml).

After that, you can refer to the outputs from Smart_CI in validation jobs, as described in
[Adding validation for a component](#adding-validation-for-a-component) section. To learn more about the syntax of
GitHub Actions Workflows, see the [workflows](https://docs.github.com/en/actions/using-workflows/about-workflows) page.

### Skipping an Entire Workflow for Specific Changes

To skip entire workflows based on specific conditions, you can use the
[paths-ignore](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions#onpushpull_requestpull_request_targetpathspaths-ignore)
feature offered by GitHub by default. However, this feature cannot be used in workflows with jobs
marked as "Required" for merge. See [skipped but required checks](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/collaborating-on-repositories-with-code-quality-features/troubleshooting-required-status-checks#handling-skipped-but-required-checks) for details.

As a workaround, Smart CI returns an indicator that enables the workflow to be completely skipped,
if the PR is labeled _only_ by given labels and/or changes only the files matching
given [fnmatch](https://docs.python.org/3/library/fnmatch.html) patterns. These labels and patterns are passed as inputs
to the Smart CI action in the `with` block. The indicator is returned as a separate output called `skip_workflow`,
for example:
```yaml
  Smart_CI:
    runs-on: ubuntu-latest
    outputs:
      ...
      # The output below is set only if the workflow can be completely skipped, and is empty otherwise
      skip_workflow: "${{ steps.smart_ci.outputs.skip_workflow }}"
    steps:
      ...
      - name: Get affected components
        id: smart_ci
        uses: ./.github/actions/smart-ci
        with:
          ...
          # Comma-separated rules for skipping the entire workflow
          skip_when_only_listed_labels_set: 'docs'
          skip_when_only_listed_files_changed: '*.md,*.rst,*.png,*.jpg,*.svg'
```
Then the `skip_workflow` output can be used to conditionally run a **parent** job in a workflow.
A parent job is the job required to pass before all other jobs. It is specified in the "needs" block
for all of them. For example, the Build job can be considered a parent job.

The `skip_workflow` condition looks as follows:
```yaml
Build:
    needs: Smart_CI
    ...
    if: "!needs.smart_ci.outputs.skip_workflow"
    ...
```
>**NOTE**: If a workflow has more than one parent job, the condition must be added to each of them.

This approach works because skipped checks are processed as successful by GitHub. They do not block
the merge, unlike required workflows skipped by path filtering.

### Adding Smart CI for Components Outside the OpenVINO Repository

Some components, like the NVIDIA plugin or ONNX Runtime, are stored in their own repositories.
Therefore they cannot be defined via pattern matching on source code in the OpenVINO repository.
However, they still need to be validated together with the core OpenVINO.
To add Smart CI rules for such components, skip the first step with modifying the labeler configuration
in the [Adding a New Component](#adding-a-new-component) instruction and proceed directly to the next step:
1. Add a new record to [.github/components.yml](../../../../.github/components.yml)
with empty values for the `revalidate` and `build` keys:
    ```yaml
    NEW_EXTERNAL_COMPONENT:
      revalidate: []
      build: []
    ```
2. Review other components in components.yml, find those that need to be validated together with
the new component. Add the name of the new component under the `revalidate` or `build` sections of these components.

3. Add or find a job that performs integration validation of the new external component
with OpenVINO. Provide it with an "if" condition: `if: fromJSON(needs.smart_ci.outputs.affected_components).NEW_EXTERNAL_COMPONENT`,
as described in step 3 of [Adding a New component](#adding-a-new-component) instruction.

This ensures that integration validation for this external component starts only on changes
made to the selected components in the OpenVINO repository.