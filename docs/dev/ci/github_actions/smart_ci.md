# Smart CI Overview

Smart CI is a feature aiming to optimize pre-commit CI workflow by running only those builds and tests that are 
actually required to validate changes in a given PR (Pull Request). As an example, if PR changes only CPU plugin, 
GPU plugin tests in the pre-commit for this PR will be skipped, since they are unrelated. This allows to decrease 
execution time for isolated changes in product components, and to minimize the load on limited hardware resources.

> **Product component** is a distinct functional unit, responsible for a specific product feature. It is defined by a 
set of files in repository (e.g. openvino/src/_some_folder_/**) containing the feature implementation.

This document describes how Smart CI is implemented in our GitHub Actions pre-commit workflows and how to add or modify 
rules for it.


### Prerequisites
Basic understanding of [GitHub Actions workflows](https://docs.github.com/en/actions)

## Implementation

Smart CI is implemented as a [custom GitHub Action](https://docs.github.com/en/actions/creating-actions/about-custom-actions) 
stored in openvino repository: [.github/actions/smart-ci](../../../../.github/actions/smart-ci). In GitHub Actions 
workflows this action is called as a first step in a separate job:
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
It takes PR data as an input and [outputs](https://docs.github.com/en/actions/using-jobs/defining-outputs-for-jobs) 
a list of product components affected by this PR and a validation scope for each of these components (either only 
"build" or both "build" and "test" - by design we assume that testing component requires it to be built). Example of 
such output for PR that changes only Tensorflow Frontend component (files inside src/frontends/tensorflow):
```
changed_component_names: {'TF_FE'}  # TF_FE is an alias we chose for Tensorflow Frontend component
affected_components={
    "TF_FE": {"test": true, "build": true}, 
    "MO": {"test": true, "build": true}, 
    "CPU": {"build": true}, 
    "Python_API": {"build": true}, 
    ...
}
```

Once Smart CI job is finished, validation jobs are started. Based on the output from Smart CI, some jobs can be skipped 
entirely; in other jobs only separate steps can be skipped. This is done via GitHub Actions 
[conditions](https://docs.github.com/en/actions/using-jobs/using-conditions-to-control-job-execution). For example,
the following job called TensorFlow_Hub_Models_Tests will be executed only if PR affects "TF_FE" component and
requires running "test" scope for it:
```yaml
TensorFlow_Hub_Models_Tests:
  needs: [Build, Smart_CI]
  ...
  if: fromJSON(needs.smart_ci.outputs.affected_components).TF_FE.test
  steps:
    - ...
```

The way how we define product components and "smart" rules for them is described further.

## Configuration of Smart CI rules

Smart CI operates based on the set of rules described in two configuration files, stored in openvino repository.

### Product components definition: [.github/labeler.yml](../../../../.github/labeler.yml)
This file contains mapping of source code paths to corresponding component names. Essentially, this a configuration 
for [actions/labeler](https://github.com/marketplace/actions/labeler?version=v4.3.0) GitHub Action, which we use to
automatically assign labels to pull requests based on PR changeset. We reuse it for Smart CI purposes, so that each 
label described in this configuration is considered a component name, and the labeler action automatically determines
which components were changed in each PR. For example:
```yaml
'category: CPU':
- 'src/plugins/intel_cpu/**/*'
- 'src/common/snippets/**/*'
- 'thirdparty/xbyak/**/*'
```
If PR changes at least one file matching any of the [minimatch glob patterns](https://github.com/isaacs/minimatch#readme) 
above, label "category: CPU" will be assigned to this PR, and GitHub Actions workflows that use Smart CI feature will
consider component named "CPU" changed ("category:" prefix is omitted in component name). 

### Definition of dependencies between components: [.github/components.yml](../../../../.github/components.yml)
Some components are not entirely independent, and changes in them may affect other components as well. In this case, 
in addition to the validation for the changed component itself (build + tests), validation for dependent components 
is also required (either only build or both build and tests). This file describes these relationships between components,
for example:
```yaml
PyTorch_FE:       # Component name
  revalidate:     # Defines list of components to revalidate (build + test) if the component above was changed
    - MO          # This component depends on PyTorch_FE and requires full revalidation
  build:          # Defines list of components to build if the PyTorch_FE was changed (test runs for them are skipped)
    - CPU         # This component and the component below must be built if PyTorch_FE was changed
    - Python_API
```
With the example above, the following pipeline will be executed on changes only to PyTorch_FE:

* Build for PyTorch_FE
* Tests for PyTorch_FE
* Build for MO
* Tests for MO
* Build for CPU
* Build for Python_API

>**NOTE**: the dependencies are **not** transitive - if a component "A" depends on component "B", and component "B" 
depends on component "C", we don't implicitly assume that "A" depends on "C". Each component must specify all his 
dependents explicitly.

### Specifics of pipeline behavior
* If the changed component **is not defined** in components.yml, we assume that it affects all other components, 
and the full validation scope will be executed.

* If **more than one** component is affected by PR, all jobs required to validate all these components will be executed.

* If PR changes files that **are not related to any known component** - the full validation scope will be executed, 
since we don't want to skip anything for the unlabeled changes - they are under our control and may potentially 
introduce regressions. For that we use a [patched](https://github.com/akladiev/labeler/releases/tag/v4.3.1) version of 
[actions/labeler v4.3.0](https://github.com/marketplace/actions/labeler?version=v4.3.0) with the same functionality, 
but with an additional feature implemented, allowing us to detect cases when PR changes files that do not match
any of the patterns in labeler.yml configuration.

## How to contribute

### Adding a new component

1. Add a new record to [.github/labeler.yml](../../../../.github/labeler.yml).
Root-level key is a component (label) name, and value is a set of globs to define which source code paths are related to 
this component. See [labeler usage](https://github.com/marketplace/actions/labeler?version=v4.3.0) to get familiar with
globs syntax.
2. Add a new record to [.github/components.yml](../../../../.github/components.yml). 
Root-level key is a component name, which is the same as the label name defined in the previous step, but with prefix 
"category:" omitted (if any). If there were spaces present in label name - replace them with underscores. Example:
`'category: LP transformations'` in labeler.yml -> `LP_transformations` in components.yml.  To fill the value, review 
other components in components.yml and choose the ones that can be affected by changes in a new component. 
Put those that require full revalidation (build and test) under `revalidate` key; and those requiring
only build - under `build` key. Example record:
    ```yaml
    your_component:
      revalidate: 
        - component_1
        - component_2
      build: 
        - component_3
    ```
    If your component does not affect anything else, specify empty list under both
`revalidate` and `build`:
    ```yaml
    your_component:
      revalidate: []
      build: []
    ```
    If you wish to explicitly show that a component affects all other components, use "all" notation as a value under 
`revalidate` (this will cause full pipeline to be executed on changes to your component - equivalent to completely 
omitting the record about it in components.yml) or `build` (this will mean that changes to your component will cause 
building - but not testing - all other components):
    ```yaml
    your_component_1:
      build: 'all'
    
    your_component_2:
      revalidate: 'all'
    ```
4. Review other components in components.yml - does your component itself require to be validated when there are changes 
in any of the listed components? If yes, add your component name under `revalidate` or `build` sections of the 
respective components.

### Adding validation for a component
You may wish to add a new validation job to test your new component, or choose an existing one. For that, go to the 
desired workflow in [.github/workflows](../../../../.github/workflows) (the main ones are 
[linux.yml](../../../../.github/workflows/linux.yml), [windows.yml](../../../../.github/workflows/windows.yml) and 
[mac.yml](../../../../.github/workflows/mac.yml)). If Smart CI is enabled for the pipeline, you will find Smart_CI job 
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
The following pages will be helpful: 
- [Adding support for Smart CI to a workflow](#adding-support-for-smart-ci-to-a-workflow)
- [using-workflows/about-workflows](https://docs.github.com/en/actions/using-workflows/about-workflows) - 
official GitHub documentation


Once you have a job that validates your component:
* Add Smart_CI to "[needs](https://docs.github.com/en/actions/using-jobs/using-jobs-in-a-workflow#defining-prerequisite-jobs)" 
  block for this job - this will ensure that it will get access to Smart CI outputs:
  ```yaml
  job_that_validates_your_component:
    needs: Smart_CI  # if other job was already specified here, add Smart_CI to list like that: [Other_Job_ID, Smart_CI]
    ...
  ```
* Add ["if" condition](https://docs.github.com/en/actions/using-jobs/using-conditions-to-control-job-execution) to
  refer to the Smart CI output. To run the whole job conditionally - add it on the same level as "needs" key:
  ```yaml
  # The job below will be started if YOUR_COMPONENT was affected and "test" scope is required
  job_that_validates_your_component:
    needs: [Build, Smart_CI]
    ...
    if: fromJSON(needs.smart_ci.outputs.affected_components).YOUR_COMPONENT.test # or <...>.build, if needed
    steps:
      - ...
  ```
  If only a separate step within the job must be executed on changes to your component - add "if" to desired steps
  (syntax is described [here](https://docs.github.com/en/actions/creating-actions/metadata-syntax-for-github-actions#runsstepsif)):
  ```yaml
  job_that_validates_your_component:
    needs: [Build, Smart_CI]
    ...
    steps:
      # The step below will be started if YOUR_COMPONENT was affected and "build" scope is required
      - name: step_name
        if: fromJSON(needs.smart_ci.outputs.affected_components).YOUR_COMPONENT.build # or <...>.test, if needed
  ```
  >**NOTE**: when adding Smart CI condition to step within a job, make sure that the job itself won't be skipped
  on changes only to your component. For that, look at the "if" condition on the job level - it either must be 
  absent (in this case the job will always be executed); or return "True" in all cases you wish your 
  conditional step to be executed.
  
  You can also use any boolean operators to write complex conditions, for example:
  ```yaml
    # The below condition will force the job/step to run when either COMPONENT_1 or COMPONENT_2 was changed
    if: fromJSON(needs.smart_ci.outputs.affected_components).COMPONENT_1.test ||
        fromJSON(needs.smart_ci.outputs.affected_components).COMPONENT_2.test
  ```
  See [learn-github-actions/expressions](https://docs.github.com/en/actions/learn-github-actions/expressions) page
  to learn more about expressions in conditions.

### Adding support for Smart CI to a workflow
To use Smart CI in a workflow, add the following code under `jobs` block before all other jobs that will use it, 
like that:
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
If needed, more parameters can be passed to "Get affected components" step, full list is available here:
[.github/actions/smart-ci/action.yml](../../../../.github/actions/smart-ci/action.yml).

After that, you can refer to the outputs from Smart_CI in validation jobs, as described in 
[Adding validation for a component](#adding-validation-for-a-component) section. To learn more about the syntax of
GitHub Actions Workflows, see also
[using-workflows/about-workflows](https://docs.github.com/en/actions/using-workflows/about-workflows).

### Skipping the whole workflow for specific changes

For cases, when you want to skip not just a few jobs, but the entire workflow, GitHub by default offers 
[paths-ignore](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions#onpushpull_requestpull_request_targetpathspaths-ignore)
feature. But it has a limitation - it cannot be used in workflows that have jobs marked as "Required" 
for merge (see [details](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/collaborating-on-repositories-with-code-quality-features/troubleshooting-required-status-checks#handling-skipped-but-required-checks)).
Since we want to keep our workflows required, a workaround on Smart CI side was added - it returns an indicator 
that workflow can be completely skipped, if PR was labeled _only_ by given labels and/or changes only files matching 
given [fnmatch](https://docs.python.org/3/library/fnmatch.html) patterns. These labels and patterns are passed as inputs 
to Smart CI action in `with` block, and the indicator is returned as a separate output called `skip_workflow`, 
for example:
```yaml
  Smart_CI:
    runs-on: ubuntu-latest
    outputs:
      ...
      # The output below is set only if the workflow can be completely skipped, and empty otherwise
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
The `skip_workflow` output can then be used to conditionally run a **parent** job in a workflow (the job that is 
required to pass before all other jobs and is specified in "needs" block for all of them, for example, Build).
The condition looks like that:
```yaml
Build:
    needs: Smart_CI
    ...
    if: "!needs.smart_ci.outputs.skip_workflow"
    ...
```
>**NOTE**: If a workflow has more than one parent job, the condition must be added to each of them.

This approach works because skipped checks are processed as successful by GitHub, so they do not block merge, unlike
required workflows skipped by paths filtering.

### Adding Smart CI for components outside openvino repository

Some components (like NVIDIA plugin or ONNX Runtime) are stored in their own repositories and therefore cannot be 
defined via pattern matching on source code in openvino repository, while they still need to be validated together with 
core OpenVINO. To add Smart CI rules for such components, skip the first step with modifying labeler configuration 
in [Adding a new component](#adding-a-new-component) instruction and go directly to the next step:
1. Add a new record to [.github/components.yml](../../../../.github/components.yml),
with empty values for `revalidate` and `build` keys, like that:
    ```yaml
    NEW_EXTERNAL_COMPONENT:
      revalidate: []
      build: []
    ```
2. Review other components in components.yml, find those that have to be validated together with a new component and 
add a new component's name under `revalidate` or `build` sections of these components.

3. Add or find a job that does integration validation of a new external component with OpenVINO and provide it with an 
"if" condition: `if: fromJSON(needs.smart_ci.outputs.affected_components).NEW_EXTERNAL_COMPONENT` like 
described in step 3 of [Adding a new component](#adding-a-new-component) instruction.

This will ensure that integration validation for this external component is started only on changes to chosen 
components in openvino repository.
