# Development Flow for Adding New Changes to the Inference Engine {#openvino_docs_Inference_Engine_Development_Procedure_IE_Dev_Procedure}

## Develop Your Feature
1. Create a branch based on the latest version of a target branch (`master` or `release`).
   Use the following branch naming:
   * *feature/<userid>/<name>* - stands for temporary work area for creating a feature or performing bug fixes
   * *scout/<name>* - is used for shared development if several developers work on one feature
   **IMPORTANT**: Do not use long branch name, because it may lead to failed CI jobs on Windows. Name length must be less 40 characters.
2. Commit changes on your branch and push it to remote.


## Create a Merge Request
1. Go to the [GitLab\\\* merge request page](https://gitlab-icv.inn.intel.com/inference-engine/dldt/merge_requests)

2. Create a merge request by pressing on **New Merge Request** or **Create merge request**. Choose an `inference-engine` project from the drop-down list.
    <br/>
    ![mr1]

    a. Fill the **Title** and **Description** fields with meaningful information. This information should be enough to understand the changes you made:

    Use this template for the **Title** field. If you did not finish your work on current bug/feature, add `[WIP]` to the beginning of the title:
    ```
    [Domain] Small description for merge request (can use first string from the commit message)
    ```
    Use domain from the following list:
    * [IE COMMON] - if a solution impacts common Inference Engine functionality
    * [IE EXTENSION]
    * [IE PYTHON]
    * [IE SAMPLES]
    * [IE TESTS]
    * [IE DOCS]
    * [IE MKLDNN]
    * [IE FPGA]
    * [IE GNA]
    * [IE CLDNN]
    * [IE MYRIAD]
    * [IE HDDL]
    * [IE HETERO]

    You can use several domains in one commit message. For example: [COMMON][MKLDNN] <Text>

    Use this template to fill **Description** field:
    ```
    <Multi-line description of the commit>

    JIRA: <url_to_jira_ticket>
    CI: <url_to_validation_result_in_CI>
        <url_to_passed_rebuilt_changes_in_CI>
    ```

    b. Add **Milestone** and **Labels** to the MR if it is possible.

    c. If your work is finished, assign the MR to a reviewer. If it is in progress, assing the MR to yourself (`[WIP]` case).

    Example of an [MR](https://gitlab-icv.inn.intel.com/inference-engine/inference-engine/merge_requests/2512):
    <br/>
       ![mr_example1]

   >**NOTE**: When creating an MR, please remember that even a person who does not know what this feature/merge request stands for >should be able to understand the reasons why this feature/change was created.

3. Change the history of your branch:

    a. Squash commits on your branch into one (you can have some number of logically separated commits if it is needed). Use the following template for commit message:

       ```
       [Domain] Small description of commit
   
       Multiline commit additional comments:
       * additional comment 1
       * additional comment 2
   
       JIRA: CVS-xxxx
       ```
   

    Example of the commit message: 
    <br/>
      ![commit_message]

    >**NOTE**: if there is no JIRA ticket for your task, you can leave this line empty.

    b. Rebase your branch to the latest version of the target branch. The difference between the last validated version of the branch must be no more than one day.

    c. Push your updates to the remote branch. You can use force push if needed, for example, when you rebase your branch to latest version of the target branch.

4. Add required people as reviewers of the MR. Please list appropriate GitLab accounts in the discussion section of the MR.

5. Run validation build (refer to the ["Run tests under CI"](#run-tests-under-ci))

6. When review is complited and you have one or several **Thumb up** in your MR:
   a. Make sure that all comments are fixed and all discussions are resolved
   b. Run validation build again, make sure that all tests are passed, and update link to CI in the MR description
   c. Create cherry-picked MR if needed (refer to the ["Create a Cherry-Picked MR (Optional)"](#create-a-cherry-picked-mr-(optional))) and validate it by CI.

    Example of an [MR](https://gitlab-icv.inn.intel.com/inference-engine/inference-engine/merge_requests/2111):
        <br/>
          ![mr_example]
 
7. Reassign the MR to the someone from integration managers.

8. An integration manager will close or merge your MR. 
    <br/>
      ![mr2]

## Create a Cherry-Picked MR (Optional)
1. If you need to merge your changes in both target branch (`release` and `master`), create new merge request to another target branch containing cherry-pick with approved commit.
    Follow the rules above to create the MR (sections 1 and 2 in (**Create a Merge Request**)[#create-a-merge-request]), but add line "\*\*cherry-picked from MR: !xxxx**" to the MR description and assign the MR to someine from the integration managers.

2. Run validation build (refer to "Run tests under CI" section).

3. Assign the MR to an integration manager.

4. The integration manager will merge or close your MR.

## Run Tests under CI
1. Go to the CI page: [TeamCity](https://teamcity01-ir.devtools.intel.com/project.html?projectId=DeepLearningSdk_DeepLearningSdk_InferenceEngineUnifiedRepo)

2. Click the **Run Engineering Validation** (if you want to merge your changes into `master` branch) or **Run Engineering Validation for XXXX RX Release branch** (if you want merge changes into `release` branch):
    <br/>
       ![run_engineering]

3. Select your branch in the top left corner:
    <br/>
       ![select_branch]

4. If you have not committed anything to your branch for the past several days, you might not see your branch in the list. In this case, you can choose it in the properties of **Run Engineering Validation** task (if you want merge changes into `master` branch) or **Run Release Validation** task (if you want merge changes into `release` branch). Click on **Run** button. On the third tab, choose your branch and click the **Run Build**:
    <br/>
      ![run_tests]

5. Click on an arrow right after **1 queued**. On a new dialog window, click on a build number. In this case, it is 1825:
    <br/>
      ![view_results]

   You will see the current status of tests:
    <br/>
      ![analyze_results]

6. Make sure that all tests are passed.
   If some test failed, see build log in TeamCity: choose failed build in dependencies, click on the result and go to **Build log** tab.  
   * If it looks like an infrastructure issue (for example, an absence of a software or a network issue), restart the build. 
   * If you think that your changes could not break the test, rebase your branch on latest version of `master` restart build. If the build failed again, explore build history: an incorrect code might have merged into the target branch before branching that is not fixed till current moment.
   * If you have an issue with code style, run code style checks `<IE_repo>/scripts/run_code_check.sh` locally to analyze the reason of failed CI jobs (see the picture below) and restart them if required:
       <br/>
        ![code_style_artifacts]
   
    Commit your changes.

   **Please add link to restarted build in MR description**

## Merge Changes to a Target Branch (master or release branches)
1. The `master` and `release` branches are protected. Only integration managers can merge to these branches

2. Assigned integration manager checks if all the requirements are met. If so, they can merge MR with the  **Merge** button or manually.

3. An integration manager removes a branch if an MR author has set an appropriate flag for this MR in GitLab GUI.

[mr_example]: img/mr_example.png
[mr_example1]: img/mr_example1.png
[commit_message]: img/commit_message.png
[code_style_artifacts]: img/code_style_artifacts.png
[select_branch]: img/select_branch.png
[run_engineering]: img/run_engineering.png
[run_tests]: img/run_tests.png
[view_results]: img/view_results.png
[analyze_results]: img/analyze_results.png
[mr1]: img/mr1.png
[mr2]: img/mr2.png
