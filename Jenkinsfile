#!groovy

echo "GitHub branch source plugin demo"

echo "Ensure pipeline is running from PR"
if (env.CHANGE_ID) { // available only when triggered from PR
    echo "Change ID: ${env.CHANGE_ID}"

    echo "Head commit info"
    echo "head's SHA ${pullRequest.head}"
    echo "head's ref ${pullRequest.headRef}"
    echo "head's base (target branch) ${pullRequest.base}"

    echo "Pull Request number ${pullRequest.number}"

    echo "Multibranch pipeline's BRANCH_NAME ${env.BRANCH_NAME}"
    //echo "Publish status at PR's head commit"
    // pullRequest.createStatus(status: 'pending',
    //                         context: 'ci/another_status',
    //                         description: 'Test status from Jenkinsfile',
    //                         targetUrl: "${env.BUILD_URL}/consoleFull")

    sleep 10
    // pullRequest.createStatus(status: 'success',
    //                         context: 'ci/another_status',
    //                         description: 'Test status from Jenkinsfile',
    //                         targetUrl: "${env.BUILD_URL}/consoleFull")
}

node {
    checkout scm
}
