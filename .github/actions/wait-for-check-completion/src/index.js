const core = require('@actions/core');
const github = require('@actions/github');

const CONCLUSION_STATES = {
    SUCCESS: 'success',
    FAILURE: 'failure',
    MIXED: 'mixed',
    ACTION_REQUIRED: 'action_required',
    TIMED_OUT: 'timed_out',
    CANCELLED: 'cancelled',
    COMPLETED: 'completed'
};

/**
 * Wait for multiple checks to complete
 * @param {Object} octokit - GitHub API client
 * @param {string} owner - Repository owner
 * @param {string} repo - Repository name
 * @param {string} ref - Git reference (commit SHA)
 * @param {string[]} checkNames - Array of check names to wait for
 * @param {number} waitInterval - Wait interval in seconds
 * @param {number} timeout - Maximum timeout in seconds
 * @returns {Promise<Object>} Combined check results
 */
async function waitForChecks(octokit, owner, repo, ref, checkNames, waitInterval, timeout) {
    const startTime = Date.now();
    const timeoutMs = timeout * 1000;
    const waitIntervalMs = waitInterval * 1000;

    core.info(`Waiting for checks [${checkNames.join(', ')}] on ref "${ref}" to complete...`);
    core.info(`Timeout: ${timeout}s, Check interval: ${waitInterval}s`);

    const checkResults = {};
    const pendingChecks = new Set(checkNames);

    while ((Date.now() - startTime < timeoutMs) && pendingChecks.size) {
        try {
            // Get all check runs for the specific commit
            const { data: checkRuns } = await octokit.rest.checks.listForRef({
                owner,
                repo,
                ref,
                per_page: 100
            });

            core.info(`Found ${checkRuns.check_runs.length} total check runs`);

            // Process each pending check
            for (const checkName of pendingChecks) {
                const matchingRuns = checkRuns.check_runs.filter(run => run.name === checkName);

                if (!matchingRuns.length) {
                    core.info(`No check runs found for "${checkName}" yet, waiting...`);
                    continue;
                }

                // Find the most recent check run for this check
                const latestCheckRun = matchingRuns.sort((a, b) =>
                    new Date(b.started_at) - new Date(a.started_at)
                )[0];

                core.info(`Check "${checkName}" status: ${latestCheckRun.status}, conclusion: ${latestCheckRun.conclusion || 'N/A'}`);

                if (latestCheckRun.status === 'completed') {
                    core.info(`Check "${checkName}" completed with conclusion: ${latestCheckRun.conclusion}`);
                    checkResults[checkName] = {
                        status: latestCheckRun.status,
                        conclusion: latestCheckRun.conclusion,
                        checkRun: latestCheckRun
                    };
                    pendingChecks.delete(checkName);
                } else if (latestCheckRun.status === 'in_progress') {
                    core.info(`Check "${checkName}" is still in progress...`);
                } else if (latestCheckRun.status === 'queued') {
                    core.info(`Check "${checkName}" is queued...`);
                }
            }
            
            if (pendingChecks.size) {
                core.info(`Still waiting for [${Array.from(pendingChecks).join(', ')}]. Waiting ${waitInterval} seconds before next check...`);
                await new Promise(resolve => setTimeout(resolve, waitIntervalMs));
            } else {
                core.info('All checks completed, parsing conclusions...');
                break;
            }
        } catch (error) {
            core.warning(`Error fetching check runs: ${error.message}`);

            // If it's a 404, the checks might not exist yet
            if (error.status === 404) {
                core.info('Checks not found yet, continuing to wait...');
            } else {
                // For other errors
                core.error(`API error: ${error.message}`);
            }
        }
    }

    if (pendingChecks.size) {
        const pendingChecksList = Array.from(pendingChecks).join(', ');
        throw new Error(`Timeout: Checks [${pendingChecksList}] did not complete within ${timeout} seconds`);
    }
    return checkResults;
}

/**
 * Main function
 */
async function run() {
    try {
        // Get inputs
        const ref = core.getInput('ref', { required: true });
        const checkNames = core.getInput('check-names', { required: true });
        const token = core.getInput('repo-token', { required: true });
        const waitInterval = parseInt(core.getInput('wait-interval') || '10', 10);
        const timeout = parseInt(core.getInput('timeout') || '600', 10);

        // Get repository info (default to current repository)
        const owner = core.getInput('owner') || github.context.repo.owner;
        const repo = core.getInput('repo') || github.context.repo.repo;

        // Parse check names
        const checkNamesArray = checkNames.split(',').map(name => name.trim()).filter(name => name.length > 0);

        if (checkNamesArray.length === 0) {
            throw new Error('No valid check names provided');
        }

        core.info(`Repository: ${owner}/${repo}`);
        core.info(`Reference: ${ref}`);
        core.info(`Check names: [${checkNamesArray.join(', ')}]`);

        // Validate inputs
        if (waitInterval < 1 || waitInterval > 300) {
            throw new Error('wait-interval must be between 1 and 300 seconds');
        }

        if (timeout < 1 || timeout > 3600) {
            throw new Error('timeout must be between 1 and 3600 seconds');
        }

        // Create GitHub API client
        const octokit = github.getOctokit(token);

        // Wait for the checks to complete
        const results = await waitForChecks(octokit, owner, repo, ref, checkNamesArray, waitInterval, timeout);

        // Analyze overall results
        const allConclusions = Object.values(results).map(r => r.conclusion);
        const allStatuses = Object.values(results).map(r => r.status);

        // Determine overall conclusion
        let overallConclusion;
        if (allConclusions.every(c => c === CONCLUSION_STATES.SUCCESS)) {
            overallConclusion = CONCLUSION_STATES.SUCCESS;
        } else if (allConclusions.some(c => [CONCLUSION_STATES.FAILURE, CONCLUSION_STATES.CANCELLED, CONCLUSION_STATES.TIMED_OUT].includes(c))) {
            overallConclusion = CONCLUSION_STATES.FAILURE;
        } else if (allConclusions.some(c => c === CONCLUSION_STATES.ACTION_REQUIRED)) {
            overallConclusion = CONCLUSION_STATES.ACTION_REQUIRED;
        } else {
            overallConclusion = CONCLUSION_STATES.MIXED;
        }

        // Set outputs
        core.setOutput('status', allStatuses.every(s => s === CONCLUSION_STATES.COMPLETED) ? CONCLUSION_STATES.COMPLETED : CONCLUSION_STATES.MIXED);
        core.setOutput('conclusion', overallConclusion);
        core.setOutput('results', JSON.stringify(results));

        // Log results
        for (const [checkName, result] of Object.entries(results)) {
            core.info(`${checkName}: Status is "${result.status}", Conclusion is "${result.conclusion}"`);
        }

        // Exit with appropriate code based on overall conclusion
        if (overallConclusion === CONCLUSION_STATES.SUCCESS) {
            core.info('All checks completed with successful conclusions');
        } else if (overallConclusion === CONCLUSION_STATES.FAILURE) {
            const failedChecks = Object.entries(results)
                .filter(([_, result]) => [CONCLUSION_STATES.FAILURE, CONCLUSION_STATES.CANCELLED, CONCLUSION_STATES.TIMED_OUT].includes(result.conclusion))
                .map(([name, _]) => name);
            core.setFailed(`Some checks failed: [${failedChecks.join(', ')}]`);
        } else if (overallConclusion === CONCLUSION_STATES.ACTION_REQUIRED) {
            const actionRequiredChecks = Object.entries(results)
                .filter(([_, result]) => result.conclusion === CONCLUSION_STATES.ACTION_REQUIRED)
                .map(([name, _]) => name);
            core.setFailed(`Some checks require action: [${actionRequiredChecks.join(', ')}]`);
        } else {
            core.warning(`Checks completed with mixed conclusions: ${overallConclusion}`);
        }
    } catch (error) {
        core.setFailed(error.message);
        core.error(error.stack || error.toString());
    }
}

// Only run if this file is executed directly (not imported)
if (require.main === module) {
    run();
}

module.exports = { run, waitForChecks };
