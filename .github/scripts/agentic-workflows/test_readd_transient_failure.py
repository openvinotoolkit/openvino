# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

from readd_transient_failure import (
    BOT_LOGIN,
    dequeue_reason,
    find_eligibility_marker,
    latest_queue_events,
    marker_is_current,
    marker_matches_analysis_run,
    parse_marker,
    pr_number_from_event,
    process_event,
    source_run_matches_pr,
)


HEAD_SHA = "a" * 40
MARKER_BODY = (
    "<!-- ci-doctor-mq-eligible pr=123 head_sha="
    f"{HEAD_SHA} source_run_id=456 analysis_run_id=789 -->"
)
NOW = datetime(2026, 11, 6, 8, 0, tzinfo=timezone.utc)


class MarkerTests(unittest.TestCase):
    def test_parse_valid_marker(self):
        marker = parse_marker(MARKER_BODY, NOW)

        self.assertEqual(marker.pr_number, "123")
        self.assertEqual(marker.head_sha, HEAD_SHA)
        self.assertEqual(marker.source_run_id, "456")
        self.assertEqual(marker.analysis_run_id, "789")

    def test_parse_rejects_malformed_marker(self):
        self.assertIsNone(parse_marker("<!-- ci-doctor-mq-eligible pr=123 -->", NOW))

    def test_only_trusted_author_and_current_head_are_accepted(self):
        comments = [
            SimpleNamespace(
                user=SimpleNamespace(login="contributor"),
                body=MARKER_BODY,
                created_at=NOW,
            ),
            SimpleNamespace(
                user=SimpleNamespace(login=BOT_LOGIN),
                body=MARKER_BODY.replace(HEAD_SHA, "b" * 40),
                created_at=NOW,
            ),
            SimpleNamespace(
                user=SimpleNamespace(login=BOT_LOGIN),
                body=MARKER_BODY,
                created_at=NOW,
            ),
        ]

        marker = find_eligibility_marker(comments, 123, HEAD_SHA)

        self.assertIsNotNone(marker)
        self.assertEqual(marker.source_run_id, "456")

    def test_user_authored_marker_is_rejected(self):
        comment = SimpleNamespace(
            user=SimpleNamespace(login="contributor"),
            body=MARKER_BODY,
            created_at=NOW,
        )

        self.assertIsNone(find_eligibility_marker([comment], 123, HEAD_SHA))

    def test_stale_marker_before_latest_enqueue_is_rejected(self):
        marker = parse_marker(MARKER_BODY, NOW)

        self.assertFalse(marker_is_current(marker, NOW))
        self.assertTrue(marker_is_current(marker, NOW.replace(hour=7)))
        self.assertFalse(marker_is_current(marker, None))

    def test_completion_fallback_must_match_marker_analysis_run(self):
        marker = parse_marker(MARKER_BODY, NOW)

        self.assertTrue(marker_matches_analysis_run(marker, {"id": 789}))
        self.assertFalse(marker_matches_analysis_run(marker, {"id": 790}))


class EventTests(unittest.TestCase):
    def test_dequeued_event_requires_failed_checks_reason(self):
        self.assertEqual(dequeue_reason({"reason": "failed_checks"}), "failed_checks")
        self.assertEqual(
            dequeue_reason({"pull_request": {"reason": "manual"}}),
            "manual",
        )

    def test_manual_dequeue_is_ignored_before_pr_lookup(self):
        repo = Mock()

        process_event({"action": "dequeued", "reason": "manual"}, Mock(), repo, "owner/repo")

        repo.get_pull.assert_not_called()

    def test_dequeue_without_reason_checks_authoritative_queue_timeline(self):
        pull = Mock()
        pull.number = 123
        pull.head.sha = HEAD_SHA
        pull.get_issue_comments.return_value = []
        repo = Mock()
        repo.get_pull.return_value = pull

        process_event(
            {"action": "dequeued", "pull_request": {"number": 123}},
            Mock(),
            repo,
            "owner/repo",
        )

        repo.get_pull.assert_called_once_with(123)

    def test_analysis_completion_accepts_merge_group_origin(self):
        pull = Mock()
        pull.number = 123
        pull.head.sha = HEAD_SHA
        pull.get_issue_comments.return_value = []
        repo = Mock()
        repo.get_pull.return_value = pull

        process_event(
            {
                "action": "completed",
                "workflow_run": {
                    "name": "CI Failure Doctor — Merge Queue",
                    "event": "merge_group",
                    "conclusion": "success",
                    "display_title": "CI Doctor MQ: gh-readonly-queue/master/pr-123-abcdef123456",
                },
            },
            Mock(),
            repo,
            "owner/repo",
        )

        repo.get_pull.assert_called_once_with(123)

    def test_dequeued_event_resolves_pr_number(self):
        self.assertEqual(
            pr_number_from_event({"action": "dequeued", "pull_request": {"number": 123}}),
            "123",
        )

    def test_analysis_completion_resolves_pr_number_from_run_title(self):
        event = {
            "workflow_run": {
                "name": "CI Failure Doctor — Merge Queue",
                "conclusion": "success",
                "display_title": "CI Doctor MQ: gh-readonly-queue/master/pr-123-abcdef123456",
            }
        }

        self.assertEqual(pr_number_from_event(event), "123")

    def test_analysis_completion_rejects_non_analysis_workflow(self):
        with self.assertRaisesRegex(SystemExit, "did not originate"):
            pr_number_from_event({"workflow_run": {"name": "Other", "conclusion": "success"}})

    def test_source_run_must_be_failed_merge_group_for_same_pr(self):
        valid = SimpleNamespace(
            event="merge_group",
            conclusion="failure",
            head_branch="gh-readonly-queue/master/pr-123-abcdef123456",
        )
        wrong_pr = SimpleNamespace(
            event="merge_group",
            conclusion="failure",
            head_branch="gh-readonly-queue/master/pr-124-abcdef123456",
        )

        self.assertTrue(source_run_matches_pr(valid, "123"))
        self.assertFalse(source_run_matches_pr(wrong_pr, "123"))
        self.assertFalse(source_run_matches_pr(SimpleNamespace(
            event="pull_request",
            conclusion="failure",
            head_branch=valid.head_branch,
        ), "123"))

    def test_latest_queue_event_includes_removal_reason_and_enqueue_time(self):
        added_at = NOW.replace(hour=6)
        removed_at = NOW.replace(hour=7)
        pull = SimpleNamespace(
            as_issue=lambda: SimpleNamespace(
                get_timeline=lambda: [
                    SimpleNamespace(event="added_to_merge_queue", created_at=added_at),
                    SimpleNamespace(
                        event="removed_from_merge_queue",
                        created_at=removed_at,
                        reason="failed_checks",
                    ),
                ]
            )
        )

        latest, latest_added = latest_queue_events(pull)

        self.assertEqual(latest, ("removed_from_merge_queue", removed_at, "failed_checks"))
        self.assertEqual(latest_added, added_at)


if __name__ == "__main__":
    unittest.main()
