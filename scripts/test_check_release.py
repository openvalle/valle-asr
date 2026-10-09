"""A release must have green native CI for all six OS/CPU combinations."""

import copy
import unittest
from unittest.mock import patch
import check_release


class ReleaseChecks(unittest.TestCase):
    def setUp(self):
        self.sha = "a" * 40
        self.runs = [dict(path=".github/workflows/" + name, head_sha=self.sha,
                          head_branch="main", event="push", id=index,
                          status="completed", conclusion="success")
                     for index, name in enumerate(check_release.WORKFLOWS)]

    def test_all_six_are_required_at_the_exact_main_commit(self):
        self.assertEqual(len(check_release.WORKFLOWS), 6)
        self.assertEqual(len(check_release.passed_runs(self.runs, self.sha)), 6)
        for index in range(6):
            with self.subTest(missing=check_release.WORKFLOWS[index]):
                with self.assertRaises(RuntimeError):
                    check_release.passed_runs(self.runs[:index] + self.runs[index + 1:], self.sha)
        for field, value in (("head_sha", "b" * 40), ("head_branch", "feature"),
                             ("event", "pull_request"), ("status", "in_progress"),
                             ("conclusion", "failure"), ("conclusion", "skipped")):
            with self.subTest(field=field, value=value):
                runs = copy.deepcopy(self.runs)
                runs[-1][field] = value
                with self.assertRaises(RuntimeError):
                    check_release.passed_runs(runs, self.sha)

    def test_newer_failed_attempt_cannot_reuse_old_success(self):
        failed = dict(self.runs[-1], id=99, conclusion="failure")
        with self.assertRaises(RuntimeError):
            check_release.passed_runs(self.runs + [failed], self.sha)

    def test_version_mismatch_rejects_before_network(self):
        with patch.dict(check_release.os.environ, {"RELEASE_TAG": "v0.2.0"}), patch.object(check_release, "request") as request:
            with self.assertRaises(RuntimeError):
                check_release.check_ci({"version": "0.1.0"})
            request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
