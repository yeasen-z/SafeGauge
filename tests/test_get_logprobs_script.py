import unittest

from scripts.get_logprobs import validate_resume


def input_record():
    return {
        "id": "sample",
        "label": 1,
        "messages": [{"role": "user", "content": "input"}],
    }


def output_record(**extra):
    record = {
        **input_record(),
        "suffix_id": "suffix",
        "suffix": "semantic suffix",
        "thinking_bypass_prefill": "none",
    }
    record.update(extra)
    return record


class ResumeTests(unittest.TestCase):
    def validate(self, record, backend, version=None):
        validate_resume(
            [record],
            [input_record()],
            suffix_id="suffix",
            suffix="semantic suffix",
            thinking_bypass_prefill="none",
            server_backend=backend,
            server_version=version,
        )

    def test_legacy_output_can_resume_with_vllm(self):
        self.validate(output_record(), "vllm")

    def test_legacy_output_cannot_resume_with_sglang(self):
        with self.assertRaisesRegex(ValueError, "predates backend tracking"):
            self.validate(output_record(), "sglang", "0.5.17")

    def test_backend_or_version_change_requires_overwrite(self):
        record = output_record(
            server_backend="sglang",
            server_version="0.5.17",
        )
        with self.assertRaisesRegex(ValueError, "backend mismatch"):
            self.validate(record, "sglang", "0.5.18")


if __name__ == "__main__":
    unittest.main()
