from __future__ import annotations

from smsp import SuffixLogProbsExtractor


def _field(value, name):
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


class ActualTokenLogProbsExtractor(SuffixLogProbsExtractor):
    """Select the observed prompt token explicitly by token ID.

    Current vLLM inserts the observed token first, so the legacy extractor's
    first-value behavior is normally correct for that implementation. Explicit
    ID matching avoids coupling this experiment to dictionary ordering and
    makes the intended feature auditable. It does not by itself remove GPU
    numerical variation caused by different concurrent batching schedules.
    """

    def get_logprobs(self, messages, logprobs_num=5):
        # Alternatives are useful only as a container in which we locate the
        # known observed token ID; their order is deliberately ignored.
        return super().get_logprobs(messages, logprobs_num=5)

    def _match_observed(self, token_dict, expected_token_id, position):
        direct = token_dict.get(expected_token_id)
        if direct is None:
            direct = token_dict.get(str(expected_token_id))
        if direct is not None:
            return direct

        expected_decoded = self.tokenizer.decode(
            [expected_token_id], clean_up_tokenization_spaces=False
        )
        expected_piece = self.tokenizer.convert_ids_to_tokens(expected_token_id)
        matches = []
        for key, value in token_dict.items():
            decoded = _field(value, "decoded_token")
            if decoded == expected_decoded or key in (expected_decoded, expected_piece):
                matches.append(value)
        if len(matches) == 1:
            return matches[0]
        summary = [
            {
                "key": str(key),
                "decoded_token": _field(value, "decoded_token"),
                "rank": _field(value, "rank"),
            }
            for key, value in token_dict.items()
        ]
        raise ValueError(
            f"Could not uniquely match observed token at suffix position {position}; "
            f"expected id={expected_token_id}, decoded={expected_decoded!r}, "
            f"piece={expected_piece!r}, candidates={summary}"
        )

    def _get_logprobs_server(self, all_input_tokens, suffix_tokens, logprobs_num):
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.completions.create(
            model=self.model,
            prompt=all_input_tokens,
            max_tokens=1,
            temperature=self.temperature,
            top_p=self.top_p,
            extra_body={"prompt_logprobs": max(int(logprobs_num), 5)},
        )
        choice = response.choices[0]
        raw = choice.prompt_logprobs[-len(suffix_tokens) :]
        selected = [
            self._match_observed(values, token_id, position)
            for position, (values, token_id) in enumerate(zip(raw, suffix_tokens))
        ]
        return {
            "text": choice.text,
            "suffix_logprobs": raw,
            "all_logprobs": [float(_field(value, "logprob")) for value in selected],
            "all_rank": [int(_field(value, "rank")) for value in selected],
            "selection": "matched_observed_token_id",
        }

    def _get_logprobs_offline(self, all_input_tokens, suffix_tokens, logprobs_num):
        from vllm import SamplingParams

        output = self.llm.generate(
            prompts=[all_input_tokens],
            sampling_params=SamplingParams(
                temperature=0,
                max_tokens=1,
                prompt_logprobs=max(int(logprobs_num), 5),
                logprobs=max(int(logprobs_num), 5),
            ),
        )[0]
        raw = output.prompt_logprobs[-len(suffix_tokens) :]
        selected = [
            self._match_observed(values, token_id, position)
            for position, (values, token_id) in enumerate(zip(raw, suffix_tokens))
        ]
        return {
            "text": output.outputs[0].text,
            "suffix_logprobs": raw,
            "all_logprobs": [float(_field(value, "logprob")) for value in selected],
            "all_rank": [int(_field(value, "rank")) for value in selected],
            "selection": "matched_observed_token_id",
        }
