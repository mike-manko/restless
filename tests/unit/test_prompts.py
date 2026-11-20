import json

from restless.actor.prompts import build_executor_task, build_judge_prompt
from tests.fakes import http_call


def test_executor_task_puts_the_operation_list_first_and_states_the_budget() -> None:
    task = build_executor_task("name: create order", "orders  POST   /orders  Create an order", max_http_calls=5)

    # The operation list is the same for every scenario in a suite, so it goes first, where the prompt cache can reuse it:
    assert task.index("<operations>") < task.index("<scenario>")
    assert "at most 5 HTTP requests" in task


def test_judge_prompt_holds_the_call_log_as_numbered_json() -> None:
    calls = [http_call(method="POST", status=201, body={"id": 7, "city": "Zürich", "paid": True}), http_call(status=204)]

    prompt = build_judge_prompt("name: create order", calls)

    # Real JSON (null, true) rather than Python's repr (None, True), so it parses back:
    call_log = json.loads(prompt.split("<call_log>")[1].split("</call_log>")[0])
    assert [call["index"] for call in call_log] == [0, 1]
    assert call_log[0]["body"] == {"id": 7, "city": "Zürich", "paid": True}
    # Non-ASCII text stays readable instead of turning into \u escapes:
    assert '"city": "Zürich"' in prompt
