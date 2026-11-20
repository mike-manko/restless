from typing import Any

import pytest

from restless.openapi.refs import resolve_refs

# A spec with one reusable schema, and a path that a ref can point into:
SPEC: dict[str, Any] = {
    "paths": {"/orders/{id}": {"get": {"summary": "Get one order"}}},
    "components": {"schemas": {"Address": {"type": "object", "properties": {"city": {"type": "string"}}}}},
}
ADDRESS = SPEC["components"]["schemas"]["Address"]


def test_same_schema_used_twice_side_by_side_expands_both_times() -> None:
    order = {"properties": {"billing": {"$ref": "#/components/schemas/Address"}, "shipping": {"$ref": "#/components/schemas/Address"}}}

    resolved = resolve_refs(order, SPEC)

    # Only a schema that contains itself is cut short, not one that appears twice:
    assert resolved["properties"]["billing"] == ADDRESS
    assert resolved["properties"]["shipping"] == ADDRESS


def test_keys_next_to_a_ref_are_kept_on_top_of_the_target() -> None:
    shipping = {"$ref": "#/components/schemas/Address", "description": "Where to ship the order."}

    resolved = resolve_refs(shipping, SPEC)

    assert resolved == {**ADDRESS, "description": "Where to ship the order."}


def test_escaped_json_pointer_is_followed() -> None:
    # The path key /orders/{id} is written as ~1orders~1{id} inside a JSON pointer:
    resolved = resolve_refs({"$ref": "#/paths/~1orders~1{id}/get"}, SPEC)

    assert resolved == {"summary": "Get one order"}


@pytest.mark.parametrize(
    ("node", "expected"),
    [
        ({"$ref": "#/components/schemas/Missing"}, {"$ref": "#/components/schemas/Missing", "note": "Broken reference in the spec."}),
        # External files aren't loaded, so the ref stays as it is:
        ({"$ref": "common.yaml#/components/schemas/Money"}, {"$ref": "common.yaml#/components/schemas/Money"}),
    ],
    ids=["broken", "external"],
)
def test_ref_that_cannot_be_followed_is_left_for_the_model_to_read(node: dict[str, Any], expected: dict[str, Any]) -> None:
    assert resolve_refs(node, SPEC) == expected
