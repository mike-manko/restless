from typing import Any

import pytest

from restless.openapi.catalog import ApiCatalog, UnknownOperationError

# A small spec in the shape Swashbuckle produces: schemas under components, a path-level parameter, and one operation without operationId.
SPEC: dict[str, Any] = {
    "openapi": "3.0.1",
    "paths": {
        "/orders": {
            "post": {
                "operationId": "CreateOrder",
                "summary": "Create an order",
                "requestBody": {"content": {"application/json": {"schema": {"$ref": "#/components/schemas/CreateOrder"}}}},
                "responses": {"201": {"description": "Created", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Order"}}}}},
            },
            "get": {"summary": "List orders", "responses": {"200": {"description": "OK"}}},
        },
        "/orders/{id}": {
            "parameters": [{"name": "id", "in": "path", "required": True, "schema": {"type": "integer"}}],
            "get": {"description": "Get one order.\nReturns 404 when it doesn't exist.", "responses": {"200": {"description": "OK"}}},
        },
        "/categories": {
            "get": {"summary": "Category tree", "responses": {"200": {"content": {"application/json": {"schema": {"$ref": "#/components/schemas/Category"}}}}}},
        },
    },
    "components": {
        "schemas": {
            "CreateOrder": {"type": "object", "properties": {"productId": {"type": "integer"}, "quantity": {"type": "integer"}}},
            "Order": {"type": "object", "properties": {"id": {"type": "integer"}, "lines": {"type": "array", "items": {"$ref": "#/components/schemas/CreateOrder"}}}},
            "Category": {"type": "object", "properties": {"name": {"type": "string"}, "children": {"type": "array", "items": {"$ref": "#/components/schemas/Category"}}}},
        }
    },
}


@pytest.fixture
def catalog() -> ApiCatalog:
    return ApiCatalog({"orders": SPEC})


def test_index_lists_every_operation_with_its_summary(catalog: ApiCatalog) -> None:
    lines = catalog.index().splitlines()

    assert lines == [
        "orders  POST   /orders  Create an order",
        "orders  GET    /orders  List orders",
        "orders  GET    /orders/{id}  Get one order.",
        "orders  GET    /categories  Category tree",
    ]


@pytest.mark.parametrize("name", ["POST /orders", "post  /orders", "CreateOrder"])
def test_find_accepts_key_lowercase_method_and_operation_id(catalog: ApiCatalog, name: str) -> None:
    assert catalog.find("orders", name).key == "POST /orders"


def test_find_suggests_close_names(catalog: ApiCatalog) -> None:
    with pytest.raises(UnknownOperationError, match=r"Did you mean: GET /orders/\{id\}"):
        catalog.find("orders", "GET /orders/{orderId}")


def test_find_rejects_unknown_api(catalog: ApiCatalog) -> None:
    with pytest.raises(UnknownOperationError, match="Unknown API 'billing'"):
        catalog.find("billing", "GET /invoices")


def test_details_resolve_refs(catalog: ApiCatalog) -> None:
    details = catalog.details(catalog.find("orders", "POST /orders"))

    request_schema = details["requestBody"]["content"]["application/json"]["schema"]
    order_lines = details["responses"]["201"]["content"]["application/json"]["schema"]["properties"]["lines"]

    assert request_schema["properties"]["quantity"] == {"type": "integer"}
    assert order_lines["items"]["properties"]["productId"] == {"type": "integer"}


def test_details_include_path_level_parameters(catalog: ApiCatalog) -> None:
    details = catalog.details(catalog.find("orders", "GET /orders/{id}"))

    assert [parameter["name"] for parameter in details["parameters"]] == ["id"]


def test_recursive_schema_stops_with_a_note(catalog: ApiCatalog) -> None:
    details = catalog.details(catalog.find("orders", "GET /categories"))

    category = details["responses"]["200"]["content"]["application/json"]["schema"]
    child = category["properties"]["children"]["items"]

    assert child["$ref"] == "#/components/schemas/Category"
    assert "Recursive" in child["note"]
