"""将 MemOS 内部元信息按 Neo4j 支持的属性类型保存并恢复。"""

import json
from functools import wraps
from typing import Any


def _encode_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    encoded = dict(metadata)
    if isinstance(encoded.get("internal_info"), dict):
        encoded["internal_info"] = json.dumps(encoded["internal_info"], ensure_ascii=False)
    return encoded


def install_internal_info_compat(graph_store: Any) -> None:
    """只调整当前服务 graph_store 的内部字段，不修改原库或 API。"""
    add_node = graph_store.add_node
    add_nodes_batch = graph_store.add_nodes_batch
    parse_node = graph_store._parse_node

    @wraps(add_node)
    def add_node_encoded(id, memory, metadata, user_name=None):
        return add_node(id, memory, _encode_metadata(metadata), user_name=user_name)

    @wraps(add_nodes_batch)
    def add_nodes_encoded(nodes, user_name=None):
        encoded = [
            {**node, "metadata": _encode_metadata(node.get("metadata", {}))}
            for node in nodes
        ]
        return add_nodes_batch(encoded, user_name=user_name)

    @wraps(parse_node)
    def parse_node_decoded(node_data):
        result = parse_node(node_data)
        internal_info = result["metadata"].get("internal_info")
        if isinstance(internal_info, str):
            result["metadata"]["internal_info"] = json.loads(internal_info)
        return result

    graph_store.add_node = add_node_encoded
    graph_store.add_nodes_batch = add_nodes_encoded
    graph_store._parse_node = parse_node_decoded
