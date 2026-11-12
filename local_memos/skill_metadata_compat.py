"""为 C 服务保存并恢复原生 Skill／工具经验的结构化元数据。"""

import copy
import json
from functools import wraps
from typing import Any


JSON_FIELDS = (
    "tool_experience", "tool_used_status", "correctness",
    "procedure", "experience", "preference", "examples", "scripts", "others",
)


def _encode_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    encoded = copy.deepcopy(metadata)
    # info 会由原生写入路径展开；其中同名结构字段也须先转为属性值。
    containers = [encoded]
    if isinstance(encoded.get("info"), dict):
        containers.append(encoded["info"])
    for container in containers:
        for field in JSON_FIELDS:
            if isinstance(container.get(field), (dict, list)):
                container[field] = json.dumps(container[field], ensure_ascii=False)
    return encoded


def _restore_source(source: Any) -> Any:
    """原生 prepare 和 add 各编码一次，只还原到来源字典。"""
    candidate = source
    for _ in range(2):
        if isinstance(candidate, dict):
            return candidate
        if not isinstance(candidate, str):
            break
        try:
            candidate = json.loads(candidate)
        except ValueError:
            break
    return candidate if isinstance(candidate, dict) else source


def install_skill_metadata_compat(graph_store: Any) -> None:
    """仅调整传入的 C 服务实例；保留原生来源编码和其它字段处理。"""
    add_node = graph_store.add_node
    add_nodes_batch = graph_store.add_nodes_batch
    parse_node = graph_store._parse_node

    @wraps(add_node)
    def add_node_encoded(id, memory, metadata, user_name=None):
        return add_node(id, memory, _encode_metadata(metadata), user_name=user_name)

    @wraps(add_nodes_batch)
    def add_nodes_encoded(nodes, user_name=None):
        encoded = [{**node, "metadata": _encode_metadata(node.get("metadata") or {})}
                   for node in nodes]
        return add_nodes_batch(encoded, user_name=user_name)

    @wraps(parse_node)
    def parse_node_decoded(node_data):
        result = parse_node(copy.deepcopy(node_data))
        metadata = result["metadata"]
        for field in JSON_FIELDS:
            value = metadata.get(field)
            if not isinstance(value, str):
                continue
            try:
                decoded = json.loads(value)
            except ValueError:
                continue
            if isinstance(decoded, (dict, list)):
                metadata[field] = decoded
        if isinstance(metadata.get("sources"), list):
            metadata["sources"] = [_restore_source(source) for source in metadata["sources"]]
        return result

    graph_store.add_node = add_node_encoded
    graph_store.add_nodes_batch = add_nodes_encoded
    graph_store._parse_node = parse_node_decoded
