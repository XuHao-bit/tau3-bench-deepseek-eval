"""用原生 Neo4j 写入／解析方法验证 Skill 和工具经验的结构往返。"""

import ast
import copy
import json
import logging
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from local_memos.internal_info_compat import install_internal_info_compat
from local_memos.skill_metadata_compat import install_skill_metadata_compat


def native_methods():
    """加载纯方法，驱动由 mock 提供，不导入数据库或模型依赖。"""
    path = Path(__file__).resolve().parents[1] / "MemOS/src/memos/graph_dbs/neo4j.py"
    module = ast.parse(path.read_text())
    functions = [node for node in module.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"_prepare_node_metadata", "_flatten_info_fields",
                                   "_sanitize_neo4j_value", "_sanitize_neo4j_metadata"}]
    graph_class = next(node for node in module.body
                       if isinstance(node, ast.ClassDef) and node.name == "Neo4jGraphDB")
    functions += [node for node in graph_class.body if isinstance(node, ast.FunctionDef)
                  and node.name in {"add_node", "add_nodes_batch", "_parse_node"}]
    namespace = {"Any": Any, "json": json, "datetime": datetime,
                 "logger": logging.getLogger("skill_metadata_test")}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


NATIVE = native_methods()


class PropertyDriver:
    def __init__(self):
        self.saved = {}

    def session(self, **kwargs):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def run(self, query, **parameters):
        rows = parameters.get("nodes") or [parameters]
        for row in rows:
            for key, value in row["metadata"].items():
                primitive = lambda item: item is None or isinstance(item, (str, int, float, bool))
                if not (primitive(value) or isinstance(value, list) and all(primitive(item) for item in value)):
                    raise TypeError(f"Neo4j 属性不接受嵌套结构：{key}")
        for row in rows:
            self.saved[row["id"]] = copy.deepcopy({
                "id": row["id"], "memory": row["memory"],
                "created_at": row["created_at"], "updated_at": row["updated_at"],
                **row["metadata"],
            })


class NativeGraphWithPropertyDriver:
    add_node = NATIVE["add_node"]
    add_nodes_batch = NATIVE["add_nodes_batch"]
    _parse_node = NATIVE["_parse_node"]

    def __init__(self):
        self.config = SimpleNamespace(user_name=None, use_multi_db=False)
        self.db_name = "test"
        self.driver = PropertyDriver()


def records():
    return [
        {"id": "skill-1", "memory": "先核对授权再执行业务。", "metadata": {
            "memory_type": "SkillMemory", "name": "核对授权",
            "procedure": ["检查当前状态", {"then": ["获得授权", "执行并验证"]}],
            "experience": [{"condition": "状态已改变", "lesson": "重新查询"}],
            "preference": [], "examples": [{"input": "业务请求", "output": {"action": "核验"}}],
            "scripts": {"check.py": {"content": "print('check')", "args": [1, 2]}},
            "others": {"checks": [True, {"required": None}]},
            "tags": ["授权", "验证"], "embedding": [],
            "sources": [{"role": "tool", "message_id": "m0005", "content": "工具结果",
                         "file_info": {"path": "evidence"}}],
            "info": {"task_id": "task_001"},
        }},
        {"id": "tool-1", "memory": "业务执行轨迹", "metadata": {
            "memory_type": "ToolTrajectoryMemory",
            "tool_experience": [{"tool_name": "check", "conditions": {"authorized": True}}],
            "experience": [{"failure": "缺少前置状态", "fix": ["先查询"]}],
            "tool_used_status": [{"name": "check", "success": True}],
            "correctness": {"observed": True},
            "info": {"task_id": "task_001"},
        }},
    ]


class SkillMetadataCompatTests(unittest.TestCase):
    def test_native_batch_rejects_real_tool_experience_and_commits_no_skill(self):
        graph = NativeGraphWithPropertyDriver()
        # 首条 Skill 使用原生可存的简单字段，后条工具经验使整个 batch 失败。
        rows = [{"id": "skill-1", "memory": "方法", "metadata": {"memory_type": "SkillMemory"}}, records()[1]]
        with self.assertRaisesRegex(TypeError, "tool_experience"):
            graph.add_nodes_batch(rows, user_name="cube-c")
        self.assertEqual(graph.driver.saved, {})

    def test_native_batch_round_trips_structured_skill_and_tool_fields(self):
        graph = NativeGraphWithPropertyDriver()
        install_skill_metadata_compat(graph)
        original = records()
        rows = copy.deepcopy(original)
        graph.add_nodes_batch(rows, user_name="cube-c")
        self.assertEqual(rows, original)
        for row in original:
            stored = graph.driver.saved[row["id"]]
            restored = graph._parse_node(stored)["metadata"]
            for field, value in row["metadata"].items():
                if field == "info":
                    self.assertEqual(restored["task_id"], value["task_id"])
                else:
                    self.assertEqual(restored[field], value, field)
        self.assertEqual(json.loads(graph.driver.saved["tool-1"]["tool_experience"]),
                         original[1]["metadata"]["tool_experience"])

    def test_single_node_is_not_double_encoded_by_native_sanitizer(self):
        graph = NativeGraphWithPropertyDriver()
        install_skill_metadata_compat(graph)
        row = records()[0]
        graph.add_node(row["id"], row["memory"], row["metadata"], user_name="cube-c")
        saved = graph.driver.saved[row["id"]]
        self.assertIsInstance(json.loads(saved["scripts"]), dict)
        restored = graph._parse_node(saved)["metadata"]
        self.assertEqual(restored["scripts"], row["metadata"]["scripts"])
        self.assertEqual(restored["sources"], row["metadata"]["sources"])

    def test_sources_restore_both_native_serialization_layers(self):
        graph = NativeGraphWithPropertyDriver()
        original = records()[0]["metadata"]["sources"]
        graph.add_nodes_batch([{"id": "source-only", "memory": "工具结果", "metadata": {
            "sources": copy.deepcopy(original),
        }}], user_name="cube-c")
        raw = graph.driver.saved["source-only"]
        self.assertEqual(json.loads(json.loads(raw["sources"][0])), original[0])
        install_skill_metadata_compat(graph)
        self.assertEqual(graph._parse_node(raw)["metadata"]["sources"], original)
        raw["sources"] = copy.deepcopy(original)
        self.assertEqual(graph._parse_node(raw)["metadata"]["sources"], original)

    def test_composes_with_internal_info_and_native_info_flattening(self):
        graph = NativeGraphWithPropertyDriver()
        install_internal_info_compat(graph)
        install_skill_metadata_compat(graph)
        experience = [{"lesson": {"retry": "检查当前状态"}}]
        metadata = {"internal_info": {"ingest_batch_id": "batch-c"},
                    "info": {"tool_experience": experience, "task_id": "task_005"}}
        graph.add_nodes_batch([{"id": "m1", "memory": "方法", "metadata": metadata}], user_name="cube-c")
        restored = graph._parse_node(graph.driver.saved["m1"])["metadata"]
        self.assertEqual(restored["internal_info"], {"ingest_batch_id": "batch-c"})
        self.assertEqual(restored["tool_experience"], experience)
        self.assertEqual(restored["task_id"], "task_005")

    def test_plain_text_and_already_encoded_values_are_not_encoded_again(self):
        graph = NativeGraphWithPropertyDriver()
        install_skill_metadata_compat(graph)
        metadata = {"procedure": "先检查，再执行。", "experience": ["查询当前状态"],
                    "scripts": json.dumps({"check.py": "pass"}), "tags": ["保留原生标签"]}
        graph.add_nodes_batch([{"id": "m1", "memory": "方法", "metadata": metadata}], user_name="cube-c")
        raw = graph.driver.saved["m1"]
        self.assertEqual(raw["procedure"], metadata["procedure"])
        self.assertEqual(raw["scripts"], metadata["scripts"])
        self.assertEqual(raw["tags"], metadata["tags"])
        restored = graph._parse_node(raw)["metadata"]
        self.assertEqual(restored["scripts"], {"check.py": "pass"})
        self.assertEqual(restored["experience"], metadata["experience"])


if __name__ == "__main__":
    unittest.main()
