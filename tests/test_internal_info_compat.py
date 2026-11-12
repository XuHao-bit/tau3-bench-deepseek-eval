"""检查真实故障字段的存储类型和往返恢复，不连接数据库。"""

import json
import unittest

from local_memos.internal_info_compat import install_internal_info_compat


class GraphWithNeo4jPropertyRules:
    def __init__(self):
        self.saved = {}

    def add_node(self, id, memory, metadata, user_name=None):
        if isinstance(metadata.get("internal_info"), dict):
            raise TypeError("Neo4j property cannot be a map")
        self.saved[id] = {"id": id, "memory": memory, "metadata": dict(metadata)}

    def add_nodes_batch(self, nodes, user_name=None):
        for node in nodes:
            self.add_node(node["id"], node["memory"], node["metadata"], user_name)

    def _parse_node(self, node):
        return {**node, "metadata": dict(node["metadata"])}


class InternalInfoCompatTest(unittest.TestCase):
    def test_unpatched_batch_rejects_real_failure_field(self):
        graph = GraphWithNeo4jPropertyRules()
        with self.assertRaises(TypeError):
            graph.add_nodes_batch([
                {"id": "m1", "memory": "tool result", "metadata": {
                    "internal_info": {"ingest_batch_id": "batch-1"}
                }}
            ])

    def test_batch_round_trip_preserves_nested_internal_info(self):
        graph = GraphWithNeo4jPropertyRules()
        install_internal_info_compat(graph)
        original = {"ingest_batch_id": "批次-1", "parts": [1, {"source": "tool"}]}
        metadata = {"internal_info": original, "task_id": "task_020"}
        graph.add_nodes_batch([{"id": "m1", "memory": "tool result", "metadata": metadata}])
        stored = graph.saved["m1"]
        self.assertIsInstance(stored["metadata"]["internal_info"], str)
        self.assertEqual(json.loads(stored["metadata"]["internal_info"]), original)
        restored = graph._parse_node(stored)
        self.assertEqual(restored["metadata"]["internal_info"], original)
        self.assertEqual(restored["metadata"]["task_id"], "task_020")
        self.assertIsInstance(metadata["internal_info"], dict)

    def test_single_node_round_trip(self):
        graph = GraphWithNeo4jPropertyRules()
        install_internal_info_compat(graph)
        graph.add_node("m1", "memory", {"internal_info": {"ingest_batch_id": "b"}})
        self.assertEqual(
            graph._parse_node(graph.saved["m1"])["metadata"]["internal_info"],
            {"ingest_batch_id": "b"},
        )

    def test_normal_metadata_does_not_gain_internal_info(self):
        graph = GraphWithNeo4jPropertyRules()
        install_internal_info_compat(graph)
        graph.add_node("m1", "memory", {"task_id": "task_001"})
        self.assertEqual(graph._parse_node(graph.saved["m1"])["metadata"], {"task_id": "task_001"})


if __name__ == "__main__":
    unittest.main()
