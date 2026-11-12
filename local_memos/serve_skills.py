"""使用独立目录和端口运行 C 的原生 Skill 服务。"""

import argparse
import os
import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    os.environ["MEMOS_EVAL_PROFILE"] = "skills"
    os.environ["MEMOS_EVAL_PORT"] = str(args.port)
    # 原生 LOCAL 文件 URL 从进程参数读取端口。
    sys.argv = [sys.argv[0], "--port", str(args.port)]
    runpy.run_path(str(Path(__file__).with_name("serve.py")), run_name="__main__")
