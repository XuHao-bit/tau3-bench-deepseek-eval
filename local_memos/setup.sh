#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
# 本地数据库启动器使用 macOS Apple Silicon 的 Java 发行包。
if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
  echo "本地启动器要求 macOS Apple Silicon；其他平台需适配 Java/Neo4j 启动方式。" >&2
  exit 1
fi
SOURCE="$ROOT/MemOS"
REVISION="a7367d07e55db61099f7b4e2c1108bc5831a24f3"
if [ ! -x "$ROOT/.venv/bin/python" ]; then
  echo "请先在仓库根目录运行 ./setup.sh。" >&2
  exit 1
fi
if [ ! -d "$SOURCE/.git" ]; then
  git init "$SOURCE"
  git -C "$SOURCE" remote add origin https://github.com/MemTensor/MemOS.git
fi
CURRENT="$(git -C "$SOURCE" rev-parse --verify HEAD 2>/dev/null || true)"
if [ -n "$CURRENT" ] && [ "$CURRENT" != "$REVISION" ]; then
  echo "MemOS 已有其他版本，请使用新的克隆目录；要求 commit $REVISION。" >&2
  exit 1
fi
if [ -z "$CURRENT" ]; then
  git -C "$SOURCE" fetch --depth 1 origin "$REVISION"
  git -C "$SOURCE" checkout --detach FETCH_HEAD
fi
if ! git -C "$SOURCE" diff --quiet HEAD -- src pyproject.toml; then
  echo "MemOS 的 Python 源码有本地修改，请使用固定版本的干净代码。" >&2
  exit 1
fi
mkdir -p "$HERE/runtime" "$HERE/downloads"
if [ ! -x "$HERE/.venv/bin/python" ]; then
  "$ROOT/.venv/bin/python" -m venv "$HERE/.venv"
fi
"$HERE/.venv/bin/python" -m pip install -c "$HERE/constraints.txt" \
  -e "$SOURCE[tree-mem,mem-scheduler,mem-reader]" tiktoken jieba alibabacloud-oss-v2

# 下载中断时保留完整缓存；下次可直接重新运行安装脚本。
download_archive() {
  local url="$1" archive="$2"
  if [ ! -s "$archive" ]; then
    curl -fL --retry 2 --connect-timeout 30 "$url" -o "$archive.part"
    mv "$archive.part" "$archive"
  fi
}

shopt -s nullglob
java_dirs=("$HERE"/runtime/jdk-21*-jre)
if [ "${#java_dirs[@]}" -eq 0 ]; then
  download_archive \
    'https://api.adoptium.net/v3/binary/latest/21/ga/mac/aarch64/jre/hotspot/normal/eclipse' \
    "$HERE/downloads/java21.tar.gz"
  tar -xzf "$HERE/downloads/java21.tar.gz" -C "$HERE/runtime"
  java_dirs=("$HERE"/runtime/jdk-21*-jre)
fi
if [ ! -x "$HERE/runtime/apache-maven-3.9.9/bin/mvn" ]; then
  download_archive \
    'https://repo.maven.apache.org/maven2/org/apache/maven/apache-maven/3.9.9/apache-maven-3.9.9-bin.tar.gz' \
    "$HERE/downloads/maven.tar.gz"
  tar -xzf "$HERE/downloads/maven.tar.gz" -C "$HERE/runtime"
fi
export JAVA_HOME="${java_dirs[0]}/Contents/Home"
"$HERE/runtime/apache-maven-3.9.9/bin/mvn" -f "$HERE/neo4j-pom.xml" \
  -Dmaven.repo.local="$HERE/runtime/m2" dependency:copy-dependencies \
  -DincludeScope=runtime -DoutputDirectory="$HERE/runtime/neo4j/lib"
