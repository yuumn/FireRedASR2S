#!/bin/bash

#===============================================
# FunASR WebSocket Service 启动脚本
# 支持命令行参数配置
#===============================================

# 默认配置
WORK_DIR="/workspace/FireRedASR2S"
HOST="0.0.0.0"
PORT=8081
WORKERS=8
TIMEOUT=900

# 帮助信息
show_help() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -w, --workers NUM     工作进程数 (默认: ${WORKERS})"
    echo "  -p, --port PORT       监听端口 (默认: ${PORT})"
    echo "  -t, --timeout SEC     超时时间 (默认: ${TIMEOUT})"
    echo "  -h, --help            显示帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 -w 8 -p 8081 -t 3000"
    echo "  $0 --workers 8 --port 8081 --timeout 3000"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -w|--workers)
            WORKERS="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -t|--timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

# 切换到工作目录
cd ${WORK_DIR} || {
    echo "错误: 无法进入目录 ${WORK_DIR}"
    exit 1
}

echo "=========================================="
echo "启动 FireRedASR2S WebSocket 服务"
echo "工作目录: ${WORK_DIR}"
echo "监听地址: ${HOST}:${PORT}"
echo "工作进程: ${WORKERS}"
echo "超时时间: ${TIMEOUT}s"
echo "=========================================="

# 启动服务
exec gunicorn \
    -w ${WORKERS} \
    -b ${HOST}:${PORT} \
    --timeout ${TIMEOUT} \
    -k gevent \
    service:app
