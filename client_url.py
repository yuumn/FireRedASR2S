import argparse
import requests
import json
import time

parser = argparse.ArgumentParser(description="音频文件识别客户端")
parser.add_argument("--input", type=str, default="input/test.wav", help="音频文件路径")
parser.add_argument("--output", type=str, default="output/output.json", help="输出文件路径")
parser.add_argument("--ip", type=str, default="0.0.0.0", help="ip")
parser.add_argument("--port", type=int, default=8081, help="port")
args = parser.parse_args()

url = f"http://{args.ip}:{args.port}/v1/audio/transcriptions"

headers = {
    "api-key": "EMPTY"
}

audio_file_path = args.input
output_file_path = args.output

try:
    with open(audio_file_path, "rb") as audio_file:
        files = {
            # "file": audio_file
        }
        data = {
            "url": "https://image-url-2-feature-1251524319.cos.ap-shanghai.myqcloud.com/zhongyi/input.wav"
            # "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav"
        }
        print("正在发送请求，请稍候...")
        start_time = time.time()
        response = requests.post(url, headers=headers, files=files, data=data)
        end_time = time.time()
        print(f"time: {end_time - start_time}")
    # if output_file_path is not "None"
    with open(output_file_path, "w", encoding="utf-8") as out_file:
        out_file.write(json.dumps(response.json(), indent=4, ensure_ascii=False))

    if response.status_code == 200:
        print(f"保存成功，已写入 {output_file_path}")
    else:
        print(f"请求失败，服务器返回信息: {response.text}")

except FileNotFoundError:
    print(f"错误: 找不到文件 {audio_file_path}，请检查路径是否正确。")
except Exception as e:
    print(f"发生错误: {e}")

