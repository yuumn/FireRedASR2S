import os
import json
import time
import requests

# url = "https://maas.devops.xiaohongshu.com/qsmaas-funasrx1-offxquota/v1/chat/completions"
url = "http://127.0.0.1:8881/v1/chat/completions"

headers = {
    "api-key": "<api-key>"
}

# audio_file_path = "./chenxiangliudianban.wav"
audio_file_path = "/root/yuanerhang/asr/longtime_example/time_34min.wav"
output_file_path = os.path.basename(audio_file_path).split(".")[0] + "2.json"
# output_file_path = "output.json"

try:
    with open(audio_file_path, "rb") as audio_file:
        files = {
            "file": audio_file
        }
        data = {
            "hotword": "wechat 微信 小红书"
        }
        print("正在发送请求，请稍候...")
        start = time.time()
        response = requests.post(url, headers=headers, files=files, data=data)
        end = time.time()

    with open(output_file_path, "w", encoding="utf-8") as out_file:
        out_file.write(json.dumps(response.json(), indent=4, ensure_ascii=False))

    if response.status_code == 200:
        print(f"保存成功，已写入 {output_file_path}")
        print(f"time: {end - start}")
    else:
        print(f"请求失败，服务器返回信息: {response.text}")

except FileNotFoundError:
    print(f"错误: 找不到文件 {audio_file_path}，请检查路径是否正确。")
except Exception as e:
    print(f"发生错误: {e}")
