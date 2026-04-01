import requests
import json

url = "http://0.0.0.0:9081/v1/chat/completions"

headers = {
    "api-key": "<api-key>"
}

# audio_file_path = "./test.wav"
audio_file_path = "test_39940_42556.wav"
output_file_path = f"output_{audio_file_path.split(".")[0]}.json"

try:
    with open(audio_file_path, "rb") as audio_file:
        files = {
            "file": audio_file
        }
        print("正在发送请求，请稍候...")
        response = requests.post(url, headers=headers, files=files)
        
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
