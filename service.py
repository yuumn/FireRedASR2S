'''
功能概述：音频分段摘要
步骤：
1、传入mp3,wav格式音频文件
2、
3、
4、输出结果

'''
from flask import Flask, request, jsonify
import os
import json
import ffmpeg
# import re
from werkzeug.utils import secure_filename
import torch
import torchaudio

from omegaconf import OmegaConf
from default_config import FireRedASR2Config

import glob
import logging

import soundfile as sf
from textgrid import IntervalTier, TextGrid

from fireredasr2s.fireredasr2 import FireRedAsr2Config
from fireredasr2s.fireredasr2system import (FireRedAsr2System,
                                            FireRedAsr2SystemConfig)
from fireredasr2s.fireredlid import FireRedLidConfig
from fireredasr2s.fireredpunc import FireRedPuncConfig
from fireredasr2s.fireredvad import FireRedVadConfig
import time
import httpx
from urllib.parse import urlparse
import filetype

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # 兼容旧版 Flask (< 2.2)
app.json.ensure_ascii = False        # 兼容新版 Flask (>= 2.2)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")

logger = logging.getLogger("fireredasr2s.asr_system")

args = OmegaConf.structured(FireRedASR2Config)


# 配置上传文件夹
UPLOAD_FOLDER = 'uploads'
UPLOAD_CONVERT_FOLDER = 'uploads_convert'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_CONVERT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['UPLOAD_CONVERT_FOLDER'] = UPLOAD_CONVERT_FOLDER

# Build Models
# VAD
vad_config = FireRedVadConfig(
    args.vad_use_gpu,
    args.smooth_window_size,
    args.speech_threshold,
    args.min_speech_frame,
    args.max_speech_frame,
    args.min_silence_frame,
    args.merge_silence_frame,
    args.extend_speech_frame,
    args.vad_chunk_max_frame,
    args.spk_model_dir,
)
# LID
lid_config = FireRedLidConfig(args.lid_use_gpu)
# ASR
asr_config = FireRedAsr2Config(
    args.asr_use_gpu,
    args.asr_use_half,
    args.beam_size,
    args.nbest,
    args.decode_max_len,
    args.softmax_smoothing,
    args.aed_length_penalty,
    args.eos_penalty,
    args.return_timestamp,
    0, 1.0, 0.0, 1.0,
    args.elm_dir,
    args.elm_weight
)
# Punc
punc_config = FireRedPuncConfig(
    args.punc_use_gpu,
    args.punc_sentence_max_length
)

asr_system_config = FireRedAsr2SystemConfig(
    args.vad_model_dir, args.lid_model_dir,
    args.asr_type,
    args.asr_model_dir if args.asr_type == "aed" else "/workspace/models/FireRedASR2S/FireRedASR2-LLM",
    args.punc_model_dir,
    vad_config, lid_config, asr_config, punc_config,
    args.asr_batch_size, args.punc_batch_size,
    args.enable_vad, args.enable_lid, args.enable_punc, 
    args.spk_mode
)
asr_system = FireRedAsr2System(asr_system_config)

def get_wav_info(wav_path):
    """
    Returns:
        wavs: list of (uttid, wav_path)
    """
    # def base(p): return os.path.basename(p).replace(".wav", "")
    def base(p): return os.path.basename(p).rsplit(".", 1)[0]
    if wav_path:
        wavs = (base(wav_path), wav_path)
    else:
        raise ValueError("Please provide valid wav info")
    # logger.info(f"#wavs={len(wavs)}")
    return wavs

def convert_audio(input_audio_path, output_wav_path):
    try:
        (
            ffmpeg
            .input(input_audio_path)
            .output(
                output_wav_path, 
                ar=16000, 
                ac=1, 
                acodec='pcm_s16le', 
                f='wav'
            )
            .overwrite_output() 
            .run(capture_stdout=True, capture_stderr=True)
        )
        print(f"转换成功：{output_wav_path}")
    except ffmpeg.Error as e:
        print(f"转换失败，错误信息：\n{e.stderr.decode('utf8')}")

def post_process_merge_result(result):
    # return result
    # 合并条件：相同说话人且时间间隔≤100ms
    def get_start_end_spk_text(sentence):
        return {
            "start": sentence["start_ms"],
            "end": sentence["end_ms"],
            "spk": sentence["spk"],
            "text": sentence["text"],
        }

    sentences = result["sentences"]
    sentences = [get_start_end_spk_text(sentence) for sentence in sentences]
    if len(sentences) <= 1:
        result["sentences"] = sentences
        return result
    
    # processed_result = []
    processed_sentences = []
    sentences_length = len(sentences)

    cur_chunk = sentences[0]
    for idx in range(1, sentences_length):
        chunk = sentences[idx]
        
        if chunk["spk"] == cur_chunk["spk"] and chunk["start"] - cur_chunk["end"] <= 100:
            cur_chunk["text"] += chunk["text"]
            cur_chunk["end"] = chunk["end"]
            if idx == sentences_length - 1:
                processed_sentences.append(cur_chunk)
            continue
        
        processed_sentences.append(cur_chunk)
        cur_chunk = chunk
        if idx == sentences_length - 1:
            processed_sentences.append(cur_chunk)
    
    result["sentences"] = processed_sentences
    return result

# 会议撰写
# @app.route('/AsrCamWithIdentify', methods=['POST'])
@app.route('/v1/audio/transcriptions', methods=['POST'])
def speech_recognition_Timestamp_cam_identify_speakers():
    # start_time = time.time()
    # 检查文件上传

    # if 'file' not in request.files:
    #     return jsonify({"error": "No audio file provided"}), 400

    # file = request.files['file']
    filepath = None
    filename = None
    if 'file' not in request.files:
        audio_url = request.form.get('url', None)
        if audio_url == None:
            return jsonify({"error": "Empty file"}), 400
        
        audio_file = httpx.get(audio_url).content

        kind = filetype.guess(audio_file)

        if not kind.mime.startswith("audio/"):
            return jsonify({"error": "Not an audio file"}), 400

        parsed_url = urlparse(audio_url)
        path = parsed_url.path 
        filename = os.path.basename(path) or "input.wav"
        filename = f"pid_{os.getpid()}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        with open(filepath, "wb") as f:
            f.write(audio_file)

    else:
        file = request.files['file']
        if file == "":
            return jsonify({"error": "Empty filename"}), 400
        # 保存上传文件
        filename = secure_filename(file.filename)
        filename = f"pid_{os.getpid()}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

    if filepath == None or filename == None:
        return jsonify({"error": "Empty file"}), 400
        
    filepath_convert = os.path.join(app.config['UPLOAD_CONVERT_FOLDER'], filename.rsplit(".", 1)[0] + ".wav")

    convert_audio(filepath, filepath_convert)


    # 提取 num_speakers min_speakers max_speakers
    num_speakers = request.form.get('num_speakers', None)
    if num_speakers is not None:
        num_speakers = int(num_speakers)

    min_speakers = request.form.get('min_speakers', None)
    if min_speakers is not None:
        min_speakers = int(min_speakers)

    max_speakers = request.form.get('max_speakers', None)
    if max_speakers is not None:
        max_speakers = int(max_speakers)
    
    print(f"num_speakers: {num_speakers}")
    print(f"min_speakers: {min_speakers}")
    print(f"max_speakers: {max_speakers}")


    try:
        # 执行语音识别
        uttid, _ = get_wav_info(filepath_convert)
        uttid = uttid[len(f"pid_{os.getpid()}_"):]
        with torch.inference_mode():
            result = asr_system.process(
                filepath_convert, 
                uttid,
                num_speakers,
                min_speakers,
                max_speakers,
            )

        # result = post_process_merge_result(result)
        # 处理结果
        # processed_result = process_cam_result_with_identify_speakers(result,speaker_db,filepath,identify_speakers)

        os.remove(filepath)
        os.remove(filepath_convert)
        # end_time = time.time()
        if len(result) == 0:
            return jsonify({
                "status": "error",
                "result": "音频解析结果为空"
            })
        else:
            # result["usage"]["seconds"] = round(end_time - start_time, 1)
            return jsonify(result)
            # return jsonify({
            #     "status": "success",
            #     "result": result
            # })

    except Exception as e:
        # 清理文件
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {str(e)}")
        if os.path.exists(filepath):
            os.remove(filepath)
        if os.path.exists(filepath_convert):
            os.remove(filepath_convert)
        
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8081))
    app.run(host='0.0.0.0', port=port, debug=False)
