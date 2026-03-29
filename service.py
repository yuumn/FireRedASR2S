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



app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # 兼容旧版 Flask (< 2.2)
app.json.ensure_ascii = False        # 兼容新版 Flask (>= 2.2)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
# logging.basicConfig(level=logging.WARNING,
#                     format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
logger = logging.getLogger("fireredasr2s.asr_system")

args = OmegaConf.structured(FireRedASR2Config)

# logging.basicConfig(level=logging.WARNING)

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
    args.asr_type, args.asr_model_dir, args.punc_model_dir,
    vad_config, lid_config, asr_config, punc_config,
    args.asr_batch_size, args.punc_batch_size,
    args.enable_vad, args.enable_lid, args.enable_punc
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

# 会议撰写
# @app.route('/AsrCamWithIdentify', methods=['POST'])
@app.route('/v1/chat/completions', methods=['POST'])
def speech_recognition_Timestamp_cam_identify_speakers():
    # 检查文件上传
    if 'file' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    # 保存上传文件
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    filepath_convert = os.path.join(app.config['UPLOAD_CONVERT_FOLDER'], filename.rsplit(".", 1)[0] + ".wav")
    file.save(filepath)

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
        result = asr_system.process(
            filepath_convert, 
            uttid,
            num_speakers,
            min_speakers,
            max_speakers,
        )


        # 处理结果
        # processed_result = process_cam_result_with_identify_speakers(result,speaker_db,filepath,identify_speakers)

        os.remove(filepath)
        os.remove(filepath_convert)
        if len(result) == 0:
            return jsonify({
                "status": "error",
                "result": "音频解析结果为空"
            })
        else:
            return jsonify({
                "status": "success",
                "result": result
            })

    except Exception as e:
        # 清理文件
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {str(e)}")
        if os.path.exists(filepath):
            os.remove(filepath)
        if os.path.exists(filepath_convert):
            os.remove(filepath_convert)
        
        return jsonify({"error": str(e)}), 500


def _extract_audio_segment(audio_path, start_sec, end_sec):
    """根据时间戳提取音频片段"""
    waveform, sample_rate = torchaudio.load(audio_path)

        # 计算起止采样点
    start_sample = int(start_sec * sample_rate)
    end_sample = int(end_sec * sample_rate)

        # 提取片段
    segment = waveform[:, start_sample:end_sample]

    # 保存为临时文件（pipeline需要文件路径）
    temp_path = f"/tmp/temp_segment_{start_sec}_{end_sec}.wav"
    torchaudio.save(temp_path, segment, sample_rate)

    return temp_path
# 是否使用声纹转化的结果处理

def process_cam_result_with_identify_speakers(result,speaker_db,filepath,identify_speakers=False,threshold=0.45):
    """处理ASR结果，返回包含时间和内容的JSON对象列表"""
    if not isinstance(result, list) or len(result) == 0:
        return []

    data = result[0]
    best_match = "unknown"
    best_score = 0.0
    # 创建JSON格式的输出
    output = []
    sentence_infos = data.get('sentence_info',[])

    current_sentence = {
        "spk": sentence_infos[0]["spk"],
        "spk_name": best_match,
        "confidence":best_score,
        "start": sentence_infos[0]["start"],
        "end": sentence_infos[0]["end"],
        "text": sentence_infos[0]["text"]
    }


    for i in range(1, len(sentence_infos)):
        sentence_info = sentence_infos[i]
        # 检查合并条件：相同说话人且时间间隔≤1000ms
        if (current_sentence["spk"] == sentence_info["spk"] and
                sentence_info["start"] - current_sentence["end"] <= 1000):

            # 合并文本内容（中文无需加空格）
            current_sentence["text"] += sentence_info["text"]

            # 更新整句结束时间
            current_sentence["end"] = sentence_info["end"]

        else:
            # 保存合并完成的句子
            if identify_speakers: # 提取说话人
                segment_audio = _extract_audio_segment( #提取对应的音频
                    filepath, current_sentence['start']/1000, current_sentence['end']/1000
                )
                result_b = sv_pipeline([segment_audio], output_emb=True)['embs'][0] #获取音频向量
                os.remove(segment_audio)
                # 遍历声纹库

                for name, db_emb in speaker_db.items():
                    # 计算余弦相似度
                    data_list = json.loads(db_emb)
                    arr = np.array(data_list, dtype=np.float32)
                    similarity = 1 - cosine(result_b, arr)
                    similarity = float(similarity)
                    if similarity > best_score and similarity > threshold:
                        best_score = similarity
                        best_match = name
            output.append({
                "spk": current_sentence["spk"],
                "spk_name": best_match,
                "confidence":best_score,
                "text": current_sentence["text"],
                "start": current_sentence["start"],
                "end": current_sentence["end"]
            })
            # 重新开始新句子
            current_sentence = sentence_info.copy()
            best_match = "unknown"
            best_score = 0.0

    # 保存合并完成的句子
    if identify_speakers:  # 提取说话人
        segment_audio = _extract_audio_segment(  # 提取对应的音频
                    filepath, current_sentence['start']/1000, current_sentence['end']/1000
        )
        result_b = sv_pipeline([segment_audio], output_emb=True)['embs'][0]  # 获取音频向量
        os.remove(segment_audio)
        # 遍历声纹库

        for name, db_emb in speaker_db.items():
            # 计算余弦相似度
            data_list = json.loads(db_emb)
            arr = np.array(data_list, dtype=np.float32)
            similarity = 1 - cosine(result_b, arr)
            similarity = float(similarity)
            if similarity > best_score and similarity > threshold:
                best_score = similarity
                best_match = name
    output.append({
        "spk": current_sentence["spk"],
        "spk_name": best_match,
        "confidence":best_score,
        "text": current_sentence["text"],
        "start": current_sentence["start"],
        "end": current_sentence["end"]
    })

    return output  # 返回JSON对象列表



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8881, debug=False)

