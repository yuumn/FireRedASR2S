# Copyright 2026 Xiaohongshu. (Author: Kaituo Xu, Kai Huang, Yan Jia, Junjie Chen, Wenpeng Li)

import logging
import re
from dataclasses import dataclass, field

import soundfile as sf

from fireredasr2s.fireredasr2 import FireRedAsr2, FireRedAsr2Config
from fireredasr2s.fireredlid import FireRedLid, FireRedLidConfig
from fireredasr2s.fireredpunc import FireRedPunc, FireRedPuncConfig
from fireredasr2s.fireredvad import FireRedVad, FireRedVadConfig
from typing import List, Dict
from collections import defaultdict

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
logger = logging.getLogger("fireredasr2s.asr_system")

def is_chinese_char(ch: str) -> bool:
    cp = ord(ch)
    return (0x4E00 <= cp <= 0x9FFF        # CJK Unified Ideographs
         or 0x3400 <= cp <= 0x4DBF        # CJK Extension A
         or 0x20000 <= cp <= 0x2A6DF      # CJK Extension B
         or 0xF900 <= cp <= 0xFAFF        # CJK Compatibility
         or 0x2F800 <= cp <= 0x2FA1F)     # CJK Compatibility Supplement

def classify_text(text: str) -> str:
    cn, en = 0, 0
    for ch in text:
        if is_chinese_char(ch):
            cn += 1
        elif ch.isascii() and ch.isalpha():
            en += 1
    if cn and not en:
        return "zh"
    if en and not cn:
        return "en"
    if cn > en:
        return "zh"
    return "en"

def merge_text_func(
    merge_text: str,
    t: str,
):
    x = classify_text(merge_text[-1])
    y = classify_text(t)
    if x == "en" and y == "en" and merge_text[-1] != " " and t[0] != " ":
        return x + " " + y
    
    return x + y


def assign_speaker_to_segments(
    segments: List[Dict],
    speaker_turns: List[Dict],
) -> List[Dict]:
    """为每个分割片段分配说话人 ID。

    Parameters
    ----------
    segments : list of dict
    每个元素形如 {"start": float, "end": float, "text": str}
    speaker_turns : list of dict
    每个元素形如 {"start": float, "end": float, "speaker": str}

    Returns
    -------
    list of dict
    每个元素形如 {"start", "end", "text", "speaker"}
    """

    def overlap(a_start, a_end, b_start, b_end) -> float:
        left = max(a_start, b_start)
        right = min(a_end, b_end)
        return max(0.0, right - left)

    results = []

    for seg in segments:
        seg_start, seg_end = seg
        best_speaker = "UNKNOWN"
        best_overlap = 0.0
        speaker_time_map = defaultdict(float)

        for ((spk_start, spk_end), spk) in speaker_turns:
            if spk_start > seg_end:
                break
            ov = overlap(seg_start, seg_end, spk_start, spk_end)
            speaker_time_map[spk] += ov
            # if ov >= best_overlap:
            #     best_overlap = ov
            #     best_speaker = spk
        
        for spk_key, overlap_value in speaker_time_map.items():
            if overlap_value >= best_overlap:
                best_overlap = ov
                best_speaker = spk_key


        results.append(((seg_start, seg_end), best_speaker))
        # results.append({
        #     "start": seg_start,
        #     "end": seg_end,
        #     # "text": seg["text"],
        #     "speaker": best_speaker,
        # })

    return results


def assign_words_to_speakers(
    words: list[tuple[str, float, float]],
    segments: list[tuple[tuple[float, float], str]],
    uttid: str,
) -> list[tuple[str, list[str]]]:
    """
    将带时间戳的单词分配到对应的说话人片段中。
    匹配逻辑：计算每个单词与每个片段的时间重叠量，
    将单词分配给重叠最大的片段。
    Args:
        words:    [("word", start_time, end_time), ...]
        segments: [((start_time, end_time), spk), ...]
    Returns:
        [(spk, [word1, word2, ...]), ...] 按片段原始顺序排列
    """
    def overlap(word_start, word_end, spk_start, spk_end, ratio: float = 0.7):
        start = max(word_start, spk_start)
        end = min(word_end, spk_end)
        overlap_time = max(0.0, end - start)
        return overlap_time
        # if overlap_time / (word_end - word_start) >= ratio:
        #     return True
        # return False

    # 为每个片段初始化空单词列表，保持原始顺序
    result: dict[int, list[str]] = {i: [] for i in range(len(segments))}
    timestamp_result: dict[int, list[tuple(str, float, float)]] = {i: [] for i in range(len(segments))}
    for word, w_start, w_end in words:
        if word == "<sil>" or word == "<blank>":
            continue

        best_idx = -1
        best_overlap = 0.0
        for i, ((s_start, s_end), _spk) in enumerate(segments):
            # 计算时间重叠
            ov = overlap(w_start, w_end, s_start, s_end)
            if ov >= best_overlap:
                best_overlap = ov
                best_idx = i
        
        if best_idx >= 0:
            result[best_idx].append(word)
            timestamp_result[best_idx].append((word, w_start, w_end))
    
    asr_spk_merged = []
    for i, ((s_start, s_end), _spk) in enumerate(segments):
        if len(result[i]) == 0:
            continue

        merge_text = result[i][0]
        for t in result[i][1:]:
            merge_text = merge_text_func(merge_text, t)


        segment = {
                    "uttid": f"{uttid}_s{int(s_start * 1000)}_e{int(s_end * 1000)}",
                    # "text": " ".join(result[i]),
                    "text": merge_text,
                    "dur_s": s_end - s_start,
                    "spk": _spk,
                    "timestamp": timestamp_result[i],
                }
        
        asr_spk_merged.append(segment)
    return asr_spk_merged


@dataclass
class FireRedAsr2SystemConfig:
    vad_model_dir: str = "pretrained_models/FireRedVAD/VAD"
    lid_model_dir: str = "pretrained_models/FireRedLID"
    asr_type: str = "aed"
    asr_model_dir: str = "pretrained_models/FireRedASR2-AED"
    punc_model_dir: str = "pretrained_models/FireRedPunc"
    vad_config: FireRedVadConfig = field(default_factory=FireRedVadConfig)
    lid_config: FireRedLidConfig = field(default_factory=FireRedLidConfig)
    asr_config: FireRedAsr2Config = field(default_factory=FireRedAsr2Config)
    punc_config: FireRedPuncConfig = field(default_factory=FireRedPuncConfig)
    asr_batch_size: int = 1
    punc_batch_size: int = 1
    enable_vad: bool = True
    enable_lid: bool = True
    enable_punc: bool = True
    spk_mode: str = "pyannote"


class FireRedAsr2System:
    def __init__(self, config):
        c = config
        self.vad = FireRedVad.from_pretrained(c.vad_model_dir, c.vad_config) if c.enable_vad else None
        self.lid = FireRedLid.from_pretrained(c.lid_model_dir, c.lid_config) if c.enable_lid else None
        self.asr = FireRedAsr2.from_pretrained(c.asr_type, c.asr_model_dir, c.asr_config)
        self.punc = FireRedPunc.from_pretrained(c.punc_model_dir, c.punc_config) if c.enable_punc else None
        self.config = config

    def process(
        self, 
        wav_path, 
        uttid="tmpid",
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ):
        wav_np, sample_rate = sf.read(wav_path, dtype="int16")
        dur = wav_np.shape[0]/sample_rate
        
        # 1. VAD
        if self.config.enable_vad:
            vad_result, prob = self.vad.detect(
                                    wav_path,
                                    num_speakers=num_speakers,
                                    min_speakers=min_speakers,
                                    max_speakers=max_speakers,
                                )
            vad_segments = vad_result["timestamps"] # [(start, end), ...]
            pyannote_spk_segments = vad_result["timestamps_with_spk"] # [((start, end), spk), ...]

            logger.info(f"VAD: {vad_result}")
        else:
            vad_segments = [(0, dur)]
            vad_result = {"timestamps" : vad_segments}
        # if self.config.spk_mode != "pyannote":
        #     vad_segments = assign_speaker_to_segments(vad_segments, pyannote_spk_segments)
        # else:
        #     vad_segments = pyannote_spk_segments
        
        logger.info(f"pyannote_spk_segments: {pyannote_spk_segments}")
        logger.info(f"vad_segments_with_speaker: {vad_segments}")
        # with open(f"pyannote_spk_segments.txt", "w") as f:
            # f.write
        # 2. VAD output to ASR input
        asr_results = []
        lid_results = []
        # spk_results = []
        assert sample_rate == 16000
        batch_asr_uttid = []
        batch_asr_wav = []
        # for j, ((start_s, end_s), spk) in enumerate(vad_segments):
        for j, (start_s, end_s) in enumerate(vad_segments):
            wav_segment = wav_np[int(start_s*sample_rate):int(end_s*sample_rate)]
            vad_uttid = f"{uttid}_s{int(start_s*1000)}_e{int(end_s*1000)}"
            batch_asr_uttid.append(vad_uttid)
            batch_asr_wav.append((sample_rate, wav_segment))
            if len(batch_asr_uttid) < self.config.asr_batch_size and j != len(vad_segments) - 1:
                continue

            # 3. ASR
            batch_asr_results = self.asr.transcribe(batch_asr_uttid, batch_asr_wav)
            logger.info(f"ASR: {batch_asr_results}")

            if self.config.enable_lid:
                batch_lid_results = self.lid.process(batch_asr_uttid, batch_asr_wav)
                logger.info(f"LID: {batch_lid_results}")
            else:
                # Note: The original batch size is used here to ensure alignment with the initial number of ASR results
                batch_lid_results = [None] * len(batch_asr_results)

            # Synchronously traverse and filter to ensure that asr_results and lid_results always maintain a one-to-one correspondence
            for a_res, l_res in zip(batch_asr_results, batch_lid_results):
                text = a_res.get("text", "").strip()
                # Filter out <blank>, <sil> and completely empty strings ""
                if not text or re.search(r"(<blank>)|(<sil>)", text):
                    continue
                asr_results.append(a_res)
                lid_results.append(l_res)
                # spk_results.append(spk)

            batch_asr_uttid = []
            batch_asr_wav = []

        # ------------------------------------------------------------------------------------------
        # 3.5 按词分配说话人
        

        spk_results = []
        if self.config.asr_config.return_timestamp:
            words_results = []
            for asr_result in asr_results:
                if "timestamp" in asr_result:
                    start_ms, end_ms = asr_result["uttid"].split("_")[-2:]
                    assert start_ms.startswith("s") and end_ms.startswith("e")
                    start_ms, end_ms = int(start_ms[1:]), int(end_ms[1:])
                    
                    words_results += [(word, start_s + start_ms * 1.0 / 1000, end_s + start_ms * 1.0 / 1000) for word, start_s, end_s in asr_result["timestamp"]]
            
            logger.info(f"words_results: {words_results}")

            asr_results = assign_words_to_speakers(words_results, pyannote_spk_segments, uttid)
            lid_results = [None] * len(asr_results)
            spk_results = [asr_result["spk"] for asr_result in asr_results]
            logger.info(f"assign_words_to_speakers asr_results: {asr_results}")
        # ------------------------------------------------------------------------------------------


        # 4. ASR output to Postprocess input
        if self.config.enable_punc:
            punc_results = []
            batch_asr_text = []
            batch_asr_uttid = []
            batch_asr_timestamp = []
            for j, asr_result in enumerate(asr_results):
                batch_asr_text.append(asr_result["text"])
                batch_asr_uttid.append(asr_result["uttid"])
                if self.config.asr_config.return_timestamp:
                    batch_asr_timestamp.append(asr_result.get("timestamp", []))
                elif "timestamp" in asr_result:
                    batch_asr_timestamp.append(asr_result["timestamp"])
                if len(batch_asr_text) < self.config.punc_batch_size and j != len(asr_results) - 1:
                    continue

                # 5. Punc
                logger.info(f"batch_asr_text: {batch_asr_text}")
                if self.config.asr_config.return_timestamp:
                    batch_punc_results = self.punc.process_with_timestamp(batch_asr_timestamp, batch_asr_uttid)
                else:
                    batch_punc_results = self.punc.process(batch_asr_text, batch_asr_uttid)
                # logger.info(f"batch_asr_text: {batch_asr_text}")
                logger.info(f"Punc: {batch_punc_results}")

                punc_results.extend(batch_punc_results)
                batch_asr_text = []
                batch_asr_uttid = []
                batch_asr_timestamp = []
        else:
            punc_results = asr_results

        # 6. Put all together & Format
        sentences = []
        words = []
        for asr_result, punc_result, lid_result, spk_result in zip(asr_results, punc_results, lid_results, spk_results):
            assert asr_result["uttid"] == punc_result["uttid"], f"fix code: {asr_result} | {punc_result}"
            start_ms, end_ms = asr_result["uttid"].split("_")[-2:]
            assert start_ms.startswith("s") and end_ms.startswith("e")
            start_ms, end_ms = int(start_ms[1:]), int(end_ms[1:])
            if self.config.asr_config.return_timestamp:
                sub_sentences = []
                if self.config.enable_punc:
                    for i, punc_sent in enumerate(punc_result["punc_sentences"]):
                        start = start_ms + int(punc_sent["start_s"]*1000)
                        end = start_ms + int(punc_sent["end_s"]*1000)
                        if i == 0:
                            start = start_ms
                        if i == len(punc_result["punc_sentences"]) - 1:
                            end = end_ms
                        sub_sentence = {
                            "start_ms": start,
                            "end_ms": end,
                            "spk": spk_result,
                            "text": punc_sent["punc_text"],
                            # "asr_confidence": asr_result["confidence"],
                            "lang": None,
                            "lang_confidence": 0
                        }
                        if lid_result:
                            sub_sentence["lang"] = lid_result["lang"]
                            sub_sentence["lang_confidence"] = lid_result["confidence"]
                        sub_sentences.append(sub_sentence)
                else:
                    sub_sentences = [{
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                        "spk": spk_result,
                        "text": asr_result["text"],
                        # "asr_confidence": asr_result["confidence"],
                        "lang": None,
                        "lang_confidence": 0
                    }]
                sentences.extend(sub_sentences)
            else:
                text = punc_result["punc_text"] if self.config.enable_punc else asr_result["text"]
                sentence = {
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "text": text,
                    "spk": spk_result,
                    # "asr_confidence": asr_result["confidence"],
                    "lang": None,
                    "lang_confidence": 0
                }
                if lid_result:
                    sentence["lang"] = lid_result["lang"]
                    sentence["lang_confidence"] = lid_result["confidence"]
                sentences.append(sentence)
            
            # if "timestamp" in asr_result:
            #     for w, s, e in asr_result["timestamp"]:
            #         word = {"start_ms": int(s*1000+start_ms), "end_ms":int(e*1000+start_ms), "text": w}
            #         words.append(word)
        # vad_segments_ms = [(int(s*1000), int(e*1000)) for ((s, e), spk) in vad_segments]
        # vad_segments_ms = [(int(s*1000), int(e*1000)) for (s, e) in vad_result["timestamps"]]
        text = "".join(s["text"] for s in sentences)
        # Add space after English punctuation when followed by a letter
        text = re.sub(r'([.,!?])\s*([a-zA-Z])', r'\1 \2', text)

        openai_format_segments = [
            {
                "type": "transcript.text.segment",
                "id": f"seg_{i+1:03d}",
                "start": round(chunk["start_ms"] * 1.0 / 1000.0, 1),
                "end": round(chunk["end_ms"] * 1.0 / 1000.0, 1),
                "text": chunk["text"],
                "speaker": chunk["spk"],
            } for i, chunk in enumerate(sentences)
        ]
        # result = {
        #     "uttid": uttid,
        #     "text": text,
        #     "sentences": sentences,
        #     # "vad_segments_ms": vad_segments_ms,
        #     "dur_s": dur,
        #     # "words": words,
        #     # "wav_path": wav_path
        # }
        logger.info(f"openai_format_segments: {openai_format_segments}")
        return {
            "task": "transcribe",
            "duration": round(dur, 1),
            "text": text,
            "segments": openai_format_segments,
            "usage": {
                "type": "duration",
                "seconds": 0.0
            }
        }
