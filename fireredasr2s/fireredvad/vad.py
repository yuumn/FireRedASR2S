# Copyright 2026 Xiaohongshu. (Author: Kaituo Xu, Kai Huang)

import logging
import os
from dataclasses import dataclass

import torch
import soundfile as sf

from .core.audio_feat import AudioFeat
from .core.detect_model import DetectModel
from .core.vad_postprocessor import VadPostprocessor
from .pyannoteclass import PyannoteModel

logger = logging.getLogger(__name__)


@dataclass
class FireRedVadConfig:
    use_gpu: bool = True
    smooth_window_size: int = 5
    speech_threshold: float = 0.4
    min_speech_frame: int = 20
    max_speech_frame: int = 2000  # 20s
    min_silence_frame: int = 20
    merge_silence_frame: int = 0
    extend_speech_frame: int = 0
    chunk_max_frame: int = 30000  # 300s
    spk_model_dir: int = "/workspace/models/FireRedASR2S/pyannote/speaker-diarization-community-1"
    def __post_init__(self):
        if self.speech_threshold < 0 or self.speech_threshold > 1:
            raise ValueError("speech_threshold must be in [0, 1]")
        if self.min_speech_frame <= 0:
            raise ValueError("min_speech_frame must be positive")




class FireRedVad:
    @classmethod
    def from_pretrained(cls, model_dir, config=FireRedVadConfig()):
        # Build Feat Extractor
        cmvn_path = os.path.join(model_dir, "cmvn.ark")
        audio_feat = AudioFeat(cmvn_path)

        # Build Model
        vad_model = DetectModel.from_pretrained(model_dir)
        spk_model = PyannoteModel(
            model_path=config.spk_model_dir,
            use_gpu=config.use_gpu,
        )
        if config.use_gpu:
            vad_model.cuda()
        else:
            vad_model.cpu()

        # Build Postprocessor
        vad_postprocessor = VadPostprocessor(
            config.smooth_window_size,
            config.speech_threshold,
            config.min_speech_frame,
            config.max_speech_frame,
            config.min_silence_frame,
            config.merge_silence_frame,
            config.extend_speech_frame)
        return cls(audio_feat, vad_model, spk_model, vad_postprocessor, config)

    def __init__(self, audio_feat, vad_model, spk_model, vad_postprocessor, config):
        self.audio_feat = audio_feat
        self.vad_model = vad_model
        self.spk_model = spk_model
        self.vad_postprocessor = vad_postprocessor
        self.config = config

    def detect(
        self, 
        audio, 
        do_postprocess=True,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ):
        # # Extract feat
        # feats, dur = self.audio_feat.extract(audio)
        # if self.config.use_gpu:
        #     feats = feats.cuda()

        # # Model inference
        # if feats.size(0) <= self.config.chunk_max_frame:
        #     probs, _ = self.vad_model.forward(feats.unsqueeze(0))
        #     probs = probs.cpu().squeeze()  # (T,)
        # else:
        #     logger.warning(f"Too long input, split every {self.config.chunk_max_frame} frames")
        #     chunk_probs = []
        #     chunks = feats.split(self.config.chunk_max_frame, dim=0)
        #     for chunk in chunks:
        #         chunk_prob, _ = self.vad_model.forward(chunk.unsqueeze(0))
        #         chunk_probs.append(chunk_prob.cpu())
        #     probs = torch.cat(chunk_probs, dim=1)
        #     probs = probs.squeeze()  # (T,)

        # if not do_postprocess:
        #     return None, probs

        # # Prob Postprocess
        # decisions = self.vad_postprocessor.process(probs.tolist())
        # starts_ends_s = self.vad_postprocessor.decision_to_segment(decisions, dur)
        # # print(f"starts_ends_s: {starts_ends_s}")

        spk_result = self.spk_model(
            audio,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        
        starts_ends_s_with_spk = [((chunk["start"], chunk["end"]), chunk["speaker"]) for chunk in spk_result["exclusive_diarization"]]
        
        
        # def merge_result(result, gap_threshold_ms=500):
        #     if len(result) <= 1:
        #         return result
            
        #     result_merge = []
        #     result_length = len(result)
        #     cur_chunk = result[0]
        #     for idx in range(1, result_length):
        #         chunk = result[idx]
        #         if chunk["speaker"] == cur_chunk["speaker"] and chunk["start"] - cur_chunk["end"] <= gap_threshold_ms:
        #             cur_chunk["end"] = chunk["end"]
        #             if idx == result_length - 1 or cur_chunk["end"] - cur_chunk["start"] >= 30000:
        #                 result_merge.append(cur_chunk)
        #             continue
        #         result_merge.append(cur_chunk)
        #         cur_chunk = chunk
        #         if idx == result_length - 1:
        #             result_merge.append(cur_chunk)
        #     return result_merge
        
        # starts_ends_s_with_spk = merge_result(
        #     spk_result["exclusive_diarization"], 
        #     # spk_result["diarization"], 
        #     500
        # )

        # starts_ends_s_with_spk = [((chunk["start"], chunk["end"]), chunk["speaker"]) for chunk in starts_ends_s_with_spk]


        """
        starts_ends_s: [(0.75, 9.74), (9.75, 11.21), (11.93, 14.77), (15.39, 16.51), (18.32, 22.55), (22.86, 24.03), (25.24, 33.95), (33.96, 41.32), (41.33, 48.76), (48.77, 53.52), (54.9, 56.28), (58.39, 59.13), (60.01, 61.43), (62.55, 65.19), (66.49, 75.6), (75.61, 82.59), (83.59, 89.9), (89.91, 99.52), (99.53, 108.31), (108.32, 117.25), (117.26, 122.86), (122.87, 132.86), (132.87, 138.66), (138.67, 146.77), (147.1, 156.86), (157.53, 165.23), (165.24, 167.63), (168.13, 175.17), (175.18, 180.69), (180.7, 185.91), (191.23, 197.45), (197.46, 201.54), (203.18, 210.53), (210.54, 220.26), (220.27, 222.1), (222.42, 224.69), (225.48, 228.01), (228.63, 235.66), (236.94, 243.3), (245.18, 248.02), (249.01, 249.91), (250.24, 255.76), (257.69, 259.24), (259.66, 265.79), (265.8, 274.42), (274.79, 284.66), (286.43, 290.51), (291.7, 298.64), (300.29, 304.896)]
        """
        # Format result
        # result = {"dur": round(dur, 3),
        #           "timestamps": starts_ends_s}
        result = {
            # "dur": round(dur, 3),
            "timestamps": starts_ends_s_with_spk
        }
        if isinstance(audio, str):
            result["wav_path"] = audio
        return result, None
        # return result, probs
