import os
import sys
import json
import shutil
import logging
import argparse

import torchaudio

import whisper

sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav

def options():
    parser = argparse.ArgumentParser(description='Inference code to clone and drive text to speech')
    parser.add_argument('--reference_speaker', '-r', type=str, required=True)
    parser.add_argument('--src_language', '-s', type=str, required=True)
    parser.add_argument('--target_language', '-t', type=str, required=True)
    parser.add_argument('--work_dir', '-w', type=str, required=True)
    args = parser.parse_args()
    return args

def main():
    args = options()
    logging.info(f"CosyVoice args {args}")
    
    cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False, use_flow_cache=False)

    logging.info(f"reference_audio_path {args.reference_speaker}")

    # Transcribing the audio sample
    model = whisper.load_model("medium")
    ref_audio_transcribe = model.transcribe(args.reference_speaker)

    ref_sample_rate = 16000

    phrases = None

    # Get the text to synthesize
    with open(f"{args.work_dir}/text.json", "r") as f:
        phrases = json.load(f)

    # texts_to_sythesize = [phrase["translated"] for phrase in phrases if phrase['type'] == "phrase"]

    # logging.info(f"texts_to_sythesize {texts_to_sythesize}")

    # Delete and recreate output folder
    out_folder = f"{args.work_dir}/phrases/"
    if os.path.exists(out_folder):
        shutil.rmtree(out_folder)

    os.mkdir(out_folder)


    # Check if cross linugal or not
    if(args.src_language == args.target_language):
        prompt_speech_16k = load_wav(args.reference_speaker, ref_sample_rate)

        for idx, phrase in enumerate(phrases):
            if phrase['type'] != "phrase":
                continue
            
            out_audio_path = None
            for i, j in enumerate(cosyvoice.inference_zero_shot(phrase["translated"], ref_audio_transcribe['text'], prompt_speech_16k, stream=False)):
                out_audio_path = f"{args.work_dir}/phrases/{idx}_out.wav"
                torchaudio.save(out_audio_path, j['tts_speech'], cosyvoice.sample_rate)

            phrases[idx]["file_path"] = out_audio_path
    else:
        prompt_speech_16k = load_wav(args.reference_speaker, ref_sample_rate)

        for idx, phrase in enumerate(phrases):
            if phrase['type'] != "phrase":
                continue

            for i, j in enumerate(cosyvoice.inference_cross_lingual(tts_text=phrase["translated"],prompt_speech_16k=prompt_speech_16k, stream=False)):
                out_audio_path = f"{args.work_dir}/phrases/{idx}_out.wav"
                torchaudio.save(out_audio_path, j['tts_speech'], cosyvoice.sample_rate)

            phrases[idx]["file_path"] = out_audio_path


    logging.info(f"Phrases {phrases}")

    with open(f"{args.work_dir}/text.json", "w") as f:
        json.dump(phrases, f)

if __name__ == "__main__":
    main()
