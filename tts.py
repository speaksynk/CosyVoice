import os
import sys
import json
import shutil
import logging
import argparse

import torchaudio

import whisper

import pydub
from pydub import AudioSegment

sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav

import whisper_timestamped as whisper_ts

ENGLISH_PREPEND = "test hey,"

PERCENT_CUT_BACK = .2

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

    phrase_generation_info = None

    # Get the text to synthesize
    with open(f"{args.work_dir}/text.json", "r") as f:
        phrase_generation_info = json.load(f)


    phrases = []
    joined_text = ""

    for i, phrase in enumerate(phrase_generation_info):
        if phrase['type'] == 'phrase':
            phrase['og_idx'] = i
            phrases.append(phrase)
            joined_text += f"{phrase['translated']} "

    print(f"phrases len {len(phrases)}")
    print(f"joined_text {joined_text}")

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

        if args.target_language == "English":
            phrase_text = f"{ENGLISH_PREPEND} {joined_text}"

            print(args.target_language)
                
            out_audio_path_temp = None
            for i, j in enumerate(cosyvoice.inference_zero_shot(phrase_text, ref_audio_transcribe['text'], prompt_speech_16k, stream=False)):
                out_audio_path_temp = f"{args.work_dir}/phrases/out.temp.wav"
                torchaudio.save(out_audio_path_temp, j['tts_speech'], cosyvoice.sample_rate)


            audio = whisper_ts.load_audio(out_audio_path_temp)

            model = whisper_ts.load_model("medium", device="cuda")

            result = whisper_ts.transcribe(model, audio, language='en')

            # print(json.dumps(result, indent = 2, ensure_ascii = False))

            first_real_word = joined_text.split()[0].lower()

            print(f"first real word {first_real_word}")

            all_word_info = [word for segment in result["segments"] for word in segment["words"]]
            
            print(json.dumps(all_word_info, indent = 2, ensure_ascii = False))
            
            song = AudioSegment.from_wav(out_audio_path_temp)

            foundStart = False

            gen_phrase_count = 0
            cut_start_time = None

            for i, word in enumerate(all_word_info):

                current_word = word['text'].lower()
                next_phrase_first = phrases[0]['translated'].split()[0].lower()

                print(f"current_word {current_word} next_phrase_first {next_phrase_first}")

                # print(f"word['text'] {word['text'].lower()}; all_word_info[i + 1]['text'] {all_word_info[i + 1]['text'].lower()}")
                if not foundStart and i > 0 and current_word == first_real_word:
                    print(f"FOUND Start")

                    time_two = all_word_info[i - 1]['end']
                    time_one = word['start']

                    diff = time_two - time_one

                    cut_time = time_one - 0.06

                    print(f"cut_time {cut_time}")

                    cut_start_time = cut_time

                    phrases.pop(0)

                    foundStart = True

                    # print(f"time_one {time_one} time_two {time_two} cut_time {cut_time}")
                    
                elif i > 0 and len(phrases) > 0 and current_word == next_phrase_first:
                    print(f"FOUND End")

                    time_one = word['start']
                    cut_end_time = time_one - 0.06

                    print(f"cut_start_time {cut_start_time} cut_end_time {cut_end_time}")

                    out_audio_path = f"{args.work_dir}/phrases/{gen_phrase_count}.wav"

                    print(f"out_audio_path {out_audio_path}")

                    audio_without_prepend = song[cut_start_time * 1000 : cut_end_time * 1000]
                    audio_without_prepend.export(out_audio_path, format="wav")

                    phrase_generation_info[phrases[0]['og_idx']]["file_path"] = out_audio_path
                    # found = True

                    cut_start_time = cut_end_time

                    gen_phrase_count = gen_phrase_count + 1

                    p = phrases.pop(0)

                    if len(phrases) == 0:
                        out_audio_path = f"{args.work_dir}/phrases/{gen_phrase_count}.wav"

                        audio_without_prepend = song[cut_end_time * 1000 : ]
                        audio_without_prepend.export(out_audio_path, format="wav")

                        phrase_generation_info[p['og_idx']]["file_path"] = out_audio_path
                        break
                        
        else:

            for idx, phrase in enumerate(phrase_generation_info):
                if phrase['type'] != "phrase":
                    continue

                phrase_text = None

                print(args.target_language)
                
                out_audio_path_temp = None
                for i, j in enumerate(cosyvoice.inference_zero_shot(phrase_text, ref_audio_transcribe['text'], prompt_speech_16k, stream=False)):
                    out_audio_path_temp = f"{args.work_dir}/phrases/{idx}_out.temp.wav"
                    torchaudio.save(out_audio_path_temp, j['tts_speech'], cosyvoice.sample_rate)

                # phrases[idx]["file_path"] = out_audio_path

                # if args.target_language == "English":
                #     audio = whisper_ts.load_audio(out_audio_path_temp)

                #     model = whisper_ts.load_model("medium", device="cuda")

                #     result = whisper_ts.transcribe(model, audio, language='en')

                #     # print(json.dumps(result, indent = 2, ensure_ascii = False))



                #     first_real_word = phrase["translated"].split()[0].lower()

                #     print(f"first real word {first_real_word}")

                #     all_word_info = [word for segment in result["segments"] for word in segment["words"]]
                #     print(json.dumps(all_word_info, indent = 2, ensure_ascii = False))
                    
                #     for i, word in enumerate(all_word_info):
                #         # print(f"word['text'] {word['text'].lower()}; all_word_info[i + 1]['text'] {all_word_info[i + 1]['text'].lower()}")

                #         if i > 0 and word['text'].lower() == first_real_word:
                #             print(f"FOUND FOUND FOUND FOUND FOUND")

                #             time_two = all_word_info[i - 1]['end']
                #             time_one = word['start']

                #             diff = time_two - time_one

                #             cut_time = time_one - 0.06

                #             # print(f"time_one {time_one} time_two {time_two} cut_time {cut_time}")

                #             song = AudioSegment.from_wav(out_audio_path_temp)

                #             out_audio_path = f"{args.work_dir}/phrases/{idx}.wav"

                #             audio_without_prepend = song[cut_time * 1000:]
                #             audio_without_prepend.export(out_audio_path, format="wav")

                #             phrases[idx]["file_path"] = out_audio_path
                #             found = True
                #             break
              

    else:
        prompt_speech_16k = load_wav(args.reference_speaker, ref_sample_rate)

        for idx, phrase in enumerate(phrase_generation_info):
            if phrase['type'] != "phrase":
                continue

            for i, j in enumerate(cosyvoice.inference_cross_lingual(tts_text=phrase["translated"],prompt_speech_16k=prompt_speech_16k, stream=False)):
                out_audio_path = f"{args.work_dir}/phrases/{idx}_out.wav"
                torchaudio.save(out_audio_path, j['tts_speech'], cosyvoice.sample_rate)

            phrase_generation_info[idx]["file_path"] = out_audio_path

            audio = whisper_ts.load_audio(out_audio_path)

            model = whisper_ts.load_model("tiny", device="cpu")

            result = whisper_ts.transcribe(model, audio, language=args.target_language.lower())

            print(json.dumps(result, indent = 2, ensure_ascii = False))

            assert False


    print(f"phrase_generation_info {phrase_generation_info}")


    with open(f"{args.work_dir}/text.json", "w") as f:
        json.dump(phrase_generation_info, f)

if __name__ == "__main__":
    main()
