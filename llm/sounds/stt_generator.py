#!/usr/bin/env python

#SPEECH TO TEXT GENERATOR

import modules.settings as settings
from openai import OpenAI
import numpy as np
import wave
import math
import contextlib
import time
import azure.cognitiveservices.speech as speechsdk
import sys
import datetime
import glob
import os
import pveagle
import modules.audio_visual as audio_visual
import modules.commons as commons

class SpeechToTextGenerator:
    def __init__(self):
        self.openai_client = OpenAI(api_key=settings.OPENAI_API_TOKEN)
        self.azure_config = speechsdk.SpeechConfig(subscription=settings.AZURE_SPEECH_API_KEY, region=settings.AZURE_SPEECH_REGION)
        self.azure_stream_speech  = None
        
        #Deepgram
        self.deepgram_client = None
        if  sys.version_info >= (3,10,0):
            from deepgram import DeepgramClient,DeepgramClientOptions,PrerecordedOptions,LiveTranscriptionEvents,LiveOptions,Microphone
            deepgram_client_options = DeepgramClientOptions(
                options={"keepalive": "true"}
            )
            self.deepgram_client = DeepgramClient(settings.DEEPGRAM_API_KEY,deepgram_client_options)
            self.deepgram_microphone = None
            self.deepgram_connection = None
            self.deepgram_utterance_end = False

        #current transcription
        self.speech_transcription = ''

        self.speaker_recogniser, self.speaker_labels = self.setup_speaker_recognition()

        print('SpeechToTextGenerator init')


    ############ COMMON FUNCTIONS ##################
    def transcribe_audio(self, audio_file):
        curr_time = datetime.datetime.now()
        sentiment = ''
        transcription = ''
        if settings.CONFIG['transcription_type'] == 'deepgram' and self.deepgram_client:
            transcription,sentiment = self.transcribe_deepgram(audio_file) 
        elif settings.CONFIG['transcription_type'] == 'whisper':
            transcription = self.transcribe_whisper(audio_file)
        elif settings.CONFIG['transcription_type'] == 'azure':
            transcription = self.transcribe_azure(audio_file)
        else:
            print('transcription_type not set')
            return '',''
        print("transcription_time: " + str(int((datetime.datetime.now() - curr_time).total_seconds() * 1000)) + 'ms')

        return transcription,sentiment    

    def setup_stream_transcribe(self):

        if settings.CONFIG['stream_transcription_type'] == 'deepgram' and self.deepgram_client:
            self.setup_stream_transcribe_deepgram()
        elif settings.CONFIG['stream_transcription_type'] == 'azure':    
            self.azure_stream_speech = speechsdk.SpeechRecognizer(speech_config=self.azure_config,language='en-us')
        else:
            print('Error: stream_transcription_type not set: ' +  settings.CONFIG['stream_transcription_type'])
            return False

        return True
    
    def get_stream_transcription(self,multi_lingual=settings.CONFIG['transcription_mulitlingual']):
        #start recording transcription
        audio_visual.start_audio_record(settings.DEFAULT_AUDIO_SAMPLE_PATH,16000,512)

        transcription = ''
        if settings.CONFIG['stream_transcription_type'] == 'deepgram' and self.deepgram_client:
            transcription = self.get_stream_transcribe_deepgram()
        elif settings.CONFIG['stream_transcription_type'] == 'azure':    
            transcription = self.get_stream_transcribe_azure()
        else:
            print('Error: stream_transcription_type not set: ' +  settings.CONFIG['stream_transcription_type'])
            return ''
    
        audio_visual.stop_audio_record()

        if multi_lingual:
            language = self.language_detection(settings.DEFAULT_AUDIO_SAMPLE_PATH)
            transcription = self.transcribe_whisper(settings.DEFAULT_AUDIO_SAMPLE_PATH,language)
    
        print('stream transcription result: ' + transcription)    
        return transcription
    

    ############ WHISPER ##################
    def transcribe_whisper(self, audio_file,language='en'):
        print('transcribe_whisper')

        audio= open(audio_file, "rb")
        transcription = self.openai_client.audio.transcriptions.create(
                model="whisper-1", 
                language=language,
                file=audio
        )

        print(transcription)
        #apply whisper filters
        transcription = self.whisper_output_filter(transcription.text)
        
        return transcription
    
    def whisper_output_filter(self, transcription):
        transcription = transcription.lower()

        #remove hallucinations
        # transcription = transcription.replace("thank you","")#remove any thank you txt
        # transcription = transcription.replace("bye","")#remove any thank you txt

        #remove non english words
        # words = set(nltk.corpus.words.words())
        # transcription = " ".join(w for w in nltk.wordpunct_tokenize(transcription) \
        #      if w.lower() in words or not w.isalpha())

        return transcription
    ###############################################


    ############ DEEPGRAM ##########################
    def setup_stream_transcribe_deepgram(self):
        
        self.deepgram_connection = self.deepgram_client.listen.live.v("1")

        def on_open(open, **kwargs):
            print(f"\n\n{open}\n\n")

        def on_message(s,result, **kwargs):
            sentence = result.channel.alternatives[0].transcript
            if len(sentence) == 0:
                return
            self.speech_transcription  = sentence
            # print(f"speaker: {sentence}")

        def on_metadata(s, metadata, **kwargs):
            print(f"\n\n{metadata}\n\n")

        def on_speech_started(s,speech_started, **kwargs):
            print(f"\n\n{speech_started}\n\n")

        def on_utterance_end(s,utterance_end, **kwargs):
            # print(f"\n\n{utterance_end}\n\n")
            self.deepgram_utterance_end = True

        def on_error(s,error, **kwargs):
            print(f"\n\n{error}\n\n")

        def on_close(s,close, **kwargs):
            print(f"\n\n{close}\n\n")

        # self.deepgram_connection.on(LiveTranscriptionEvents.Open, on_open)
        self.deepgram_connection.on(LiveTranscriptionEvents.Transcript, on_message)
        self.deepgram_connection.on(LiveTranscriptionEvents.Metadata, on_metadata)
        # self.deepgram_connection.on(LiveTranscriptionEvents.SpeechStarted, on_speech_started)
        self.deepgram_connection.on(LiveTranscriptionEvents.UtteranceEnd, on_utterance_end)
        self.deepgram_connection.on(LiveTranscriptionEvents.Error, on_error)
        # self.deepgram_connection.on(LiveTranscriptionEvents.Close, on_close)

        options = LiveOptions(
            model="nova-2", 
            language="en-US", 
            smart_format=True,
            encoding="linear16",
            channels=1,
            sample_rate=16000,
            interim_results=True,
            utterance_end_ms="1000",
            vad_events=True,
        )
        self.deepgram_connection.start(options)

        # create microphone
        self.deepgram_microphone = Microphone(self.deepgram_connection.send)

    def get_stream_transcribe_deepgram(self):
        print('getting deepgram transcription')
        self.speech_transcription = ''
        self.deepgram_utterance_end = False
        self.deepgram_microphone.start()
        current_time = datetime.datetime.now()
        utterance_timeout = 15
        while not self.deepgram_utterance_end and int((datetime.datetime.now() - current_time).total_seconds()) < utterance_timeout:
            time.sleep(0.1)

        self.deepgram_microphone.finish()
        if int((datetime.datetime.now() - current_time).total_seconds()) > utterance_timeout:
            print('utterance timeout')
            return ''

        return self.speech_transcription
    
    def transcribe_deepgram(self, audio_file):
        print('transcribe_deepgram')

        transcription = ''
        sentiment = ''
        curr_time = datetime.datetime.now()
        try:
           with open(audio_file, 'rb') as audio:
                buffer_data = audio.read()
                payload = { "buffer": buffer_data,}

                options: PrerecordedOptions = PrerecordedOptions(
                    model="nova",
                    smart_format=True,
                    # summarize="v2",
                    sentiment=True,
                )
                response = self.deepgram_client.listen.prerecorded.v("1").transcribe_file(
                    payload, options
                )

                if response and response.results:
                    if len(response.results.channels)>0:
                        if len(response.results.channels[0].alternatives)>0:
                            transcription = response.results.channels[0].alternatives[0].transcript
                    
                    print(response.results.sentiments)
                    if response.results.sentiments:
                        sentiment_average = response.results.sentiments.average
                        sentiment = sentiment_average.sentiment

                print('transcribe_deepgram: ' + transcription)
        except Exception as e:
            print(f"Exception: {e}")

        print("deepgram transcription_time: " + str(int((datetime.datetime.now() - curr_time).total_seconds() * 1000)) + 'ms')

        return transcription,sentiment
    
    ###############################################

    ############ AZURE ##########################
    def transcribe_azure(self, audio_file):
        print('transcribe_azure')
        audio_config = speechsdk.audio.AudioConfig(filename=audio_file)
        speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=self.azure_config, audio_config=audio_config)
        result = speech_recognizer.recognize_once()
        transcription = self.process_azure_trancription_result(result)
        print(transcription)
        return transcription
    
    def get_stream_transcribe_azure(self):
        print('getting azure transcription')


        self.speech_transcription  = ''
        if self.azure_stream_speech:
            result = self.azure_stream_speech.recognize_once_async().get()
            print(result)
            if result:
                if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                    print("Recognized: {}".format(result.text))
                    self.speech_transcription  = result.text
                elif result.reason == speechsdk.ResultReason.NoMatch:
                    print("No speech could be recognized")
                elif result.reason == speechsdk.ResultReason.Canceled:
                    cancellation_details = result.cancellation_details
                    print("Speech Recognition canceled: {}".format(cancellation_details.reason))
                    if cancellation_details.reason == speechsdk.CancellationReason.Error:
                        print("Error details: {}".format(cancellation_details.error_details))


        return self.speech_transcription       
 

    def process_azure_trancription_result(self,result):
        transcription = ''
               # Check the result
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            print("Recognized: {}".format(result.text))
            transcription = result.text
        elif result.reason == speechsdk.ResultReason.NoMatch:
            print("No speech could be recognized")
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            print("Speech Recognition canceled: {}".format(cancellation_details.reason))
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                print("Error details: {}".format(cancellation_details.error_details))
        return transcription

    ###############################################################


    ###############PICO VOICE SPEAKER RECOGNITION################
    def setup_speaker_recognition(self):
        speaker_profiles = []
        speaker_labels = []
        
        files = glob.glob(settings.HUMANINFO_FOLDER+'*.profile')
        for input_profile_path in files:
            print("speakers to recognise: "+input_profile_path)
            speaker_labels.append(os.path.splitext(os.path.basename(input_profile_path))[0])
            with open(input_profile_path, 'rb') as f:
                speaker_profiles.append(pveagle.EagleProfile.from_bytes(f.read()))

        speaker_recogniser = None
        if speaker_profiles:
            try:
                speaker_recogniser = pveagle.create_recognizer(
                    access_key=settings.PICO_API_KEY,
                    speaker_profiles=speaker_profiles)
            except pveagle.EagleActivationLimitError:
                print('AccessKey has reached its processing limit.')
            except pveagle.EagleError as e:
                print("Failed to initialize Eagle: ", e)
                raise
                
        return speaker_recogniser,speaker_labels
    
    #If speaker recongition is enabled
    def speaker_detection(self,audio_file=settings.DEFAULT_AUDIO_SAMPLE_PATH):
        if not self.speaker_recogniser:
            print('speaker recognition disabled')
            return ''
        
        print('using speaker detecter')
        detected_voice = self.detect_speaker_name(audio_file,self.speaker_recogniser.frame_length)
        if not detected_voice:
            print('failed to detect speaker')
            return ''
            
        return detected_voice
    
    def detect_speaker_name(self, audio_file, frame_length):
        if not self.speaker_recogniser:
            print('speaker recognition disabled')
            return ''
        
        audio_file = audio_visual.combine_two_audio_files(audio_file, audio_file)
        audio_frames = audio_visual.audio_file_to_frames(audio_file, self.speaker_recogniser.sample_rate)
        # num_frames = len(audio_frames)
        num_frames = len(audio_frames) // frame_length
        frame_to_second = frame_length / self.speaker_recogniser.sample_rate
        window_size = frame_length
        step_size = frame_length // 2

        probabilities = [0.0] * len(self.speaker_labels)
        majority_voting = [0] * len(self.speaker_labels)
        weighted_probabilities = [0.0] * len(self.speaker_labels)
        total_weight = 0
    
        threshold = 0.01
        # for i in range(0, num_frames - window_size + 1, step_size):
        #     window = audio_frames[i:i + window_size]
        #     scores = self.speaker_recogniser.process(window)
        #     window_start_time = i * frame_to_second
        #     window_end_time = (i + window_size) * frame_to_second
        #     audio_visual.print_scores(window_start_time, scores, self.speaker_labels)

        for i in range(num_frames):
            frame = audio_frames[i * frame_length:(i + 1) * frame_length]
            scores = self.speaker_recogniser.process(frame)
            time = i * frame_to_second
            # audio_visual.print_scores(time, scores, self.speaker_labels)
            for j, score in enumerate(scores):
                probabilities[j] += score

            for j, score in enumerate(scores):     
                if score > threshold:           
                     majority_voting[j] += 1

            weight = len(frame)
            total_weight += weight
            for j, score in enumerate(scores):
                weighted_probabilities[j] += score * weight

        #Weighted Probabilities        
        weighted_probabilities = [prob / total_weight for prob in weighted_probabilities]
        print('weighted probabilities:', weighted_probabilities)

        #Majority Voting
        print('majority voting', majority_voting)

        #Probability Sum
        print('probability sum', probabilities)

        method_detection = majority_voting
        
        if max(method_detection) == 0:
            print('could not detect voice')
            return None

        speaker_detected = self.speaker_labels[method_detection.index(max(method_detection))]
        human_info = commons.Human(settings.HUMANINFO_FOLDER + speaker_detected + ".json")
        print('speaker name: ' + human_info.info.name + ' with prediction: ' + str(max(method_detection)))
        return human_info.info.name

    ################UTILITIES#######################
    def band_filter(self, fname, outname):
     
        cutOffFrequency = 3000.0

        with contextlib.closing(wave.open(fname,'rb')) as spf:
            sampleRate = spf.getframerate()
            ampWidth = spf.getsampwidth()
            nChannels = spf.getnchannels()
            nFrames = spf.getnframes()

            # Extract Raw Audio from multi-channel Wav File
            signal = spf.readframes(nFrames*nChannels)
            spf.close()
            channels = self.interpret_wav(signal, nFrames, nChannels, ampWidth, True)

            # get window size
            # from http://dsp.stackexchange.com/questions/9966/what-is-the-cut-off-frequency-of-a-moving-average-filter
            freqRatio = (cutOffFrequency/sampleRate)
            N = int(math.sqrt(0.196196 + freqRatio**2)/freqRatio)

            # Use moviung average (only on first channel)
            filtered = self.running_mean(channels[0], N).astype(channels.dtype)

            wav_file = wave.open(outname, "w")
            wav_file.setparams((1, ampWidth, sampleRate, nFrames, spf.getcomptype(), spf.getcompname()))
            wav_file.writeframes(filtered.tobytes('C'))
            wav_file.close()
     
    # from http://stackoverflow.com/questions/13728392/moving-average-or-running-mean
    def running_mean(self, x, windowSize):
        cumsum = np.cumsum(np.insert(x, 0, 0)) 
        return (cumsum[windowSize:] - cumsum[:-windowSize]) / windowSize

    # from http://stackoverflow.com/questions/2226853/interpreting-wav-data/2227174#2227174
    def interpret_wav(self, raw_bytes, n_frames, n_channels, sample_width, interleaved = True):

        if sample_width == 1:
            dtype = np.uint8 # unsigned char
        elif sample_width == 2:
            dtype = np.int16 # signed 2-byte short
        else:
            raise ValueError("Only supports 8 and 16 bit audio formats.")

        channels = np.fromstring(raw_bytes, dtype=dtype)

        if interleaved:
            # channels are interleaved, i.e. sample N of channel M follows sample N of channel M-1 in raw data
            channels.shape = (n_frames, n_channels)
            channels = channels.T
        else:
            # channels are not interleaved. All samples from channel M occur before all samples from channel M-1
            channels.shape = (n_channels, n_frames)

        return channels

    #identify speakers during interaction
    def audio_diarization(self,audio_file):

        audio_config = speechsdk.audio.AudioConfig(filename=audio_file)
        conversation_transcriber = speechsdk.transcription.ConversationTranscriber(speech_config=self.azure_config, audio_config=audio_config)

        transcribing_stop = False
        transcription = []

        def stop_cb(evt: speechsdk.SessionEventArgs):
            #"""callback that signals to stop continuous recognition upon receiving an event `evt`"""
            print('CLOSING on {}'.format(evt))
            nonlocal transcribing_stop
            transcribing_stop = True

        def conversation_transcriber_transcribed_cb(evt: speechsdk.SpeechRecognitionEventArgs):
            # print('TRANSCRIBED:')
            nonlocal transcription
            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                transcription.append({'speaker':evt.result.speaker_id,'text':evt.result.text})
            elif evt.result.reason == speechsdk.ResultReason.NoMatch:
                print('\tNOMATCH: Speech could not be TRANSCRIBED: {}'.format(evt.result.no_match_details))

        # Connect callbacks to the events fired by the conversation transcriber
        conversation_transcriber.transcribed.connect(conversation_transcriber_transcribed_cb)
        # conversation_transcriber.session_started.connect(conversation_transcriber_session_started_cb)
        # conversation_transcriber.session_stopped.connect(conversation_transcriber_session_stopped_cb)
        # conversation_transcriber.canceled.connect(conversation_transcriber_recognition_canceled_cb)
        # stop transcribing on either session stopped or canceled events
        # conversation_transcriber.session_stopped.connect(stop_cb)
        # conversation_transcriber.canceled.connect(stop_cb)

        conversation_transcriber.start_transcribing_async()
        # Waits for completion.
        while not transcribing_stop:
            time.sleep(.5)

        conversation_transcriber.stop_transcribing_async()

        return transcription
    ###############################################


    def filter_audio(self, audio_file, save_file, use_band_filter=False):
        #apply band filter
        if use_band_filter:
            self.band_filter(audio_file, audio_file)

        save_file = audio_file
        # audio,value = load_audio(audio_file, sr=df_state.sr())
        # enhanced = enhance(model, df_state, audio)
        # save_audio(save_file, enhanced, df_state.sr())


    def language_detection(self,audio_file):
     
        detected_src_lang = ''
        # "en-US", "zh-CN","fr","de-DE","it-IT","ja-JP","ko-KR","pt-BR","ru-RU","es-ES","ar-EG","hi-IN","tr-TR","vi-VN"
        auto_detect_source_language_config = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(languages=["en-US", "zh-CN"])

        # Creates an AudioConfig from a given WAV file
        audio_config = speechsdk.audio.AudioConfig(filename=audio_file)

        # Creates a source language recognizer using a file as audio input, also specify the speech language
        source_language_recognizer = speechsdk.SourceLanguageRecognizer(
            speech_config=self.azure_config,
            auto_detect_source_language_config=auto_detect_source_language_config,
            audio_config=audio_config)
        
        detected_src_lang = ''

        result = source_language_recognizer.recognize_once()
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                detected_src_lang = result.properties[
                    speechsdk.PropertyId.SpeechServiceConnection_AutoDetectSourceLanguageResult]
                print("Detected Language: {}".format(detected_src_lang))
        elif result.reason == speechsdk.ResultReason.NoMatch:
            print("No speech could be recognized: {}".format(result.no_match_details))
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            print("Speech Language Detection canceled: {}".format(cancellation_details.reason))
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                print("Error details: {}".format(cancellation_details.error_details))

        if detected_src_lang:
            #remove everything after -
            detected_src_lang = detected_src_lang.split('-')[0]     

        return detected_src_lang
