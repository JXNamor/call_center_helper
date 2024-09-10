import sys
import subprocess
import tkinter as tk
from tkinter import ttk
import threading
from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from openai_key import OPENAI_API_KEY

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Ensure required packages are installed
try:
    import openai
except ImportError:
    install("openai")
try:
    import whisper
except ImportError:
    install("whisper")
try:
    import speech_recognition as sr
except ImportError:
    install("speechrecognition")
try:
    import numpy as np
except ImportError:
    install("numpy")
try:
    import torch
except ImportError:
    install("torch")


# Initialize OpenAI API
openai.api_key = OPENAI_API_KEY

def extract_key_information(text):
    messages = [
        {"role": "system", "content": "You are an assistant that extracts key information from text."},
        {"role": "user", "content": f"Extract all key information from this transcription of a phone call between a client and a call operator. The first paragraph is the information that has already been extracted. Only extract new key elements about the customer to help the call operator and update the elements using the new information. Do not drastically change the output, keep it similar to what was already used in the first paragraph. Use small phrases or only words:\n{text}"}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=100,
        temperature=0.5
    )
    return response.choices[0].message['content'].strip()

def post_call_summary(text):
    messages = [
        {"role": "system", "content": "You are an assistant that evaluate how a call operator handled a client."},
        {"role": "user", "content": f"Evaluate how well the call operator handled a client using this transcript of a phone call. The information needed is : was the problem solved (answer with yes/no/partly), how well was the customer greeted, was the agent knowledgeable, were the agent solving skills useful, were there customer complaints, was the agent's tone apropriate. You should rate all of these points out of 10 and then make a briefing about the points that should be improved and what was done nicely. For each rating that wasn't 10, explain what could be improved in order to make that a 10. You also need to add a confidence rate out of 10 of how much you are confident in these notations. \n{text}"}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=200,
        temperature=0.5
    )
    return response.choices[0].message['content'].strip()

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-time Speech-to-Text and Summarization")

        # Styling
        self.style = ttk.Style()
        self.style.theme_use('winnative')

        # Parameters frame
        self.parameters_frame = ttk.Frame(root, padding="20", style="TFrame")
        self.parameters_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))

        ttk.Label(self.parameters_frame, text="Model:").grid(row=0, column=0, sticky=tk.W)
        self.model_combobox = ttk.Combobox(self.parameters_frame, values=["tiny", "base", "small", "medium", "large"])
        self.model_combobox.set("small")
        self.model_combobox.grid(row=0, column=1)

        ttk.Label(self.parameters_frame, text="Energy Threshold:").grid(row=1, column=0, sticky=tk.W)
        self.energy_threshold_entry = ttk.Entry(self.parameters_frame)
        self.energy_threshold_entry.insert(0, "1000")
        self.energy_threshold_entry.grid(row=1, column=1)

        ttk.Label(self.parameters_frame, text="Record Timeout:").grid(row=2, column=0, sticky=tk.W)
        self.record_timeout_entry = ttk.Entry(self.parameters_frame)
        self.record_timeout_entry.insert(0, "3")
        self.record_timeout_entry.grid(row=2, column=1)

        ttk.Label(self.parameters_frame, text="Phrase Timeout:").grid(row=3, column=0, sticky=tk.W)
        self.phrase_timeout_entry = ttk.Entry(self.parameters_frame)
        self.phrase_timeout_entry.insert(0, "2")
        self.phrase_timeout_entry.grid(row=3, column=1)

        ttk.Label(self.parameters_frame, text="Device to listen to:").grid(row=4, column=0, sticky=tk.W)
        self.device_entry = ttk.Entry(self.parameters_frame)
        self.device_entry.insert(0, "Line 1 (Virtual Audio Cable)")
        self.device_entry.grid(row=4, column=1)

        self.start_button = ttk.Button(self.parameters_frame, text="Start", command=self.start)
        self.start_button.grid(row=5, column=0, padx=5, pady=5)

        self.stop_button = ttk.Button(self.parameters_frame, text="Stop", command=self.stop)
        self.stop_button.grid(row=5, column=1, padx=5, pady=5)

        self.quit_button = ttk.Button(self.parameters_frame, text="Quit", command=self.quit)
        self.quit_button.grid(row=5, column=2, columnspan=2, pady=10)

        # Output frame
        self.output_frame = ttk.Frame(root, padding="20", style="TFrame")
        self.output_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))

        font = ("Helvetica", 12)
        self.transcription_text = tk.Text(self.output_frame, height=10, width=80, font=font, bg="#ffffff", fg="#000000", borderwidth=2, relief="sunken")
        self.transcription_text.grid(row=0, column=0, padx=5, pady=5)

        self.summary_text = tk.Text(self.output_frame, height=10, width=80, font=font, bg="#ffffff", fg="#000000", borderwidth=2, relief="sunken")
        self.summary_text.grid(row=1, column=0, padx=5, pady=5)

        # Threading setup
        self.stop_thread_flag = threading.Event()

    def start(self):
        self.stop_thread_flag.clear()
        threading.Thread(target=self.run_transcription, daemon=True).start()

    def stop(self):
        self.stop_thread_flag.set()

    def quit(self):
        self.stop()
        self.root.quit()

    def run_transcription(self):
        # Fetch parameters from UI
        model_name = self.model_combobox.get()
        energy_threshold = int(self.energy_threshold_entry.get())
        record_timeout = float(self.record_timeout_entry.get())
        phrase_timeout = float(self.phrase_timeout_entry.get())
        device = str(self.device_entry.get())

        self.transcription_text.delete('1.0', tk.END)
        self.summary_text.delete('1.0', tk.END)
        self.transcription_text.insert(tk.END, "\nDownloading the model... Please wait.\n")

        data_queue = Queue()
        recorder = sr.Recognizer()
        recorder.energy_threshold = energy_threshold
        recorder.dynamic_energy_threshold = False

        mic_index = None
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            if device in name:
                mic_index = index
                break

        # Use the selected microphone or default to device index 0 if not found
        if mic_index is None:
            mic_index = 0
            self.transcription_text.insert(tk.END, "\n'Line 1' microphone not found. Using default microphone.\n")
        else:
            self.transcription_text.insert(tk.END, f"\nUsing microphone: {device} (device index: {mic_index}).\n")

        # Initialize the microphone source
        source = sr.Microphone(sample_rate=16000, device_index=mic_index)

        model = model_name
        audio_model = whisper.load_model(model)

        transcription = ['']
        phrase_time = None

        def record_callback(_, audio: sr.AudioData) -> None:
            data = audio.get_raw_data()
            data_queue.put(data)

        recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

        self.transcription_text.insert(tk.END, "\nListening...")
        key_info = ""
        while not self.stop_thread_flag.is_set():
            now = datetime.utcnow()

            if not data_queue.empty():
                phrase_complete = False
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    phrase_complete = True
                phrase_time = now

                audio_data = b''.join(data_queue.queue)
                data_queue.queue.clear()

                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                result = audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
                text = result['text'].strip()

                if phrase_complete:
                    transcription.append(text)

                    # Update transcription text in UI
                    self.transcription_text.delete('1.0', tk.END)
                    self.transcription_text.insert(tk.END, "\n".join(transcription))

                    # Summarize transcribed text so far
                    full_text = key_info + "\n ".join(transcription)
                    if len(full_text.split()) > 50:
                        key_info = extract_key_information(full_text)
                        self.summary_text.delete('1.0', tk.END)
                        self.summary_text.insert(tk.END, key_info)
                else:
                    transcription[-1] = text

                sleep(0.25)

        self.transcription_text.insert(tk.END, "\n\nTranscription Ended")
        self.summary_text.delete('1.0', tk.END)
        self.summary_text.insert(tk.END, key_info)
        self.summary_text.insert(tk.END, "\n\nSummary Ended. Here is the post call update :\n")
        call_summary = post_call_summary(full_text)
        self.summary_text.insert(tk.END, call_summary)

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.quit)
    root.mainloop()
