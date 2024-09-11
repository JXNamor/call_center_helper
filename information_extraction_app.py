import sys
import subprocess
import tkinter as tk
from tkinter import ttk
import threading
from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from openai_key import OPENAI_API_KEY
import json
import os


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
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except ImportError:
    install("matplotlib")


# Initialize OpenAI API
openai.api_key = OPENAI_API_KEY

def extract_key_information(text):
    messages = [
        {"role": "system", "content": "You are an assistant that extracts key information from text."},
        {"role": "user", "content": f"Extract all key information from this transcription of a phone call between a client and a call operator. The first paragraph is the information that has already been extracted. Only extract new key elements about the customer to help the call operator and update the elements using the new information. The output should respect this format :\n 'Client first name : Unknown\nClient last name : Unknown\nClient number : Unknown\nClient phone number : Unknown\nClient email : Unknown\nClient date of birth : Unknown\nReason for client's call : Unknown\n'. Fill the missing information. Use small phrases or only words:\n{text}"}
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
        {"role": "user", "content": f"Evaluate how well the call operator handled a client using this transcript of a phone call. The information needed is : 'Problem solved : x/100\nCustomer greeting : x/100\nAgent knowledge : x/100\nAgent solving skills : x/100\nCustomer complaints : x/100\nAgent's engagement : x/100\nAgent sentiment score : x/100\nTalking pace estimation: x WPM'. You should rate all of these points out of 100 and then make a briefing about the points that should be improved and what was done nicely, along with a word per minute estimation for the agent (the length of the call is given at the end). For each rating that wasn't 100, explain what could be improved in order to make that a 100. You also need to add a confidence rate out of 100 of how much you are confident in these notations. \n{text}"}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=500,
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
        self.device_entry.insert(0, "CABLE Output")
        self.device_entry.grid(row=4, column=1)

        self.start_button = ttk.Button(self.parameters_frame, text="Start", command=self.start)
        self.start_button.grid(row=5, column=0, padx=5, pady=5)

        self.stop_button = ttk.Button(self.parameters_frame, text="Stop", command=self.stop)
        self.stop_button.grid(row=5, column=1, padx=5, pady=5)

        self.show_graph_button = ttk.Button(self.parameters_frame, text="Show Graph", command=self.show_graph)
        self.show_graph_button.grid(row=5, column=2, padx=5, pady=5)

        self.quit_button = ttk.Button(self.parameters_frame, text="Quit", command=self.quit)
        self.quit_button.grid(row=6, column=0, columnspan=1, pady=5)

        self.ratings_data = []
        self.load_ratings_data()

        self.graph_window = None

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

    def load_ratings_data(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, "ratings_data.json")
        try:
            with open(file_path, "r") as f:
                content = f.read()
                if content.strip():  # Check if the file is not empty
                    self.ratings_data = json.loads(content)
                else:
                    self.ratings_data = []
        except (FileNotFoundError, json.JSONDecodeError):
            self.ratings_data = []

    def save_ratings_data(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, "ratings_data.json")
        with open(file_path, "w") as f:
            json.dump(self.ratings_data, f)

    def extract_ratings(self, summary_text):
        ratings = {}
        for line in summary_text.split('\n'):
            if ":" in line:
                key, value = line.split(':')
                value = value.strip()
                if "- " in key:
                    key = key.split(': ')[0].strip()
                try:
                    if '/' in value:
                        value = value.split('/')[0].strip()
                    ratings[key.strip()] = int(value)
                except ValueError:
                    if "WPM" in value:
                        ratings[key.strip()] = value
                    else:
                        ratings[key.strip()] = None
        return ratings
    
    def show_graph(self):
        self.graph_window = tk.Toplevel(self.root)
        self.graph_window.title("Ratings Evolution")

        # Create a frame for checkboxes and the graph
        checkbox_frame = ttk.Frame(self.graph_window)
        checkbox_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10)

        graph_frame = ttk.Frame(self.graph_window)
        graph_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Define metrics
        metrics = ['Problem solved', 'Customer greeting', 'Agent knowledge', 'Agent solving skills',
                'Customer complaints', 'Agent\'s engagement', 'Agent sentiment score']

        # Dictionary to hold checkbox variables
        self.metric_vars = {}
        
        # Create a checkbox for each metric
        for metric in metrics:
            var = tk.BooleanVar(value=True)  # By default, all checkboxes are checked
            checkbox = ttk.Checkbutton(checkbox_frame, text=metric, variable=var)
            checkbox.pack(anchor=tk.W)
            self.metric_vars[metric] = var

        # Button to refresh the graph based on selected metrics
        refresh_button = ttk.Button(checkbox_frame, text="Show Selected", command=self.update_graph)
        refresh_button.pack(pady=10)

        # Set up the initial graph
        self.fig, self.ax = plt.subplots(figsize=(8, 6))

        # Create a canvas to embed the graph in the window
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Initially display the graph with all metrics selected
        self.update_graph()

    def update_graph(self):
        """Update the graph based on selected metrics."""
        self.ax.clear()  # Clear the previous graph

        metrics = ['Problem solved', 'Customer greeting', 'Agent knowledge', 'Agent solving skills',
                'Customer complaints', 'Agent\'s engagement', 'Agent sentiment score']

        for metric in metrics:
            if self.metric_vars[metric].get():  # Only plot the metric if its checkbox is selected
                values = [call.get(metric) for call in self.ratings_data if metric in call]
                if values:
                    self.ax.plot(range(1, len(values) + 1), values, label=metric)

        self.ax.set_xlabel('Call Number')
        self.ax.set_ylabel('Ratings')
        self.ax.set_title('Operator Behavior Metrics Evolution')
        self.ax.legend()

        # Redraw the canvas with the updated graph
        self.canvas.draw()

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
            print(f"microphone name : {name} associated with index {index}")
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
        start = datetime.utcnow()
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
        end = datetime.utcnow()
        self.transcription_text.insert(tk.END, "\n\nTranscription Ended")
        self.summary_text.delete('1.0', tk.END)
        self.summary_text.insert(tk.END, key_info)
        self.summary_text.insert(tk.END, "\n\nSummary Ended. Here is the post call update :\n")
        call_time = end - start
        text_for_summary = full_text + "\n" + str(call_time)
        call_summary = post_call_summary(text_for_summary)
        self.summary_text.insert(tk.END, call_summary)

        ratings = self.extract_ratings(call_summary)
        self.ratings_data.append(ratings)
        self.save_ratings_data()

    

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.quit)
    root.mainloop()
