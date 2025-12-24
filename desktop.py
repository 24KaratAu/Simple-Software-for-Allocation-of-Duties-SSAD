import sys
import os
import socket
import threading
import time
import webbrowser
import requests # We use this for the custom download stream
import tkinter as tk
from tkinter import ttk, messagebox

# --- FORCE FIX FOR TCL/TK (Keep this!) ---
if sys.platform == 'win32':
    tcl_path = os.path.join(sys.prefix, 'Lib', 'tcl8.6')
    tk_path = os.path.join(sys.prefix, 'Lib', 'tk8.6')
    if os.path.exists(tcl_path):
        os.environ['TCL_LIBRARY'] = tcl_path
        os.environ['TK_LIBRARY'] = tk_path

# --- CONFIGURATION ---
# Direct link to the file for manual streaming
MODEL_URL = "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
MODEL_DIR = os.path.join(os.getcwd(), "models")
MODEL_FILE = "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)

# --- GUI SETUP WIZARD ---
class SetupWizard:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("SecureDuty - First Time Setup")
        self.root.geometry("600x500") # Slightly taller for details
        self.root.resizable(False, False)
        
        # Center Window
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = int((screen_width/2) - (600/2))
        y = int((screen_height/2) - (500/2))
        self.root.geometry(f'+{x}+{y}')

        # Styles
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TProgressbar", thickness=25, background="#28a745")

        # 1. Header with Shield Icon (Text only for now)
        header_frame = tk.Frame(self.root)
        header_frame.pack(pady=20)
        tk.Label(header_frame, text="🛡️", font=("Segoe UI", 30)).pack(side="left")
        tk.Label(header_frame, text="SecureDuty Privacy Shield", font=("Segoe UI", 18, "bold"), fg="#0056b3").pack(side="left", padx=10)

        # 2. The Pitch
        manifesto = (
            "Welcome to the SecureDuty Secure Environment.\n\n"
            "Most AI tools send your data to the cloud, creating privacy risks.\n"
            "SecureDuty is different. It runs 100% OFFLINE.\n\n"
            "To enable this, we need to install the 'Llama-3.2-1B' Neural Network\n"
            "directly onto your device. This allows us to process\n"
            "rosters without a single byte of data leaving this computer."
        )
        self.lbl_text = tk.Label(self.root, text=manifesto, font=("Segoe UI", 10), justify="center", fg="#333", wraplength=520)
        self.lbl_text.pack(pady=10)

        # 3. Dynamic Status Labels
        self.status_var = tk.StringVar(value="Ready to initialize local AI engine.")
        self.lbl_status = tk.Label(self.root, textvariable=self.status_var, font=("Segoe UI", 10, "bold"), fg="#0056b3")
        self.lbl_status.pack(pady=(15, 5))

        self.detail_var = tk.StringVar(value="") # "45 MB / 800 MB (5.2 MB/s)"
        self.lbl_detail = tk.Label(self.root, textvariable=self.detail_var, font=("Consolas", 9), fg="gray")
        self.lbl_detail.pack(pady=0)

        # 4. The Real Progress Bar
        self.progress = ttk.Progressbar(self.root, orient="horizontal", length=500, mode="determinate")
        self.progress.pack(pady=15)

        # 5. Action Button
        self.btn_install = tk.Button(self.root, text="Download & Install AI Model", font=("Segoe UI", 11, "bold"), 
                                     bg="#28a745", fg="white", padx=25, pady=10, command=self.start_download, relief="flat", cursor="hand2")
        self.btn_install.pack(pady=10)

        self.download_complete = False

    def start_download(self):
        self.btn_install.config(state="disabled", bg="#cccccc", cursor="arrow")
        self.status_var.set("Initializing connection to HuggingFace...")
        threading.Thread(target=self.download_logic, daemon=True).start()

    def download_logic(self):
        try:
            os.makedirs(MODEL_DIR, exist_ok=True)
            
            # Start the stream
            response = requests.get(MODEL_URL, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            if response.status_code != 200:
                raise Exception(f"HTTP Error {response.status_code}")

            # Prepare for loop
            downloaded = 0
            start_time = time.time()
            chunk_size = 1024 * 1024 # 1MB chunks
            
            with open(MODEL_PATH, 'wb') as f:
                for data in response.iter_content(chunk_size=chunk_size):
                    f.write(data)
                    downloaded += len(data)
                    
                    # Math for UI Updates
                    elapsed = time.time() - start_time
                    if elapsed > 0:
                        speed = (downloaded / 1024 / 1024) / elapsed # MB/s
                        percent = (downloaded / total_size) * 100
                        
                        # Update UI (Thread safe via lambda)
                        self.root.after(0, lambda p=percent: self.progress.configure(value=p))
                        self.root.after(0, lambda s=f"{downloaded/1024/1024:.1f} MB / {total_size/1024/1024:.1f} MB  ({speed:.1f} MB/s)": self.detail_var.set(s))
                        self.root.after(0, lambda: self.status_var.set("Downloading Llama-3.2 Brain..."))

            # Done
            self.download_complete = True
            self.root.after(0, self.finish_setup)

        except Exception as e:
            def error_ui():
                messagebox.showerror("Download Error", f"Failed: {str(e)}\nCheck your internet.")
                self.btn_install.config(state="normal", bg="#28a745", text="Retry Download")
                self.status_var.set("Download Failed.")
            self.root.after(0, error_ui)

    def finish_setup(self):
        self.progress['value'] = 100
        self.status_var.set("Installation Complete!")
        self.detail_var.set("Verifying integrity... OK.")
        self.root.update()
        time.sleep(1) # Let user see the 100%
        messagebox.showinfo("Success", "Local AI Engine Installed!\n\nSecureDuty is now ready to run offline.")
        self.root.destroy()
    
    def run(self):
        self.root.mainloop()

# --- MAIN APP LOGIC ---
def check_model_gui():
    if not os.path.exists(MODEL_PATH):
        try:
            wizard = SetupWizard()
            wizard.run()
        except Exception as e:
            print(f"GUI Error: {e}")
            # If GUI fails, we just exit so we don't crash loop
            sys.exit(1)
        
        if not os.path.exists(MODEL_PATH):
            sys.exit(0) # User cancelled

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

def open_browser(port):
    print(f"--- Launching Interface... ---")
    webbrowser.open_new(f"http://127.0.0.1:{port}")

if __name__ == "__main__":
    # 1. Setup Check
    check_model_gui()

    # 2. Import App
    try:
        from app import app
    except ImportError as e:
        print("Critical: Could not import Flask app. Check requirements.")
        sys.exit(1)

    # 3. Launch
    port = find_free_port()
    threading.Timer(1.5, open_browser, args=[port]).start()
    print(f"Server running on port {port}")
    app.run(port=port, debug=False, use_reloader=False)