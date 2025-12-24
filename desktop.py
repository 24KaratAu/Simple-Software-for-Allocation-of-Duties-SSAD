import sys
import os
import socket
from threading import Timer
import webbrowser
from app import app

# Function to find a free port automatically
def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

def open_browser(port):
    """Wait 1.5 seconds then open the browser."""
    print(f"--- Launching Browser on port {port} ---")
    webbrowser.open_new(f"http://127.0.0.1:{port}")

if __name__ == "__main__":
    # 1. Find a free port
    port = find_free_port()

    # 2. Schedule the browser to open in a background timer (Safe)
    Timer(1.5, open_browser, args=[port]).start()

    # 3. Run Flask in the MAIN thread (Crucial for Windows)
    # This prevents 'Windows Error 6' because the main thread owns the console.
    print(f"Starting Server on port {port}...")
    try:
        app.run(port=port, debug=False, use_reloader=False)
    except OSError as e:
        if e.winerror == 10013: # Port denied error
             print("Port locked. Trying default 5000...")
             app.run(port=5000, debug=False, use_reloader=False)
        else:
            raise e