import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import subprocess
import threading

# Create the main window
root = tk.Tk()
root.title("Project GUI")

# Set the window to open in full-screen mode
root.attributes("-fullscreen", True)

# Load the background image
bg_image_path = "C:/Users/ASUS/Downloads/ui_image1.jpg"  # Replace with your image path
bg_image = Image.open(bg_image_path)
bg_photo = ImageTk.PhotoImage(bg_image.resize((root.winfo_screenwidth(), root.winfo_screenheight())))

# Create a canvas for the background
canvas = tk.Canvas(root, width=root.winfo_screenwidth(), height=root.winfo_screenheight())
canvas.pack(fill="both", expand=True)
canvas.create_image(0, 0, image=bg_photo, anchor="nw")  # Set the background image

# Create a frame to center content (button only)
frame = tk.Frame(root, highlightbackground="white", highlightthickness=5)
frame.place(relx=0.5, rely=0.5, anchor="center")

label = tk.Label(root, text=" Smart Traffic Light  Management System", font=("Helvetica", 40, "bold"), fg="black")
label.place(relx=0.5, rely=0.2, anchor="center")  # Position the label outside the frame

# Create a "Please Wait..." label
wait_label = tk.Label(root, text="Please Wait...", font=("Helvetica", 20), fg="black")
wait_label.place(relx=0.5, rely=0.3, anchor="center")
wait_label.place_forget()  # Initially hide the label

# Button click event handler
def on_button_click():
    messagebox.showinfo("Info", "Starting the project...")
    wait_label.place(relx=0.5, rely=0.3, anchor="center")  # Show the "Please Wait..." message
    threading.Thread(target=start_project).start()  # Run the project script in a separate thread

# Function to start the project by running the script and capturing output
def start_project():
    try:
        # Run the Python script and capture the output
        result = subprocess.run(
            ["python", "C:/Users/ASUS/Desktop/project/yolov5/algorithim.py"], 
            check=True, 
            capture_output=True, 
            text=True
        )  # 'capture_output=True' captures stdout and stderr
        
        # If there's any error output, display it in a messagebox
        if result.stderr:
            messagebox.showerror("Error", f"An error occurred:\n{result.stderr}")
        else:
            messagebox.showinfo("Success", "Project completed successfully!")
            
    except Exception as e:
        print(f"Error: {e}")
        messagebox.showerror("Error", f"An error occurred: {e}")
    finally:
        wait_label.place_forget()  # Hide the "Please Wait..." message
        # Optionally, display the result or process further here
        # For example, you can show the result in a new window or a label

# Create the "Start Project" button
button = tk.Button(frame, text="Start Project", command=on_button_click, font=("Helvetica", 14), bg="white", fg="black")
button.pack(pady=20)

# Animation for the "Start Project" button
def animate_button():
    current_color = button.cget("bg")
    new_color = "lightblue" if current_color == "white" else "white"  # Toggle between white and light blue
    button.config(bg=new_color)
    root.after(500, animate_button)  # Repeat every 500 milliseconds

# Start the button animation
animate_button()

# Exit full-screen mode with 'Esc' key
def exit_fullscreen(event):
    root.attributes("-fullscreen", False)

# Bind the 'Esc' key to exit full-screen mode
root.bind("<Escape>", exit_fullscreen)

# Run the Tkinter event loop
root.mainloop()
