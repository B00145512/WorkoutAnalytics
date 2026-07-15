from tkinter import *
from Exercises.Curl import curl

# Start by showing a window to select exercise
curl()
'''
select_window = Tk()
select_window.title("Workout Analyser")
select_window.geometry("700x450")
icon = PhotoImage(file="WA.png")
select_window.iconphoto(True, icon)
select_window.configure(bg="#ffeaea")

title = Label(select_window, text="Welcome, select what you'd like to train!",
              font=("Blaka", 25, "bold"), bg="#ffeaea",image=icon, compound="bottom")
title.pack()
curl_button = Button(select_window, text="Bicep Curl")
curl_button.config(command=curl, font=("Ariel", 15, "bold"), activebackground="#ffeaea", compound="left")
curl_button.pack()

tricep_button = Button(select_window, text="Tricep Extension")
tricep_button.config(command=tricep_extension, font=("Ariel", 15, "bold"), activebackground="#ffeaea", compound="left")
tricep_button.pack()

squat_button = Button(select_window, text="Squat")
squat_button.config(command=squat, font=("Ariel", 15, "bold"), activebackground="#ffeaea", compound="left")
squat_button.pack()

select_window.mainloop()
'''