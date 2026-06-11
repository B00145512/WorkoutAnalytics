from tkinter import *
from Exercise import curl, squat, tricep_extension

# Works best on Python 3.12
"""
0 - nose
1 - left eye (inner)
2 - left eye
3 - left eye (outer)
4 - right eye (inner)
5 - right eye
6 - right eye (outer)
7 - left ear
8 - right ear
9 - mouth (left)
10 - mouth (right)
11 - left shoulder
12 - right shoulder
13 - left elbow
14 - right elbow
15 - left wrist
16 - right wrist
17 - left pinky
18 - right pinky
19 - left index
20 - right index
21 - left thumb
22 - right thumb
23 - left hip
24 - right hip
25 - left knee
26 - right knee
27 - left ankle
28 - right ankle
29 - left heel
30 - right heel
31 - left foot index
32 - right foot index
"""
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