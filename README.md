# Workout Analytics
Workout Analytics is a computer vision-based application with machine learning made to analyse exercises for new gym users. This current system analyses **Bicep Curls**, using pose estimation by MediaPipe to track landmarks and an LSTM neural network to classify each repetition.

## Problem Statement
Incorrect exercise techniques can reduce exercise efficiency and could cause injury. Finding out what the proper form is per exercise could pose a challenge, especially for a beginner.
**Workout Analytics** aims to analyse each repetition the user does using a camera and provide written and visual feedback.

### *Features*
- GUI for workout selection
- Live Video feed of user with relevant landmarks connected and measured
- Live Spine Correction
- Accuracy score based on posture and repetition effort
- Written feedback for the user to improve in future sets
- Visualisations of their exercise with graphs based on previous uses, providing a comparison

## How it works
1. User selects exercise to then select repetitions
2. Person performs the bicep curl
3. MediaPipe Pose module detects relevant body landmarks
4. OpenCV processes camera feed and visualises the detected landmarks
5. Joints position, joint angles, velocity and movement is calculated
6. Calculated data is then stored into a CSV file
7. An LSTM Neural Network analyses the CSV and outputs one of the following classes: 1. perfect 2. elbow sway 3. Fatigue
8. Altair and StreamLit provide a post-workout summary with visualisations

   
# Flow Chart
![Flow Chart](https://github.com/B00145512/WorkoutAnalytics/blob/main/FlowChart.png)

### Technologies
| Technology        | Usage  |
| ------------- |:-------------:|
| Python      | Main language |
| OpenCV      | Camera capture processing and real-time visualisation      | 
| MediaPipe   | Human Pose and landmark detection      | 
| PyTorch      | LSTM implementation |
| LSTM      | Rep classification      | 
| Pandas | Processing and storing data     | 
| NumPy      | Calculations |
| CSV      | Storing frame data and workout summary data      | 
| Altair | Interactive data visualisation      | 
| StreamLit      | Workout dashboard |
| Scikit-Learn      | Dataset splitting and ML exaluation      | 


