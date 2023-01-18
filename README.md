# AMLS_assignment22_23-
Python environment: 3.7  
In this project, A1 and A2 use the same SVM model,  
B1 and B2 use the same MLP model, with only slight variations in  
the feature extraction algorithm.  

# About the shape_predictor_68_face_landmarks.dat
Both A1 and A2 codes utilize the shape_predictor_68_face_landmarks.dat from week 6 lab  
However, size of the predictor is too large (95MB) and can't be uploaded to github, and I  
use a text file called 'shape_predictor_68_face_landmarks' to represent this predictor.  
In order to run the uploaded A1 and A2 code, please replace the text file in A1 and A2 folder  
with real predictor.  

# Folder A1
A1_SVM -- SVM algorithm for gender detection.  
landmarks_HOG_A1 -- HOG 68 landmarks extraction algorithm.  
shape_predictor_68_face_landmarks -- Assist the extraction of landmarks_HOG_A1.  

# Folder A2
A2_SVM -- SVM algorithm for emotion detection.  
landmarks_HOG_A2 -- HOG 68 landmarks extraction algorithm.  
shape_predictor_68_face_landmarks -- Assist the extraction of landmarks_HOG_A2.  

# Folder B1
B1_MLP -- MLP algorithm for face shape recognition.

# Folder B2
B2_MLP -- MLP algorithm for eye color recognition.


# libraries required to run the code:
os, numpy, tensorflow, keras, OpenCV-Python, cv2, dlib, importlib, sklearn

# You can run all the programs in A1, A2, B1, B2 directly in main.py.
Or run the programs in the corresponding folders  
It is recommended to run only one task at a time in mian and comment out the other tasks.  

# Notice that in order to run A1 and A2 in main
you need to update the path of landmarks_HOG_A1/A2 in A1_SVM and A2_SVM  
you also need to update the path of basedir and predictor in corresponding landmarks_HOG_A1 and landmarks_HOG_A2 file  

# Please update the path of corresponding files in mian.py before execution
