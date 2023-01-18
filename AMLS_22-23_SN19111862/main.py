# You can run all the programs in A1, A2, B1, B2 directly in main.py.
# Or run the programs in the corresponding folders
# It is recommended to run only one task at a time in mian and comment out the other tasks.

# Notice that in order to run A1 and A2 in main
# you need to update the path of landmarks_HOG_A1/A2 in A1_SVM and A2_SVM
# you also need to update the path of basedir and predictor in corresponding landmarks_HOG_A1 and landmarks_HOG_A2 file

import A1, A2

# Task A1
# In order to run A1 in main.py
# please update the path of A1_SVM stored in A1 folder to path_A1
path_A1 = r'C:\Users\Wei Dai\PycharmProjects\AMLS_22-23_SN19111862\A1\A1_SVM.py'
exec(open(path_A1).read())

# Task A2
# In order to run A2 in main.py
# please update the path of A2_SVM stored in A2 folder to path_A2
path_A2 = r'C:\Users\Wei Dai\PycharmProjects\AMLS_22-23_SN19111862\A2\A2_SVM.py'
exec(open(path_A2).read())

# Task B1
# In order to run B1 in main.py
# please update the path of B1_MLP stored in B1 folder to path_B1
path_B1 = r'C:\Users\Wei Dai\PycharmProjects\AMLS_22-23_SN19111862\B1\B1_MLP.py'
exec(open(path_B1).read())

# Task B2
# In order to run B2 in main.py
# please update the path of B2_MLP stored in B2 folder to path_B2
path_B2 = r'C:\Users\Wei Dai\PycharmProjects\AMLS_22-23_SN19111862\B2\B2_MLP.py'
exec(open(path_B2).read())