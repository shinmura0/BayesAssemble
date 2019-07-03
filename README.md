# Baysian Optimization Assembly Checker
![gif1](https://github.com/shinmura0/BayesAssemble/blob/add-license-1/35a55-42c6i.gif "gif1")  

# Environment
+ Raspberry Pi 3 model B (or PC)
+ USB camera
+ NCS2 or Movidius

# How to use
## Liblary 
+ [OpenVINO](https://qiita.com/shinmura0/items/318c775544fae64ae0db)
+ [dtw](https://github.com/pierre-rouanet/dtw)
+ [pykalman](https://github.com/pykalman/pykalman)
+ [GPy](https://github.com/SheffieldML/GPy)
+ [GPyOpt](https://github.com/SheffieldML/GPyOpt)

##Procedure
+ Download this repository
+ Command as follows
+ python3 main.py -wd 640 -ht 480 -vidfps 15
+ Push [m] to record train data.
+ Push [e] to finish recording.
+ Push [s] to record test data.
+ Push [e] to finish recording.
+ Result is saved as [result_1.avi] in the folder.

Click [here](https://qiita.com/shinmura0/items/09ed686466f3d7141d07) for details(Japanese)
