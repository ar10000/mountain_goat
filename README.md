# Data analysis
-Project: mountain_goat  
  This was our final project during Le Wagon Data Science course
- Description: Virtual assistant for inexperienced climbers : when a an amateur climber is stuck on a route
  we can then recommend a move based on what an experienced climber would do

- Data Source: Youtube Videos of experienced climbers , roboflow for grip dataset

# Steps followed
- First ,we built a grip detector to detect all the grips on a climbing wall. We used detectron2 for this task and we archieved accuracy of about 93%  
-We then used cv2 to do some color analysis and determine all the colors of the grips, We also archieved realtively high accuaracy  for this task.  
-We then took frames of when an climbing instructor was holding a grip from a video (still trying to automate this step) and trained an LSTM model to predict the next    move that a climbing instructor would do
-We are currently trying to automate the step of getting signifiacant frames from a video(frames where a person is holding a grip i.e every move)

# Install
Go to `https://github.com/ar10000/mountain_goat` to see the project
