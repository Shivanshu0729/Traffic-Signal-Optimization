# Traffic-Signal-Optimization
Traffic Signal Optimization model is based on a hybrid system that combines a Genetic Algorithm (GA) with Machine Learning (ML) using a Random Forest Regressor to optimize traffic signal timings at intersections. The system considers real-time traffic data like vehicle count, pedestrian presence, and emergency vehicles to dynamically adjust green, yellow, and red light durations. It improves traffic flow by predicting the best signal timings based on previous data and current conditions. Key features of your model include: dynamic signal optimization, pedestrian-friendly logic, emergency vehicle handling, real-time fitness evaluation, and learning from historical data to make smarter decisions over time.

🔧 Elements used in this model:
Genetic Algorithm for optimization

Random Forest Regressor for learning traffic patterns

Historical traffic dataset for training

Vehicle count input

Emergency and pedestrian condition flags

Fitness function for signal evaluation

ML model training and saving using joblib

Signal timing parameters: green, yellow, red

Python libraries: random, numpy, sklearn, joblib

Web display using HTML/CSS/JS

Real-time clock and condition display on webpage
