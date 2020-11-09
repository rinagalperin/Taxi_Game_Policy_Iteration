# Reinforcement learning project: Taxi-v3 task from openAI 

## :dart: Goal ([source](https://gym.openai.com/envs/Taxi-v3/))
This task was introduced to illustrate some issues in hierarchical reinforcement learning.
There are 4 locations (labeled by different letters) and our job is to pick up the passenger at one location and drop him off in another.
We receive +20 points for a successful dropoff, and lose 1 point for every timestep it takes.
There is also a 10 point penalty for illegal pick-up and drop-off actions.

<br><br>
![code output example](taxi_example.png?raw=true "Taxi example")
<br><br>

## :bulb: Solution
We implement a solution using policy iteration, i.e. - finding an optimal policy if exists.

## :clipboard: Code
At each run the code does the following: 
1. Computes the optimal policy (using the ["learn" function](policy_iteration_agent.py) to train the agent)
2. Performs a single simulation run for the optrimal policy ([the simulator](simulation_runner.py) gets the agent as input and uses function "get_action")
3. for each state in the check_states array, we print the number representing the state and the value function of the state.

## :email: Contact
- rinag@post.bgu.ac.il
- schnapp@post.bgu.ac.il
