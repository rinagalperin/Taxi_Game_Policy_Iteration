# Reinforcement learning project: Taxi-v3 task from openAI 

## :dart: Goal ([source](https://gym.openai.com/envs/Taxi-v3/))
This task was introduced in to illustrate some issues in hierarchical reinforcement learning.
There are 4 locations (labeled by different letters) and our job is to pick up the passenger at one location and drop him off in another.
We receive +20 points for a successful dropoff, and lose 1 point for every timestep it takes.
There is also a 10 point penalty for illegal pick-up and drop-off actions.

## :bulb: Solution
We implement a solution using policy iteration, i.e. - finding an optimal policy if exists.

## :email: Contact
- rinag@post.bgu.ac.il
- schnapp@post.bgu.ac.il
