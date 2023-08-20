# JingDian-RL
JingDian (经典), adjective, means it's classical in Mandarin. In this repo, all OpenAI Gym environments are implemented in Python. Some of the RL algortihms are implemented in Julia as an experiment of this rising language in Machine Learning. The rest of the algorithms are implemented in Python.

A side note: divide between Julia and Python may not be most efficient in development. Because
- Loading a Gym Environment into Julia can take 5s, or more
- Debugging Python in Julia can be difficult in IPython Notebook, because both languages do not share the same stdio. In that case, we want flush output in Julia. But if there are bugs in Python, the program will break before flushing. I'm not sure if there are other effective debugging means, other than testing the code in python.

## Assignment 3
We added a Monte-Carlo and TD prediction assignment from [INF8953DE](https://github.com/mina-parham/INF8953DE/blob/master/assignment%202/Assignment_02_questions.pdf), to complement the syllabus.  
