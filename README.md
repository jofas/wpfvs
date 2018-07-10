# wpfvs

## execute

1. install dependencies:
  
  - tensorflow: <code>pip install tensorflow</code>

  - keras: <code>pip install keras</code>
  
  - gym:
  
    + <code> git clone https://github.com/openai/gym </code>
      
    + <code> cd gym </code>
  
    + <code> pip install -e '.[all]' </code>

2. execute: 

  - Set WPFVS\_HOME variable: <code>soucre .env</code>

  - start program with: <code>python3 run.py</code>
 
## todo

### doc

**topics we have to or could cover**

- What our program does and benchmark analysis (how well we improved using the cluster)

- The programs architecture

- Our source code (listing for the documentation)

- Tensorflow

- Python

- OpenAI Gym

- Neural Nets

- (RabbitMQ)

- (Numpy) (fast linear algebra in python)

- ...

### proc

- optimize dataset

- add RabbitMQ for executing on the cluster

## links

- **[openai gym](http://gym.openai.com)** for training a RL agent.

- **[distributed multi-layered ai cluster architecture](https://medium.com/adhive/distributed-multi-layered-ai-cluster-architecture-4576497ec27c)**
for a distributed system.

- **[awesome-mashine-learning](https://github.com/josephmisiti/awesome-machine-learning)** a lot of reading material.

- **[awesome-reinforcement-learning](https://github.com/aikorea/awesome-rl#human-computer-interaction)** for stuff about reinforcement learning.

- **[multi-agent system](https://en.wikipedia.org/wiki/Multi-agent_system)** theoretical model of something we could build.

- **[deep q learning](http://rll.berkeley.edu/deeprlcourse/f17docs/lecture_7_advanced_q_learning.pdf)** holy sh\*t mathematical but basically describes how our agent works and how to optimize it further.

- **[double deep q learining](https://arxiv.org/pdf/1509.06461.pdf)**

- **[time to beat LunarLander-v2](https://gym.openai.com/evaluations/eval_FbKq5MxAS9GlvB7W6ioJkg/)** 22 minutes.
