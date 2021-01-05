
import numpy as np
from hmmlearn import hmm
# 상태 s1 = 비, s2 = 해
states = ["Rain", "Sunn"]
n_states = len(states)
# 여자친구의 삶 
observations = ["walk", "shop", "clean"]
n_observations = len(observations)
# 초기 상태 확률           P(비)=0.6 P(해)=0.4
start_probability = np.array([0.6, 0.4])
# 상태 전이도
transition_probability = np.array([
  [0.7, 0.3],           # 비가 왔을때 -> 비 : 0.7, 비가 왔을때 -> 해 : 0.3
  [0.4, 0.6]            # 해가 떴을때 -> 비 : 0.4, 해가 떴을때 -> 해 : 0.6
])

# 해온일 
schedule = [0, 0, 2, 1] # O = (o1=산책,o2=산책,o3=청소,o=쇼핑)

# 관측 확률 B = |bj(vk)|    emission_probability(schedule)
emission_probability = np.array([
  [0.1, 0.4, 0.5],
  [0.6, 0.3, 0.1]
])

model = hmm.MultinomialHMM(n_components=n_states)
model.startprob_=start_probability
model.transmat_=transition_probability
model.emissionprob_=emission_probability

# ----------------------------------------------------평가-------------------------------------------------------------
import math

print('평가 문제')
print('그녀가 오늘, 내일, 모레, 글피에 산책,산책,청소,쇼핑을 할 확률')
print(math.exp(model.score(np.array([schedule]))))

print('-------------------------------------------')
# predict a sequence of hidden states based on visible states
bob_says = np.array([schedule]).T
model = model.fit(bob_says)
logprob, alice_hears = model.decode(bob_says, algorithm="viterbi")
print()
# ----------------------------------------------------디코딩-------------------------------------------------------------
print('디코딩 문제')
print("여자친구가 지난 나흘동안 :", ", ".join(map(lambda x: observations[int(x)], bob_says)), '을 했을 때')
print("그 나흘의 날씨를 추정해 보면 :", ", ".join(map(lambda x: states[int(x)], alice_hears)))

