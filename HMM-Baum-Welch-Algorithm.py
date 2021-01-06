import numpy as np
import matplotlib.pyplot as plt

#generating initial probabilities

#transition probabilities 
# 상태 전이 확률 a
transition = np.array([[0.7,0.3],
                       [0.4,0.6]])
#Emission probabilities
# 관측 확률 B = |bj(vk)|
emission = np.array([[0.1,0.4,0.5],     # 상태1(비가올때) 기호 vk(산책, 쇼핑, 청소)가 관측될 확률 
                     [0.6,0.3,0.1]])    # 상태2(해가뜰떄) 기호 vk(산책, 쇼핑, 청소)가 관측될 확률

#defining states and sequence symbols
states = ['R','S']
states_dic = {'R':0, 'S':1}
sequence_syms = {'1':0,'2':1,'3':2} # vk 1(산책):0, 2(쇼핑), 3(청소)
# emission[j,sequence_syms] : bj(vk)
sequence = ['1','2','3']    #

#test sequence
# O = (o1, o2, o3, o4)
test_sequence = '1132'
test_sequence = [x for x in test_sequence]

#probabilities of going to end state
end_probs = [1, 1]
#probabilities of going from start state
# pi 파이1=0.6 파이2=0.4
start_probs = [0.6, 0.4]

# 합이 1인 랜덤(난수) 행렬  p.247 : 이들이 확률이기 때문에 합이 1이 되는 조건을 지켜야한다.
def random_func(column,raw):
    reference = np.random.uniform(low=0, high=1.0, size=(column,raw - 1)) 
    reference.sort(axis = 1) 
    diffs = np.diff(reference, prepend=0, append=1, axis=1) 
    return diffs

#function to find forward probabilities
# 전방 변수 알파 (7.13) (7.14)
def forward_probs():
    # node values stored during forward algorithm
    # A 상태전이 확률
    # alpha = np.zeros((len(states), len(test_sequence)))
    alpha = random_func(len(states), len(test_sequence))
    for t, sequence_val in enumerate(test_sequence):
        for i in range(len(states)):
            # if first sequence value then do this
            if (t == 0):        # (7.13) alpha[j,0] = pi[j]*b[j,o1]
                alpha[i, t] = start_probs[i] * emission[i, sequence_syms[sequence_val]]
            # else perform this
            else:
                values = [alpha[k, t - 1] * emission[i, sequence_syms[sequence_val]] * transition[k, i] for k in range(len(states))]
                # (7.14)
                alpha[i, t] = sum(values) 

    #end state value
    end_state = np.multiply(alpha[:,-1], end_probs)
    end_state_val = sum(end_state)
    return alpha, end_state_val



#function to find backward probabilities
# 후방변수 베타 (7.24) (7.25)
def backward_probs():
    # node values stored during forward algorithm
    #Beta = np.zeros((len(states), len(test_sequence)))
    Beta = random_func(len(states), len(test_sequence))
    #for i, sequence_val in enumerate(test_sequence):
    for t in range(1,len(test_sequence)+1):
        for i in range(len(states)):
            # if first sequence value then do this
            if (-t == -1):
                Beta[i, -t] = end_probs[i]
            # else perform this
            else:
                values = [Beta[k, -t+1] * emission[k, sequence_syms[test_sequence[-t+1]]] * transition[i, k] for k in range(len(states))]
                Beta[i, -t] = sum(values)

    #start state value
    start_state = [Beta[m,0] * emission[m, sequence_syms[test_sequence[0]]] for m in range(len(states))]
    start_state = np.multiply(start_state, start_probs)
    start_state_val = sum(start_state)
    return Beta, start_state_val


#function to find si probabilities
# 감마 κ (7.26)
def si_probs(forward, backward, forward_val):

    si_probabilities = np.zeros((len(states), len(test_sequence)-1, len(states)))

    for t in range(len(test_sequence)-1):
        for i in range(len(states)):
            for k in range(len(states)):
                si_probabilities[i,t,k] = ( forward[i,t] * backward[k,t+1] * transition[i,k] * emission[k,sequence_syms[test_sequence[t+1]]] ) / forward_val
    return si_probabilities

#function to find gamma probabilities
# 감마 γ (7.22)
def gamma_probs(forward, backward, forward_val):

    #gamma_probabilities = np.zeros((len(states), len(test_sequence)))
    gamma_probabilities= random_func(len(states), len(test_sequence))
    for t in range(len(test_sequence)):
        for i in range(len(states)):
            #gamma_probabilities[i,t] = ( forward[i,t] * backward[i,t] * emission[i,sequence_syms[test_sequence[t]]] ) / forward_val
            gamma_probabilities[i, t] = (forward[i, t] * backward[i, t]) / forward_val

    return gamma_probabilities





#performing iterations until convergence
model_prob_list = []
model_list = []
for iteration in range(10000):

    print('\nIteration No: ', iteration + 1, '-----------------------------------------------------------')
    # print('\nTransition:\n ', transition)
    # print('\nEmission: \n', emission)

    #Calling probability functions to calculate all probabilities
    # E단계
    fwd_probs, fwd_val = forward_probs()    # 전방변수 알파
    bwd_probs, bwd_val = backward_probs()   # 후방변수 베타
    si_probabilities = si_probs(fwd_probs, bwd_probs, fwd_val)  # 감마 κ
    gamma_probabilities = gamma_probs(fwd_probs, bwd_probs, fwd_val)    # 감마 γ

    # print('Forward Probs:')
    # print(np.matrix(fwd_probs))
    #
    # print('Backward Probs:')
    # print(np.matrix(bwd_probs))
    #
    # print('Si Probs:')
    # print(si_probabilities)
    #
    # print('Gamma Probs:')
    # print(np.matrix(gamma_probabilities))

    #caclculating 'a' and 'b' matrices
    # 세타 = (A,B,pi)  세타 초기화 -> 난수를 이용해서 A,B,pi 설정 합이 1이 되게

    a = random_func(len(states), len(states))
    b = random_func(len(states), len(sequence_syms))
    pi = random_func(1,len(states))

    #'A' matrix
    # (7.27)
    for j in range(len(states)):
        for i in range(len(states)):
            for t in range(len(test_sequence)-1):
                a[j,i] = a[j,i] + si_probabilities[j,t,i]   # 감마 κ t=1부터 T-1까지 합, si에서 sj로 이전할 기대값 

            denomenator_a = [si_probabilities[j, t_x, i_x] for t_x in range(len(test_sequence) - 1) for i_x in range(len(states))]
            denomenator_a = sum(denomenator_a)
            # a[j,i]new
            if (denomenator_a == 0):
                a[j,i] = 0
            else:
                a[j,i] = a[j,i]/denomenator_a

    #'B' matrix
    # (7.28)
    for j in range(len(states)): #states
        for i in range(len(sequence)): #seq
            indices = [idx for idx, val in enumerate(test_sequence) if val == sequence[i]]
            numerator_b = sum( gamma_probabilities[j,indices] )
            denomenator_b = sum( gamma_probabilities[j,:] )
            # bi(vk)new
            if (denomenator_b == 0):
                b[j,i] = 0
            else:
                b[j, i] = numerator_b / denomenator_b

    # 'pi' matrix
    # (7.29)    pi는 t=0일때 si에 있을 확률 감마0(i)
    pi = gamma_probabilities[:,0]   # 하지만 for문을 멈추거나 모델을 만드는데 실제로는 안쓰임

    print('\nMatrix a:')
    print(np.matrix(a.round(decimals=4)))
    print('\nMatrix b:')
    print(np.matrix(b.round(decimals=4)))
    print()

    # 새 A, B 업데이트 (상태 전이 확률 A = |a(i,j)| 업데이트, 관측확률 B = |b(j,vk)|
    transition = a
    emission = b
    start_probs = pi

    new_fwd_temp, new_fwd_temp_val = forward_probs()    # 알파가 모델, 알파중에 가장 큰거 
    print('New forward probability: ', new_fwd_temp_val)
    model_list.append(new_fwd_temp)
    model_prob_list.append(new_fwd_temp_val)
    diff =  np.abs(fwd_val - new_fwd_temp_val)  # 새로 추정한 값과 이전 값의 차이
    print('Difference in forward probability: ', diff)  

    # 차이가 임계 값보다 작으면 더 이상 변화가 없다고 보고 멈춘다.
    if (diff < 0.00001):      
        break

print()
print('모델')
print(max(model_prob_list))
print()
print(model_list[max(range(len(model_prob_list)), key=lambda i:model_prob_list[i])])
