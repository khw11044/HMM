# 알고리즘 [7.1]
# 예제 7.6 전방 계산에 의한 확률 평가

# 입력
pi = [0,0.6,0.4] # 초기값 파이 (0.6,0.4) (더해서 1)

o1 = '산책'
o2 = '산책'
o3 = '청소'
o4 = '쇼핑'

O = [0,o1,o2,o3,o4] # 주어진 여자친구의 일상 오늘 산책 내일 산책 모레 청소 글피 쇼핑 --> 날씨는?

a = [[0,0,0],[0,0.7,0.3],[0,0.4,0.6]]   # 전이 상태 확률 a11=0.7, a12=0.3, a21=0.4, a22=0.6

def b(num,o):
    if num == 1:        # 비일때
        if o == '산책':
            return 0.1
        elif o == '쇼핑':
            return 0.4
        else :  # 청소
            return 0.5
    else :  # num = 2   # 해일때
        if o == '산책':
            return 0.6
        elif o == '쇼핑':
            return 0.3
        else : 
            return 0.1
# ------------------------------------------------------------------------------

# 알고리즘 [7.1]
# 예제 7.6 전방 계산에 의한 확률 평가
# A는 알파t(i)
# _sum은 P(O|theta)
def forward(T,n):
    A = [[0 for j in range(n+1)] for i in range(T+1)]
    for i in range(1,n+1):
        A[1][i] = pi[i]*b(i,o1) 

    for t in range(2,T+1):
        for i in range(1,n+1):
            _sum = 0
            for j in range(1,n+1):
                _sum = _sum + A[t-1][j]*a[j][i]
         
            A[t][i] = round(_sum*b(i,O[t]),5)

    _sum = 0

    for j in range(1,n+1):
        _sum = _sum + A[T][j]
    
    return _sum, A

result_sum, A = forward(len(O)-1,len(pi)-1)

print(result_sum)

for i in A:
    print(i)


print('-------------------------------')

# 알고리즘 [7.2]
# 예제 7.4 Viterbi 알고리즘에 의한 디코딩

def viterbi(T,n):               # T=4,n=2
    Q = [0 for _ in range(T+1)]
    X = [[0 for j in range(n+1)] for i in range(T+1)]
    tau = [[0 for j in range(n+1)] for i in range(T+1)]

    for i in range(1,n+1):
        X[1][i] = pi[i]*b(i,o1)
    
    for t in range(2,T+1):
        for i in range(1,n+1):
            k_max = X[t-1][1]*a[1][i]
            k = 1
            for j in range(2,n+1):
                if k_max < X[t-1][j]*a[j][i]:
                    k_max = X[t-1][j]*a[j][i]
                    k = j
            X[t][i] = round(X[t-1][k]*a[k][i]*b(i,O[t]),5)
            tau[t][i] = k

    # 경로 역추적
    j = X[T].index(max(X[T]))
    Q[T] = tau[T][j]
    for t in range(T-1,0,-1):
        Q[t] = tau[t+1][Q[t+1]]

    return Q[1:]


Q_hat = viterbi(4,2)
print(Q_hat)

