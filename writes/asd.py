import numpy as np
S = "Será que hoje vai chover, eu não sei não"
S = S.lower()
S = S.replace(',', '')
S = S.split()
V = list(set(S))
V.sort()
V = dict(zip(V, range(0, len(V))))

n = len(V)
M = np.zeros((4,n), dtype=int)
for k, w in enumerate(S[0:2] + S[3:5]):
    i = V[w]
    wr = np.zeros(n)
    wr[i] = 1
    M[k] = wr

# V
# V
# S
# list(set(S))
# list(set(S)).sorted()
# V = list(set(S))
# V
# V.sort()
# V
# for w in S[0:2] + S[3:5]:
#     print(V[w])
# S
# S[0:2]
# S[0:2] + S[3:5]
# V
# V = dict(zip(V,range(1,len(V))))
# V
# V = dict(zip(V,range(1,len(V+1))))
# V = "Será que hoje vai chover, eu não sei não"
# V = V.lower()
# V = V.split()
# S
# V
# V = dict(zip(V,range(1,len(V+1))))
# V
# V = dict(zip(V,range(1,len(V)+1)))
# V
# %history
