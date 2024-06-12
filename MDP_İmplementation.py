import numpy as np

# T[s,a,s']
T = [
    [[0.7,0.3,0.],[1.,0.,0.],[0.8,0.2,0.]],
    [[0.,1.,0.],None,[0.,0.,1.]],
    [None,[0.8,0.1,0.1]]
]
# R = [s,a,s']
R = [
[[+10, 0, 0],[0,0,0],[0,0,0]],
[[0,0,0],[0,0,0],[0,0,-50]],
[[0,0,0],[+40,0,0],[0,0,0]]
]

possibleActions=[[0,1,2],[0,2],[1]]

q_values =np.full((3,3),-np.inf)

for state, action in enumerate(possibleActions):
    q_values[state,action]=0.
    print(state,action)

gamma=0.90
forSum = 0

for iteration in range(50):
    q_prev=q_values.copy()
    for s in range(3):
        for a in possibleActions[s]:
            q_values[s,a]=np.sum([T[s][a][sp]*(R[s][a][sp]+gamma*np.max(q_prev[sp])) for sp in range(3)])

print(q_values)
print("\n")
print(np.argmax(q_values,axis=1))