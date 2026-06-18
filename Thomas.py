#Thomas algorithm

import numpy as np 

def thomas(a, b, c, d): 
    """
    A is the tridiagonal coefficient matrix and d is the RHS matrix
    Structure of A: 
        [b_1, ..., b_n] along primary diagonal 
        [a_1, ..., a_n-1] along lower diagonal 
        [c_1, ..., c_n-1] along upper diagonal 
    """
    a = np.asarray(a, dtype=float)  #(n-1)  
    b = np.asarray(b, dtype=float)  #(n) 
    c = np.asarray(c, dtype=float)  #(n-1)
    d = np.asarray(d, dtype=float)  #(n) 

    n = b.size
    if a.size != n-1 or c.size != n-1 or d.size != n:
        raise ValueError("Need a,c:(n-1,), b,d:(n,)")

    cp = np.zeros(n-1, dtype=float)
    dp = np.zeros(n, dtype=float)
    x = np.zeros(n, dtype=float) 

    #Perform forward sweep
    denom = b[0]
    cp[0] = c[0] / denom
    dp[0] = d[0] / denom

    for i in range(1, n): 
        denom = b[i] - (a[i-1] * cp[i-1]) 
        
        if i < n-1: 
            cp[i] = c[i] / denom
        dp[i] = (d[i] - (a[i-1] * dp[i-1])) / denom 

    #Perform Back substitution 
    x[-1] = dp[-1] 

    for i in range(n-2, -1, -1):
        x[i] = dp[i] - (cp[i] * x[i+1]) 

    return(x) 



