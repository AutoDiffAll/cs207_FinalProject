import numpy as np
try:
    from variables import Variable
except:
    from AutoDiff.variables import Variable



#example loss function:
def loss_function(lambdas, X_data):
    # x1, x2 will be vector inputs that Shane and Josh will implement
    # for example, x1, x2 = X_data[:,0], X_data[:,1]

    #lambda1 = 2.05384, lambda2 = 0 in this case
    lambda1, lambda2 = lambdas[0][0], lambdas[1][0]
    a = 0.000045*lambda2**2*np.sum(x2)
    b = -0.000098*lambda1**2*np.sum(x1)
    c = 0.003926*np.sum(x1)*lambda1*np.exp(-0.1*(lambda1**2+lambda2**2))
    return a + b + c

# gradient of the loss function
def gradient_func(lambdas, X_data):
    x1, x2 = X_data[:,0], X_data[:,1]
    lambda1, lambda2 = lambdas[0][0], lambdas[1][0]
    
    # this will be 
    df_dlambda1 = -0.000098 * 2 * lambda1 * np.sum(x1) + 0.003926 *  np.sum(x1) * np.exp(-0.1 * (lambda1**2 + lambda2**2)) - 0.003926 * 2 * 0.1 *  np.sum(x1) * lambda1**2 * np.exp(-0.1 * (lambda1**2 + lambda2**2))
    df_dlambda2 = 2 * 0.000045 * lambda2 * np.sum(x2) - 0.1 * 2* 0.003926 *  np.sum(x1) * lambda1 * lambda2 * np.exp(-0.1 * (lambda1**2 + lambda2**2))
    return np.array([[df_dlambda1], [df_dlambda2]])

# Obtain Stochastic Gradient Descent
def stochastic_gradient_descent(lambda_init, X_data, step_size, scale, max_iterations, 
                     precision, loss):
    history = []
    oldloss = -np.inf
    m, n = X_data.shape
    lambdas = lambda_init
    currentloss = loss(lambdas, X_data)
    counter = 0
    history.append(currentloss)
    while abs(currentloss - oldloss) > precision:
        np.random.shuffle(X_data)
        for i in range(m):
            gradient = gradient_func(lambdas, np.array([X_data[i,:]]))

            lambdas = lambdas - step_size * gradient * scale
        oldloss = currentloss
        currentloss = loss(lambdas, X_data)
        history.append(currentloss)
        counter += 1
        if counter == max_iterations:
            break
    return {'lambdas': lambdas, 'history': history}