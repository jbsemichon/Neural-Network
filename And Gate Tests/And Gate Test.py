x = [1,0]
expected = 0

##x = [0,1]
##expected = 0
##
##x = [1,1]
##expected = 1

def sigmoid(a):
    s = 1/(1 + (e**-a))
    return s
def derivative_of_sigmoid(value):
    return value - (1-value)
def gradient_function_cost(value,expected):
    return value-expected

lr = 0.05

w = [3,0.3]

b = 0.009

e = 2.718


derivative_of_a_wrt_weight1 = w[0]
derivative_of_a_wrt_weight2 = w[1]

for i in range(0,100):
    a = x[0]*w[0]+x[1]*w[1]

    print("Output before sigmoid:",a)

    value = sigmoid(a)

    print("Output after sigmoid:",value)

    cost = 0.5 * (value - expected)**2

    print("Cost:",cost)


    w[0] = w[0] - gradient_function_cost(value,expected) * derivative_of_sigmoid(value) * derivative_of_a_wrt_weight1 * lr
    w[1] = w[1] - gradient_function_cost(value,expected) * derivative_of_sigmoid(value) * derivative_of_a_wrt_weight2 * lr

print(w)
