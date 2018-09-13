# Neural nets in Julia

# The data container for a dense layer
mutable struct DenseLayer
    W::Array{Float32}
    b::Array{Float32}
end

# Initializes a dense layer with random weights
function initDense(input_size, output_size)
    W = (2 .* rand(input_size, output_size) .- 1)./((output_size)^0.5)
    b = 2 .* rand(output_size) .- 1
    layer = DenseLayer(W, b)
    return layer
end

# Performs a dense (fully connected) layer forward pass
function forwardDense(x, layer)
    x = x' * layer.W
    x = reshape(x, :)
    x = x + layer.b
    return x
end

# Backpropagates the gradients through the dense layer. Updates the weights and biases and returns the new grads.
function backpropDense(x, layer, grads)
    new_grads = layer.W * grads
    weight_updates = x * reshape(grads, 1, :)
    bias_updates = grads
    layer.W += 0.001 .* weight_updates
    layer.b += 0.001 .* bias_updates
    return new_grads
end
   
# Forwardpass of the ReLU activation function
function ReLU(x)
    return clamp!(x, 0, Inf)
end

# Backpropagates gradients through the ReLU activation function
function backpropReLU(x, grads)
       grads = copy(grads)
       grads[x .< 0] .= 0
    return grads
end

# Computes the sum squared error between x (prediction) and y (target)
function sse(x, y)
    return sum((x-y).^2)
end

# Computes the gradients for gradient decend on the SSE cost function
function backpropSSE(x, y)
    return y - x
end

# ______________________________Tests____________________________________
d1 = initDense(2, 50)
d2 = initDense(50, 1500)
d3 = initDense(1500, 1)

X = [[0 0]; [0 1]; [1 0]; [1 1];]
Y = [[0]; [1]; [1]; [0];]

for i = 1:10000
    for j = 1:4
        x, y = X[j, :], Y[j, :]
        z1 = forwardDense(x, d1)
        z1r = ReLU(z1)
        z2 = forwardDense(z1r, d2)
        z2r = ReLU(z2)
        z3 = forwardDense(z2r, d3)
        cost = sse(z3, y)
        delta = backpropSSE(z3, y)
        delta = backpropDense(z2r, d3, delta)
        delta = backpropReLU(z2, delta)
        delta = backpropDense(z1r, d2, delta)
        delta = backpropReLU(z1, delta)
        delta = backpropDense(x, d1, delta)
        if i%100 == 0
            println(cost)
        end
    end
end

for j = 1:4
    x, y = X[j, :], Y[j, :]
    println("Result for ", x, " : ")

    z1 = forwardDense(x, d1)
    z1r = ReLU(z1)
    z2 = forwardDense(z1r, d2)
    z2r = ReLU(z2)
    z3 = forwardDense(z2r, d3)
    println("Prediction: ", z3, ", actual: ", y)
    println("")
end
