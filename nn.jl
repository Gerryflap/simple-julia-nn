# Neural nets in Julia

mutable struct DenseLayer
    W::Array{Float32}
    b::Array{Float32}
end

function initDense(input_size, output_size)
    W = 2 .* rand(input_size, output_size) .- 1
    b = 2 .* rand(output_size) .- 1
    layer = DenseLayer(W, b)
    return layer
end

function forwardDense(x, layer)
    x = x' * layer.W
    x = reshape(x, :)
    x = x + layer.b
    return x
end

function backpropDense(x, layer, grads)
    new_grads = layer.W * grads
    weight_updates = x * reshape(grads, 1, :)
    bias_updates = grads
    layer.W += 0.0001 .* weight_updates
    layer.b += 0.0001 .* bias_updates
    return new_grads
end
   
function ReLU(x)
    return clamp!(x, 0, Inf)
end


function backpropReLU(x, grads)
       grads = copy(grads)
       grads[x .< 0] .= 0
    return grads
end

function sse(x, y)
    return sum((x-y).^2)
end

function backpropSSE(x, y)
    return y - x
end

# Tests

d1 = initDense(2, 50)
d2 = initDense(50, 1)

X = [[0 0]; [0 1]; [1 0]; [1 1];]
Y = [[0]; [1]; [1]; [0];]

for i = 1:100000
    for j = 1:4
        x, y = X[j, :], Y[j, :]
        z1 = forwardDense(x, d1)
        z1r = ReLU(z1)
        z2 = forwardDense(z1r, d2)
        cost = sse(z2, y)
        delta = backpropSSE(z2, y)
        delta = backpropDense(z1r, d2, delta)
        delta = backpropReLU(z1, delta)
        delta = backpropDense(x, d1, delta)
        println(cost)
    end
end

for j = 1:4
    x, y = X[j, :], Y[j, :]
    println("Result for ", x, " : ")

    z1 = forwardDense(x, d1)
    z1r = ReLU(z1)
    z2 = forwardDense(z1r, d2)
    println("Prediction: ", z2, ", actual: ", y)
    println("")
end
