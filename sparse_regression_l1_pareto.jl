## FIRST EXPERIMENT
# Sparse Linear Regression - L1 & Pareto

## PART 0) SETUP THE ENVINRONEMT
# 1) set your folder and path
# 2) activate environment  '] activate .' and packages

cd("C:/VEJŠKA/Ing/5. ROČNÍK/DIPLOMOVÁ PRÁCE/programy")

using Flux
using Flux.Data: DataLoader
using LinearAlgebra
using PlotlyJS
using DataFrames
using MLDataUtils
using Random
using Flux: @epochs
Random.seed!(42069)

# function to find out the number of non-zero elements
function non_zeros(A)
    B = []
    for i = 1:length(A)
        if A[i] != 0.0
            push!(B,i)
        end
    end
    return length(B)
end

# function to replace minimal elemenent for zero
function find_replace(A)
    min = minimum(A)
    if min != 0.0
       B = replace(A, min=>0.0)
       return B
    else
        C = findall(x -> x > 0.0, A)
        A2 = A[C]
        min2 = minimum(A2)
        for i = 1:length(A)
            if A[i] == min2
                A[i] = 0.0
            end
        end
        return A
    end
end

# function to replace based on ε neighborhood
function find_replace_epsilon(A, ϵ)
    for i = 1:length(A)
        if A[i] > -ϵ && A[i] < ϵ
            A[i] = 0.0
        end
    end
    return A
end


## PART 1) DATA GENERATING - ARTIFICAL
"
function prepare_known_dataset(n_observations, n_variables, split_ratio, batch_size, noise, θ_true, bias)
⋅ n_observations = number of observation in dataset D (N)
⋅ n_variables = number of observed predictors x_{k} (K)
⋅ split_ratio = ratio in which you would like to cut the dataset to train/test data
⋅ batch_size = number of batches you would like to use during training
⋅ noise = how much noisy you want to have a response variable y
⋅ θ_true = your groundtruth vector parameter
⋅ bias = bool (false/true), if you want to estimate the bias in regression model
"
function prepare_known_dataset(n_observations, n_variables, split_ratio, batch_size, noise, θ_true, bias)
    println("Preparing artificial known dataset with batchsize $(batch_size) & split ratio $(split_ratio)...")
    #SETS DIMENSIONS
    n_observations = n_observations;
    n_variables = n_variables;
    #ARTIFICIAL DATA generating
    if bias == true
        v = ones(n_observations);
        X = rand(n_observations, n_variables);
        X = hcat(v,X);
        θ_true = θ_true;
        y = X*θ_true + noise*randn(n_observations);
    elseif bias == false
        X = rand(n_observations, n_variables);
        θ_true = θ_true;
        y = X*θ_true + noise*randn(n_observations);
    end
    #SPLITTING DATASET acc. ratio
    (X_train, y_train), (X_test, y_test) = splitobs((X',y'); at=split_ratio);
    #TRAIN DATALOADER
    data_train = DataLoader((X_train, y_train); batchsize=batch_size, shuffle=true);
    data_test = DataLoader((X_test, y_test), batchsize=batch_size);
    println("Done!")
    return n_observations, n_variables, data_train, data_test, X_train, y_train, X_test, y_test, θ_true, X, y
end

θ_true = [2.9, 1.1, 0.02, 0.05, 10.0, 7.2, 0.06, 9.1, 0.001, 0.2, 0.76]
n_obs, n_var, data_train, data_test, X_train, y_train, X_test, y_test, θ_true, X, y = prepare_known_dataset(100, length(θ_true)-1, 0.8, 16, 0.5, θ_true, true)

## PART 2) selecting optimizer, λ range and training
lr = 0.001
opt = ADAM(lr, (0.9, 0.999))

# 2a) function to call with appropriate λ and selected number of epochs
function train_model_l1_linear(n_epochs, λ)
    model = Dense(n_var+1, 1, identity, bias=false)
    ps = Flux.params(model)
    loss_l1(X_train,y_train) = Flux.mse(model(X_train),y_train) + λ*mapreduce(x -> norm(x,1), +, ps)
    Flux.@epochs n_epochs Flux.train!(loss_l1, ps, data_train, opt; cb=()->println("$(loss_l1(X_train, y_train))"))
    θ_hat = vec(Flux.params(model)[1])
    return θ_hat
end

# 2b) train the models with appropriate hyperparameter
lambdas_thetas = []
for λ = 0.0:0.1:4.0
    θ_hat = train_model_l1_linear(7000, λ)
    push!(lambdas_thetas, [λ, θ_hat])
end
zaloha = deepcopy(lambdas_thetas)

CSV.write("training.csv", DataFrame(zaloha), header = false)
# 2b) λ curves

# function to sort λ and Θ values in trained and gained field
function get_lambdas_thetas(lambdas_thetas)
    B = deepcopy(lambdas_thetas)
    only_lambdas = []
    only_thetas = []
    for i = 1:length(lambdas_thetas)
        push!(only_lambdas, B[i][1])
        push!(only_thetas, B[i][2])
    end
    return only_lambdas, only_thetas
end

only_lambdas, only_thetas = get_lambdas_thetas(lambdas_thetas)
#CSV.write("training.csv", DataFrame([only_lambdas, only_thetas]), header = false)

# function to plot dependency of λ on parameter values
function plot_lambda_curve()
    rolled = hcat(only_thetas...)
    data = [scatter(;x=only_lambdas, y=rolled[j,:], mode="lines",name="param $j") for j=1:size(rolled)[1]]
    layout = Layout(;title="λ curve",xaxis_range=only_lambdas,
                        xaxis=attr(title="λ", showgrid=false, zeroline=false),
                        yaxis=attr(title="parameter values"))
    plot(data, layout)
end

function plot_lambda_curve_restrict(num)
    rolled = hcat(only_thetas...)
    data = [scatter(;x=only_lambdas[1:num], y=rolled[j,:], mode="lines",name="param $j") for j=1:num]
    layout = Layout(;title="λ curve (restricted)",xaxis_range=only_lambdas[1:num],
                        xaxis=attr(title="λ", showgrid=false, zeroline=false),
                        yaxis=attr(title="parameter values"))
    plot(data, layout)
end

# function to plot two figs for comparing (based on number of points in restricted interval)
function plot_neighborhood(num)
    lambda_fig1 = plot_lambda_curve()
    lambda_fig2 = plot_lambda_curve_restrict(num)
    lambda_fig = [lambda_fig1 lambda_fig2]
    lambda_fig.plot.layout["showlegend"] = false
    lambda_fig.plot.layout["width"] = 1000
    lambda_fig.plot.layout["height"] = 600
    lambda_fig
end

zkouska = plot_lambda_curve()

lambda_fig_1 = plot_neighborhood(10)
savefig(lambda_fig_1, "lambda_curve.pdf")

# function to plot dependency of λ on number of non-zero paramters
function sparsity_lambda(lambdas, thetas, tolerance)
    no_param = []
    ϵ = tolerance
    C = deepcopy(thetas)
    for i in 1:length(thetas)
        push!(no_param, non_zeros(find_replace_epsilon(C[i], ϵ)))
    end
    trace1 = scatter(;x=lambdas, y=no_param, mode="markers",name="ϵ=$ϵ")
    layout = Layout(;title="$ϵ-neighborhood",
                    xaxis=attr(title="λ", showgrid=false, zeroline=false),
                    yaxis=attr(title="number of non-zero parameters"))
    plot(trace1, layout)
end

# customible function to plot one big plot with various neighborhoods
function plot_neighborhood()
    eps_fig1 = sparsity_lambda(only_lambdas, only_thetas, 0.0)
    eps_fig2 = sparsity_lambda(only_lambdas, only_thetas, 0.01)
    eps_fig3 = sparsity_lambda(only_lambdas, only_thetas, 0.001)
    eps_fig4 = sparsity_lambda(only_lambdas, only_thetas, 0.0001)
    eps_fig = [eps_fig1 eps_fig2; eps_fig3 eps_fig4]
    eps_fig.plot.layout["showlegend"] = false
    eps_fig.plot.layout["width"] = 1000
    eps_fig.plot.layout["height"] = 600
    eps_fig
end

eps_fig = plot_neighborhood()
savefig(eps_fig, "eps_fig_sparse_reg_0_4_27_04_00_DP.pdf")

# 2c) Pareto frontiers
function data_to_pareto_tolerance(lambdas_thetas, tolerance)
    eval_data = []
    ϵ = tolerance
    B = deepcopy(lambdas_thetas)
    #FIRST LOOP TO EVAL ALL FULL MODELS
    for k = 1:size(B)[1]
        θ_hat = B[k][2]
        suma = []
        for i = 1:size(X_test)[2]
            y_hat = X_test[:,i]'*θ_hat
            quad = (y_hat - y_test[:,i][1])^2
            push!(suma, quad)
        end

        MSE_test = sum(suma) / size(X_test)[2]
        n_params = non_zeros(θ_hat)

    #    push!(eval_data, [MSE_test, n_params, B[k][1]])

        println("Test data MSE: $(MSE_test) with total of $(n_params) (full) parameters for λ = $(B[k][1]).")
    end
    # with function based on ε
    for k = 1:size(B)[1]
        θ_hat = B[k][2]
        for j = 1:length(θ_hat)-1
            θ_hat_j = find_replace_epsilon(θ_hat, ϵ)

            suma_j = []
            for i = 1:size(X_test)[2]
                y_hat = X_test[:,i]'*θ_hat_j
                quad = (y_hat - y_test[:,i][1])^2
                push!(suma_j, quad)
            end
            push!(eval_data, [sum(suma_j / size(X_test)[2]), non_zeros(θ_hat_j),B[k][1]])
            θ_hat = θ_hat_j
        end

    end
    return eval_data
end

# first (and easier) version on function how to get all the neccessarry data to Pareto frontier
function data_to_pareto(lambdas_thetas)
    eval_data = []
    C = deepcopy(lambdas_thetas)
    #FIRST LOOP TO EVAL ALL FULL MODELS
    for k = 1:size(C)[1]
        θ_hat = C[k][2]
        suma = []
        for i = 1:size(X_test)[2]
            y_hat = X_test[:,i]'*θ_hat
            quad = (y_hat - y_test[:,i][1])^2
            push!(suma, quad)
        end

        MSE_test = sum(suma) / size(X_test)[2]
        n_params = non_zeros(θ_hat)

        push!(eval_data, [MSE_test, n_params, C[k][1]])

        println("Test data MSE: $(MSE_test) with total of $(n_params) (full) parameters for λ = $(C[k][1]).")
    end

    for k = 1:size(C)[1]
        θ_hat = C[k][2]
        for j = 1:length(θ_hat)-1
            θ_hat_j = find_replace(θ_hat)

            suma_j = []
            for i = 1:size(X_test)[2]
                y_hat = X_test[:,i]'*θ_hat_j
                quad = (y_hat - y_test[:,i][1])^2
                push!(suma_j, quad)
            end
            push!(eval_data, [sum(suma_j / size(X_test)[2]), non_zeros(θ_hat_j),C[k][1]])
            θ_hat = θ_hat_j
        end

    end
    return eval_data
end

input_tolerance = 0.001

eval_data_neighborhood = data_to_pareto_tolerance(lambdas_thetas, input_tolerance)
eval_data_minimum_replaced = data_to_pareto(lambdas_thetas)

# sorting the data for easier way to plot them
function data_to_plot(eval_data)
    D = deepcopy(eval_data)
    MSEs = []
    param = []
    lambdas = []
    lambdas_string = []

    for i = 1:size(D)[1]
        push!(MSEs, D[i][1])
        push!(param, D[i][2])
        push!(lambdas, D[i][3])
        push!(lambdas_string, "λ="*string(D[i][3]))
    end

    return MSEs, param, lambdas, lambdas_string
end

MSEs_eps, param_eps, lambdas_eps, lambdas_eps_string = data_to_plot(eval_data_neighborhood)
MSEs, param, lambdas = data_to_plot(eval_data_minimum_replaced)

## PLOTTING
function pareto_plot(param, MSEs, lambdas)
    trace1 = scatter(;x=param, y=MSEs, text=lambdas, mode="markers+text", textposition="left center",textfont_size=8)
    layout = Layout(;title="Pareto frontier",
                    xaxis=attr(title="Number of parameters in model", showgrid=false, zeroline=false),
                    yaxis=attr(title="MSE (test data)"))
    plot(trace1, layout)
end

pareto_eps_fig = pareto_plot(param_eps, MSEs_eps, lambdas_eps_string)
pareto_min_fig = pareto_plot(param, MSEs)

df = DataFrame(x=MSEs_eps, y=param_eps)
sort!(df, rev=false)
pareto = df[1:1, :]
foreach(row -> row.y < pareto.y[end] && push!(pareto, row), eachrow(df))
#scatter!(pareto.y, pareto.x, label="Pareto Frontier", markersize=8, legend=true)

# function to plot final Pareto frontier
function pareto_frontier(param, MSEs, lambdas)
    trace1 = scatter(;x=param, y=MSEs, text=lambdas, mode="markers+text", textposition="center left",textfont_size=8,
            name="Various models")
    trace2 = scatter(;x=pareto.y, y=pareto.x, mode="markers",name="Pareto frontier")

    data = [trace1, trace2]
    layout = Layout(;title="Pareto frontier, ϵ=$input_tolerance",
                    xaxis=attr(title="Number of parameters in model", showgrid=false, zeroline=false),
                    yaxis=attr(title="MSE (test data)"))
    plot(data, layout)
end

pareto_fig = pareto_frontier(param_eps, MSEs_eps, lambdas_eps_string)
savefig(pareto_fig, "pareto_L1_eps_DP_001.pdf")

## Methods evaluation
# trace2 = saved values from VADAM prior -> easier than train it again
function evaluate_methods_reg()
    trace1 = scatter(;x=pareto.y, y=pareto.x, mode="markers+lines",name="L1")
    trace2 = scatter(;x=[4.0, 5.0, 6.0, 8.0, 11.0], y=[2.0, 0.59, 0.4, 0.382, 0.371], mode="markers+lines",name="VADAM with prior")
    data = [trace1, trace2]
    layout = Layout(;title="Evaluation of methods using Pareto frontiers",
                    xaxis=attr(title="Number of parameters in model", showgrid=false, zeroline=false),
                    yaxis=attr(title="MSE (test data)"))
    plot(data, layout)
end

evaluation1 = evaluate_methods_reg()
savefig(evaluation1, "experiment1_eval.pdf")
