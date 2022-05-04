## FIRST EXPERIMENT
# Sparse Linear Regression

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

## PART 2) MODEL
"
function get_model()
⋅ returns the linear model without any hidden layers and output linear activation
⋅ for beter handling with bias it is better to estimate one vector instead of tupple in Flux pipeline
"
function get_model()
    model = Dense(n_var+1, 1, identity, bias=false)
end

model = get_model()

# 1) set the targeted loss and empty fields to remembering the loss values
loss(x, y) = Flux.Losses.mse(model(x), y)

function initialize_fields()
    test_losses = []
    train_losses = []
    param_history = []

    test_losses_l1 = []
    train_losses_l1 = []
    param_history_l1 = []

    return test_losses, train_losses, param_history, test_losses_l1, train_losses_l1, param_history_l1
end

test_losses, train_losses, param_history, test_losses_l1, train_losses_l1, param_history_l1 = initialize_fields()

# 2) set the optimizer in case of MLE/L1 (RMSprop, ADAM, SGD..)
lr = 0.001
opt = ADAM(lr, (0.9, 0.999))

# 3) functions to evaluating loss values as used in callbacks during training
function loss_all(data_loader)
    sum([loss(x, y) for (x,y) in data_loader]) / length(data_loader)
end

# 4) initialize the callbacks of what you want to monitor
callbacks = [
        () -> push!(train_losses, loss_all(data_train)),
        () -> push!(test_losses, loss_all(data_test)),
        () -> push!(param_history, Flux.destructure(model))
        ]

# 5) save initialize params (He initialization) in Flux structure
ps = Flux.params(model)

# 6) training function with one argument of number of epochs you want to train the model
function train(epochs)
    @epochs epochs Flux.train!(loss, ps, data_train, opt, cb = callbacks)

    @show train_loss = loss_all(data_train)
    @show test_loss = loss_all(data_test)
end

# 7) train the model
train(7000)

## PART 3) EVALUATING

# 1) getting the train history of parameters
function get_train_history(param_history)
    train_history = []
    for i in 1:length(param_history)
        parameters, res = param_history[i]
        push!(train_history, parameters)
    end
    train_history = mapreduce(permutedims, vcat, train_history)
    return train_history
end

train_history = get_train_history(param_history)

function plot_param_history(train_history)
    "
    ⋅ function to plot the train history of params
    ⋅ fill=tozerox in quotes to add errorbar
    "
    rolled = [train_history[:,i] for i in 1:size(train_history)[2]]
    data = [scatter(;y=rolled[j], mode="lines",name="param $j") for j in 1:size(rolled)[1]]
    layout = Layout(;title="Param history", width=700, height=400,
                        xaxis=attr(title="epochs", showgrid=false, zeroline=false),
                        yaxis=attr(title="value"))
    plot(data, layout)
end

# optional - analytical MLE estimate
θ_true_hat_train = inv(X_train*X_train')*X_train*y_train'
# plotting param history
param_history_fig = plot_param_history(train_history)
# adding plot of analytical solution
regression_full_fig = addtraces(param_history_fig,
                                scatter(;x=35020 .+ zeros(length(θ_true_hat_train)),
                                y=vec(θ_true_hat_train), mode="markers",lwd=5,color="red",
                                name="analytical estimation"))
# saving the figure
savefig(regression_full_fig, "regression_full_history.pdf")

# 2) getting the losses and plotting them
function plot_losses(train_losses, test_losses)
    p1 = scatter(;y=train_losses, color="blue", mode="lines", name="train")
    p2 = scatter(;y=test_losses, color="red", mode="lines", name="test")
    data = [p1, p2]
    layout = Layout(;title="Losses",
                    xaxis=attr(title="epochs", showgrid=false, zeroline=false),
                    yaxis=attr(title="loss (MSE)"))
    plot(data, layout)
end

# alternative to plot losses in log shape
function plot_log_losses(train_losses, test_losses)
    p1 = scatter(;y=log.(train_losses), color="blue", mode="lines", name="train")
    p2 = scatter(;y=log.(test_losses), color="red", mode="lines", name="test")
    data = [p1, p2]
    layout = Layout(;title="Log losses",
                    xaxis=attr(title="epochs", showgrid=false, zeroline=false),
                    yaxis=attr(title="log loss (MSE)"))
    plot(data, layout)
end

# plotting losses
losses_fig = plot_losses(train_losses, test_losses)
savefig(losses_fig, "regression_full_losses.pdf")

#plotting log losses
losses_log_fig = plot_log_losses(train_losses, test_losses)
savefig(losses_log_fig, "regression_full_losses.pdf")

losses_all_fig = hcat(losses_fig, losses_log_fig)
savefig(losses_all_fig, "regression_full_losses_both.pdf")
