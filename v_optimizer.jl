## FIRST EXPERIMENT
# Sparse Linear Regression - Variational Methods

## PART 0) SETUP THE ENVINRONEMT
# 1) set your folder and path
# 2) activate environment  '] activate .' and packages
cd("C:/VEJŠKA/Ing/5. ROČNÍK/DIPLOMOVÁ PRÁCE/programy")

using Flux
using LinearAlgebra
using PlotlyJS
using Base.Iterators
using DataFrames
using CSV
using Random
using MLDataUtils
using Flux.Data: DataLoader
using Flux: @epochs
using Flux.Optimise:  batchmemaybe, update!
import Zygote: Params, gradient
Random.seed!(42069)


## PART 0) initialize the functions
# 0a) initialize the empty Gaussian Posterior structure
struct GaussPs
    ps
    σps
end

# 0b) function to evaluate loss in each epoch
function loss_all(data_loader)
    sum([loss(x, y) for (x,y) in data_loader]) / length(data_loader)
end

# 0c) function to evaluate loss with ard prior in each epoch
function loss_all_ard(data_loader)
    sum([loss(x, y) + loss_ard() for (x,y) in data_loader]) / length(data_loader)
end

# 0d) variational inference without prior
function vtrain!(loss, ps, data, opt; cb = () -> (), σ0=1e4)
    ps_mean = deepcopy(ps) # ps_mean = μw
    σps = deepcopy(ps)     # σps = σw
    map((x)->(x.=σ0) , σps) # σps .= σ0

    losses = []
    #cb = runall(cb)
    k = 1
    for d in data
        # reparametrization trick
        for i=1:length(ps)
            ps_mean[i].=ps[i] # backup w-> psmean
            ps[i].=ps[i].+randn(size(ps[i])).* σps[i] # reparam. trick
        end
        gs = gradient(ps) do  # gradient(loss(batchmemaybe(d)...) ,ps)
          loss(batchmemaybe(d)...)
        end
        for i=1:length(ps)
            ps[i].=ps_mean[i]
        end
        update!(opt, ps, gs)
        # update!(opt, ps_mean, gs)

        for i=1:length(ps)
            σps[i].=1.0./sqrt.(opt.state[ps[i]][2])
        end
        cb(ps)
        loss_actual = loss_all(data)
        println("Epoch: $k.")
        k = k+1
        push!(losses, loss_actual)
    end
    return GaussPs(ps,σps), losses
end

# 0e) variational inference with zero-mean Gaussian prior with precision

# initialize ard prior loss for vector form of params
loss_ard() = mapreduce(pl->ard(pl[1],pl[2]),+, (ps[1]',ψ[1]'))
ard(x,ψ) = 0.5*sum(x.^2 .* ψ)

# proposed algorithm
function vtrain_ardvb!(loss, ps, data, opt, N; cb = (a) -> (), σ0=1e-3, ψ0=1e-2, clip=0.0)
    ps_mean = deepcopy(ps)    # mean (i.e. ps as it is in adam) - μw
    σps = deepcopy(ps)        # variances - σw
    ψ = deepcopy(ps)         # precision of prior of every ps in ARD
    map((x)->(x.=σ0) , σps)   # only mapping function to assign first run variances
    map((x)->(x.=ψ0) , ψ)     # only mapping function to assign first run precisions
    ard(x,ψ) = 0.5*sum(x.^2 .* ψ) # Gaussian Prior on weights with λ precision

    loss_ard() = mapreduce(pl->ard(pl[1],pl[2]),+, (ps[1]',ψ[1]'))

    losses = []
    k = 1
    # training via Flux.jl pipeline
    for d in data #for every epoch
        # reparametrization trick
        for i=1:length(ps)
            ps_mean[i].=ps[i]
            ps[i].=ps[i].+randn(size(ps[i])).* σps[i] # ϴ -> μ + σ ∘ ϵ
        end # ps is random now - gradients on data ∇logp(d_i\textbar ϴ)
        gs = gradient(ps) do
          loss(batchmemaybe(d)...) + loss_ard()
        end
        # only if clip in function greater than one
        if clip>0.0
            for i=1:length(ps)
                clipnorm!(gs[ps[i]],clip)
            end # ps is random now
        end

        for i=1:length(ps)
            ps[i].=ps_mean[i]
        end # ps is the mean again
        update!(opt, ps, gs) # update the mean

        # store ADAM internals in σ - updating variance
        for i=1:length(ps)
            σps[i].=1.0./sqrt.(N*opt.state[ps[i]][2])
        end

        # VB update for ψ
        for (ψ,μ,σ) in zip(ψ,ps,σps)
            ψ .= 1.0 ./ (μ.^2 .+ σ.^2 )
        end
        cb(ps,ψ) #callbacks
        loss_actual = loss_all_ard(data)
        println("Epoch $k")
        k = k+1
        push!(losses, loss_actual)
    end
    return GaussPs(ps,σps), ψ, losses
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

# Variational Methods are written with another type of dataloader
function create_data(n_epochs)
    n_epochs = n_epochs
    data_train = repeated((X_train, y_train), n_epochs)
    data_test = repeated((X_test, y_test), n_epochs)
    return n_epochs, data_train, data_test
end

n_epochs, data_train, data_test = create_data(10000)

## PART 2) MODEL SET UP

function get_model()
    model = Dense(n_var+1, 1, identity, bias=false)
end

model = get_model()
loss(x, y) = Flux.Losses.mse(model(x), y)
ps = Flux.params(model)
ψ = deepcopy(ps)

# 2a) select optimizer -> if ADAM + v_train! or v_train_ardvb! -> VADAM
#                      -> if RMSprop + v_train! or v_train_ardvb! -> Variational RMSprop
lr = 0.1
opt = ADAM(lr, (0.9, 0.999))

# 2b) initialize empty vector for training history
PS=Vector()

## PART 3) TRAIN VIA VARIATIONAL INFERENCE
# 3a) without prior
qps, losses = vtrain!(loss, ps, data_train, opt, cb = (ps)->(push!(PS,deepcopy(hcat(ps...)))))

pps = vcat(PS...)

# 3b) with ARD prior
qps, ψ, losses = vtrain_ardvb!(loss, ps, data_train, opt, 80; cb = (ps,ψ)->(push!(PS,deepcopy(hcat(ps...)))))

ψ_posterior = ψ[1] #VADAM estimated precisions
pps_prior = vcat(PS...)

## PART 4) MODEL EVALUATION
function plot_param_history(pps)
    rolled = [pps[:,i] for i in 1:size(pps)[2]]
    data = [scatter(;y=rolled[j], mode="lines",name="param $j") for j in 1:size(rolled)[1]]
    layout = Layout(;title="VADAM param history",
                        xaxis=attr(title="epochs", showgrid=false, zeroline=false),
                        yaxis=attr(title="value"))
    plot(data, layout)
end

function plot_param_history_prior(pps)
    rolled = [pps[:,i] for i in 1:size(pps)[2]]
    data = [scatter(;y=rolled[j], mode="lines",name="param $j") for j in 1:size(rolled)[1]]
    layout = Layout(;title="VADAM param history with ARD prior",
                        xaxis=attr(title="epochs", showgrid=false, zeroline=false),
                        yaxis=attr(title="value"))
    plot(data, layout)
end

# check analytical solution
θ_true_hat = inv(X_train*X_train')*X_train*y_train'

vadam_without_prior_fig = plot_param_history(pps)
vadam_with_prior_fig = plot_param_history_prior(pps_prior)
regression_vadam_no_prior_fig = addtraces(vadam_without_prior_fig, scatter(;x=10020 .+ zeros(length(θ_true_hat)),
                                y=vec(θ_true_hat), mode="markers",lwd=5,color="red", name="analytical estimation"))
regression_vadam_prior_fig = addtraces(vadam_with_prior_fig, scatter(;x=10020 .+ zeros(length(θ_true_hat)),
                                y=vec(θ_true_hat), mode="markers",lwd=5,color="red", name="analytical estimation"))

savefig(regression_vadam_prior_fig, "regression_vadam_prior_history.pdf")

# collecting trained means μ and variance σμ from gaussian structure
μ,σμ = qps.ps, qps.σps

# 4a) finding the relevant parameters
function finding_relevant_parameters(n_std)
    n_std = n_std
    i_relevant = μ.>n_std*σμ
    θ_true_relevant = θ_true[vec(i_relevant')]
    X_relevant = X[:,vec(i_relevant)]
    μ_plot = []
    σμ_plot = []
    for i in 1:size(i_relevant)[2]
        if i_relevant[i] == true
            push!(μ_plot, convert(Float64, μ[i]))
            push!(σμ_plot, convert(Float64,σμ[i]))
        else
            push!(μ_plot, convert(Float64,0.0))
            push!(σμ_plot, convert(Float64,0.0))
        end
    end
    return X_relevant, θ_true_relevant, i_relevant, n_std, μ_plot, σμ_plot
end

# function with optional parameter which assign user's prefered number of std to find relevant param
X_relevant, θ_true_relevant, i_relevant, n_std, μ_plot, σμ_plot = finding_relevant_parameters(3)

last_epoch_vadam_fig = plot(μ', mode="markers",name="last epoch", error_y=attr(array=n_std*σμ, visible=true),
    Layout(;title="VADAM trained parameters (ARD)", xaxis=attr(title="No. of parameter component", showgrid=false, zeroline=false),
                    yaxis=attr(title="value")))

last_epoch_relevant_vadam_fig = plot(hcat(μ_plot), mode="markers", name="relevant parameters", error_y=attr(array=n_std*hcat(σμ_plot'), visible=true),
    Layout(;title="Relevant parameters", xaxis=attr(title="No. of parameter component", showgrid=false, zeroline=false),
                    yaxis=attr(title="value")))

vadam_fig = hcat(last_epoch_vadam_fig, last_epoch_relevant_vadam_fig)

savefig(vadam_fig, "regression_vadam_last_3_prior.pdf")

# function to plot losses
function plot_losses(losses_input)
    p1 = scatter(;y=losses_input, color="blue", mode="lines", name="train")
#    p2 = scatter(;y=test_losses, color="red", mode="lines", name="test")
    data = [p1]
    layout = Layout(;title="Loss",
                    xaxis=attr(title="epochs", showgrid=false, zeroline=false),
                    yaxis=attr(title="loss"))
    plot(data, layout)
end

plot_losses(losses)

# function to find and replace based on user's preference of number of stds
function find_replace_n_std(trained_μ_params, trained_σμ_params, n_std)
	copy_μ = deepcopy(trained_μ_params)
	copy_σμ = deepcopy(trained_σμ_params)
	for i in 1:length(copy_μ)
		if (typeof(copy_μ[i]) == Matrix{Float32}) == true
			for l in 1:size(copy_μ[i])[1]
				for k in 1:size(copy_μ[i])[2]
					if abs(copy_μ[i][l,k]) > n_std*copy_σμ[i][l,k] #&& copy_μ[i][l,k] < n_std*copy_σμ[i][l,k]
						copy_μ[i][l,k] = copy_μ[i][l,k]
					else
						copy_μ[i][l,k] = convert(Float32, 0.0)
					end
				end
			end
		elseif (typeof(copy_μ[i]) == Vector{Float32}) == true
			for j in 1:length(copy_μ[i])
				if abs(copy_μ[i][j]) > n_std*copy_σμ[i][j] #&& copy_μ[i][j] < n_std*copy_σμ[i][j]
					copy_μ[i][j] = copy_μ[i][j]
				else
					copy_μ[i][j] = convert(Float32, 0.0)
				end
			end

		end
	end
	return copy_μ
end

# function to gain only posterior vector from params structure to vector
function extract_posterior_means(trained_μ_params)
	copy = deepcopy(trained_μ_params)
	μ_vec = []

	for i in 1:length(copy)
		append!(μ_vec, vcat(copy[i]...))
	end
	return μ_vec
end

# function to gain only posterior vector of means and stds from params structure to vector
function extract_posterior_params(qps)
	copy = deepcopy(qps)
	μ = []
	σμ = []

	for i in 1:length(copy.ps)
		append!(μ, vcat(copy.ps[i]...))
		append!(σμ, vcat(copy.σps[i]...))
	end
	return μ,σμ
end

# function to gain only posterior vector of precisions from params structure to vector
function extract_posterior_precision(ψ)
	copy = deepcopy(ψ)
	ψ_posterior = []

	for i in 1:length(copy)
		append!(ψ_posterior, vcat(copy[i]...))
	end
	return ψ_posterior
end

# loop for Pareto
model = get_model()

function reg_to_pareto(trained_μ_params, trained_σμ_params, n)
	n_std = n
	model = get_model()
	updated_means = find_replace_n_std(trained_μ_params, trained_σμ_params, n_std)
	Flux.loadparams!(model,updated_means)
    loss_eval(X_test,y_test) = Flux.mse(model(X_test), y_test)
	loss = loss_eval(X_test,y_test)
	return n_std, non_zeros(updated_means[1]), loss
end

## Pareto data loading & looping
data_to_pareto = []
lower, step_neco, upper = 0.0, .5, 10.0
for n in lower:step_neco:upper
	n_std, not_null, loss_test = reg_to_pareto(μ, σμ, n)
	push!(data_to_pareto, [n_std, not_null, loss_test])
end

function get_attributes_pareto(data_to_pareto)
	losses = []
	stds = []
	stds_string = []
	no_params = []
	A = deepcopy(data_to_pareto)
	for i in 1:size(A)[1]
		std = A[i][1]
		non = A[i][2]
		loss_actual = A[i][3]
		push!(stds, std)
		push!(no_params, non)
		push!(losses, loss_actual)
		push!(stds_string, "d="*string(A[i][1]))
	end
	return stds, stds_string, no_params, losses
end

# sorry for this, PlotlyJS is useless in some way and stucked all the text together -> this way to
# rearrange the vector to make the plot more clear and readible
# data from data_to_pareto (saved)
stds_string_better = ["d=0.0","d=0.5","d=1.0, d=1.5, d=2.0, d=2.5", "d=3.0, d=3.5, d=4.0, d=4.5, d=5.0",
                      "d=5.5, d=6.0, d=6.5, d=7.0, d=7.5, d=8.0, d=8.5, d=9.0, d=9.5, d=10.0"]
losses_better = [0.40017245751669933, 0.3995972293298057, 0.34642633557325675, 0.41917986151477427,
                1.6491944748884086]
no_params_better = [11.0, 9.0, 6.0, 5.0, 4.0]

stds_string_better_ard = ["d=0.0","d=0.5","d=1.0, d=1.5, d=2.0, d=2.5, d=3.0","d=3.5, d=4.0, d=4.5, d=5.0",
                        "d=5.5, d=6.0, d=6.5, d=7.0, d=7.5, d=8.0, d=8.5, d=9.0, d=9.5, d=10.0"]
losses_better_ard = [0.34573882391026556, 0.3576148296465297, 0.3995403063659276, 0.590529023099644, 2.003326162288243]
no_params_better_ard = [11.0, 8.0, 6.0, 5.0, 4.0]

stds, stds_string, no_params, losses = get_attributes_pareto(data_to_pareto)

# function to plot losses
function plot_loss()
    data = scatter(;y=losses,x=collect(range(lower, upper, length=size(losses)[1])), mode="markers+lines")
    layout = Layout(;title="Reg VADAM",
                        xaxis=attr(title="n_std", showgrid=false, zeroline=false),
                        yaxis=attr(title="loss (test data)"))
    plot(data, layout)
end

plot_loss()

## Pareto
# function with optional parameter which assign user's prefered number of std to find relevant param
function pareto_plot(param, MSEs, lambdas)
    trace1 = scatter(;x=param, y=MSEs, text=lambdas, mode="markers+text", textposition="left center")
    layout = Layout(;title="Pareto frontier VADAM",
                    xaxis=attr(title="Number of parameters in model", showgrid=false, zeroline=false),
                    yaxis=attr(title="MSE (test data)"))
    plot(trace1, layout)
end

pareto_eps_fig = pareto_plot(no_params_better_ard, losses_better_ard, stds_string_better_ard)
pareto_min_fig = pareto_plot(param, MSEs)

# df only with points on Pareto frontier
df = DataFrame(x=losses, y=no_params)
sort!(df, rev=false)
pareto = df[1:1, :]
foreach(row -> row.y < pareto.y[end] && push!(pareto, row), eachrow(df))
#scatter!(pareto.y, pareto.x, label="Pareto Frontier", markersize=8, legend=true)

# final Pareto function to plot Pareto frontier
function pareto_frontier(param, MSEs, lambdas)
    trace1 = scatter(;x=param, y=MSEs, text=lambdas, mode="markers+text", textposition="right",textfont_size=8,
            name="Various models")
    trace2 = scatter(;x=pareto.y, y=pareto.x, mode="markers",name="Pareto frontier")

    data = [trace1, trace2]
    layout = Layout(;title="Pareto frontier VADAM with ARD",
                    xaxis=attr(title="Number of parameters in model", showgrid=false, zeroline=false),
                    yaxis=attr(title="loss (test data)"))
    plot(data, layout)
end

pareto_fig = pareto_frontier(no_params_better_ard, losses_better_ard, stds_string_better_ard)
savefig(pareto_fig, "reg_VADAM_ard_pareto.pdf")
