## THIRD EXPERIMENT
# MILL - MLE + Variational Methods

# main source -> https://ctuavastlab.github.io/Mill.jl/stable/examples/musk/musk/
# doc. Pevný + Ing. Mandlík (main contributors)

## PART 0) SETUP THE ENVINRONEMT
# 1) set your folder and path
# 2) activate environment  '] activate .' and packages
cd("C:/VEJŠKA/Ing/5. ROČNÍK/DIPLOMOVÁ PRÁCE/programy")

using StatsBase
using PlotlyJS
using DelimitedFiles
using FileIO, JLD2, Statistics, Mill, Flux
using Flux: throttle, @epochs
using Mill: reflectinmodel
using Base.Iterators: repeated
using Random
using Flux: Data.DataLoader
using CSV
using LinearAlgebra
using DataFrames
using Flux.Optimise: batchmemaybe, update!
import Zygote: Params, gradient
Random.seed!(42069);

# function to find number of non-zero elements
function non_zeros(A)
    B = []
    for i = 1:length(A)
        if A[i] != 0.0
            push!(B,i)
        end
    end
    return length(B)
end

# function to find and replace minimal element
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

# function to find the element in ε neighborhood
function find_replace_epsilon(A, ϵ)
    for i = 1:length(A)
        if A[i] > -ϵ && A[i] < ϵ
            A[i] = 0.0
        end
    end
    return A
end


## PART 1) LOADING AND SORTING THE DATASET MUSK w/ splitting function
# 1a) changing the loading types
fMat = load("musk.jld2", "fMat")
writedlm("C:/VEJŠKA/Ing/5. ROČNÍK/DIPLOMOVÁ PRÁCE/programy/data/Musk/data.csv", fMat, '\t')

bagids = load("musk.jld2", "bagids")
writedlm("C:/VEJŠKA/Ing/5. ROČNÍK/DIPLOMOVÁ PRÁCE/programy/data/Musk/bagids.csv", bagids, '\t')

y = load("musk.jld2", "y")
writedlm("C:/VEJŠKA/Ing/5. ROČNÍK/DIPLOMOVÁ PRÁCE/programy/data/Musk/labels.csv", y, '\t')

function seqids2bags(bagids)
	c = countmap(bagids)
	Mill.length2bags([c[i] for i in sort(collect(keys(c)))])
end

function csv2mill(problem)
	x=readdlm("$(problem)/data.csv",'\t',Float32)
	bagids = readdlm("$(problem)/bagids.csv",'\t',Int)[:]
	bags = seqids2bags(bagids)
	y = readdlm("$(problem)/labels.csv",'\t',Int)
	y = map(b -> maximum(y[b]), bags)
	(samples = BagNode(ArrayNode(x), bags), labels = y)
end

data = "C:/VEJŠKA/Ing/5. ROČNÍK/DIPLOMOVÁ PRÁCE/programy/data/Musk"
(x,y) = csv2mill(data)

function split_train_test(x,y,ratio)
	A = length(x.data.data[:,1])
	b = length(y)
	n = floor(Int, length(y)*(1-ratio))
	tr_set = zeros(Int, b-n)
	te_set = zeros(Int, n)
	r1 = shuffle(1:b)
	r2 = sample(1:b, n, replace = false)
	q = symdiff(r1, r2)
	tr_set[:] = q
	te_set[:] = r2
	x_train = x[tr_set]
	x_test = x[te_set]
	y_train = y[tr_set]
	y_oh_train = Flux.onehotbatch((y_train.+1)[:],1:2)
	y_test = y[te_set]
	y_oh_test = Flux.onehotbatch((y_test.+1)[:],1:2)
	(x_train,y_train,y_oh_train), (x_test, y_test,y_oh_test)
end

(x_train,y_train,y_oh_train), (x_test, y_test, y_oh_test) = split_train_test(x,y,0.8)


## PART 2) L1 penalization

# 2a) model architecture -> remember, we are in Multi-instance Learning
function get_model()
    model = BagModel(
            Dense(166, 10, Flux.tanh), # input layer with 166 dimensions (of every instance)
            SegmentedMeanMax(10), # pooling/aggregation layer with Meanmax operator
            Chain(Dense(20, 10, Flux.tanh), # hidden layer with tanh activation
                  Dense(10, 2)) # output layer with 2 dimensions (0/1)
                  )
end

model = get_model()

# 2c) params of model and checking how many of them are there
ps = params(model)

function no_of_param(model)
	params = Flux.destructure(model)[1]
	len = length(params)
	println("There is $len trainable parameters in total in the model.")
	return len
end

no_params = no_of_param(model)

# 2b) define targeted loss (classic logitbinarycrossentropy) & optimizer
loss(x_train, y_oh_train) = Flux.logitcrossentropy(model(x_train).data, y_oh_train)

evalcb = throttle(() -> @show(loss(x_train, y_oh_train)), 1)
opt = Flux.ADAM(0.01)
@epochs 10 Flux.train!(loss, params(model), repeated((x_train, y_oh_train), 2000), opt, cb=evalcb)

loss(x_test, y_oh_test)

## PART 2) L1

function train_model_l1_mill(λ)
    model = get_model()
    ps = Flux.params(model)
    loss_l1(x_train, y_oh_train) = Flux.logitcrossentropy(model(x_train).data, y_oh_train) + λ*mapreduce(x -> norm(x,1), +, ps)
    Flux.train!(loss_l1, ps, repeated((x_train, y_oh_train), 10000), opt; cb=()->println("$(loss_l1(x_train, y_oh_train))"))
	last_test_loss = loss_l1(x_test,y_oh_test)
	θ_hat = vec(Flux.destructure(model)[1])
    return θ_hat, last_test_loss
end

# 2a) main training loop - set range for λ and your preferable ε
input_tolerance = 0.001
lambdas_thetas = []
losses_params = []
for λ = 0.0:0.001:0.1
    θ_hat, last_test_loss = train_model_l1_mill(λ)
	θ_pom = deepcopy(θ_hat)
    push!(lambdas_thetas, [λ, θ_hat])
	push!(losses_params, [last_test_loss, non_zeros(find_replace_epsilon(θ_pom, input_tolerance))])
end

# only backup
zaloha, zaloha2 = deepcopy(lambdas_thetas), deepcopy(losses_params)

# function to get attributes if other criterion is met
function make_it_smaller(lambdas_thetas, losses_params)
	A = deepcopy(lambdas_thetas)
	B = deepcopy(losses_params)
	only_lambdas_0_1000 = []
	only_lambdas_otherwise = []
	only_lambdas_string_0_1000 = []
	only_lambdas_string_otherwise = []
	only_thetas_0_1 = []
	only_thetas_otherwise = []
	only_losses_0_1 = []
	only_losses_otherwise = []
	only_num_params_0_1000 = []
	only_num_params_otherwise = []
	for i = 1:length(A)
		if B[i][1] < 1.0 && B[i][2] < 1000.0
			push!(only_lambdas_0_1000, A[i][1])
			push!(only_lambdas_string_0_1000, "λ="*string(A[i][1]))
			push!(only_thetas_0_1, A[i][2])
			push!(only_losses_0_1, B[i][1])
			push!(only_num_params_0_1000, B[i][2])
		else
			push!(only_lambdas_otherwise, A[i][1])
			push!(only_lambdas_string_otherwise, "λ="*string(A[i][1]))
			push!(only_thetas_otherwise, A[i][2])
			push!(only_losses_otherwise, B[i][1])
			push!(only_num_params_otherwise, B[i][2])
		end
	end
	return only_lambdas_0_1000, only_lambdas_string_0_1000, only_thetas_0_1, only_losses_0_1, only_num_params_0_1000,
			only_lambdas_otherwise, only_lambdas_string_otherwise, only_thetas_otherwise, only_losses_otherwise, only_num_params_otherwise
end

lambdas_0_1000, string_0_1000, thetas_0_1, losses_0_1, num_params_0_1000, params_otherwise, string_otherwise, thetas_otherwise,losses_otherwise, params_otherwise = make_it_smaller(lambdas_thetas, losses_params)

# 2b) λ curves and Pareto
# function to get λ, Θ and their corresponding losses and no.of non-zero param within ϵ
function get_attributes(lambdas_thetas, losses_params)
	A = deepcopy(lambdas_thetas)
    B = deepcopy(losses_params)
    only_lambdas = []
	only_lambdas_string = []
    only_thetas = []
	only_losses = []
	only_num_params = []
    for i = 1:length(A)
        push!(only_lambdas, A[i][1])
		push!(only_lambdas_string, "λ="*string(A[i][1]))
        push!(only_thetas, A[i][2])
		push!(only_losses, B[i][1])
		push!(only_num_params, B[i][2])
    end
    return only_lambdas, only_lambdas_string, only_thetas, only_losses, only_num_params
end

only_lambdas, only_lambdas_string, only_thetas, only_losses, only_num_params = get_attributes(lambdas_thetas, losses_params)

# function to plot dependency of λ on parameter values
function plot_lambda_curve()
    rolled = hcat(only_thetas...)
    data = [scatter(;x=only_lambdas, y=rolled[j,:], mode="markers",marker=attr(color="red")) for j=1:size(rolled)[1]]
    layout = Layout(;title="λ curve",xaxis_range=only_lambdas,
                        xaxis=attr(title="λ", showgrid=false, zeroline=false),
                        yaxis=attr(title="parameter values"))
    plot(data, layout)
end

zkouska = plot_lambda_curve()

function plot_lambda_curve_restrict(num)
    rolled = hcat(only_thetas...)
    data = [scatter(;x=only_lambdas[1:num], y=rolled[j,:], mode="markers",marker=attr(color="red")) for j=1:size(rolled)[1]]
    layout = Layout(;title="λ curve (restricted)",xaxis_range=only_lambdas[1:num],
                        xaxis=attr(title="λ", showgrid=false, zeroline=false),
                        yaxis=attr(title="parameter values"))
    plot(data, layout)
end

# function to plot two figs (one origin, second restricted)
function plot_neighborhood(num)
    lambda_fig1 = plot_lambda_curve()
    lambda_fig2 = plot_lambda_curve_restrict(num)
    lambda_fig = [lambda_fig1 lambda_fig2]
    lambda_fig.plot.layout["showlegend"] = false
    lambda_fig.plot.layout["width"] = 1000
    lambda_fig.plot.layout["height"] = 600
    lambda_fig
end

lambda_fig_mill = plot_neighborhood(10)
savefig(lambda_fig_mill, "lambda_curve_mill_kuličky.pdf")

# plot of optinional ε neighborhood
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
    eps_fig1 = sparsity_lambda(only_lambdas, only_thetas, 0.001)
    eps_fig2 = sparsity_lambda(only_lambdas, only_thetas, 0.002)
    eps_fig3 = sparsity_lambda(only_lambdas, only_thetas, 0.005)
    eps_fig4 = sparsity_lambda(only_lambdas, only_thetas, 0.01)
    eps_fig = [eps_fig1 eps_fig2; eps_fig3 eps_fig4]
    eps_fig.plot.layout["showlegend"] = false
    eps_fig.plot.layout["width"] = 1000
    eps_fig.plot.layout["height"] = 600
    eps_fig
end

eps_fig = plot_neighborhood()
savefig(eps_fig, "eps_fig_sparse_reg_0_1_mill.pdf")

##PARETO

function pareto_plot(param, MSEs, lambdas)
    trace1 = scatter(;x=param, y=MSEs, text=lambdas, mode="markers+text", textposition="left center")
    layout = Layout(;title="Pareto frontier",
                    xaxis=attr(title="Number of parameters in model", showgrid=false, zeroline=false),
                    yaxis=attr(title="loss (test data)"))
    plot(trace1, layout)
end

# backup again
A, B, C = deepcopy(only_num_params), deepcopy(only_losses), deepcopy(only_lambdas_string)

pareto_eps_fig = pareto_plot(only_num_params, only_losses, only_lambdas_string)

df = DataFrame(x=only_losses, y=only_num_params)
sort!(df, rev=false)
pareto = df[1:1, :]
foreach(row -> row.y < pareto.y[end] && push!(pareto, row), eachrow(df))
#scatter!(pareto.y, pareto.x, label="Pareto Frontier", markersize=8, legend=true)

# function to plot pareto frontier
function pareto_frontier(param, MSEs, lambdas)
    trace1 = scatter(;x=param, y=MSEs, text=lambdas, mode="markers+text", textposition="right",textfont_size=8,
            name="Various models")
    trace2 = scatter(;x=pareto.y, y=pareto.x, mode="markers",name="Pareto frontier")

    data = [trace1, trace2]
    layout = Layout(;title="Pareto frontier (zoomed), ϵ=0.001",
                    xaxis=attr(title="Number of parameters in model", showgrid=false, zeroline=false),
                    yaxis=attr(title="loss (test data)"))
    plot(data, layout)
end

pareto_fig = pareto_frontier(only_num_params, only_losses, only_lambdas_string)
savefig(pareto, "pareto_L1_eps_mill_0_1.pdf")

pareto_fig_0_1000 = pareto_frontier(num_params_0_1000, losses_0_1, string_0_1000)
savefig(pareto_fig_0_1000, "pareto_L1_eps_mill_0_1_0_1000.pdf")

savefig(pareto_fig_MILL_all, "pareto_fig_MILL_all.png")
pareto_fig_MILL_all

pareto_MILL = hcat(pareto_fig_0_1000, pareto_fig_MILL_all)
## PART 3) - VADAM

# initialize Gaussian posterior structure
struct GaussPs
    ps
    σps
end

# initialize ard prior
loss_ard(ps, ψ) = 5e-1*sum(sum(p .^ 2 .* ψ) for (p, ψ) in zip(ps, ψ))

# Author's proposed variational method
function vtrain_ardvb!(loss, model, data, opt, N; cb = () -> (), σ0=1e-7, ψ0=1e-4, clip=0.0)
    ps = Flux.params(model)
    ps_mean = deepcopy(ps)
    σps = deepcopy(ps)
    ψ = map(p->copy(p[:, 1]), ps)
    foreach(x->x.=σ0, σps)
    foreach(x->x.=ψ0, ψ)

    losses = []
    k = 1
    for d in data
        for i=1:length(ps)
            ps_mean[i] .= ps[i]
            ps[i] .= ps[i] .+ randn(size(ps[i])) .* σps[i]
        end
        display(loss(batchmemaybe(d)...) + loss_ard(ps, ψ))
        px = [p for p in Flux.params(model)]
        gs = Flux.gradient(()->loss(batchmemaybe(d)...) + loss_ard(px, ψ), ps)

        if clip>0.0
            for i=1:length(ps)
                clipnorm!(gs[ps[i]],clip)
            end
        end

        for i=1:length(ps)
            ps[i].=ps_mean[i]
        end
        Flux.Optimise.update!(opt, ps, gs)

        for i=1:length(ps)
            σps[i].=1.0./sqrt.(N*opt.state[ps[i]][2] .+ 1e-6)
        end

        for (ψ, μ, σ) in zip(ψ, ps, σps)
            ψ .= 1.0 ./ sum(μ.^2 .+ σ.^2)
        end
        #loss_actual = loss_all_ard(data)
        # println("Epoch $k")
        k = k+1
        #push!(losses, loss_actual)
    end
    return GaussPs(ps,σps), ψ
end

model = get_model()
loss(x_train, y_oh_train) = Flux.logitcrossentropy(model(x_train).data, y_oh_train)

lr = 0.01
#opt = ADAM(lr, (0.9, 0.999))
opt = Flux.ADAM(lr)

qps, ψ = vtrain_ardvb!(loss, model, repeated((x_train, y_oh_train), 10000), opt, size(y_oh_train, 2); cb = ()->())

μ, σμ = qps.ps, qps.σps

## TRAIN VIA VARIATIONAL INFERENCE
# σ is an std already
function find_replace_n_std(trained_μ_params, trained_σμ_params, n_std)
	copy_μ = deepcopy(trained_μ_params)
	copy_σμ = deepcopy(trained_σμ_params)
	for i in 1:length(copy_μ)
		if (typeof(copy_μ[i]) == Matrix{Float32}) == true
			for l in 1:size(copy_μ[i])[1]
				for k in 1:size(copy_μ[i])[2]
					##ZEPTAT SE!!!
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

# function to get vector from struture of params
function extract_posterior_means(trained_μ_params)
	copy = deepcopy(trained_μ_params)
	μ_vec = []

	for i in 1:length(copy)
		append!(μ_vec, vcat(copy[i]...))
	end
	return μ_vec
end

# function to get vector from struture of params
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

# function to get vector from struture of params
function extract_posterior_precision(ψ)
	copy = deepcopy(ψ)
	ψ_posterior = []

	for i in 1:length(copy)
		append!(ψ_posterior, vcat(copy[i]...))
	end
	return ψ_posterior
end

## continuing

# loop for pareto in MIL
function MIL_to_pareto(trained_μ_params, trained_σμ_params, n)
	n_std = n
	updated_means = find_replace_n_std(trained_μ_params, trained_σμ_params, n_std)
	model = get_model()
	Flux.loadparams!(model,updated_means)
	loss = Flux.logitcrossentropy(model(x_test).data, y_oh_test)
	means_vec = extract_posterior_means(updated_means)
	return n_std, non_zeros(means_vec), loss
end

## Pareto data loading
data_to_pareto = []
lower, step_neco, upper = 0.0, 0.0001, 0.015
for n in lower:step_neco:upper
	n_std, not_null, loss_test = MIL_to_pareto(μ, σμ, n)
	push!(data_to_pareto, [n_std, not_null, loss_test])
end

# funtion to get attributes from fields from loop
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

stds, stds_string, no_params, losses = get_attributes_pareto(data_to_pareto)

# ploting loss of gained results
function plot_loss()
    data = scatter(;y=losses,x=collect(range(lower, upper, length=size(losses)[1])), mode="markers+lines")
    layout = Layout(;title="MILL VADAM",
                        xaxis=attr(title="n_std", showgrid=false, zeroline=false),
                        yaxis=attr(title="loss (test data)"))
    plot(data, layout)
end

plot_loss()


# Posterior qps and precision in vector form
μ_vec,σμ_vec = extract_posterior_params(qps)
ψ_posterior_vec = extract_posterior_precision(ψ)

# function to reveal relevant components of parameter vector
function finding_relevant_parameters_mill(means, deviations, n_std)
    n_std = n_std
	μ_hat = deepcopy(means)
	σ_hat = deepcopy(deviations)
    μ_plot = []
    σμ_plot = []
    for i in 1:size(μ_hat)[1]
        if μ_hat[i]> - n_std*σ_hat[i] && μ_hat[i] < n_std*σ_hat[i]
            push!(μ_plot, convert(Float64, μ_hat[i]))
			push!(σμ_plot, convert(Float64, σ_hat[i]))
			#push!(μ_plot, μ_hat[i])
            #push!(σμ_plot, σ_hat[i])
        else
            push!(μ_plot, convert(Float64,0.0))
            push!(σμ_plot, convert(Float64,0.0))
			#push!(μ_plot, 0.0)
            #push!(σμ_plot, 0.0)
        end
    end
	println("Found $(non_zeros(μ_plot)) relevant parameters within $n_std std interval.")
    return n_std, μ_plot, σμ_plot
end

## Pareto
# function with optional parameter which assign user's prefered number of std to find relevant param
function pareto_plot(param, MSEs, lambdas)
    trace1 = scatter(;x=param, y=MSEs, text=lambdas, mode="markers+text", textposition="right center")
    layout = Layout(;title="Pareto frontier (MIL)",
                    xaxis=attr(title="Number of parameters in model", showgrid=false, zeroline=false),
                    yaxis=attr(title="loss (test data)"))
    plot(trace1, layout)
end

pareto_eps_fig = pareto_plot(no_params, losses, stds_string)
pareto_min_fig = pareto_plot(param, MSEs)

df = DataFrame(x=losses, y=no_params)
sort!(df, rev=false)
pareto = df[1:1, :]
foreach(row -> row.y < pareto.y[end] && push!(pareto, row), eachrow(df))
#scatter!(pareto.y, pareto.x, label="Pareto Frontier", markersize=8, legend=true)

function pareto_frontier(param, MSEs, lambdas)
    trace1 = scatter(;x=param, y=MSEs, text=lambdas, mode="markers+text", textposition="right",textfont_size=8,
            name="Various models")
    trace2 = scatter(;x=pareto.y, y=pareto.x, mode="markers",name="Pareto frontier")

    data = [trace1, trace2]
    layout = Layout(;title="Pareto frontier",
                    xaxis=attr(title="Number of parameters in model", showgrid=false, zeroline=false),
                    yaxis=attr(title="loss (test data)"))
    plot(data, layout)
end

pareto_fig = pareto_frontier(no_params, losses, stds_string)
savefig(pareto_fig, "mill_pareto.pdf")

## Methods evaluation
# i have saved data -> or pareto save to df (as after loop)

function evaluate_methods_mill()
    trace1 = scatter(;x=[821.0, 809.0,1257.0], y=[0.768, 0.77, 0.693], mode="markers+lines",name="L1")
    trace2 = scatter(;x=[991.0, 792.0], y=[0.681,0.683], mode="markers+lines",name="VADAM with prior")
    data = [trace1, trace2]
    layout = Layout(;title="Evaluation of methods using Pareto frontiers",
                    xaxis=attr(title="Number of parameters in model", showgrid=false, zeroline=false),
                    yaxis=attr(title="loss (test data)"))
    plot(data, layout)
end

# saving 
evaluation_mill = evaluate_methods_mill()
savefig(evaluation_mill, "experiment3_eval.pdf")
