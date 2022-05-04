## SECOND EXPERIMENT
# MLE, L1 + Variational Methods

## PART 0) SETUP THE ENVINRONEMT
# 1) set your folder and path
# 2) activate environment  '] activate .' and packages

cd("C:/VEJŠKA/Ing/5. ROČNÍK/DIPLOMOVÁ PRÁCE/programy")
##SET WD and environment "] activate .
using Base.Iterators
using PlotlyJS
using UCIData
using CSV, DataFrames
using Flux, Random
using Flux: Data.DataLoader
using Flux: @epochs
using IterTools: ncycle
using LinearAlgebra
using StatsBase
using Flux.Optimise: batchmemaybe, update!
import Zygote: Params, gradient
Random.seed!(42069);

# function to find non-zero elements
function non_zeros(A)
    B = []
    for i = 1:length(A)
        if A[i] != 0.0
            push!(B,i)
        end
    end
    return length(B)
end

# function to find the minimal element and zero it out
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

# function to find element from ε neighborhood
function find_replace_epsilon(A, ϵ)
    for i = 1:length(A)
        if A[i] > -ϵ && A[i] < ϵ
            A[i] = 0.0
        end
    end
    return A
end


## PART 1) GETTING DATA

# dataloading
df = UCIData.dataset("iris")
rename!(df,["index","sepal_len", "sepal_wid", "petal_len", "petal_wid", "class"])
df = select!(df, Not("index"))
setosa, versicolor, virginica = groupby(df, :class)

function plot_data()
    p1 = scatter(;x=df.sepal_len, y=df.sepal_wid, color="blue", mode="markers", name="sepal")
    p2 = scatter(;x=df.petal_len, y=df.petal_wid, color="red", mode="markers", name="petal")
    data = [p1, p2]
    layout = Layout(;title="len vs wid",
                    xaxis=attr(title="width", showgrid=false, zeroline=false),
                    yaxis=attr(title="length"))
    plot(data, layout)
end

plot_data()


# 1a) Data preparation

function data_preprocess(df, ratio, batch_size)
    data = deepcopy(df)
    data = data[shuffle(1:end),:]
    split_ratio = ratio
    idx = Int(floor(size(df,1)*split_ratio))
    data_train = data[1:idx,:]
    data_test = data[idx+1:end,:]

    x_train = hcat([data_train[:,j] for j=1:4]...)
    x_test = hcat([data_test[:,j] for j=1:4]...)

    onehot(d) = Flux.onehotbatch(d[:,end], unique(df.class))
    y_train = onehot(data_train)
    y_test = onehot(data_test)

    batch_size = batch_size
    train_dataloader = DataLoader((x_train',y_train), batchsize=batch_size, shuffle=true)
    test_dataloader = DataLoader((x_test',y_test), batchsize=batch_size)
    return x_train, y_train, train_dataloader, x_test, y_test, test_dataloader
end

x_train, y_train, train_dataloader, x_test, y_test, test_dataloader = data_preprocess(df, 0.8, 4)

# 1b) loading to VADAM
function create_data(n_epochs)
    n_epochs = n_epochs
    data_train = repeated((x_train', y_train), n_epochs)
    data_test = repeated((x_test', y_test), n_epochs)
    return n_epochs, data_train, data_test
end

n_epochs, data_train, data_test = create_data(100)

## PART 2) SETTING THE MODEL, loss, etc.

# 2a) Model architecture
function get_model()
    nn = Chain(
           Dense(4,8,relu),
           Dense(8,3),
           softmax
           )
end

model = get_model()

# 2b) set loss and initialize empty fields to store the values during training
loss(x,y) = Flux.Losses.logitbinarycrossentropy(model(x), y)

function initialize_fields()
    train_losses = []
    test_losses = []
    train_acces = []
    test_acces = []
    return train_losses, test_losses, train_acces, test_acces
end

train_losses, test_losses, train_acces, test_acces = initialize_fields()

# 2c) set optimizer
lr = 0.001
opt = ADAM(lr, (0.9, 0.999))

# 2d) set callbacks
function loss_all(data_loader)
    sum([loss(x, y) for (x,y) in data_loader]) / length(data_loader)
end

function acc(data_loader)
    f(x) = Flux.onecold(cpu(x))
    acces = [sum(f(model(x)) .== f(y)) / size(x,2)  for (x,y) in data_loader]
    sum(acces) / length(data_loader)
end

callbacks = [
        () -> push!(train_losses, loss_all(train_dataloader)),
        () -> push!(test_losses, loss_all(test_dataloader)),
        () -> push!(train_acces, acc(train_dataloader)),
        () -> push!(test_acces, acc(test_dataloader)),
        ]

# 2e) set params
ps = Flux.params(model)

# 2f) train the model
function train(epochs)
    @epochs epochs Flux.train!(loss, ps, train_dataloader, opt, cb = callbacks)

    @show train_loss = loss_all(train_dataloader)
    @show test_loss = loss_all(test_dataloader)
    @show train_acc = acc(train_dataloader)
    @show test_acc = acc(test_dataloader)
end

train(1000)

# 2g) evaluate via plots
function plot_losses(input_train_loss, input_test_loss)
    p1 = scatter(;y=input_train_loss, color="blue", mode="lines", name="train")
    p2 = scatter(;y=input_test_loss, color="red", mode="lines", name="test")
    data = [p1, p2]
    layout = Layout(;title="Losses",
                    xaxis=attr(title="epochs", showgrid=false, zeroline=false),
                    yaxis=attr(title="loss"))
    plot(data, layout)
end

losses = plot_losses(train_losses, test_losses)

function plot_acc(input_train_acc, input_test_acc)
    p1 = scatter(;y=input_train_acc, color="blue", mode="lines", name="train")
    p2 = scatter(;y=input_test_acc, color="red", mode="lines", name="test")
    data = [p1, p2]
    layout = Layout(;title="Accuracy",
                    xaxis=attr(title="epochs", showgrid=false, zeroline=false),
                    yaxis=attr(title="acc"))
    plot(data, layout)
end

accuracy = plot_acc(train_acces, test_acces)
fig_basis = hcat(losses, accuracy)
savefig(fig_basis, "iris_acc_losses_30k.pdf")

loss(x,y) = Flux.Losses.logitbinarycrossentropy(model(x), y) + λ*mapreduce(x -> norm(x,1), +, ps)

## PART 3) L1 & Pareto

function train_model_l1_logistic(n_epochs, λ)
    model = get_model()
    ps = Flux.params(model)
    loss(x,y) = Flux.Losses.logitbinarycrossentropy(model(x), y) + λ*mapreduce(x -> norm(x,1), +, ps)
    Flux.@epochs n_epochs Flux.train!(loss, ps, train_dataloader, opt; cb=()->println("learning"))
    last_test_loss = loss(x_test',y_test)
    θ_hat = vec(Flux.destructure(model)[1])
    return θ_hat, last_test_loss
end


# 3a) main training loop - set range for λ and your preferable ε
input_tolerance = 0.001
lambdas_thetas = []
losses_params = []
for λ = 0.0:0.0005:0.04
    θ_hat, last_test_loss = train_model_l1_logistic(1000, λ)
    θ_pom = deepcopy(θ_hat)
    push!(lambdas_thetas, [λ, θ_hat])
    push!(losses_params, [last_test_loss, non_zeros(find_replace_epsilon(θ_pom, input_tolerance))])
end

# 3b) λ curves

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

# function to plot dependency of λ on parameter values
function plot_lambda_curve()
    rolled = hcat(only_thetas...)
    data = [scatter(;x=only_lambdas, y=rolled[j,:], mode="markers+lines",name="param $j",marker=attr(color="red")) for j=1:size(rolled)[1]]
    layout = Layout(;title="λ curve",xaxis_range=only_lambdas,
                        xaxis=attr(title="λ", showgrid=false, zeroline=false),
                        yaxis=attr(title="parameter values"))
    plot(data, layout)
end

zkouska = plot_lambda_curve()

function plot_lambda_curve_restrict(num)
    rolled = hcat(only_thetas...)
    data = [scatter(;x=only_lambdas[1:num], y=rolled[j,:], mode="markers+lines",name="param $j",marker=attr(color="red")) for j=1:size(rolled)[1]]
    layout = Layout(;title="λ curve (restricted)",xaxis_range=only_lambdas[1:num],
                        xaxis=attr(title="λ", showgrid=false, zeroline=false),
                        yaxis=attr(title="parameter values"))
    plot(data, layout)
end

function plot_neighborhood(num)
    lambda_fig1 = plot_lambda_curve()
    lambda_fig2 = plot_lambda_curve_restrict(num)
    lambda_fig = [lambda_fig1 lambda_fig2]
    lambda_fig.plot.layout["showlegend"] = false
    lambda_fig.plot.layout["width"] = 1000
    lambda_fig.plot.layout["height"] = 600
    lambda_fig
end

lambda_fig_1 = plot_neighborhood(20)
savefig(lambda_fig_1, "lambda_curve_LR_better_kuličky_lines.pdf")

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
savefig(eps_fig, "eps_fig_sparse_reg_0_1_smoother_LR_DP.pdf")

# L1 Pareto

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

# function to gain different values based on ε
function get_different_epsilon(input_theta, tolerance)
    num_param = []
    ϵ = tolerance
    C = deepcopy(input_theta)
    for i in 1:length(C)
        push!(num_param, non_zeros(find_replace_epsilon(C[i], ϵ)))
    end
    return num_param
end

only_num_params_00001 = get_different_epsilon(only_thetas, 0.0001)

# function to plot pareto frontier
function pareto_plot(param, MSEs, lambdas)
    trace1 = scatter(;x=param, y=MSEs, text=lambdas, mode="markers+text", textposition="right center")
    layout = Layout(;title="Pareto frontier",
                    xaxis=attr(title="Number of parameters in model", showgrid=false, zeroline=false),
                    yaxis=attr(title="loss (test data)"))
    plot(trace1, layout)
end


pareto_eps_fig = pareto_plot(only_num_params, only_losses, only_lambdas_string)

input_tolerance = 0.001

# dataframe to find out the point on Pareto frontier
df = DataFrame(x=only_losses, y=only_num_params)
sort!(df, rev=false)
pareto = df[1:1, :]
foreach(row -> row.y < pareto.y[end] && push!(pareto, row), eachrow(df))
#scatter!(pareto.y, pareto.x, label="Pareto Frontier", markersize=8, legend=true)

# complete Pareto function
function pareto_frontier(param, MSEs, lambdas)
    trace1 = scatter(;x=param, y=MSEs, text=lambdas, mode="markers+text", textposition="top right",textfont_size=8, angle=45,
            name="Various models")
    trace2 = scatter(;x=pareto.y, y=pareto.x, mode="markers",name="Pareto frontier")

    data = [trace1, trace2]
    layout = Layout(;title="Pareto frontier, ϵ=$input_tolerance",
                    xaxis=attr(title="Number of parameters in model", showgrid=false, zeroline=false),
                    yaxis=attr(title="loss (test data)"))
    plot(data, layout)
end

pareto_fig = pareto_frontier(only_num_params, only_losses, only_lambdas_string)
savefig(pareto_fig, "pareto_L1_eps_LR_0_1_00001.pdf")


## PART 4) VADAM

## TRAIN VIA VARIATIONAL INFERENCE
# σ is an std already
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


function extract_posterior_means(trained_μ_params)
	copy = deepcopy(trained_μ_params)
	μ_vec = []

	for i in 1:length(copy)
		append!(μ_vec, vcat(copy[i]...))
	end
	return μ_vec
end

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

function extract_posterior_precision(ψ)
	copy = deepcopy(ψ)
	ψ_posterior = []

	for i in 1:length(copy)
		append!(ψ_posterior, vcat(copy[i]...))
	end
	return ψ_posterior
end

# initialize empty gaussian structure
struct GaussPs
    ps
    σps
end

# initialize the ard prior
loss_ard(ps, ψ) = 5e-1*sum(sum(p .^ 2 .* ψ) for (p, ψ) in zip(ps, ψ))

# function to perfom author's proposed variational inference via his algorithm
function vtrain_ardvb!(loss, model, data, opt, N; cb = () -> (), σ0=1e-8, ψ0=1e-5, clip=0.0)
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
loss(x,y) = Flux.Losses.logitbinarycrossentropy(model(x), y)
ps = Flux.params(model)
ψ = deepcopy(ps)

lr = 0.001
opt = ADAM(lr, (0.9, 0.999))

qps, ψ = vtrain_ardvb!(loss, model, repeated((x_train', y_train), 20000), opt, size(y_train, 2); cb = ()->())

μ, σμ = qps.ps, qps.σps

foreach(p->display(p), Flux.params(model))

# function for looping
function LR_to_pareto(trained_μ_params, trained_σμ_params, n)
	n_std = n
	updated_means = find_replace_n_std(trained_μ_params, trained_σμ_params, n_std)
    model = get_model()
    Flux.loadparams!(model,updated_means)
    loss_last = Flux.Losses.logitbinarycrossentropy(model(x_test'), y_test)
	return n_std, non_zeros(extract_posterior_means(updated_means)), loss_last
end

## Pareto data loading & looping
data_to_pareto = []
lower, step_neco, upper = 0.0, 0.1, 3.0
for n in lower:step_neco:upper
	n_std, not_null, loss_test = LR_to_pareto(μ, σμ, n)
	push!(data_to_pareto, [n_std, not_null, loss_test])
end

# function to get attributes from loop
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

# function to plot losses gained from loop
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


## Pareto
# function with optional parameter which assign user's prefered number of std to find relevant param
function pareto_plot(param, MSEs, lambdas)
    trace1 = scatter(;x=param, y=MSEs, text=lambdas, mode="markers+text", textposition="left center")
    layout = Layout(;title="Pareto frontier",
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
    trace1 = scatter(;x=param, y=MSEs, text=lambdas, mode="markers+text",textangle=60,textposition="top center",textfont_size=8,
            name="Various models")
    trace2 = scatter(;x=pareto.y, y=pareto.x, mode="markers",name="Pareto frontier")

    data = [trace1, trace2]
    layout = Layout(;title="Pareto frontier",
                    xaxis=attr(title="Number of parameters in model", showgrid=false, zeroline=false),
                    yaxis=attr(title="loss (test data)"),
                    width=1200, height=600)
    plot(data, layout)
end

pareto_fig = pareto_frontier(no_params, losses, stds_string)
savefig(pareto_fig, "LR_Vadam_pareto.pdf")

## PART 5) Methods evaluation
# 1) Pareto from L1 (the first one in df zeroes out the whole model (irrelevant to use))
# 2) Saved values from VADAM prior (or df_vadam .. the exact procedure as in L1)

function evaluate_methods()
    trace1 = scatter(;x=pareto.y, y=pareto.x, mode="markers+lines",name="L1")
    trace2 = scatter(;x=[3.0], y=[0.762528], mode="markers+lines",name="VADAM with prior")
    data = [trace1, trace2]
    layout = Layout(;title="Evaluation of methods using Pareto frontiers",
                    xaxis=attr(title="Number of parameters in model", showgrid=false, zeroline=false),
                    yaxis=attr(title="loss (test data)"))
    plot(data, layout)
end

evaluation = evaluate_methods()
savefig(evaluation, "experiment2_eval.pdf")
